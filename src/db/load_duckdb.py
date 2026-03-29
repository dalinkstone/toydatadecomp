"""Load all data into DuckDB.

Creates tables from parquet files, a VIEW over transaction files (auto-detecting
parquet / csv.zst / csv), indexes, and analytical views.  The VIEW approach keeps
the .duckdb file small while DuckDB scans transaction columns on disk with
predicate pushdown.

Usage:
    python src/db/load_duckdb.py --db-path data/db/cvs_analytics.duckdb
    python src/db/load_duckdb.py --materialize   # materialize expensive views
"""

import os
import time
from pathlib import Path

import click
import duckdb
from rich.console import Console
from rich.table import Table

console = Console()


# -- Analytical view SQL (shared between load and schema.sql) ----------------

ANALYTICAL_VIEWS = {
    "customer_purchase_summary": """
        SELECT
            customer_id,
            COUNT(*)                    AS total_transactions,
            COUNT(DISTINCT product_id)  AS unique_products,
            COUNT(DISTINCT store_id)    AS unique_stores,
            SUM(total)                  AS total_spend,
            AVG(total)                  AS avg_transaction,
            MIN(date)                   AS first_purchase,
            MAX(date)                   AS last_purchase
        FROM transactions
        GROUP BY customer_id
    """,
    "product_monthly_sales": """
        SELECT
            product_id,
            DATE_TRUNC('month', date)   AS month,
            SUM(quantity)               AS units_sold,
            SUM(total)                  AS revenue
        FROM transactions
        GROUP BY product_id, DATE_TRUNC('month', date)
    """,
    "store_performance": """
        SELECT
            store_id,
            COUNT(DISTINCT customer_id) AS unique_customers,
            COUNT(*)                    AS total_transactions,
            SUM(total)                  AS total_revenue,
            AVG(total)                  AS avg_transaction_value
        FROM transactions
        GROUP BY store_id
    """,
    "customer_product_matrix": """
        SELECT
            customer_id,
            product_id,
            COUNT(*)    AS purchase_count,
            SUM(total)  AS total_spent_on_product,
            MAX(date)   AS last_purchased
        FROM transactions
        GROUP BY customer_id, product_id
    """,
    "product_cooccurrence": """
        SELECT
            t1.product_id AS product_a,
            t2.product_id AS product_b,
            COUNT(*)      AS co_purchase_count
        FROM transactions t1
        JOIN transactions t2
            ON  t1.customer_id = t2.customer_id
            AND t1.date        = t2.date
            AND t1.product_id  < t2.product_id
        GROUP BY t1.product_id, t2.product_id
    """,
}

EXPENSIVE_VIEWS = {"customer_product_matrix", "product_cooccurrence"}


# -- Helpers -----------------------------------------------------------------

def detect_transaction_format(data_dir: str) -> tuple[str, str]:
    """Auto-detect transaction file format.

    Checks for .parquet first, then .csv.zst, then .csv.
    Returns (glob_pattern, duckdb_reader_function).
    """
    txn_dir = Path(data_dir) / "synthetic" / "transactions"

    for ext, reader in [
        ("*.parquet", "read_parquet"),
        ("*.csv.zst", "read_csv"),
        ("*.csv", "read_csv"),
    ]:
        if list(txn_dir.glob(ext)):
            return str(txn_dir / ext), reader

    raise FileNotFoundError(f"No transaction files found in {txn_dir}")


def _resolve(data_dir: str, *parts: str) -> str:
    """Build an absolute path string from data_dir + parts."""
    return str(Path(data_dir).resolve().joinpath(*parts))


# -- Load command ------------------------------------------------------------

@click.command()
@click.option("--db-path", default="data/db/cvs_analytics.duckdb",
              help="DuckDB database path.")
@click.option("--data-dir", default="data",
              help="Root data directory.")
@click.option("--materialize", is_flag=True, default=False,
              help="Materialize expensive views (customer_product_matrix, "
                   "product_cooccurrence) as tables.")
def main(db_path: str, data_dir: str, materialize: bool) -> None:
    """Load all data into DuckDB."""
    console.print(f"[bold]Loading data into DuckDB -> {db_path}[/bold]\n")

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = duckdb.connect(db_path)
    t0 = time.time()

    # ── Stores ──────────────────────────────────────────────────────────
    console.print("[cyan]Loading stores...[/cyan]")
    stores_path = _resolve(data_dir, "real", "stores.parquet")
    con.execute("DROP TABLE IF EXISTS stores")
    con.execute(f"""
        CREATE TABLE stores AS
        SELECT CAST(store_id AS INTEGER) AS store_id,
               name, address, city, state, zip_code,
               latitude, longitude, phone,
               store_type, hours_mon_fri, hours_sat, hours_sun
        FROM read_parquet('{stores_path}')
    """)
    stores_n = con.execute("SELECT COUNT(*) FROM stores").fetchone()[0]
    console.print(f"  stores: {stores_n:,} rows")

    # ── Products ────────────────────────────────────────────────────────
    console.print("[cyan]Loading products...[/cyan]")
    products_path = _resolve(data_dir, "real", "products.parquet")
    con.execute("DROP TABLE IF EXISTS products")
    con.execute(f"""
        CREATE TABLE products AS
        SELECT * FROM read_parquet('{products_path}')
    """)
    products_n = con.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    console.print(f"  products: {products_n:,} rows")

    # ── Customers ───────────────────────────────────────────────────────
    console.print("[cyan]Loading customers...[/cyan]")
    cust_glob = _resolve(data_dir, "synthetic", "customers", "*.parquet")
    con.execute("DROP TABLE IF EXISTS customers")
    con.execute(f"""
        CREATE TABLE customers AS
        SELECT * FROM read_parquet('{cust_glob}')
    """)
    customers_n = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    console.print(f"  customers: {customers_n:,} rows")

    # ── Coupon clips ────────────────────────────────────────────────────
    clips_dir = Path(data_dir) / "synthetic" / "coupon_clips"
    clips_n = 0
    if list(clips_dir.glob("*.parquet")):
        console.print("[cyan]Loading coupon clips...[/cyan]")
        clips_glob = _resolve(data_dir, "synthetic", "coupon_clips", "*.parquet")
        con.execute("DROP TABLE IF EXISTS coupon_clips")
        con.execute(f"""
            CREATE TABLE coupon_clips AS
            SELECT * FROM read_parquet('{clips_glob}')
        """)
        clips_n = con.execute("SELECT COUNT(*) FROM coupon_clips").fetchone()[0]
        console.print(f"  coupon_clips: {clips_n:,} rows")
    else:
        console.print("[yellow]  No coupon clips data found, skipping.[/yellow]")

    # ── Transactions (VIEW) ─────────────────────────────────────────────
    console.print("[cyan]Creating transactions view...[/cyan]")
    txn_pattern, reader_fn = detect_transaction_format(data_dir)
    con.execute("DROP VIEW IF EXISTS transactions")
    con.execute("DROP TABLE IF EXISTS transactions")
    # ignore_errors handles truncated lines from interrupted zstd writes
    extra_opts = ", ignore_errors=true" if reader_fn == "read_csv" else ""
    con.execute(f"""
        CREATE VIEW transactions AS
        SELECT * FROM {reader_fn}('{txn_pattern}'{extra_opts})
    """)
    console.print(f"  transactions: VIEW via {reader_fn}('{txn_pattern}')")

    # ── Indexes ─────────────────────────────────────────────────────────
    console.print("[cyan]Creating indexes...[/cyan]")
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_customers_customer_id ON customers(customer_id)",
        "CREATE INDEX IF NOT EXISTS idx_customers_state ON customers(state)",
        "CREATE INDEX IF NOT EXISTS idx_products_product_id ON products(product_id)",
        "CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)",
    ]
    if clips_n > 0:
        indexes += [
            "CREATE INDEX IF NOT EXISTS idx_coupon_clips_loyalty ON coupon_clips(loyalty_number)",
            "CREATE INDEX IF NOT EXISTS idx_coupon_clips_product ON coupon_clips(product_id)",
        ]
    for sql in indexes:
        con.execute(sql)
    console.print(f"  {len(indexes)} indexes created")

    # ── Analytical views ────────────────────────────────────────────────
    console.print("[cyan]Creating analytical views...[/cyan]")
    for name, query in ANALYTICAL_VIEWS.items():
        con.execute(f"DROP VIEW IF EXISTS {name}")
        con.execute(f"DROP TABLE IF EXISTS {name}")

        if materialize and name in EXPENSIVE_VIEWS:
            console.print(f"  Materializing {name} (this may take a while)...")
            con.execute(f"CREATE TABLE {name} AS {query}")
            n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            console.print(f"  {name}: TABLE ({n:,} rows)")
        else:
            con.execute(f"CREATE VIEW {name} AS {query}")
            console.print(f"  {name}: VIEW")

    elapsed = time.time() - t0

    # ── Summary ─────────────────────────────────────────────────────────
    console.print()
    console.print("[bold green]Load complete[/bold green]")
    console.print(f"Time: {elapsed:.1f}s\n")

    tbl = Table(title="Database Contents")
    tbl.add_column("Name", style="cyan")
    tbl.add_column("Type", style="magenta")
    tbl.add_column("Rows", justify="right", style="green")

    tbl.add_row("stores", "TABLE", f"{stores_n:,}")
    tbl.add_row("products", "TABLE", f"{products_n:,}")
    tbl.add_row("customers", "TABLE", f"{customers_n:,}")
    if clips_n > 0:
        tbl.add_row("coupon_clips", "TABLE", f"{clips_n:,}")
    tbl.add_row("transactions", "VIEW", "(on disk)")
    for name in ANALYTICAL_VIEWS:
        vtype = "TABLE" if (materialize and name in EXPENSIVE_VIEWS) else "VIEW"
        tbl.add_row(name, vtype, "")
    console.print(tbl)

    # Sample queries
    console.print("\n[bold]Sample queries:[/bold]")

    console.print("\n[dim]Top 5 product categories:[/dim]")
    r = con.execute(
        "SELECT category, COUNT(*) AS n FROM products "
        "GROUP BY category ORDER BY n DESC LIMIT 5"
    ).fetchdf()
    console.print(r.to_string(index=False))

    console.print("\n[dim]Customer count by state (top 5):[/dim]")
    r = con.execute(
        "SELECT state, COUNT(*) AS n FROM customers "
        "GROUP BY state ORDER BY n DESC LIMIT 5"
    ).fetchdf()
    console.print(r.to_string(index=False))

    console.print("\n[dim]Store count by state (top 5):[/dim]")
    r = con.execute(
        "SELECT state, COUNT(*) AS n FROM stores "
        "GROUP BY state ORDER BY n DESC LIMIT 5"
    ).fetchdf()
    console.print(r.to_string(index=False))

    con.close()

    db_mb = os.path.getsize(db_path) / (1024 * 1024)
    console.print(f"\n[bold]Database file size: {db_mb:.1f} MB[/bold]")


# -- Validate command --------------------------------------------------------

@click.command("validate")
@click.option("--db-path", default="data/db/cvs_analytics.duckdb",
              help="DuckDB database path.")
def validate(db_path: str) -> None:
    """Run data validation checks against the loaded DuckDB database."""
    if not os.path.exists(db_path):
        console.print(f"[red]Database not found: {db_path}[/red]")
        console.print("Run the load command first.")
        raise SystemExit(1)

    con = duckdb.connect(db_path, read_only=True)
    checks: list[tuple[str, bool, str]] = []
    t0 = time.time()

    def run_check(description: str, query_fn):
        """Run a single check, catching errors."""
        start = time.time()
        try:
            ok, detail = query_fn(con)
        except Exception as e:
            ok, detail = False, f"ERROR: {e}"
        elapsed = time.time() - start
        checks.append((description, ok, detail, elapsed))

    # 1. Products row count (12,000 based on actual product catalog)
    def check_products(c):
        n = c.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        return n == 12_000, f"{n:,} rows"
    run_check("Products has 12,000 rows", check_products)

    # 2. Customers row count
    def check_customers(c):
        n = c.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        return n == 10_000_000, f"{n:,} rows"
    run_check("Customers has 10,000,000 rows", check_customers)

    # 3. Transaction count ~10B (+-1%)
    def check_txn_count(c):
        n = c.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        target = 10_000_000_000
        pct = abs(n - target) / target * 100
        ok = pct <= 1.0
        return ok, f"{n:,} rows ({pct:.2f}% off target)"
    run_check("Transactions ~10,000,000,000 (+-1%)", check_txn_count)

    # 4. Referential integrity: customer_id
    def check_fk_customer(c):
        n = c.execute("""
            SELECT COUNT(*) FROM (
                SELECT DISTINCT customer_id FROM transactions
                EXCEPT
                SELECT customer_id FROM customers
            )
        """).fetchone()[0]
        return n == 0, f"{n:,} orphan customer_ids"
    run_check("All transaction customer_ids exist in customers", check_fk_customer)

    # 5. Referential integrity: product_id
    def check_fk_product(c):
        n = c.execute("""
            SELECT COUNT(*) FROM (
                SELECT DISTINCT product_id FROM transactions
                EXCEPT
                SELECT product_id FROM products
            )
        """).fetchone()[0]
        return n == 0, f"{n:,} orphan product_ids"
    run_check("All transaction product_ids exist in products", check_fk_product)

    # 6. Date range
    def check_date_range(c):
        mn, mx = c.execute(
            "SELECT MIN(date), MAX(date) FROM transactions"
        ).fetchone()
        ok = str(mn) >= "2024-01-01" and str(mx) <= "2025-12-31"
        return ok, f"{mn} to {mx}"
    run_check("Date range is 2024-01-01 to 2025-12-31", check_date_range)

    # 7. No negative totals
    def check_no_neg(c):
        n = c.execute(
            "SELECT COUNT(*) FROM transactions WHERE total < 0"
        ).fetchone()[0]
        return n == 0, f"{n:,} negative totals"
    run_check("No negative totals", check_no_neg)

    con.close()
    total_elapsed = time.time() - t0

    # Print results
    console.print()
    console.print("[bold]Validation Results[/bold]\n")
    passed = failed = 0
    for desc, ok, detail, elapsed in checks:
        tag = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        if ok:
            passed += 1
        else:
            failed += 1
        console.print(f"  {tag}  {desc}")
        console.print(f"        {detail} ({elapsed:.1f}s)")

    console.print(f"\n[bold]{passed} passed, {failed} failed[/bold] "
                  f"(total: {total_elapsed:.1f}s)")


if __name__ == "__main__":
    main()

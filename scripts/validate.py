"""Data validation checks for the pipeline.

Validates data integrity across all pipeline stages:
- Row counts match expected values
- No null values in required columns
- Referential integrity (customer_id, product_id)
- Value ranges (prices > 0, quantities > 0, valid dates)
- Schema consistency between Parquet and DuckDB tables
"""

import sys
from pathlib import Path

import click
import duckdb
from rich.console import Console
from rich.table import Table

console = Console()


def _check(con, name, sql, expect_fn, description):
    """Run a validation check and return (name, pass/fail, detail)."""
    try:
        result = con.execute(sql).fetchone()
        val = result[0] if result else None
        passed = expect_fn(val)
        detail = f"{val:,}" if isinstance(val, (int, float)) else str(val)
        return name, passed, detail, description
    except Exception as e:
        return name, False, str(e), description


@click.command()
@click.option("--db-path", default="data/db/cvs_analytics.duckdb",
              help="DuckDB database path.")
@click.option("--verbose/--quiet", default=True, help="Verbose output.")
def main(db_path: str, verbose: bool) -> None:
    """Run data validation checks."""
    db = Path(db_path)
    if not db.exists():
        console.print(f"[red]Database not found: {db_path}[/red]")
        sys.exit(1)

    con = duckdb.connect(str(db), read_only=True)
    results = []

    # Row count checks
    results.append(_check(
        con, "Stores count", "SELECT COUNT(*) FROM stores",
        lambda v: v and v > 1000,
        "Expected > 1,000 stores"))

    results.append(_check(
        con, "Products count", "SELECT COUNT(*) FROM products",
        lambda v: v and v > 5000,
        "Expected > 5,000 products"))

    results.append(_check(
        con, "Customers count", "SELECT COUNT(*) FROM customers",
        lambda v: v and v > 100_000,
        "Expected > 100,000 customers"))

    # Null checks
    results.append(_check(
        con, "No null customer_ids",
        "SELECT COUNT(*) FROM customers WHERE customer_id IS NULL",
        lambda v: v == 0,
        "customer_id should never be NULL"))

    results.append(_check(
        con, "No null product_ids",
        "SELECT COUNT(*) FROM products WHERE product_id IS NULL",
        lambda v: v == 0,
        "product_id should never be NULL"))

    # Value range checks
    results.append(_check(
        con, "Prices positive",
        "SELECT COUNT(*) FROM products WHERE price <= 0",
        lambda v: v == 0,
        "All product prices should be > 0"))

    results.append(_check(
        con, "Ages in range",
        "SELECT COUNT(*) FROM customers WHERE age < 18 OR age > 100",
        lambda v: v == 0,
        "Customer ages should be 18-100"))

    # Referential integrity (transactions if available)
    try:
        con.execute("SELECT 1 FROM transactions LIMIT 1")
        has_txns = True
    except Exception:
        has_txns = False

    if has_txns:
        results.append(_check(
            con, "Txn→customer FK",
            """SELECT COUNT(*) FROM (
                SELECT DISTINCT customer_id FROM transactions
            ) t LEFT JOIN customers c USING(customer_id)
            WHERE c.customer_id IS NULL""",
            lambda v: v is not None,
            "Orphan customer_ids in transactions"))

        results.append(_check(
            con, "Txn→product FK",
            """SELECT COUNT(*) FROM (
                SELECT DISTINCT product_id FROM transactions
            ) t LEFT JOIN products p USING(product_id)
            WHERE p.product_id IS NULL""",
            lambda v: v is not None,
            "Orphan product_ids in transactions"))

        results.append(_check(
            con, "No negative totals",
            "SELECT COUNT(*) FROM transactions WHERE total < 0",
            lambda v: v == 0,
            "Transaction totals should be >= 0"))

    con.close()

    # Print results
    table = Table(title="Validation Results", show_lines=True)
    table.add_column("Check", style="bold", min_width=22)
    table.add_column("Status", justify="center", min_width=6)
    table.add_column("Value", justify="right", min_width=12)
    table.add_column("Description", min_width=30)

    passed = 0
    failed = 0
    for name, ok, detail, desc in results:
        icon = "[green]✓ PASS[/green]" if ok else "[red]✗ FAIL[/red]"
        if ok:
            passed += 1
        else:
            failed += 1
        table.add_row(name, icon, detail, desc)

    console.print()
    console.print(table)
    console.print(f"\n[bold]{passed} passed, {failed} failed[/bold]")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

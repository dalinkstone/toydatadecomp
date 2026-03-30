"""Product tiering from transaction data.

Classifies all 12,000 products into four tiers based on revenue contribution,
discount responsiveness, and purchase behavior. Uses DuckDB for all heavy
computation over the 10B-row transaction dataset.

Tiers:
  1 - Core Revenue Drivers (10-30 products)
  2 - Discount-Responsive (500-2,000 products)
  3 - Organic Sellers (2,000-4,000 products)
  4 - Long Tail / Breakout Candidates (5,000-9,000 products)

Usage:
    python src/cli.py tier products --db-path data/db/cvs_analytics.duckdb
"""

import json
import time
from pathlib import Path

import click
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

console = Console()


class ProductTierClassifier:
    """Computes product features from transaction data and assigns tiers."""

    def __init__(self, db_path: str):
        self.con = duckdb.connect(db_path)
        self.con.execute("SET memory_limit='32GB'")
        self.con.execute("SET temp_directory='/tmp/duckdb_temp'")
        self.con.execute("SET preserve_insertion_order=false")
        self.con.execute("SET threads=4")

    def _compute_base_features(self):
        """Stage 1: Per-product aggregates from full 10B transaction scan.

        Computes: total_units_sold, total_revenue, unique_customers,
        avg_discount_when_sold, discount_purchase_ratio, organic_purchase_ratio.
        """
        console.print(
            "[cyan]Stage 1/3: Base product features (full transaction scan)...[/cyan]"
        )
        t0 = time.time()
        self.con.execute("""
            CREATE OR REPLACE TABLE _tier_base AS
            SELECT
                product_id,
                SUM(quantity)               AS total_units_sold,
                SUM(subtotal)               AS total_revenue,
                COUNT(DISTINCT customer_id) AS unique_customers,
                AVG(discount_pct)           AS avg_discount_when_sold,
                SUM(CASE WHEN COALESCE(discount_pct, 0) > 0
                         THEN 1 ELSE 0 END)
                    * 1.0 / COUNT(*)        AS discount_purchase_ratio,
                SUM(CASE WHEN COALESCE(discount_pct, 0) = 0
                         THEN 1 ELSE 0 END)
                    * 1.0 / COUNT(*)        AS organic_purchase_ratio
            FROM transactions
            GROUP BY product_id
        """)
        n = self.con.execute("SELECT COUNT(*) FROM _tier_base").fetchone()[0]
        console.print(f"  {n:,} products  ({time.time() - t0:.1f}s)")

    def _compute_repeat_purchase_rate(self):
        """Stage 2: Repeat purchase rate per product.

        Groups 10B rows by (product_id, customer_id) to count per-customer
        purchases, then computes the fraction with 2+ purchases per product.
        DuckDB spills to disk for the ~2-3B intermediate groups.
        """
        console.print(
            "[cyan]Stage 2/3: Repeat purchase rates (heavy aggregation)...[/cyan]"
        )
        t0 = time.time()
        self.con.execute("""
            CREATE OR REPLACE TABLE _tier_repeat AS
            WITH customer_product_counts AS (
                SELECT product_id, customer_id, COUNT(*) AS purchase_count
                FROM transactions
                GROUP BY product_id, customer_id
            )
            SELECT
                product_id,
                SUM(CASE WHEN purchase_count >= 2 THEN 1 ELSE 0 END)
                    * 1.0 / COUNT(*) AS repeat_purchase_rate
            FROM customer_product_counts
            GROUP BY product_id
        """)
        n = self.con.execute("SELECT COUNT(*) FROM _tier_repeat").fetchone()[0]
        console.print(f"  {n:,} products  ({time.time() - t0:.1f}s)")

    def _compute_basket_features(self):
        """Stage 3: Average basket size when product appears (1% sample).

        A basket is all items purchased by the same customer on the same date.
        Uses 1% Bernoulli sample (~100M rows) for tractability. Basket sizes
        from the sample are approximate but preserve relative product ranking.
        """
        console.print(
            "[cyan]Stage 3/3: Basket features (1% sample)...[/cyan]"
        )
        t0 = time.time()
        self.con.execute("""
            CREATE OR REPLACE TABLE _tier_basket AS
            WITH sample_txn AS (
                SELECT customer_id, product_id, date
                FROM transactions
                USING SAMPLE 1 PERCENT (bernoulli)
            ),
            basket_sizes AS (
                SELECT customer_id, date, COUNT(*) AS basket_size
                FROM sample_txn
                GROUP BY customer_id, date
            )
            SELECT
                s.product_id,
                AVG(b.basket_size) AS avg_basket_size
            FROM sample_txn s
            JOIN basket_sizes b
                ON s.customer_id = b.customer_id AND s.date = b.date
            GROUP BY s.product_id
        """)
        n = self.con.execute("SELECT COUNT(*) FROM _tier_basket").fetchone()[0]
        console.print(f"  {n:,} products  ({time.time() - t0:.1f}s)")

    def _assemble_and_classify(self):
        """Join feature stages, compute revenue_rank, apply tier rules."""
        console.print("[cyan]Assembling features and classifying tiers...[/cyan]")
        t0 = time.time()

        # Combine all feature stages
        self.con.execute("""
            CREATE OR REPLACE TABLE _tier_assembled AS
            SELECT
                b.product_id,
                b.total_units_sold,
                b.total_revenue,
                b.unique_customers,
                b.avg_discount_when_sold,
                b.discount_purchase_ratio,
                b.organic_purchase_ratio,
                RANK() OVER (ORDER BY b.total_revenue DESC) AS revenue_rank,
                COALESCE(r.repeat_purchase_rate, 0.0)       AS repeat_purchase_rate,
                COALESCE(k.avg_basket_size, 1.0)            AS avg_basket_size
            FROM _tier_base b
            LEFT JOIN _tier_repeat r ON b.product_id = r.product_id
            LEFT JOIN _tier_basket k ON b.product_id = k.product_id
        """)

        # Thresholds for tier 2 and 3 rules
        thresholds = self.con.execute("""
            SELECT
                PERCENTILE_CONT(0.50) WITHIN GROUP
                    (ORDER BY total_units_sold) AS units_median,
                PERCENTILE_CONT(0.25) WITHIN GROUP
                    (ORDER BY total_units_sold) AS units_p25
            FROM _tier_assembled
        """).fetchone()
        units_median, units_p25 = thresholds
        console.print(
            f"  Thresholds: units_median={units_median:,.0f}, "
            f"units_p25={units_p25:,.0f}"
        )

        # Apply tier classification rules (in order of priority)
        self.con.execute(f"""
            CREATE OR REPLACE TABLE revenue_product_tiers AS
            SELECT
                f.*,
                CASE
                    -- TIER 1: Core Revenue Drivers
                    WHEN f.revenue_rank <= 12
                         OR (f.revenue_rank <= 50
                             AND f.repeat_purchase_rate > 0.3)
                    THEN 1
                    -- TIER 2: Discount-Responsive
                    WHEN f.discount_purchase_ratio > 0.5
                         AND f.total_units_sold > {units_median}
                    THEN 2
                    -- TIER 3: Organic Sellers
                    WHEN f.organic_purchase_ratio > 0.7
                         AND f.total_units_sold > {units_p25}
                    THEN 3
                    -- TIER 4: Long Tail / Breakout Candidates
                    ELSE 4
                END AS tier
            FROM _tier_assembled f
        """)

        # Clean up staging tables
        for tbl in ("_tier_base", "_tier_repeat", "_tier_basket", "_tier_assembled"):
            self.con.execute(f"DROP TABLE IF EXISTS {tbl}")

        console.print(f"  Done  ({time.time() - t0:.1f}s)")

    def _export_parquet(self, output_dir: Path):
        """Write product_tiers.parquet with specified output columns."""
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "product_tiers.parquet"

        df = self.con.execute("""
            SELECT
                product_id,
                tier,
                total_units_sold,
                total_revenue,
                unique_customers,
                avg_discount_when_sold,
                discount_purchase_ratio,
                organic_purchase_ratio,
                revenue_rank,
                repeat_purchase_rate
            FROM revenue_product_tiers
            ORDER BY revenue_rank
        """).fetchdf()

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, str(out_path), compression="snappy")
        console.print(f"  Wrote {out_path} ({len(df):,} rows)")

    def _export_summary(self, output_dir: Path):
        """Write tier_summary.json with per-tier counts, revenue, top 10."""
        out_path = output_dir / "tier_summary.json"

        # Per-tier aggregates
        tier_stats = self.con.execute("""
            SELECT
                t.tier,
                COUNT(*)          AS product_count,
                SUM(t.total_revenue) AS tier_revenue,
                AVG(p.price)      AS avg_price
            FROM revenue_product_tiers t
            JOIN products p ON t.product_id = p.product_id
            GROUP BY t.tier
            ORDER BY t.tier
        """).fetchdf()

        total_revenue = float(tier_stats["tier_revenue"].sum())

        summary = {"total_revenue": total_revenue, "tiers": {}}

        for _, row in tier_stats.iterrows():
            tier_num = int(row["tier"])

            top_products = self.con.execute(f"""
                SELECT
                    t.product_id,
                    p.name,
                    p.brand,
                    p.category,
                    t.total_revenue,
                    t.revenue_rank
                FROM revenue_product_tiers t
                JOIN products p ON t.product_id = p.product_id
                WHERE t.tier = {tier_num}
                ORDER BY t.total_revenue DESC
                LIMIT 10
            """).fetchdf()

            summary["tiers"][str(tier_num)] = {
                "product_count": int(row["product_count"]),
                "revenue": float(row["tier_revenue"]),
                "revenue_share": (
                    float(row["tier_revenue"] / total_revenue)
                    if total_revenue > 0
                    else 0.0
                ),
                "avg_price": round(float(row["avg_price"]), 2),
                "top_10_products": [
                    {
                        "product_id": int(r["product_id"]),
                        "name": r["name"],
                        "brand": r["brand"],
                        "category": r["category"],
                        "revenue": float(r["total_revenue"]),
                        "revenue_rank": int(r["revenue_rank"]),
                    }
                    for _, r in top_products.iterrows()
                ],
            }

        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        console.print(f"  Wrote {out_path}")

    def _print_summary_table(self):
        """Print a rich table showing tier distribution and revenue breakdown."""
        rows = self.con.execute("""
            SELECT
                tier,
                COUNT(*)                       AS cnt,
                SUM(total_revenue)             AS rev,
                AVG(avg_discount_when_sold)    AS avg_disc,
                AVG(repeat_purchase_rate)       AS avg_repeat,
                AVG(organic_purchase_ratio)     AS avg_organic
            FROM revenue_product_tiers
            GROUP BY tier
            ORDER BY tier
        """).fetchall()

        total_rev = sum(r[2] for r in rows)

        table = Table(title="Product Tier Distribution", show_lines=True)
        table.add_column("Tier", justify="center", style="bold")
        table.add_column("Description", min_width=25)
        table.add_column("Products", justify="right")
        table.add_column("Revenue", justify="right")
        table.add_column("Rev Share", justify="right")
        table.add_column("Avg Discount", justify="right")
        table.add_column("Repeat Rate", justify="right")
        table.add_column("Organic Rate", justify="right")

        tier_names = {
            1: "Core Revenue Drivers",
            2: "Discount-Responsive",
            3: "Organic Sellers",
            4: "Long Tail / Breakout",
        }

        for tier, cnt, rev, avg_disc, avg_repeat, avg_organic in rows:
            rev_share = rev / total_rev * 100 if total_rev > 0 else 0
            table.add_row(
                str(tier),
                tier_names.get(tier, "Unknown"),
                f"{cnt:,}",
                f"${rev:,.0f}",
                f"{rev_share:.1f}%",
                f"{avg_disc:.1f}%",
                f"{avg_repeat:.1%}",
                f"{avg_organic:.1%}",
            )

        console.print()
        console.print(table)
        console.print()

    def run(self, output_dir: str) -> None:
        """Run the full tier classification pipeline."""
        t0 = time.time()
        output_path = Path(output_dir)

        self._compute_base_features()
        self._compute_repeat_purchase_rate()
        self._compute_basket_features()
        self._assemble_and_classify()
        self._export_parquet(output_path)
        self._export_summary(output_path)
        self._print_summary_table()

        elapsed = time.time() - t0
        console.print(
            f"[bold green]Product tiering complete[/bold green] ({elapsed:.1f}s)"
        )

    def close(self):
        self.con.close()


@click.command()
@click.option(
    "--db-path",
    default="data/db/cvs_analytics.duckdb",
    help="DuckDB database path.",
)
@click.option(
    "--output-dir",
    default="data/model/",
    help="Output directory for parquet and JSON files.",
)
def main(db_path: str, output_dir: str):
    """Classify all products into four revenue-based tiers."""
    console.print("[bold]Product Tier Classification[/bold]")
    console.print(f"  DB: {db_path}")
    console.print(f"  Output: {output_dir}\n")

    classifier = ProductTierClassifier(db_path)
    try:
        classifier.run(output_dir)
    finally:
        classifier.close()


if __name__ == "__main__":
    main()

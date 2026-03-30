"""Feature engineering pipeline for two-tower model.

Computes product-level and customer-level aggregate features from DuckDB,
assigns coupon-based tiers to products, samples training pairs, and exports
lookup tables for the training loop. All heavy computation happens in DuckDB SQL.

Usage:
    python src/ml/features.py --db-path data/db/cvs_analytics.duckdb
"""

import time
from pathlib import Path

import click
import duckdb
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

# -- SQL for feature materialization -------------------------------------------

PRODUCT_FEATURES_SQL = """
CREATE OR REPLACE TABLE product_features AS
WITH txn_agg AS (
    SELECT
        product_id,
        COUNT(*)                     AS total_units_sold,
        SUM(total)                   AS total_revenue,
        COUNT(DISTINCT customer_id)  AS unique_buyers,
        AVG(discount_pct)            AS avg_discount_pct
    FROM transactions
    GROUP BY product_id
),
coupon_agg AS (
    SELECT
        product_id,
        COUNT(*)                                       AS total_clips,
        SUM(CASE WHEN redeemed THEN 1 ELSE 0 END)     AS total_redeemed,
        ROUND(SUM(CASE WHEN redeemed THEN 1 ELSE 0 END)
              * 1.0 / NULLIF(COUNT(*), 0), 4)          AS redemption_rate
    FROM coupon_clips
    GROUP BY product_id
)
SELECT
    p.product_id,
    p.brand,
    p.category,
    p.subcategory,
    p.price,
    p.unit_cost,
    p.is_store_brand,
    p.is_rx,
    p.popularity_score,
    (p.price - p.unit_cost) / NULLIF(p.price, 0)          AS margin_pct,
    COALESCE(t.total_units_sold, 0)                        AS total_units_sold,
    COALESCE(t.total_revenue, 0.0)                         AS total_revenue,
    COALESCE(t.unique_buyers, 0)                           AS unique_buyers,
    COALESCE(t.avg_discount_pct, 0.0)                      AS avg_discount_pct,
    COALESCE(c.total_clips, 0)                             AS coupon_clips,
    COALESCE(c.total_redeemed, 0)                          AS coupon_redeemed,
    COALESCE(c.redemption_rate, 0.0)                       AS coupon_redemption_rate,
    CASE WHEN COALESCE(t.unique_buyers, 0) > 0
         THEN (COALESCE(t.unique_buyers, 0) - COALESCE(c.total_redeemed, 0))
              * 1.0 / t.unique_buyers
         ELSE 1.0
    END                                                    AS organic_purchase_ratio,
    CASE WHEN COALESCE(t.unique_buyers, 0) > 0
         THEN COALESCE(c.total_clips, 0) * 1.0 / t.unique_buyers
         ELSE 0.0
    END                                                    AS coupon_clip_rate
FROM products p
LEFT JOIN txn_agg t ON p.product_id = t.product_id
LEFT JOIN coupon_agg c ON p.product_id = c.product_id
"""

# Customer features are built in stages (see build_customer_features) to avoid OOM.
# This constant is kept for documentation only.
_CUSTOMER_FEATURES_COLUMNS = """
customer_id, age, gender, state, is_student,
total_spend, avg_basket_size, total_transactions,
coupon_clips_count, coupon_redeemed_count,
coupon_redemption_rate, coupon_engagement_score,
price_sensitivity_bucket
"""

TIER_VOCAB = {
    'coupon_loyalist': 1,
    'coupon_curious': 2,
    'organic_star': 3,
    'hidden_gem': 4,
    'unclassified': 5,
}


class FeatureStore:
    """Manages feature computation and export for training."""

    def __init__(self, db_path: str):
        self.con = duckdb.connect(db_path)
        # Tune DuckDB for large out-of-core aggregations on 64GB machine.
        # memory_limit must be well below physical RAM because zstd decompression
        # buffers and jemalloc bypass DuckDB's buffer manager.
        # temp_directory enables spill-to-disk for external aggregation.
        self.con.execute("SET memory_limit='32GB'")
        self.con.execute("SET temp_directory='/tmp/duckdb_temp'")
        self.con.execute("SET preserve_insertion_order=false")
        self.con.execute("SET threads=4")

    def build_product_features(self):
        """Materialize product_features table (scans all transactions)."""
        console.print("[cyan]Building product_features (full transaction scan)...[/cyan]")
        t0 = time.time()
        self.con.execute(PRODUCT_FEATURES_SQL)
        n = self.con.execute("SELECT COUNT(*) FROM product_features").fetchone()[0]
        console.print(f"  product_features: {n:,} rows ({time.time() - t0:.1f}s)")

    def build_customer_features(self):
        """Materialize customer_features table (scans all transactions).

        Split into stages to avoid holding two large intermediates in memory.
        """
        console.print("[cyan]Building customer_features (full transaction scan)...[/cyan]")
        t0 = time.time()

        # Free memory from product_features if it exists (will rebuild in build_product_tiers)
        had_pf = self.has_table("product_features")
        if had_pf:
            console.print("  Temporarily dropping product_features to free memory...")
            self.con.execute("DROP TABLE IF EXISTS product_features")

        # Stage 1: transaction aggregation (the expensive 10B-row scan)
        # temp_directory + memory_limit=32GB enables spill-to-disk.
        console.print("  Stage 1/3: aggregating transactions by customer...")
        self.con.execute("""
            CREATE OR REPLACE TABLE _cust_txn_agg AS
            SELECT
                customer_id,
                COUNT(*)           AS total_transactions,
                SUM(total)         AS total_spend,
                AVG(total)         AS avg_basket_size,
                AVG(discount_pct)  AS avg_discount_pct
            FROM transactions
            GROUP BY customer_id
        """)
        n1 = self.con.execute("SELECT COUNT(*) FROM _cust_txn_agg").fetchone()[0]
        console.print(f"    _cust_txn_agg: {n1:,} rows ({time.time() - t0:.1f}s)")

        # Stage 2: coupon aggregation (fast, 16M rows)
        console.print("  Stage 2/3: aggregating coupon clips...")
        self.con.execute("""
            CREATE OR REPLACE TABLE _cust_coupon_agg AS
            SELECT
                c.customer_id,
                COUNT(*)                                      AS total_clips,
                SUM(CASE WHEN cc.redeemed THEN 1 ELSE 0 END) AS total_redeemed
            FROM customers c
            JOIN coupon_clips cc ON c.loyalty_number = cc.loyalty_number
            GROUP BY c.customer_id
        """)
        n2 = self.con.execute("SELECT COUNT(*) FROM _cust_coupon_agg").fetchone()[0]
        console.print(f"    _cust_coupon_agg: {n2:,} rows ({time.time() - t0:.1f}s)")

        # Stage 3: final join (10M customers + two small lookup tables)
        console.print("  Stage 3/3: joining features...")
        self.con.execute("""
            CREATE OR REPLACE TABLE customer_features AS
            SELECT
                cu.customer_id,
                cu.age,
                cu.gender,
                cu.state,
                cu.is_student,
                COALESCE(ct.total_spend, 0.0)            AS total_spend,
                COALESCE(ct.avg_basket_size, 0.0)        AS avg_basket_size,
                COALESCE(ct.total_transactions, 0)       AS total_transactions,
                COALESCE(cc.total_clips, 0)              AS coupon_clips_count,
                COALESCE(cc.total_redeemed, 0)           AS coupon_redeemed_count,
                CASE WHEN COALESCE(cc.total_clips, 0) > 0
                     THEN cc.total_redeemed * 1.0 / cc.total_clips
                     ELSE 0.0
                END                                      AS coupon_redemption_rate,
                CASE WHEN COALESCE(ct.total_transactions, 0) > 0
                     THEN COALESCE(cc.total_clips, 0) * 1.0 / ct.total_transactions
                     ELSE 0.0
                END                                      AS coupon_engagement_score,
                CASE
                    WHEN COALESCE(ct.avg_discount_pct, 0) < 0.05 THEN 0
                    WHEN COALESCE(ct.avg_discount_pct, 0) < 0.10 THEN 1
                    WHEN COALESCE(ct.avg_discount_pct, 0) < 0.15 THEN 2
                    WHEN COALESCE(ct.avg_discount_pct, 0) < 0.20 THEN 3
                    ELSE 4
                END                                      AS price_sensitivity_bucket
            FROM customers cu
            LEFT JOIN _cust_txn_agg ct ON cu.customer_id = ct.customer_id
            LEFT JOIN _cust_coupon_agg cc ON cu.customer_id = cc.customer_id
        """)

        # Clean up staging tables
        self.con.execute("DROP TABLE IF EXISTS _cust_txn_agg")
        self.con.execute("DROP TABLE IF EXISTS _cust_coupon_agg")

        n = self.con.execute("SELECT COUNT(*) FROM customer_features").fetchone()[0]
        console.print(f"  customer_features: {n:,} rows ({time.time() - t0:.1f}s)")

        # Rebuild product_features if we dropped it for memory
        if had_pf:
            console.print("  Rebuilding product_features...")
            self.con.execute(PRODUCT_FEATURES_SQL)
            console.print(f"    product_features rebuilt ({time.time() - t0:.1f}s)")

    def build_product_tiers(self):
        """Compute coupon-based product tiers using percentile thresholds."""
        console.print("[cyan]Building product_tiers...[/cyan]")
        thresholds = self.con.execute("""
            SELECT
                PERCENTILE_CONT(0.30) WITHIN GROUP (ORDER BY coupon_clips)     AS clips_p30,
                PERCENTILE_CONT(0.70) WITHIN GROUP (ORDER BY coupon_clips)     AS clips_p70,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_units_sold) AS units_p50,
                PERCENTILE_CONT(0.70) WITHIN GROUP (ORDER BY total_units_sold) AS units_p70
            FROM product_features
        """).fetchone()
        clips_p30, clips_p70, units_p50, units_p70 = thresholds
        console.print(f"  Thresholds: clips_p30={clips_p30}, clips_p70={clips_p70}, "
                      f"units_p50={units_p50}, units_p70={units_p70}")

        self.con.execute(f"""
            CREATE OR REPLACE TABLE product_tiers AS
            SELECT
                pf.*,
                CASE
                    WHEN coupon_clips >= {clips_p70}
                         AND coupon_redemption_rate >= 0.40
                    THEN 'coupon_loyalist'
                    WHEN coupon_clips >= {clips_p30}
                         AND coupon_redemption_rate < 0.40
                    THEN 'coupon_curious'
                    WHEN total_units_sold >= {units_p70}
                         AND coupon_clips < {clips_p30}
                    THEN 'organic_star'
                    WHEN margin_pct >= 0.45
                         AND total_units_sold < {units_p50}
                    THEN 'hidden_gem'
                    ELSE 'unclassified'
                END AS tier
            FROM product_features pf
        """)
        tier_dist = self.con.execute(
            "SELECT tier, COUNT(*) FROM product_tiers GROUP BY tier ORDER BY COUNT(*) DESC"
        ).fetchall()
        for tier, cnt in tier_dist:
            console.print(f"    {tier}: {cnt:,}")

    def build_training_pairs(self, sample_pct: float = 1.0):
        """Sample positive training pairs from transactions."""
        console.print(f"[cyan]Sampling training pairs ({sample_pct}% of transactions)...[/cyan]")
        t0 = time.time()
        self.con.execute(f"""
            CREATE OR REPLACE TABLE training_pairs AS
            SELECT customer_id, product_id
            FROM transactions
            USING SAMPLE {sample_pct} PERCENT (bernoulli)
        """)
        n = self.con.execute("SELECT COUNT(*) FROM training_pairs").fetchone()[0]
        console.print(f"  training_pairs: {n:,} rows ({time.time() - t0:.1f}s)")

    def build_coupon_response_pairs(self, sample_pct: float = 1.0):
        """Build labeled training pairs for coupon response model.

        Three types of examples:
        1. Coupon response (from coupon_clips): label=redeemed, weight=1.0
        2. Discount purchases (discount_pct > 0): label=1, weight=1.0
        3. Organic purchases (discount_pct = 0): label=1, weight=0.5

        Target mix: ~30% coupon / 40% discount / 30% organic (positives only).
        """
        t0 = time.time()
        console.print("[cyan]Building coupon response training pairs...[/cyan]")

        # Single scan of transactions, split by discount
        console.print(f"  Sampling transactions ({sample_pct}%)...")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE _sampled_txns AS
            SELECT customer_id, product_id, discount_pct
            FROM transactions
            USING SAMPLE {sample_pct} PERCENT (bernoulli)
        """)

        counts = self.con.execute("""
            SELECT
                SUM(CASE WHEN discount_pct > 0 THEN 1 ELSE 0 END) AS n_disc,
                SUM(CASE WHEN discount_pct = 0 THEN 1 ELSE 0 END) AS n_org
            FROM _sampled_txns
        """).fetchone()
        n_disc_raw, n_org_raw = int(counts[0] or 0), int(counts[1] or 0)
        console.print(f"    Sampled txns: {n_disc_raw + n_org_raw:,} "
                      f"(disc: {n_disc_raw:,}, org: {n_org_raw:,})")

        # Coupon examples: subsample to achieve ~30% of total positives
        # Target: coupon = purchase_total * 3/7 (so 30% of combined total)
        n_purchase_total = n_disc_raw + n_org_raw
        n_coupon_target = int(n_purchase_total * 3 / 7)
        n_total_clips = self.con.execute("SELECT COUNT(*) FROM coupon_clips").fetchone()[0]

        if n_total_clips <= n_coupon_target or n_coupon_target == 0:
            coupon_sample_clause = ""
        else:
            coupon_pct = n_coupon_target / n_total_clips * 100
            coupon_sample_clause = f"USING SAMPLE {coupon_pct:.4f} PERCENT (bernoulli)"

        console.print(f"  Building coupon examples (target: {n_coupon_target:,} "
                      f"from {n_total_clips:,} clips)...")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE _coupon_examples AS
            SELECT
                cu.customer_id,
                cc.product_id::BIGINT AS product_id,
                CASE
                    WHEN cc.discount_type = 'percent_off'
                        THEN LEAST(cc.discount_value, 0.50)
                    WHEN cc.discount_type = 'bogo' THEN 0.50
                    WHEN cc.discount_type = 'dollar_off'
                        THEN LEAST(cc.discount_value / NULLIF(p.price, 0), 0.50)
                    ELSE 0.10
                END AS discount_offer,
                CASE WHEN cc.redeemed THEN 1.0 ELSE 0.0 END AS label,
                1.0 AS weight
            FROM (SELECT * FROM coupon_clips {coupon_sample_clause}) cc
            JOIN customers cu ON cc.loyalty_number = cu.loyalty_number
            JOIN products p ON cc.product_id = p.product_id
        """)
        n_coupon = self.con.execute("SELECT COUNT(*) FROM _coupon_examples").fetchone()[0]

        # Subsample disc/org to achieve 40/30 ratio relative to coupon at 30%
        # If coupon is N at 30%, disc should be N*4/3, org should be N
        n_disc_target = int(n_coupon * 4 / 3)
        n_org_target = n_coupon

        if n_disc_raw > n_disc_target and n_disc_target > 0:
            disc_frac = n_disc_target / n_disc_raw
        else:
            disc_frac = 1.0

        if n_org_raw > n_org_target and n_org_target > 0:
            org_frac = n_org_target / n_org_raw
        else:
            org_frac = 1.0

        # Union all into final table
        console.print("  Combining training data...")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE training_pairs_labeled AS
            SELECT customer_id, product_id, discount_offer, label, weight
            FROM _coupon_examples
            UNION ALL
            SELECT customer_id, product_id, discount_pct AS discount_offer,
                   1.0 AS label, 1.0 AS weight
            FROM _sampled_txns
            WHERE discount_pct > 0
            {"AND RANDOM() < " + str(disc_frac) if disc_frac < 1.0 else ""}
            UNION ALL
            SELECT customer_id, product_id, 0.0 AS discount_offer,
                   1.0 AS label, 0.5 AS weight
            FROM _sampled_txns
            WHERE discount_pct = 0
            {"AND RANDOM() < " + str(org_frac) if org_frac < 1.0 else ""}
        """)

        # Clean up staging tables
        self.con.execute("DROP TABLE IF EXISTS _sampled_txns")
        self.con.execute("DROP TABLE IF EXISTS _coupon_examples")

        total = self.con.execute("SELECT COUNT(*) FROM training_pairs_labeled").fetchone()[0]
        n_disc_final = self.con.execute(
            "SELECT COUNT(*) FROM training_pairs_labeled WHERE weight = 1.0 AND label = 1.0 AND discount_offer > 0"
        ).fetchone()[0]
        n_org_final = self.con.execute(
            "SELECT COUNT(*) FROM training_pairs_labeled WHERE weight = 0.5"
        ).fetchone()[0]
        console.print(f"  training_pairs_labeled: {total:,} examples "
                      f"(coupon: {n_coupon:,}, disc: {n_disc_final:,}, org: {n_org_final:,}) "
                      f"({time.time() - t0:.1f}s)")

    def export_customer_lookup(self) -> dict[str, np.ndarray]:
        """Export customer features as numpy arrays indexed by customer_id.

        customer_id is 1-based (1..10M), so array index 0 is unused.
        """
        console.print("[cyan]Exporting customer feature arrays...[/cyan]")
        max_id = self.con.execute("SELECT MAX(customer_id) FROM customer_features").fetchone()[0]
        size = max_id + 1

        df = self.con.execute(
            "SELECT * FROM customer_features ORDER BY customer_id"
        ).fetchdf()

        cids = df["customer_id"].values

        result = {}
        # Numeric features
        for col in ["age", "total_spend", "avg_basket_size", "total_transactions",
                     "coupon_clips_count", "coupon_redeemed_count",
                     "coupon_redemption_rate", "coupon_engagement_score"]:
            arr = np.zeros(size, dtype=np.float32)
            arr[cids] = df[col].values.astype(np.float32)
            result[col] = arr

        # Boolean
        arr = np.zeros(size, dtype=np.float32)
        arr[cids] = df["is_student"].values.astype(np.float32)
        result["is_student"] = arr

        # Categorical: gender (string -> int)
        gender_map = {"F": 0, "M": 1, "NB": 2}
        arr = np.zeros(size, dtype=np.int64)
        arr[cids] = df["gender"].map(gender_map).fillna(0).values.astype(np.int64)
        result["gender"] = arr

        # Categorical: state (string -> int)
        state_vocab = self.export_state_vocab()
        arr = np.zeros(size, dtype=np.int64)
        arr[cids] = df["state"].map(state_vocab).fillna(0).values.astype(np.int64)
        result["state"] = arr

        # Price sensitivity bucket (int 0-4)
        arr = np.zeros(size, dtype=np.int64)
        arr[cids] = df["price_sensitivity_bucket"].values.astype(np.int64)
        result["price_sensitivity_bucket"] = arr

        console.print(f"  Exported {len(result)} feature arrays, shape ({size},)")
        return result

    def export_product_lookup(self) -> dict:
        """Export product features as dict keyed by product_id."""
        console.print("[cyan]Exporting product feature lookup...[/cyan]")
        df = self.con.execute("SELECT * FROM product_tiers").fetchdf()
        lookup = {}
        for _, row in df.iterrows():
            lookup[int(row["product_id"])] = row.to_dict()
        console.print(f"  Exported {len(lookup)} products")
        return lookup

    def export_brand_vocab(self) -> dict[str, int]:
        """Map brand strings to sequential integer IDs (0 = unknown)."""
        brands = self.con.execute(
            "SELECT DISTINCT brand FROM products ORDER BY brand"
        ).fetchdf()["brand"].tolist()
        return {b: i + 1 for i, b in enumerate(brands)}

    def export_category_vocab(self) -> dict[str, int]:
        """Map category strings to sequential integer IDs (0 = unknown)."""
        cats = self.con.execute(
            "SELECT DISTINCT category FROM products ORDER BY category"
        ).fetchdf()["category"].tolist()
        return {c: i + 1 for i, c in enumerate(cats)}

    def export_state_vocab(self) -> dict[str, int]:
        """Map state codes to sequential integer IDs (0 = unknown)."""
        states = self.con.execute(
            "SELECT DISTINCT state FROM customers ORDER BY state"
        ).fetchdf()["state"].tolist()
        return {s: i + 1 for i, s in enumerate(states)}

    def export_training_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Load training pairs into numpy arrays."""
        console.print("[cyan]Loading training pairs into memory...[/cyan]")
        df = self.con.execute("SELECT customer_id, product_id FROM training_pairs").fetchdf()
        cids = df["customer_id"].values.astype(np.int64)
        pids = df["product_id"].values.astype(np.int64)
        console.print(f"  Loaded {len(cids):,} pairs")
        return cids, pids

    def export_coupon_training_data(self):
        """Export labeled training pairs as numpy arrays.

        Returns (customer_ids, product_ids, discount_offers, labels, weights).
        """
        console.print("[cyan]Loading labeled training pairs...[/cyan]")
        df = self.con.execute(
            "SELECT customer_id, product_id, discount_offer, label, weight "
            "FROM training_pairs_labeled"
        ).fetchdf()
        console.print(f"  Loaded {len(df):,} labeled pairs")
        return (
            df["customer_id"].values.astype(np.int64),
            df["product_id"].values.astype(np.int64),
            df["discount_offer"].values.astype(np.float32),
            df["label"].values.astype(np.float32),
            df["weight"].values.astype(np.float32),
        )

    def export_elasticity_lookup(self) -> dict[int, dict]:
        """Load elasticity data from parquet, keyed by product_id."""
        console.print("[cyan]Loading elasticity data...[/cyan]")
        df = self.con.execute(
            "SELECT product_id, elasticity_beta, optimal_discount "
            "FROM read_parquet('data/model/elasticity.parquet')"
        ).fetchdf()
        lookup = {}
        for _, row in df.iterrows():
            lookup[int(row["product_id"])] = {
                "elasticity_beta": float(row["elasticity_beta"]),
                "optimal_discount": float(row["optimal_discount"]),
            }
        console.print(f"  Loaded elasticity for {len(lookup)} products")
        return lookup

    def export_tier_vocab(self) -> dict[str, int]:
        """Return tier string to integer mapping (0 = unknown)."""
        return dict(TIER_VOCAB)

    def export_normalization_stats(self) -> dict[str, tuple[float, float]]:
        """Compute mean/std for features that need normalization."""
        stats = {}
        for col, table in [
            ("age", "customer_features"),
            ("total_spend", "customer_features"),
            ("avg_basket_size", "customer_features"),
            ("coupon_engagement_score", "customer_features"),
            ("price", "product_features"),
        ]:
            row = self.con.execute(
                f"SELECT AVG({col}), STDDEV({col}) FROM {table}"
            ).fetchone()
            mean, std = float(row[0]), max(float(row[1]), 1e-8)
            stats[col] = (mean, std)
        return stats

    def _existing_tables(self) -> set[str]:
        return {r[0] for r in self.con.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()}

    def has_table(self, name: str) -> bool:
        return name in self._existing_tables()

    def has_features(self) -> bool:
        """Check if all feature tables already exist."""
        needed = {"product_features", "customer_features", "product_tiers",
                  "training_pairs_labeled"}
        return needed.issubset(self._existing_tables())

    def close(self):
        self.con.close()


@click.command()
@click.option("--db-path", default="data/db/cvs_analytics.duckdb",
              help="DuckDB database path.")
@click.option("--sample-pct", default=1.0, type=float,
              help="Percent of transactions to sample for training (1.0 = 1%%).")
@click.option("--skip-txn-scan", is_flag=True,
              help="Skip transaction scans if feature tables already exist.")
def main(db_path: str, sample_pct: float, skip_txn_scan: bool):
    """Run feature engineering pipeline."""
    console.print(f"[bold]Feature Engineering Pipeline[/bold]")
    console.print(f"  DB: {db_path}")
    console.print(f"  Sample: {sample_pct}%\n")

    fs = FeatureStore(db_path)
    t0 = time.time()

    if skip_txn_scan and fs.has_features():
        console.print("[yellow]Skipping all (tables exist).[/yellow]")
    else:
        if fs.has_table("product_features"):
            console.print("[yellow]product_features exists, skipping initial build.[/yellow]")
        else:
            fs.build_product_features()

        if fs.has_table("customer_features"):
            console.print("[yellow]customer_features exists, skipping.[/yellow]")
        else:
            # This may temporarily drop product_features to free memory,
            # so we rebuild it after if needed.
            fs.build_customer_features()

        # Rebuild product_features if it was dropped for memory
        if not fs.has_table("product_features"):
            console.print("[cyan]Rebuilding product_features...[/cyan]")
            fs.build_product_features()

        fs.build_product_tiers()
        fs.build_training_pairs(sample_pct)
        fs.build_coupon_response_pairs(sample_pct)

    elapsed = time.time() - t0
    console.print(f"\n[bold green]Feature engineering complete[/bold green] ({elapsed:.1f}s)")
    fs.close()


if __name__ == "__main__":
    main()

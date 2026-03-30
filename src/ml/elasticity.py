"""Price elasticity estimation from coupon redemption and transaction data.

Uses coupon_clips (16M rows) as the primary behavioral signal: for each product,
models how redemption probability responds to discount depth. Cross-references
with organic purchase ratios from transactions to identify products where
discounting is unnecessary (high organic demand = leaving profit on the table).

The elasticity beta represents: for each unit increase in effective discount
fraction, how much does log(redemption_rate) increase? Products with high beta
AND low organic ratio are good discount candidates. Products with high organic
ratio regardless of beta should NOT be discounted.

Usage:
    python src/cli.py elasticity estimate --db-path data/db/cvs_analytics.duckdb
"""

import time
from pathlib import Path

import click
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

console = Console()


class ElasticityEstimator:
    """Estimates price elasticity per product from coupon redemption data."""

    def __init__(self, db_path: str, sample_pct: float = 2.0):
        self.con = duckdb.connect(db_path, read_only=True)
        self.con.execute("SET memory_limit='32GB'")
        self.con.execute("SET temp_directory='/tmp/duckdb_temp'")
        self.con.execute("SET preserve_insertion_order=false")
        self.con.execute("SET threads=4")
        self.sample_pct = sample_pct

    # ── Stage 1: Coupon redemption curve ──────────────────────────────

    def _query_coupon_redemption(self):
        """Build per-product redemption stats from coupon_clips.

        Normalizes all discount types to an effective discount fraction:
          - percent_off: discount_value directly (0.05-0.40)
          - dollar_off:  discount_value / product price
          - bogo:        0.50 (buy-one-get-one ≈ 50% off per unit)

        Returns count of products with coupon data.
        """
        console.print(
            "[cyan]Stage 1/5: Coupon redemption curves "
            "(16M coupon_clips)...[/cyan]"
        )
        t0 = time.time()

        self.con.execute("""
            CREATE OR REPLACE TEMP TABLE _elast_coupon AS
            SELECT
                cc.product_id,
                CASE
                    WHEN cc.discount_type = 'percent_off'
                        THEN cc.discount_value
                    WHEN cc.discount_type = 'dollar_off'
                        THEN LEAST(cc.discount_value / NULLIF(p.price, 0), 0.60)
                    WHEN cc.discount_type = 'bogo'
                        THEN 0.50
                    ELSE cc.discount_value
                END AS effective_discount,
                cc.discount_type,
                cc.redeemed
            FROM coupon_clips cc
            JOIN products p ON cc.product_id = p.product_id
        """)

        # Aggregate: per product × effective_discount level
        self.con.execute("""
            CREATE OR REPLACE TEMP TABLE _elast_coupon_agg AS
            SELECT
                product_id,
                ROUND(effective_discount, 2) AS disc_level,
                COUNT(*)  AS clips,
                SUM(CASE WHEN redeemed THEN 1 ELSE 0 END) AS redeemed,
                SUM(CASE WHEN redeemed THEN 1 ELSE 0 END) * 1.0
                    / COUNT(*) AS redemption_rate
            FROM _elast_coupon
            GROUP BY product_id, ROUND(effective_discount, 2)
        """)

        # Per-product totals
        self.con.execute("""
            CREATE OR REPLACE TEMP TABLE _elast_coupon_totals AS
            SELECT
                product_id,
                SUM(clips)    AS total_clips,
                SUM(redeemed) AS total_redeemed,
                SUM(redeemed) * 1.0 / SUM(clips) AS overall_redemption_rate
            FROM _elast_coupon_agg
            GROUP BY product_id
        """)

        stats = self.con.execute("""
            SELECT
                COUNT(*) AS n_products,
                SUM(CASE WHEN total_clips >= 50 THEN 1 ELSE 0 END) AS n_sufficient,
                SUM(total_clips) AS total_clips
            FROM _elast_coupon_totals
        """).fetchone()

        console.print(
            f"  {stats[0]:,} products with coupons, "
            f"{stats[1]:,} with ≥50 clips, "
            f"{stats[2]:,} total clip events  "
            f"({time.time() - t0:.1f}s)"
        )
        return stats[1]

    # ── Stage 2: Per-product elasticity from coupon data ──────────────

    def _fit_product_level(self):
        """Fit log-linear model on coupon redemption per product.

        For each product with ≥50 coupon clips:
            log(redemption_rate) = alpha + beta * effective_discount

        Weighted by number of clips at each discount level.
        Returns dict: product_id -> (beta, se, p_value).
        """
        console.print(
            "[cyan]Stage 2/5: Per-product coupon elasticity models...[/cyan]"
        )
        t0 = time.time()

        rows = self.con.execute("""
            SELECT
                a.product_id,
                a.disc_level,
                a.clips,
                a.redemption_rate
            FROM _elast_coupon_agg a
            JOIN _elast_coupon_totals t ON a.product_id = t.product_id
            WHERE t.total_clips >= 50
              AND a.clips >= 3
              AND a.redemption_rate > 0
            ORDER BY a.product_id, a.disc_level
        """).fetchnumpy()

        if len(rows["product_id"]) == 0:
            console.print("  No products with sufficient coupon data")
            return {}

        product_ids = rows["product_id"]
        discounts = rows["disc_level"]
        weights = rows["clips"]
        redemption = rows["redemption_rate"]

        unique_pids, inverse, counts = np.unique(
            product_ids, return_inverse=True, return_counts=True
        )

        from scipy import stats as sp_stats

        results = {}
        log_rate = np.log(np.maximum(redemption, 1e-6))

        idx = 0
        for i, pid in enumerate(unique_pids):
            n = counts[i]
            sl = slice(idx, idx + n)
            idx += n

            x = discounts[sl].astype(np.float64)
            y = log_rate[sl]
            w = weights[sl].astype(np.float64)

            if len(np.unique(x)) < 2:
                continue

            sw = np.sqrt(w)
            X = np.column_stack([np.ones(n), x]) * sw[:, None]
            Y = y * sw

            try:
                coeffs, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if rank < 2:
                continue

            alpha, beta = coeffs

            y_hat = X @ coeffs
            sse = np.sum((Y - y_hat) ** 2)
            dof = max(n - 2, 1)
            mse = sse / dof
            try:
                XtX_inv = np.linalg.inv(X.T @ X)
                se_beta = np.sqrt(max(mse * XtX_inv[1, 1], 0.0))
            except np.linalg.LinAlgError:
                se_beta = np.nan

            if se_beta > 0 and not np.isnan(se_beta):
                t_stat = beta / se_beta
                p_value = float(2.0 * sp_stats.t.sf(abs(t_stat), dof))
            else:
                p_value = np.nan

            results[int(pid)] = (float(beta), float(se_beta), float(p_value))

        console.print(
            f"  {len(results):,} product-level models fit  "
            f"({time.time() - t0:.1f}s)"
        )
        return results

    # ── Stage 3: Category-level fallback ─────────────────────────────

    def _fit_category_level(self):
        """Fit coupon elasticity at category level for sparse products.

        Returns dict: category -> (beta, se, p_value).
        """
        console.print(
            "[cyan]Stage 3/5: Category-level fallback models...[/cyan]"
        )
        t0 = time.time()

        rows = self.con.execute("""
            SELECT
                p.category,
                a.disc_level,
                SUM(a.clips) AS clips,
                SUM(a.redeemed) * 1.0 / SUM(a.clips) AS redemption_rate
            FROM _elast_coupon_agg a
            JOIN products p ON a.product_id = p.product_id
            GROUP BY p.category, a.disc_level
            HAVING SUM(a.clips) >= 10 AND SUM(a.redeemed) > 0
            ORDER BY p.category, a.disc_level
        """).fetchnumpy()

        if len(rows["category"]) == 0:
            console.print("  No category-level data")
            return {}

        categories = rows["category"]
        discounts = rows["disc_level"]
        weights = rows["clips"]
        redemption = rows["redemption_rate"]

        unique_cats, inverse, counts = np.unique(
            categories, return_inverse=True, return_counts=True
        )

        from scipy import stats as sp_stats

        results = {}
        log_rate = np.log(np.maximum(redemption, 1e-6))

        idx = 0
        for i, cat in enumerate(unique_cats):
            n = counts[i]
            sl = slice(idx, idx + n)
            idx += n

            x = discounts[sl].astype(np.float64)
            y = log_rate[sl]
            w = weights[sl].astype(np.float64)

            if len(np.unique(x)) < 2:
                continue

            sw = np.sqrt(w)
            X = np.column_stack([np.ones(n), x]) * sw[:, None]
            Y = y * sw

            try:
                coeffs, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if rank < 2:
                continue

            alpha, beta = coeffs

            y_hat = X @ coeffs
            sse = np.sum((Y - y_hat) ** 2)
            dof = max(n - 2, 1)
            mse = sse / dof
            try:
                XtX_inv = np.linalg.inv(X.T @ X)
                se_beta = np.sqrt(max(mse * XtX_inv[1, 1], 0.0))
            except np.linalg.LinAlgError:
                se_beta = np.nan

            if se_beta > 0 and not np.isnan(se_beta):
                t_stat = beta / se_beta
                p_value = float(2.0 * sp_stats.t.sf(abs(t_stat), dof))
            else:
                p_value = np.nan

            results[str(cat)] = (float(beta), float(se_beta), float(p_value))

        console.print(
            f"  {len(results):,} category-level models fit  "
            f"({time.time() - t0:.1f}s)"
        )
        return results

    # ── Stage 4: Organic strength from transactions ──────────────────

    def _compute_organic_strength(self):
        """Compute per-product organic purchase metrics.

        Uses product_features if available, otherwise samples transactions.
        Returns dict: product_id -> {organic_ratio, discount_ratio, total_revenue}.
        """
        console.print(
            "[cyan]Stage 4/5: Organic purchase strength...[/cyan]"
        )
        t0 = time.time()

        # Try product_features table first (already computed by features.py)
        try:
            rows = self.con.execute("""
                SELECT
                    product_id,
                    organic_purchase_ratio,
                    1.0 - organic_purchase_ratio AS discount_purchase_ratio,
                    total_revenue,
                    total_units_sold,
                    COALESCE(coupon_redemption_rate, 0) AS coupon_redemption_rate,
                    COALESCE(coupon_clips, 0) AS coupon_clips
                FROM product_features
            """).fetchnumpy()
            source = "product_features"
        except duckdb.CatalogException:
            # Fall back to sampling transactions
            console.print("  product_features not found, sampling transactions...")
            rows = self.con.execute(f"""
                SELECT
                    product_id,
                    SUM(CASE WHEN COALESCE(discount_pct, 0) = 0 THEN 1 ELSE 0 END)
                        * 1.0 / COUNT(*) AS organic_purchase_ratio,
                    SUM(CASE WHEN COALESCE(discount_pct, 0) > 0 THEN 1 ELSE 0 END)
                        * 1.0 / COUNT(*) AS discount_purchase_ratio,
                    SUM(subtotal) AS total_revenue,
                    SUM(quantity) AS total_units_sold,
                    0.0 AS coupon_redemption_rate,
                    0 AS coupon_clips
                FROM (
                    SELECT product_id, discount_pct, subtotal, quantity
                    FROM transactions
                    USING SAMPLE {self.sample_pct} PERCENT (bernoulli)
                ) s
                GROUP BY product_id
            """).fetchnumpy()
            source = "transaction_sample"

        organic = {}
        for i in range(len(rows["product_id"])):
            pid = int(rows["product_id"][i])
            organic[pid] = {
                "organic_ratio": float(rows["organic_purchase_ratio"][i]),
                "discount_ratio": float(rows["discount_purchase_ratio"][i]),
                "total_revenue": float(rows["total_revenue"][i]),
                "total_units": float(rows["total_units_sold"][i]),
                "coupon_redemption_rate": float(
                    rows["coupon_redemption_rate"][i]
                ),
                "coupon_clips": int(rows["coupon_clips"][i]),
            }

        console.print(
            f"  {len(organic):,} products scored from {source}  "
            f"({time.time() - t0:.1f}s)"
        )
        return organic

    # ── Stage 5: Combine into final scores ───────────────────────────

    def _compute_final_scores(self, product_results, category_results,
                              organic_data):
        """Combine coupon elasticity with organic strength into final metrics.

        For each product:
        - elasticity_beta: from coupon model (how responsive to discount depth)
        - optimal_discount: revenue-maximizing discount level
        - discount_sensitivity_score: for Tier 1, revenue impact of removing
          discounts. High sensitivity = product needs discounting. Low = profit
          left on table.

        The key insight: products with HIGH organic_ratio and LOW coupon beta
        should NOT be discounted — customers buy them at full price anyway.
        """
        console.print(
            "[cyan]Stage 5/5: Computing optimal discounts and "
            "sensitivity scores...[/cyan]"
        )
        t0 = time.time()

        products = self.con.execute("""
            SELECT p.product_id, p.category, p.price, p.unit_cost
            FROM products p
            ORDER BY p.product_id
        """).fetchnumpy()

        # Load tier data
        try:
            tier_data = self.con.execute("""
                SELECT product_id, tier
                FROM revenue_product_tiers
            """).fetchnumpy()
            tier_map = dict(zip(
                tier_data["product_id"].tolist(),
                tier_data["tier"].tolist(),
            ))
        except duckdb.CatalogException:
            tier_map = {}

        # Coupon redemption totals for coupon_lift calculation
        try:
            coupon_totals = self.con.execute("""
                SELECT product_id, overall_redemption_rate
                FROM _elast_coupon_totals
            """).fetchnumpy()
            redemption_map = dict(zip(
                coupon_totals["product_id"].tolist(),
                coupon_totals["overall_redemption_rate"].tolist(),
            ))
        except duckdb.CatalogException:
            redemption_map = {}

        n = len(products["product_id"])
        out = {
            "product_id": products["product_id"].tolist(),
            "category": [str(c) for c in products["category"]],
            "elasticity_beta": np.full(n, np.nan),
            "elasticity_se": np.full(n, np.nan),
            "p_value": np.full(n, np.nan),
            "optimal_discount": np.zeros(n),
            "revenue_at_optimal": np.zeros(n),
            "revenue_at_zero_discount": np.zeros(n),
            "discount_sensitivity_score": np.zeros(n),
            "organic_ratio": np.zeros(n),
            "coupon_redemption_rate": np.zeros(n),
            "data_source": [""] * n,
        }

        for i in range(n):
            pid = int(products["product_id"][i])
            cat = str(products["category"][i])
            price = float(products["price"][i])
            cost = float(products["unit_cost"][i])
            margin = price - cost

            org = organic_data.get(pid, {})
            organic_ratio = org.get("organic_ratio", 0.3)
            total_revenue = org.get("total_revenue", 0.0)
            total_units = org.get("total_units", 1.0)
            prod_redemption = redemption_map.get(pid, 0.0)

            out["organic_ratio"][i] = organic_ratio
            out["coupon_redemption_rate"][i] = prod_redemption

            # Choose product-level or category-level coupon elasticity
            if pid in product_results:
                beta, se, pv = product_results[pid]
                source = "product-level"
            elif cat in category_results:
                beta, se, pv = category_results[cat]
                source = "category-level"
            else:
                beta, se, pv = 0.0, np.nan, np.nan
                source = "category-level"

            out["elasticity_beta"][i] = beta
            out["elasticity_se"][i] = se
            out["p_value"][i] = pv
            out["data_source"][i] = source

            # Revenue at zero discount (organic-only baseline)
            base_demand = max(total_units, 1.0)
            rev_zero = price * base_demand
            out["revenue_at_zero_discount"][i] = rev_zero

            # Optimal discount from coupon elasticity model
            # Revenue = price * (1 - d) * demand(d)
            # demand(d) = base_demand * exp(beta * d)
            # d* = 1 - 1/beta when beta > 1, else 0
            if beta > 1.0:
                d_star = 1.0 - 1.0 / beta
                d_star = min(max(d_star, 0.0), 0.50)
            elif beta > 0:
                # Even with beta < 1, search for marginal improvement
                # Check if discount at observed coupon level helps
                # Use the average coupon discount for this product
                avg_disc = 0.15  # typical coupon level
                rev_disc = price * (1.0 - avg_disc) * base_demand * np.exp(
                    beta * avg_disc
                )
                if rev_disc > rev_zero:
                    # Find the peak numerically in [0, 0.50]
                    test_d = np.linspace(0, 0.50, 100)
                    test_rev = price * (1.0 - test_d) * base_demand * np.exp(
                        beta * test_d
                    )
                    d_star = float(test_d[np.argmax(test_rev)])
                else:
                    d_star = 0.0
            else:
                d_star = 0.0

            # Adjust: if product has high organic ratio, penalize discounting
            # Products that sell at full price don't need discounts
            if organic_ratio > 0.5:
                # Scale down optimal discount by how organic the product is
                # At organic_ratio=1.0, multiply d_star by 0
                # At organic_ratio=0.5, keep d_star as-is
                organic_penalty = 2.0 * (1.0 - organic_ratio)
                d_star *= organic_penalty

            out["optimal_discount"][i] = d_star

            # Revenue at optimal discount
            rev_optimal = price * (1.0 - d_star) * base_demand * np.exp(
                beta * d_star
            )
            out["revenue_at_optimal"][i] = rev_optimal

            # Discount sensitivity score
            # Positive = product benefits from discounting (keep coupons)
            # Negative = product is being discounted unnecessarily (cut coupons)
            #
            # Formula: combine coupon elasticity with organic strength
            # High organic + any beta  →  low sensitivity (don't discount)
            # Low organic + high beta  →  high sensitivity (needs discounts)
            # Low organic + low beta   →  moderate (discounts help but not much)
            tier = tier_map.get(pid, 4)

            if organic_ratio > 0.5:
                # Product sells well organically: discounts are wasted margin
                # Score = negative (= "you're leaving money on table")
                sensitivity = -(organic_ratio - 0.3) * (1.0 + abs(beta))
            elif beta > 0:
                # Product responds to discounts and doesn't sell organically
                sensitivity = beta * (1.0 - organic_ratio)
            else:
                # Negative beta + low organic: product struggles regardless
                sensitivity = beta * 0.1

            out["discount_sensitivity_score"][i] = sensitivity

        console.print(
            f"  {n:,} products scored  ({time.time() - t0:.1f}s)"
        )
        return out

    # ── Export ────────────────────────────────────────────────────────

    def _export_parquet(self, results, output_dir: Path):
        """Write elasticity.parquet."""
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "elasticity.parquet"

        table = pa.table({
            "product_id": pa.array(results["product_id"], type=pa.int64()),
            "category": pa.array(results["category"], type=pa.string()),
            "elasticity_beta": pa.array(
                results["elasticity_beta"], type=pa.float64()
            ),
            "elasticity_se": pa.array(
                results["elasticity_se"], type=pa.float64()
            ),
            "p_value": pa.array(results["p_value"], type=pa.float64()),
            "optimal_discount": pa.array(
                results["optimal_discount"], type=pa.float64()
            ),
            "revenue_at_optimal": pa.array(
                results["revenue_at_optimal"], type=pa.float64()
            ),
            "revenue_at_zero_discount": pa.array(
                results["revenue_at_zero_discount"], type=pa.float64()
            ),
            "discount_sensitivity_score": pa.array(
                results["discount_sensitivity_score"], type=pa.float64()
            ),
            "organic_ratio": pa.array(
                results["organic_ratio"], type=pa.float64()
            ),
            "coupon_redemption_rate": pa.array(
                results["coupon_redemption_rate"], type=pa.float64()
            ),
            "data_source": pa.array(
                results["data_source"], type=pa.string()
            ),
        })

        pq.write_table(table, str(out_path), compression="snappy")
        console.print(f"  Wrote {out_path} ({table.num_rows:,} rows)")

    def _merge_into_product_tiers(self, output_dir: Path):
        """Add elasticity columns to product_tiers.parquet."""
        tiers_path = output_dir / "product_tiers.parquet"
        elast_path = output_dir / "elasticity.parquet"

        if not tiers_path.exists():
            console.print(
                "[yellow]  product_tiers.parquet not found, "
                "skipping merge[/yellow]"
            )
            return

        merge_con = duckdb.connect()
        merged = merge_con.execute(f"""
            SELECT
                t.*,
                e.elasticity_beta,
                e.elasticity_se,
                e.optimal_discount,
                e.revenue_at_optimal,
                e.discount_sensitivity_score,
                e.organic_ratio AS elast_organic_ratio,
                e.coupon_redemption_rate AS elast_coupon_redemption,
                e.data_source AS elasticity_source
            FROM read_parquet('{tiers_path}') t
            LEFT JOIN read_parquet('{elast_path}') e
                ON t.product_id = e.product_id
            ORDER BY t.revenue_rank
        """).fetch_arrow_table()
        merge_con.close()

        pq.write_table(merged, str(tiers_path), compression="snappy")
        console.print(
            f"  Merged elasticity into {tiers_path} "
            f"({merged.num_rows:,} rows)"
        )

    # ── Summary output ───────────────────────────────────────────────

    def _print_summary(self, results):
        """Print actionable summary tables."""
        betas = np.array(results["elasticity_beta"])
        pids = np.array(results["product_id"])
        cats = np.array(results["category"])
        sources = np.array(results["data_source"])
        opt_d = np.array(results["optimal_discount"])
        organic = np.array(results["organic_ratio"])
        sens = np.array(results["discount_sensitivity_score"])
        redemption = np.array(results["coupon_redemption_rate"])

        valid = ~np.isnan(betas)

        # ── Distribution overview ──
        table = Table(title="Elasticity Distribution", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        v = betas[valid]
        table.add_row("Products with estimates", f"{valid.sum():,}")
        table.add_row(
            "  Product-level fits",
            f"{np.sum(np.array(sources) == 'product-level'):,}",
        )
        table.add_row(
            "  Category-level fallbacks",
            f"{np.sum(np.array(sources) == 'category-level'):,}",
        )
        table.add_row("Mean beta", f"{np.mean(v):.4f}")
        table.add_row("Median beta", f"{np.median(v):.4f}")
        table.add_row("Std beta", f"{np.std(v):.4f}")
        table.add_row(
            "Products with positive beta (coupon-responsive)",
            f"{np.sum(v > 0):,}",
        )
        table.add_row(
            "Products with negative beta (coupon-resistant)",
            f"{np.sum(v < 0):,}",
        )
        table.add_row("Mean organic ratio", f"{np.mean(organic[valid]):.1%}")
        table.add_row(
            "Mean optimal discount", f"{np.mean(opt_d[valid]):.1%}"
        )
        console.print()
        console.print(table)

        # ── Per-tier breakdown ──
        try:
            tier_data = self.con.execute("""
                SELECT product_id, tier FROM revenue_product_tiers
            """).fetchnumpy()
            tier_map = dict(zip(
                tier_data["product_id"].tolist(),
                tier_data["tier"].tolist(),
            ))

            tier_table = Table(
                title="Elasticity by Product Tier", show_lines=True
            )
            tier_table.add_column("Tier", justify="center", style="bold")
            tier_table.add_column("Count", justify="right")
            tier_table.add_column("Mean Beta", justify="right")
            tier_table.add_column("Mean Organic", justify="right")
            tier_table.add_column("Mean Redemption", justify="right")
            tier_table.add_column("Mean Opt Disc", justify="right")
            tier_table.add_column("Avg Sensitivity", justify="right")

            for t in [1, 2, 3, 4]:
                mask = np.array([
                    tier_map.get(int(p), 4) == t for p in pids
                ]) & valid
                if mask.sum() == 0:
                    continue
                tier_table.add_row(
                    str(t),
                    f"{mask.sum():,}",
                    f"{np.mean(betas[mask]):.4f}",
                    f"{np.mean(organic[mask]):.1%}",
                    f"{np.mean(redemption[mask]):.1%}",
                    f"{np.mean(opt_d[mask]):.1%}",
                    f"{np.mean(sens[mask]):.4f}",
                )
            console.print()
            console.print(tier_table)
        except duckdb.CatalogException:
            tier_map = {}

        # ── Top 10 most coupon-responsive (should discount) ──
        sorted_idx = np.argsort(betas[valid])[::-1]
        top_table = Table(
            title="Top 10 Most Coupon-Responsive (Discount Candidates)",
            show_lines=True,
        )
        top_table.add_column("Product ID", justify="right")
        top_table.add_column("Category")
        top_table.add_column("Beta", justify="right")
        top_table.add_column("Organic %", justify="right")
        top_table.add_column("Redemption", justify="right")
        top_table.add_column("Opt Discount", justify="right")
        top_table.add_column("Source")

        vp = pids[valid]
        vc = cats[valid]
        vb = betas[valid]
        vo = organic[valid]
        vr = redemption[valid]
        vd = opt_d[valid]
        vs = sources[valid]

        for j in sorted_idx[:10]:
            top_table.add_row(
                str(vp[j]), str(vc[j]), f"{vb[j]:.4f}",
                f"{vo[j]:.1%}", f"{vr[j]:.1%}", f"{vd[j]:.1%}", str(vs[j]),
            )
        console.print()
        console.print(top_table)

        # ── Top 10 most inelastic (discount-resistant) ──
        bot_table = Table(
            title="Top 10 Most Coupon-Resistant (Do NOT Discount)",
            show_lines=True,
        )
        bot_table.add_column("Product ID", justify="right")
        bot_table.add_column("Category")
        bot_table.add_column("Beta", justify="right")
        bot_table.add_column("Organic %", justify="right")
        bot_table.add_column("Redemption", justify="right")
        bot_table.add_column("Opt Discount", justify="right")
        bot_table.add_column("Source")

        for j in sorted_idx[-10:][::-1]:
            bot_table.add_row(
                str(vp[j]), str(vc[j]), f"{vb[j]:.4f}",
                f"{vo[j]:.1%}", f"{vr[j]:.1%}", f"{vd[j]:.1%}", str(vs[j]),
            )
        console.print()
        console.print(bot_table)

        # ── Profit leakage: high organic + currently discounted ──
        # Products where customers buy at full price but we're still
        # offering coupons/discounts — leaving money on the table
        leakage_table = Table(
            title="Profit Leakage: High-Organic Products With Active Coupons",
            show_lines=True,
        )
        leakage_table.add_column("Product ID", justify="right")
        leakage_table.add_column("Category")
        leakage_table.add_column("Organic %", justify="right")
        leakage_table.add_column("Coupon Clips", justify="right")
        leakage_table.add_column("Redemption", justify="right")
        leakage_table.add_column("Sensitivity", justify="right")
        leakage_table.add_column("Action")

        coupon_clips = np.array([
            results.get("_coupon_clips", {}).get(int(p), 0) for p in pids
        ]) if "_coupon_clips" in results else redemption

        # High organic AND has coupon program (redemption > 0)
        leakage_mask = (organic > 0.35) & (redemption > 0) & valid
        leakage_idx = np.where(leakage_mask)[0]
        # Sort by organic ratio descending (worst offenders first)
        leakage_sorted = leakage_idx[np.argsort(organic[leakage_idx])[::-1]]

        for j in leakage_sorted[:15]:
            leakage_table.add_row(
                str(pids[j]),
                str(cats[j]),
                f"{organic[j]:.1%}",
                f"{redemption[j]:.1%}",  # using redemption as proxy
                f"{redemption[j]:.1%}",
                f"{sens[j]:.4f}",
                "reduce/remove discounts",
            )
        console.print()
        console.print(leakage_table)

        # ── Tier 1 discount analysis ──
        if tier_map:
            tier1_mask = np.array([
                tier_map.get(int(p), 4) == 1 for p in pids
            ]) & valid

            if tier1_mask.sum() > 0:
                t1_table = Table(
                    title="Tier 1: Discount Dependency Analysis",
                    show_lines=True,
                )
                t1_table.add_column("Product ID", justify="right")
                t1_table.add_column("Category")
                t1_table.add_column("Beta", justify="right")
                t1_table.add_column("Organic %", justify="right")
                t1_table.add_column("Sensitivity", justify="right")
                t1_table.add_column("Verdict")

                t1_idx = np.where(tier1_mask)[0]
                t1_sorted = t1_idx[np.argsort(sens[t1_idx])]

                for j in t1_sorted:
                    if sens[j] < -0.01:
                        verdict = "REDUCE discounts — sells organically"
                    elif sens[j] > 0.01:
                        verdict = "KEEP discounts — coupon-driven"
                    else:
                        verdict = "neutral"
                    t1_table.add_row(
                        str(pids[j]),
                        str(cats[j]),
                        f"{betas[j]:.4f}",
                        f"{organic[j]:.1%}",
                        f"{sens[j]:.4f}",
                        verdict,
                    )
                console.print()
                console.print(t1_table)

    # ── Pipeline orchestration ───────────────────────────────────────

    def run(self, output_dir: str) -> None:
        """Run the full elasticity estimation pipeline."""
        t0 = time.time()
        output_path = Path(output_dir)

        self._query_coupon_redemption()
        product_results = self._fit_product_level()
        category_results = self._fit_category_level()
        organic_data = self._compute_organic_strength()
        results = self._compute_final_scores(
            product_results, category_results, organic_data
        )
        self._export_parquet(results, output_path)
        self._merge_into_product_tiers(output_path)
        self._print_summary(results)

        # Clean up temp tables
        for tbl in ("_elast_coupon", "_elast_coupon_agg",
                     "_elast_coupon_totals"):
            try:
                self.con.execute(f"DROP TABLE IF EXISTS {tbl}")
            except Exception:
                pass

        elapsed = time.time() - t0
        console.print(
            f"\n[bold green]Elasticity estimation complete[/bold green] "
            f"({elapsed:.1f}s)"
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
    help="Output directory for parquet files.",
)
@click.option(
    "--sample-pct",
    default=2.0,
    type=float,
    help="Percent of transactions to sample (fallback if product_features "
         "not available).",
)
def main(db_path: str, output_dir: str, sample_pct: float):
    """Estimate price elasticity for each product."""
    console.print("[bold]Price Elasticity Estimation[/bold]")
    console.print(f"  DB: {db_path}")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Sample fallback: {sample_pct}%\n")

    estimator = ElasticityEstimator(db_path, sample_pct=sample_pct)
    try:
        estimator.run(output_dir)
    finally:
        estimator.close()


if __name__ == "__main__":
    main()

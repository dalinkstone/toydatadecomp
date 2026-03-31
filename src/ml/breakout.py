"""Breakout candidate identification from Tier 4 products.

Uses trained product embeddings to find Tier 4 products geometrically close
to Tier 1 products in embedding space. The intuition: if a Tier 4 product has
a similar brand, category, price point, and appeals to a similar customer
demographic as a Tier 1 product, but nobody has tried promoting it, then it
might be a breakout candidate.

Scoring formula:
  breakout_score = w1 × cosine_similarity_to_tier1_centroid
                 + w2 × category_match (binary)
                 + w3 × (1 - price_ratio_deviation)
                 + w4 × brand_presence_in_tier1 (binary)
                 - w5 × estimated_discount_to_break_in

Output: data/model/breakout_candidates.parquet

Usage:
    python src/cli.py breakout identify --top-k 100
"""

import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

console = Console()

# Assumed months of transaction data for monthly unit conversion
MONTHS_OF_DATA = 12

# Scaling factor converting log-odds elasticity beta to volume response.
# The elasticity_beta from the coupon model measures log(redemption_rate)
# response to discount fraction. Multiplying by this factor approximates
# the volume-level demand elasticity used in the discount estimation.
DISCOUNT_SCALE = 7.0

DEFAULT_WEIGHTS = {
    "w1_cosine": 0.2,
    "w2_category": 0.2,
    "w3_price": 0.2,
    "w4_brand": 0.2,
    "w5_discount": 0.2,
}


class BreakoutIdentifier:
    """Identifies breakout candidates among Tier 4 products."""

    def __init__(self, model_dir: str, data_dir: str):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)

    # ── Data loading ─────────────────────────────────────────────────

    def _load_data(self):
        """Load embeddings, tiers, elasticity, and product catalog."""
        # Product embeddings (12K × 256)
        self.embeddings = np.load(self.model_dir / "product_embeddings.npy")
        self.product_ids = np.load(self.model_dir / "product_ids.npy")
        self.pid_to_idx = {
            int(pid): idx for idx, pid in enumerate(self.product_ids)
        }

        # Product tiers (handle duplicate columns from elasticity merge)
        pf = pq.ParquetFile(self.model_dir / "product_tiers.parquet")
        tier_table = pf.read().select([
            "product_id", "tier", "total_units_sold", "total_revenue",
            "unique_customers", "revenue_rank",
        ])
        self.tiers_df = tier_table.to_pandas()

        # Elasticity parameters
        self.elasticity_df = pq.ParquetFile(
            self.model_dir / "elasticity.parquet"
        ).read().to_pandas()

        # Product catalog
        self.products_df = pq.read_table(
            self.data_dir / "real" / "products.parquet"
        ).to_pandas()

    # ── Tier 1 centroid ──────────────────────────────────────────────

    def _compute_tier1_centroid(self):
        """Compute mean embedding of Tier 1 products and prepare NN lookup."""
        tier1_pids = self.tiers_df.loc[
            self.tiers_df["tier"] == 1, "product_id"
        ].tolist()
        # Filter to products present in the embedding matrix
        self.tier1_pids = [
            pid for pid in tier1_pids if pid in self.pid_to_idx
        ]
        tier1_indices = [self.pid_to_idx[pid] for pid in self.tier1_pids]
        self.tier1_embeddings = self.embeddings[tier1_indices]

        # Centroid (L2-normalized)
        centroid = self.tier1_embeddings.mean(axis=0)
        self.tier1_centroid = centroid / np.linalg.norm(centroid)

        # Normalize Tier 1 embeddings for cosine similarity lookups
        norms = np.linalg.norm(
            self.tier1_embeddings, axis=1, keepdims=True
        )
        self.tier1_emb_normed = self.tier1_embeddings / np.maximum(
            norms, 1e-8
        )

    # ── Feature computation ──────────────────────────────────────────

    def _compute_breakout_features(self):
        """Compute all breakout features for every Tier 4 product."""
        tier4_pids = self.tiers_df.loc[
            self.tiers_df["tier"] == 4, "product_id"
        ].values

        # Lookup structures
        products_idx = self.products_df.set_index("product_id")
        tiers_idx = self.tiers_df.set_index("product_id")

        # Tier 1 categories and brands
        tier1_categories = set()
        tier1_brands = set()
        tier1_prices = {}
        for pid in self.tier1_pids:
            if pid in products_idx.index:
                row = products_idx.loc[pid]
                tier1_categories.add(row["category"])
                tier1_brands.add(row["brand"])
                tier1_prices[pid] = float(row["price"])

        # Tier 2 volume threshold (25th percentile = entry-level Tier 2)
        tier2_units = self.tiers_df.loc[
            self.tiers_df["tier"] == 2, "total_units_sold"
        ]
        self.tier2_threshold = (
            float(tier2_units.quantile(0.25))
            if len(tier2_units) > 0
            else float(self.tiers_df["total_units_sold"].median())
        )

        # Category-level average |elasticity_beta|
        self.cat_elasticity = (
            self.elasticity_df
            .groupby("category")["elasticity_beta"]
            .apply(lambda x: float(x.abs().mean()))
            .to_dict()
        )

        results = []
        for pid in tier4_pids:
            if pid not in self.pid_to_idx or pid not in products_idx.index:
                continue

            prod = products_idx.loc[pid]
            emb = self.embeddings[self.pid_to_idx[pid]]
            emb_norm = emb / max(np.linalg.norm(emb), 1e-8)

            # (a) cosine similarity to tier 1 centroid
            cos_to_centroid = float(np.dot(emb_norm, self.tier1_centroid))

            # (b,c) nearest Tier 1 neighbor
            sims = self.tier1_emb_normed @ emb_norm
            best_idx = int(np.argmax(sims))
            nearest_t1_pid = self.tier1_pids[best_idx]
            nearest_t1_sim = float(sims[best_idx])
            nearest_t1_dist = 1.0 - nearest_t1_sim

            nearest_t1_name = (
                products_idx.loc[nearest_t1_pid, "name"]
                if nearest_t1_pid in products_idx.index
                else "Unknown"
            )
            nearest_t1_price = tier1_prices.get(
                nearest_t1_pid, float(prod["price"])
            )

            # (d) price ratio
            my_price = float(prod["price"])
            price_ratio = my_price / max(nearest_t1_price, 0.01)

            # (e) category match
            category = prod["category"]
            category_match = 1 if category in tier1_categories else 0

            # (f) brand presence in tier 1
            brand = prod["brand"]
            brand_in_tier1 = 1 if brand in tier1_brands else 0

            # (g) estimated discount to break in
            current_units = (
                float(tiers_idx.loc[pid, "total_units_sold"])
                if pid in tiers_idx.index
                else 0.0
            )
            current_revenue = (
                float(tiers_idx.loc[pid, "total_revenue"])
                if pid in tiers_idx.index
                else 0.0
            )
            cat_elast = self.cat_elasticity.get(category, 0.75)

            if current_units > 0 and cat_elast > 0.01:
                volume_gap = self.tier2_threshold / current_units
                if volume_gap > 1.0:
                    discount_est = float(np.clip(
                        np.log(volume_gap) / (cat_elast * DISCOUNT_SCALE),
                        0.05, 0.50,
                    ))
                else:
                    discount_est = 0.05  # already at Tier 2 volume
            else:
                discount_est = 0.50

            results.append({
                "product_id": int(pid),
                "product_name": prod["name"],
                "category": category,
                "brand": brand,
                "price": my_price,
                "cosine_to_tier1": round(cos_to_centroid, 4),
                "nearest_tier1_product_id": int(nearest_t1_pid),
                "nearest_tier1_name": nearest_t1_name,
                "nearest_tier1_distance": round(nearest_t1_dist, 4),
                "nearest_tier1_similarity": round(nearest_t1_sim, 4),
                "nearest_tier1_price": nearest_t1_price,
                "price_ratio": round(price_ratio, 4),
                "category_match": category_match,
                "brand_in_tier1": brand_in_tier1,
                "estimated_discount_to_break_in": round(discount_est, 4),
                "category_elasticity": round(cat_elast, 4),
                "current_monthly_units": int(round(current_units / MONTHS_OF_DATA)),
                "current_monthly_revenue": round(current_revenue / MONTHS_OF_DATA, 2),
            })

        return results

    # ── Scoring ──────────────────────────────────────────────────────

    def _score_candidates(self, results, weights):
        """Score each Tier 4 product and rank by breakout_score descending."""
        w1 = weights["w1_cosine"]
        w2 = weights["w2_category"]
        w3 = weights["w3_price"]
        w4 = weights["w4_brand"]
        w5 = weights["w5_discount"]

        for r in results:
            price_ratio_deviation = abs(r["price_ratio"] - 1.0)
            price_score = max(0.0, 1.0 - price_ratio_deviation)

            r["breakout_score"] = round(
                w1 * r["cosine_to_tier1"]
                + w2 * r["category_match"]
                + w3 * price_score
                + w4 * r["brand_in_tier1"]
                - w5 * r["estimated_discount_to_break_in"],
                4,
            )

        results.sort(key=lambda x: x["breakout_score"], reverse=True)
        return results

    # ── Export ────────────────────────────────────────────────────────

    def _export(self, results, top_k):
        """Write breakout_candidates.parquet with top-K candidates."""
        out_path = self.model_dir / "breakout_candidates.parquet"
        candidates = results[:top_k]

        output_cols = [
            "product_id", "product_name", "category", "brand", "price",
            "breakout_score", "cosine_to_tier1", "nearest_tier1_product_id",
            "nearest_tier1_name", "price_ratio", "category_match",
            "brand_in_tier1", "estimated_discount_to_break_in",
            "current_monthly_units", "current_monthly_revenue",
        ]
        df = pd.DataFrame(candidates)[output_cols]
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, str(out_path), compression="snappy")
        console.print(f"  Wrote {out_path} ({len(df)} candidates)")
        return candidates

    # ── Display ───────────────────────────────────────────────────────

    def _print_top(self, candidates, n=20):
        """Print a rich table of the top N breakout candidates."""
        table = Table(
            title=f"Top {min(n, len(candidates))} Breakout Candidates",
            show_lines=True,
        )
        table.add_column("Rank", justify="right", style="bold")
        table.add_column("Product", min_width=28)
        table.add_column("Score", justify="right")
        table.add_column("Category", min_width=16)
        table.add_column("Price", justify="right")
        table.add_column("Cos→T1", justify="right")
        table.add_column("Nearest Tier 1", min_width=28)
        table.add_column("Cat?", justify="center")
        table.add_column("Brand?", justify="center")
        table.add_column("Disc%", justify="right")
        table.add_column("Mo. Units", justify="right")

        for i, c in enumerate(candidates[:n]):
            table.add_row(
                str(i + 1),
                c["product_name"][:32],
                f"{c['breakout_score']:.3f}",
                c["category"][:20],
                f"${c['price']:.2f}",
                f"{c['cosine_to_tier1']:.2f}",
                c["nearest_tier1_name"][:32],
                "\u2713" if c["category_match"] else "\u2014",
                "\u2713" if c["brand_in_tier1"] else "\u2014",
                f"{c['estimated_discount_to_break_in']:.0%}",
                f"{c['current_monthly_units']:,}",
            )

        console.print()
        console.print(table)
        console.print()

    def _print_summaries(self, candidates, n=5):
        """Print human-readable narrative summaries for the top N candidates."""
        console.print("[bold]Breakout Candidate Summaries:[/bold]\n")

        for i, c in enumerate(candidates[:n]):
            cat_str = "Same category" if c["category_match"] else "Different category"
            brand_str = (
                "same brand family" if c["brand_in_tier1"]
                else "different brand family"
            )

            discount_pct = c["estimated_discount_to_break_in"]
            monthly_now = c["current_monthly_units"]
            cat_elast = c["category_elasticity"]
            volume_multiplier = np.exp(cat_elast * DISCOUNT_SCALE * discount_pct)
            monthly_projected = int(monthly_now * volume_multiplier)

            console.print(
                f"  [bold]{i + 1}. {c['product_name']}[/bold] "
                f"(Tier 4, ${c['price']:.2f}, {c['category']})\n"
                f"     Most similar to: {c['nearest_tier1_name']} "
                f"(Tier 1, ${c['nearest_tier1_price']:.2f}, "
                f"{c['category'] if c['category_match'] else 'different category'})\n"
                f"     {cat_str}, {brand_str}, "
                f"{c['cosine_to_tier1']:.0%} embedding similarity.\n"
                f"     Estimated discount of {discount_pct:.0%} could lift "
                f"monthly units from {monthly_now:,} to ~{monthly_projected:,} "
                f"based on category elasticity of {cat_elast:.1f}.\n"
            )

    # ── Main entry point ──────────────────────────────────────────────

    def run(self, top_k=100, weights=None):
        """Run the full breakout identification pipeline."""
        if weights is None:
            weights = DEFAULT_WEIGHTS

        t0 = time.time()

        console.print("[cyan]Loading data...[/cyan]")
        self._load_data()
        console.print(
            f"  Embeddings: {self.embeddings.shape[0]:,} products "
            f"\u00d7 {self.embeddings.shape[1]} dims"
        )
        n_t1 = int((self.tiers_df["tier"] == 1).sum())
        n_t4 = int((self.tiers_df["tier"] == 4).sum())
        console.print(f"  Tier 1: {n_t1:,} products | Tier 4: {n_t4:,} products")

        console.print("[cyan]Computing Tier 1 centroid...[/cyan]")
        self._compute_tier1_centroid()
        console.print(f"  Centroid from {len(self.tier1_pids)} Tier 1 embeddings")

        console.print("[cyan]Computing breakout features for Tier 4 products...[/cyan]")
        results = self._compute_breakout_features()
        console.print(f"  Computed features for {len(results):,} Tier 4 products")

        console.print("[cyan]Scoring and ranking...[/cyan]")
        results = self._score_candidates(results, weights)

        console.print(f"[cyan]Exporting top {top_k} candidates...[/cyan]")
        candidates = self._export(results, top_k)

        self._print_top(candidates, n=20)
        self._print_summaries(candidates, n=5)

        elapsed = time.time() - t0
        console.print(
            f"[bold green]Breakout identification complete[/bold green] "
            f"({elapsed:.1f}s)"
        )
        return candidates


# ── CLI ───────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--model-dir", default="data/model/",
    help="Directory with embeddings and tier/elasticity parquets.",
)
@click.option(
    "--data-dir", default="data/",
    help="Root data directory (contains real/products.parquet).",
)
@click.option(
    "--top-k", default=100, type=int,
    help="Number of top breakout candidates to export.",
)
def main(model_dir: str, data_dir: str, top_k: int):
    """Identify breakout candidates from Tier 4 products."""
    console.print("[bold]Breakout Candidate Identification[/bold]")
    console.print(f"  Model dir: {model_dir}")
    console.print(f"  Data dir:  {data_dir}")
    console.print(f"  Top-K:     {top_k}\n")

    identifier = BreakoutIdentifier(model_dir, data_dir)
    identifier.run(top_k=top_k)


if __name__ == "__main__":
    main()

"""Full inference pipeline: score 10M customers x 12K products.

Loads pre-computed embeddings, computes affinity scores via chunked matrix
multiply, layers on revenue impact scoring (margin x tier strategy), and
outputs the Top-100 products with recommended actions.

Three modes:
  full-matrix:  chunked (10M x 256) @ (256 x 12K) -> per-product scores
  per-product:  use vecdb to find top-K customers for a product
  per-customer: use vecdb to find top-K products for a customer
"""

import os
import time

import click
import duckdb
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

console = Console()

TIER_MULTIPLIERS = {
    "coupon_loyalist": 1.2,
    "coupon_curious": 1.1,
    "organic_star": 1.3,
    "hidden_gem": 1.0,
    "unclassified": 0.8,
}

TIER_ACTIONS = {
    "coupon_loyalist": (
        "REDUCE discount by 5-10%. These customers will buy anyway. "
        "Cross-sell complementary products. Protect margin."
    ),
    "coupon_curious": (
        "OPTIMIZE the discount: switch type (e.g. dollar_off -> bogo), "
        "increase value by 10-15%, or add urgency (shorter expiration). "
        "Goal: convert clips to purchases."
    ),
    "organic_star": (
        "DO NOT discount. Focus on inventory optimization to prevent stockouts. "
        "These products sell organically. Any discount destroys margin."
    ),
    "hidden_gem": (
        "LAUNCH targeted coupon campaign. High margin supports the discount. "
        "Target customers with high affinity scores. Start with 15-20% off."
    ),
    "unclassified": "Analyze further. May need category-specific strategy.",
}


def run_full_matrix(customer_embeddings: np.ndarray,
                    product_embeddings: np.ndarray,
                    product_ids: np.ndarray,
                    product_tiers_df: pd.DataFrame,
                    chunk_size: int = 100_000,
                    device: str = "cpu") -> pd.DataFrame:
    """Compute per-product aggregate affinity scores across all customers.

    Instead of storing the full 10M x 12K score matrix, we accumulate
    per-product: sum of scores and count of high-affinity customers.
    """
    num_customers = customer_embeddings.shape[0]
    num_products = product_embeddings.shape[0]
    dev = torch.device(device)

    product_score_sum = np.zeros(num_products, dtype=np.float64)
    product_high_affinity = np.zeros(num_products, dtype=np.int64)
    affinity_threshold = 0.3

    prod_tensor = torch.from_numpy(product_embeddings).to(dev)

    console.print(f"[cyan]Scoring {num_customers - 1:,} customers x {num_products:,} products[/cyan]")
    console.print(f"  Chunk size: {chunk_size:,}, Device: {dev}")

    t0 = time.time()
    chunks_done = 0
    total_chunks = (num_customers - 1 + chunk_size - 1) // chunk_size

    for start in range(1, num_customers, chunk_size):
        end = min(start + chunk_size, num_customers)
        chunk = torch.from_numpy(customer_embeddings[start:end]).to(dev)

        # (chunk, 256) @ (256, num_products) -> (chunk, num_products)
        scores = torch.mm(chunk, prod_tensor.T)

        product_score_sum += scores.sum(dim=0).cpu().numpy()
        product_high_affinity += (scores > affinity_threshold).sum(dim=0).cpu().numpy()

        chunks_done += 1
        if chunks_done % 10 == 0 or chunks_done == total_chunks:
            elapsed = time.time() - t0
            console.print(f"  Chunk {chunks_done}/{total_chunks} ({elapsed:.1f}s)")

    active_customers = num_customers - 1
    product_avg_affinity = product_score_sum / active_customers

    # Build revenue scores
    results = []
    tier_lookup = product_tiers_df.set_index("product_id")

    for i in range(num_products):
        pid = int(product_ids[i])
        if pid not in tier_lookup.index:
            continue
        row = tier_lookup.loc[pid]
        tier = row.get("tier", "unclassified")
        margin_pct = float(row.get("margin_pct", 0))
        price = float(row.get("price", 0))
        unique_buyers = int(row.get("unique_buyers", 0))

        avg_aff = float(product_avg_affinity[i])
        demand_signal = avg_aff * unique_buyers
        dollar_margin = margin_pct * price
        tier_mult = TIER_MULTIPLIERS.get(tier, 0.8)
        revenue_score = demand_signal * dollar_margin * tier_mult

        results.append({
            "product_id": pid,
            "name": row.get("subcategory", ""),
            "brand": row.get("brand", ""),
            "category": row.get("category", ""),
            "price": price,
            "unit_cost": float(row.get("unit_cost", 0)),
            "margin_pct": margin_pct,
            "is_store_brand": bool(row.get("is_store_brand", False)),
            "tier": tier,
            "avg_affinity": avg_aff,
            "demand_signal": demand_signal,
            "dollar_margin": dollar_margin,
            "revenue_score": revenue_score,
            "high_affinity_customers": int(product_high_affinity[i]),
            "total_units_sold": int(row.get("total_units_sold", 0)),
            "coupon_clips": int(row.get("coupon_clips", 0)),
            "coupon_redemption_rate": float(row.get("coupon_redemption_rate", 0)),
        })

    df = pd.DataFrame(results).sort_values("revenue_score", ascending=False).reset_index(drop=True)
    elapsed = time.time() - t0
    console.print(f"[green]Scoring complete ({elapsed:.1f}s)[/green]")
    return df


def run_geographic(customer_embeddings: np.ndarray,
                   product_embeddings: np.ndarray,
                   product_ids: np.ndarray,
                   top100_pids: np.ndarray,
                   db_path: str) -> pd.DataFrame:
    """Compute per-state affinity for top-100 products using real store joins."""
    console.print("[cyan]Computing geographic recommendations...[/cyan]")
    con = duckdb.connect(db_path, read_only=True)

    # Get customer state mapping
    state_df = con.execute(
        "SELECT customer_id, state FROM customers ORDER BY customer_id"
    ).fetchdf()
    con.close()

    max_cid = customer_embeddings.shape[0]
    customer_states = np.full(max_cid, "", dtype=object)
    customer_states[state_df["customer_id"].values] = state_df["state"].values

    # Build product_id -> embedding index mapping
    pid_to_idx = {int(pid): i for i, pid in enumerate(product_ids)}
    top100_indices = [pid_to_idx[pid] for pid in top100_pids if pid in pid_to_idx]
    top100_embs = product_embeddings[top100_indices]  # (100, 256)

    unique_states = sorted(set(customer_states[1:]) - {""})
    rows = []

    for state in unique_states:
        cids_in_state = np.where(customer_states == state)[0]
        if len(cids_in_state) == 0:
            continue
        state_embs = customer_embeddings[cids_in_state]
        # (N, 256) @ (256, 100) -> (N, 100)
        scores = state_embs @ top100_embs.T
        avg_scores = scores.mean(axis=0)

        for j, pid in enumerate(top100_pids):
            if pid in pid_to_idx:
                rows.append({
                    "state": state,
                    "product_id": int(pid),
                    "avg_affinity": float(avg_scores[j]),
                    "customers_in_state": len(cids_in_state),
                })

    df = pd.DataFrame(rows)
    console.print(f"  {len(unique_states)} states, {len(df)} state-product pairs")
    return df


def print_top100(df: pd.DataFrame):
    """Pretty-print the top-100 products."""
    tbl = Table(title="Top 100 Products — Revenue Impact Ranking")
    tbl.add_column("Rank", style="dim", width=4)
    tbl.add_column("ID", width=5)
    tbl.add_column("Category", width=20)
    tbl.add_column("Brand", width=15)
    tbl.add_column("Price", justify="right", width=7)
    tbl.add_column("Margin", justify="right", width=7)
    tbl.add_column("Tier", style="cyan", width=16)
    tbl.add_column("Rev Score", justify="right", width=10)
    tbl.add_column("High-Aff Custs", justify="right", width=12)

    for i, row in df.head(25).iterrows():
        tier_style = {
            "coupon_loyalist": "green",
            "organic_star": "blue",
            "hidden_gem": "magenta",
            "coupon_curious": "yellow",
        }.get(row["tier"], "dim")

        tbl.add_row(
            str(i + 1),
            str(row["product_id"]),
            str(row["category"])[:20],
            str(row["brand"])[:15],
            f"${row['price']:.2f}",
            f"{row['margin_pct']:.0%}",
            f"[{tier_style}]{row['tier']}[/{tier_style}]",
            f"{row['revenue_score']:.0f}",
            f"{row['high_affinity_customers']:,}",
        )

    console.print(tbl)
    console.print(f"\n  (showing top 25 of {len(df)} ranked products)")


@click.command()
@click.option("--db-path", default="data/db/cvs_analytics.duckdb")
@click.option("--model-dir", default="data/model/")
@click.option("--output-dir", default="data/results/")
@click.option("--mode", default="full-matrix",
              type=click.Choice(["full-matrix", "per-product", "per-customer"]))
@click.option("--top-k", default=100, help="Number of top products to output.")
@click.option("--chunk-size", default=100_000, help="Customers per chunk.")
@click.option("--device", default="auto",
              type=click.Choice(["auto", "mps", "cpu"]))
@click.option("--geographic/--no-geographic", default=True,
              help="Include state-level geographic grouping.")
def main(db_path: str, model_dir: str, output_dir: str, mode: str,
         top_k: int, chunk_size: int, device: str, geographic: bool):
    """Run full inference and revenue optimization."""
    console.print("[bold]Inference & Revenue Optimization Pipeline[/bold]")
    console.print(f"  Mode: {mode}, Top-K: {top_k}, Device: {device}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Load embeddings
    console.print("[cyan]Loading embeddings...[/cyan]")
    cust_emb_path = os.path.join(model_dir, "customer_embeddings.npy")
    prod_emb_path = os.path.join(model_dir, "product_embeddings.npy")
    prod_ids_path = os.path.join(model_dir, "product_ids.npy")

    customer_embeddings = np.load(cust_emb_path)
    product_embeddings = np.load(prod_emb_path)
    product_ids = np.load(prod_ids_path)

    console.print(f"  Customers: {customer_embeddings.shape}")
    console.print(f"  Products: {product_embeddings.shape}")

    # Load product tiers from DuckDB
    con = duckdb.connect(db_path, read_only=True)
    product_tiers_df = con.execute("SELECT * FROM product_tiers").fetchdf()
    con.close()

    dev = "cpu"
    if device == "auto" and torch.backends.mps.is_available():
        dev = "mps"
    elif device != "auto":
        dev = device

    if mode == "full-matrix":
        ranked_df = run_full_matrix(
            customer_embeddings, product_embeddings, product_ids,
            product_tiers_df, chunk_size=chunk_size, device=dev)

        # Add recommended actions
        ranked_df["recommended_action"] = ranked_df["tier"].map(TIER_ACTIONS)

        # Save full rankings
        full_path = os.path.join(output_dir, "product_rankings.parquet")
        ranked_df.to_parquet(full_path, index=False)
        console.print(f"\n  Full rankings: {full_path} ({len(ranked_df)} products)")

        # Save top-K
        top_df = ranked_df.head(top_k)
        top_path = os.path.join(output_dir, "top100_products.parquet")
        top_df.to_parquet(top_path, index=False)
        top_df.to_csv(os.path.join(output_dir, "top100_products.csv"), index=False)
        console.print(f"  Top {top_k}: {top_path}")

        # Print summary
        print_top100(ranked_df)

        # Tier distribution in top 100
        console.print(f"\n[bold]Tier distribution (Top {top_k}):[/bold]")
        for tier, cnt in top_df["tier"].value_counts().items():
            console.print(f"  {tier}: {cnt}")

        # Geographic grouping
        if geographic:
            top100_pids = top_df["product_id"].values
            geo_df = run_geographic(
                customer_embeddings, product_embeddings, product_ids,
                top100_pids, db_path)
            geo_path = os.path.join(output_dir, "state_recommendations.parquet")
            geo_df.to_parquet(geo_path, index=False)
            console.print(f"  Geographic: {geo_path}")

    elif mode in ("per-product", "per-customer"):
        console.print("[yellow]per-product and per-customer modes use vecdb.[/yellow]")
        try:
            from vecdb.vecdb_wrapper import VecDB
        except ImportError:
            console.print("[red]vecdb_wrapper not available. Build vecdb.dylib first.[/red]")
            raise SystemExit(1)

        if mode == "per-product":
            # Load customer embeddings into vecdb, query with product vectors
            db = VecDB(capacity=customer_embeddings.shape[0], dims=256)
            cust_ids = np.arange(1, customer_embeddings.shape[0], dtype=np.uint32)
            db.batch_insert(cust_ids, customer_embeddings[1:])
            console.print(f"  Loaded {db.count()} customer vectors into vecdb")

            results = []
            for i, pid in enumerate(product_ids):
                query = product_embeddings[i:i+1]
                ids, scores = db.batch_query_topk(query, top_k)
                for j in range(top_k):
                    results.append({
                        "product_id": int(pid),
                        "customer_id": int(ids[0, j]),
                        "score": float(scores[0, j]),
                        "rank": j + 1,
                    })
            out_path = os.path.join(output_dir, "per_product_recommendations.parquet")
            pd.DataFrame(results).to_parquet(out_path, index=False)
            console.print(f"  Output: {out_path}")

        else:  # per-customer
            db = VecDB(capacity=product_embeddings.shape[0], dims=256)
            prod_ids_u32 = product_ids.astype(np.uint32)
            db.batch_insert(prod_ids_u32, product_embeddings)
            console.print(f"  Loaded {db.count()} product vectors into vecdb")

            # Score a sample of customers (all 10M would be huge output)
            sample_size = min(100_000, customer_embeddings.shape[0] - 1)
            sample_cids = np.random.choice(
                np.arange(1, customer_embeddings.shape[0]), size=sample_size, replace=False)
            sample_embs = customer_embeddings[sample_cids]

            ids, scores = db.batch_query_topk(sample_embs, top_k)
            results = []
            for i in range(sample_size):
                for j in range(top_k):
                    results.append({
                        "customer_id": int(sample_cids[i]),
                        "product_id": int(ids[i, j]),
                        "score": float(scores[i, j]),
                        "rank": j + 1,
                    })
            out_path = os.path.join(output_dir, "per_customer_recommendations.parquet")
            pd.DataFrame(results).to_parquet(out_path, index=False)
            console.print(f"  Output: {out_path}")

    console.print(f"\n[bold green]Inference complete![/bold green]")


if __name__ == "__main__":
    main()

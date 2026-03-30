"""Decision & Ranking Layer for per-customer recommendations.

Reads pre-computed embeddings, computes raw affinity scores via chunked matrix
multiply, then applies a multi-stage business-logic ranking layer:

  1. Margin boost: higher-margin products get a configurable score lift
  2. Recency suppression: recently-purchased products are penalized
  3. Coupon eligibility: products the customer has shown coupon interest in get boosted
  4. Category diversity: no more than K products from the same category in top-N

Designed for full scale (10M customers x 12K products) with chunked processing.
Demo mode available for testing at 10K customer scale.

Usage:
    python src/ranking/decision_engine.py
    python src/ranking/decision_engine.py --demo
    python src/ranking/decision_engine.py --top-k 20 --margin-weight 0.5
"""

import os
import time
from dataclasses import dataclass

import click
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from rich.console import Console
from rich.table import Table

console = Console()


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RankingConfig:
    """All thresholds and weights for the business-logic ranking layer.

    Sensible defaults tuned for CVS-scale retail (12K products, 27 categories).
    All values are configurable via CLI options.
    """
    top_k: int = 10                    # recommendations per customer
    recency_products: int = 5          # suppress last N distinct products purchased
    recency_decay: float = 0.5         # multiply score by this for recently-purchased
    max_same_category: int = 3         # max items from same category in top-K
    margin_weight: float = 0.3         # boost = 1 + margin_pct * margin_weight
    coupon_boost: float = 0.15         # score multiplier for coupon-eligible products
    chunk_size: int = 50_000           # customers per processing chunk
    candidate_multiplier: int = 5      # get top_k * this candidates before diversity filter


# Parquet output schema
OUTPUT_SCHEMA = pa.schema([
    ("customer_id", pa.int32()),
    ("rank", pa.int8()),
    ("product_id", pa.int32()),
    ("raw_score", pa.float32()),
    ("final_score", pa.float32()),
    ("category", pa.string()),
    ("margin_pct", pa.float32()),
])


# ═══════════════════════════════════════════════════════════════════════
# Recency materialization
# ═══════════════════════════════════════════════════════════════════════

def materialize_recency(con: duckdb.DuckDBPyConnection, window: int):
    """Create a table of each customer's N most-recently-purchased distinct products.

    Three-stage approach designed for 10M customers × 10B transactions on 64GB RAM:
      1. GROUP BY customer_id, product_id to get last purchase date per pair
         (~4.4B rows, streams to disk via DuckDB spill)
      2. Chunked ROW_NUMBER: process customer_id ranges in batches so the
         window-sort never exceeds ~4GB of working memory
      3. Concatenate chunk results into the final _recent_purchases table
    """
    console.print(f"[cyan]Materializing recency data "
                  f"(last {window} distinct products per customer)...[/cyan]")
    t0 = time.time()

    # Tune DuckDB for this heavy workload
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=2")

    # Stage 1: aggregate to customer-product level (one pass over transactions)
    console.print("  Stage 1/3: aggregating customer-product pairs...")
    con.execute("""
        CREATE OR REPLACE TABLE _cust_prod_last AS
        SELECT customer_id, product_id, MAX(date) AS last_date
        FROM transactions
        GROUP BY customer_id, product_id
    """)
    n1 = con.execute("SELECT COUNT(*) FROM _cust_prod_last").fetchone()[0]
    console.print(f"    {n1:,} customer-product pairs ({time.time() - t0:.1f}s)")

    # Stage 2: chunked ROW_NUMBER to avoid OOM on the window sort.
    # With 4.4B rows and 10M customers, each customer has ~440 products on avg.
    # Processing 500K customers at a time means ~220M rows per chunk, which
    # sorts comfortably in <4GB of memory.
    console.print("  Stage 2/3: selecting most recent products per customer (chunked)...")

    max_cid = con.execute(
        "SELECT MAX(customer_id) FROM _cust_prod_last"
    ).fetchone()[0]
    chunk_size = 500_000
    con.execute("CREATE OR REPLACE TABLE _recent_purchases (customer_id INTEGER, product_id INTEGER)")

    chunks_done = 0
    total_chunks = (max_cid + chunk_size - 1) // chunk_size

    for cid_start in range(1, max_cid + 1, chunk_size):
        cid_end = min(cid_start + chunk_size, max_cid + 1)
        con.execute(f"""
            INSERT INTO _recent_purchases
            SELECT customer_id, product_id
            FROM (
                SELECT customer_id, product_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY customer_id
                           ORDER BY last_date DESC
                       ) AS rn
                FROM _cust_prod_last
                WHERE customer_id >= {cid_start}
                  AND customer_id < {cid_end}
            )
            WHERE rn <= {window}
        """)
        chunks_done += 1
        if chunks_done % max(1, total_chunks // 10) == 0 or chunks_done == total_chunks:
            elapsed = time.time() - t0
            console.print(f"    Chunk {chunks_done}/{total_chunks} "
                          f"({elapsed:.1f}s elapsed)")

    # Stage 3: cleanup
    console.print("  Stage 3/3: cleaning up intermediate table...")
    con.execute("DROP TABLE IF EXISTS _cust_prod_last")

    # Restore default thread count
    con.execute("SET threads=4")

    n = con.execute("SELECT COUNT(*) FROM _recent_purchases").fetchone()[0]
    elapsed = time.time() - t0
    console.print(f"  _recent_purchases: {n:,} rows ({elapsed:.1f}s)")


def _build_lookup(cids: np.ndarray, pids: np.ndarray) -> dict[int, set[int]]:
    """Build {customer_id: {product_ids}} dict from sorted arrays."""
    if len(cids) == 0:
        return {}

    changes = np.where(np.diff(cids) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(cids)]])

    lookup: dict[int, set[int]] = {}
    for s, e in zip(starts, ends):
        lookup[int(cids[s])] = set(int(p) for p in pids[s:e])
    return lookup


def load_recency(con: duckdb.DuckDBPyConnection) -> dict[int, set[int]]:
    """Load all recency data into memory as {customer_id: {product_ids}}.

    Loads in chunks to avoid a single 50M-row DataFrame allocation.
    Final dict: ~10M keys × set of ~5 ints = ~2-3GB.
    """
    console.print("[cyan]Loading recency lookup...[/cyan]")
    t0 = time.time()

    total = con.execute("SELECT COUNT(*) FROM _recent_purchases").fetchone()[0]
    if total == 0:
        console.print("  No recency data found")
        return {}

    recency: dict[int, set[int]] = {}
    chunk = 5_000_000
    offset = 0

    while offset < total:
        df = con.execute(
            f"SELECT customer_id, product_id "
            f"FROM _recent_purchases "
            f"ORDER BY customer_id "
            f"LIMIT {chunk} OFFSET {offset}"
        ).fetchdf()
        if len(df) == 0:
            break
        for cid, pid in zip(df["customer_id"].values, df["product_id"].values):
            recency.setdefault(int(cid), set()).add(int(pid))
        offset += chunk

    console.print(f"  {len(recency):,} customers with recency data "
                  f"({time.time() - t0:.1f}s)")
    return recency


def load_coupon_data(con: duckdb.DuckDBPyConnection) -> dict[int, set[int]]:
    """Load coupon clip data as {customer_id: {product_ids}}.

    Placeholder for future coupon classifier: uses existing coupon_clips
    as a signal of coupon eligibility (customer clipped -> likely to redeem).
    The coupon classifier (when built) will replace this lookup.
    """
    console.print("[cyan]Loading coupon eligibility lookup...[/cyan]")
    t0 = time.time()
    df = con.execute("""
        SELECT DISTINCT c.customer_id, cc.product_id
        FROM coupon_clips cc
        JOIN customers c ON c.loyalty_number = cc.loyalty_number
        ORDER BY c.customer_id
    """).fetchdf()

    if len(df) == 0:
        console.print("  No coupon data found")
        return {}

    coupons = _build_lookup(df["customer_id"].values, df["product_id"].values)
    console.print(f"  {len(coupons):,} customers with coupon data "
                  f"({time.time() - t0:.1f}s)")
    return coupons


# ═══════════════════════════════════════════════════════════════════════
# Per-chunk ranking
# ═══════════════════════════════════════════════════════════════════════

def rank_chunk(
    raw_scores: np.ndarray,
    customer_ids: np.ndarray,
    product_ids: np.ndarray,
    product_categories: np.ndarray,
    product_margins: np.ndarray,
    margin_boost: np.ndarray,
    pid_to_idx: dict[int, int],
    recency: dict[int, set[int]],
    coupon_data: dict[int, set[int]],
    config: RankingConfig,
) -> pa.Table:
    """Apply business-logic ranking to a chunk of customers.

    Steps per chunk:
      1. Margin boost (vectorized broadcast across all customers x products)
      2. Recency suppression (sparse coordinate update)
      3. Coupon eligibility boost (sparse coordinate update)
      4. Top-K candidate selection via torch.topk
      5. Category diversity constraint (greedy selection per customer)

    Returns a PyArrow table matching OUTPUT_SCHEMA.
    """
    chunk_size = raw_scores.shape[0]
    num_products = len(product_ids)
    candidate_k = min(config.top_k * config.candidate_multiplier, num_products)

    # Work on a copy so raw_scores stays pristine for output
    scores = raw_scores.astype(np.float32).copy()

    # ── 1. Margin boost (vectorized) ─────────────────────────────────
    scores *= margin_boost  # (1, num_products) broadcasts to (chunk, num_products)

    # ── 2. Recency suppression (sparse) ──────────────────────────────
    rec_rows, rec_cols = [], []
    for i in range(chunk_size):
        recent_pids = recency.get(int(customer_ids[i]))
        if recent_pids:
            for pid in recent_pids:
                col = pid_to_idx.get(pid)
                if col is not None:
                    rec_rows.append(i)
                    rec_cols.append(col)
    if rec_rows:
        scores[np.array(rec_rows), np.array(rec_cols)] *= config.recency_decay

    # ── 3. Coupon eligibility boost (sparse) ─────────────────────────
    coup_rows, coup_cols = [], []
    for i in range(chunk_size):
        coupon_pids = coupon_data.get(int(customer_ids[i]))
        if coupon_pids:
            for pid in coupon_pids:
                col = pid_to_idx.get(pid)
                if col is not None:
                    coup_rows.append(i)
                    coup_cols.append(col)
    if coup_rows:
        scores[np.array(coup_rows), np.array(coup_cols)] *= (1.0 + config.coupon_boost)

    # ── 4. Top-K candidate selection ─────────────────────────────────
    with torch.no_grad():
        top_vals, top_idx = torch.topk(
            torch.from_numpy(scores), k=candidate_k, dim=1
        )
    top_vals_np = top_vals.numpy()
    top_idx_np = top_idx.numpy()

    # ── 5. Category diversity + output assembly ──────────────────────
    out_cids = []
    out_ranks = []
    out_pids = []
    out_raw = []
    out_final = []
    out_cats = []
    out_margins = []

    for i in range(chunk_size):
        cid = int(customer_ids[i])
        selected = 0
        cat_counts: dict[str, int] = {}

        for j in range(candidate_k):
            if selected >= config.top_k:
                break
            prod_idx = int(top_idx_np[i, j])
            cat = product_categories[prod_idx]
            cc = cat_counts.get(cat, 0)
            if cc >= config.max_same_category:
                continue
            cat_counts[cat] = cc + 1
            selected += 1

            out_cids.append(cid)
            out_ranks.append(selected)
            out_pids.append(int(product_ids[prod_idx]))
            out_raw.append(float(raw_scores[i, prod_idx]))
            out_final.append(float(top_vals_np[i, j]))
            out_cats.append(cat)
            out_margins.append(float(product_margins[prod_idx]))

    return pa.table(
        {
            "customer_id": pa.array(out_cids, type=pa.int32()),
            "rank": pa.array(out_ranks, type=pa.int8()),
            "product_id": pa.array(out_pids, type=pa.int32()),
            "raw_score": pa.array(out_raw, type=pa.float32()),
            "final_score": pa.array(out_final, type=pa.float32()),
            "category": pa.array(out_cats, type=pa.string()),
            "margin_pct": pa.array(out_margins, type=pa.float32()),
        },
        schema=OUTPUT_SCHEMA,
    )


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def _print_sample(path: str, n_customers: int = 5):
    """Pretty-print sample recommendations from the output file."""
    tbl_data = pq.read_table(path)
    df = tbl_data.to_pandas()
    sample_cids = df["customer_id"].unique()[:n_customers]
    sample = df[df["customer_id"].isin(sample_cids)]

    tbl = Table(title=f"Sample Recommendations ({n_customers} customers)")
    tbl.add_column("Customer", width=10)
    tbl.add_column("Rank", width=4)
    tbl.add_column("Product", width=8)
    tbl.add_column("Raw", justify="right", width=8)
    tbl.add_column("Final", justify="right", width=8)
    tbl.add_column("Category", width=22)
    tbl.add_column("Margin", justify="right", width=8)

    for _, row in sample.iterrows():
        tbl.add_row(
            str(row["customer_id"]),
            str(row["rank"]),
            str(row["product_id"]),
            f"{row['raw_score']:.4f}",
            f"{row['final_score']:.4f}",
            str(row["category"])[:22],
            f"{row['margin_pct']:.0%}",
        )

    console.print(tbl)


def _print_summary(path: str, config: RankingConfig):
    """Print distribution statistics from the output file."""
    tbl_data = pq.read_table(path)
    df = tbl_data.to_pandas()

    console.print(f"\n[bold]Category distribution (across all rank-1 picks):[/bold]")
    rank1 = df[df["rank"] == 1]
    for cat, cnt in rank1["category"].value_counts().head(10).items():
        console.print(f"  {cat}: {cnt:,}")

    console.print(f"\n[bold]Score statistics:[/bold]")
    console.print(f"  Raw score  — mean: {df['raw_score'].mean():.4f}, "
                  f"std: {df['raw_score'].std():.4f}, "
                  f"max: {df['raw_score'].max():.4f}")
    console.print(f"  Final score — mean: {df['final_score'].mean():.4f}, "
                  f"std: {df['final_score'].std():.4f}, "
                  f"max: {df['final_score'].max():.4f}")


@click.command()
@click.option("--db-path", default="data/db/cvs_analytics.duckdb")
@click.option("--model-dir", default="data/model/")
@click.option("--output-dir", default="data/results/")
@click.option("--top-k", default=10, help="Recommendations per customer.")
@click.option("--chunk-size", default=50_000, type=int,
              help="Customers per processing chunk.")
@click.option("--device", default="auto",
              type=click.Choice(["auto", "mps", "cpu"]))
@click.option("--recency-window", default=5, type=int,
              help="Suppress last N distinct products purchased.")
@click.option("--recency-decay", default=0.5, type=float,
              help="Score multiplier for recently-purchased products.")
@click.option("--max-same-category", default=3, type=int,
              help="Max items from the same category in top-K.")
@click.option("--margin-weight", default=0.3, type=float,
              help="Margin boost weight: score *= 1 + margin_pct * weight.")
@click.option("--coupon-boost", default=0.15, type=float,
              help="Score multiplier for coupon-eligible products.")
@click.option("--demo", is_flag=True,
              help="Demo mode: process only first 10K customers.")
@click.option("--skip-recency-build", is_flag=True,
              help="Skip recency materialization if _recent_purchases table exists.")
def main(db_path, model_dir, output_dir, top_k, chunk_size, device,
         recency_window, recency_decay, max_same_category, margin_weight,
         coupon_boost, demo, skip_recency_build):
    """Run the decision & ranking layer on top of model affinity scores."""

    config = RankingConfig(
        top_k=top_k,
        recency_products=recency_window,
        recency_decay=recency_decay,
        max_same_category=max_same_category,
        margin_weight=margin_weight,
        coupon_boost=coupon_boost,
        chunk_size=chunk_size,
    )

    console.print("[bold]Decision & Ranking Layer[/bold]")
    console.print(f"  Top-K: {config.top_k}, Chunk size: {config.chunk_size:,}")
    console.print(f"  Recency: last {config.recency_products} products, "
                  f"decay={config.recency_decay}")
    console.print(f"  Diversity: max {config.max_same_category} per category in top-{config.top_k}")
    console.print(f"  Margin weight: {config.margin_weight}, "
                  f"Coupon boost: {config.coupon_boost}")
    if demo:
        console.print("[yellow]  Demo mode: limiting to 10K customers[/yellow]")
    console.print()

    os.makedirs(output_dir, exist_ok=True)

    # ── Load embeddings ──────────────────────────────────────────────
    console.print("[cyan]Loading embeddings...[/cyan]")
    customer_embeddings = np.load(os.path.join(model_dir, "customer_embeddings.npy"))
    product_embeddings = np.load(os.path.join(model_dir, "product_embeddings.npy"))
    product_ids = np.load(os.path.join(model_dir, "product_ids.npy"))

    num_customers = customer_embeddings.shape[0]   # index 0 is unused (1-based IDs)
    num_products = product_embeddings.shape[0]
    console.print(f"  Customers: {num_customers - 1:,}, Products: {num_products:,}")

    if demo:
        max_cid = min(10_001, num_customers)       # 1-based: IDs 1..10000
    else:
        max_cid = num_customers

    active_customers = max_cid - 1
    console.print(f"  Processing: {active_customers:,} customers\n")

    # ── Load product metadata ────────────────────────────────────────
    console.print("[cyan]Loading product metadata...[/cyan]")
    con = duckdb.connect(db_path)
    con.execute("SET memory_limit='24GB'")
    con.execute("SET temp_directory='/tmp/duckdb_temp'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=2")

    product_df = con.execute("""
        SELECT product_id, category,
               (price - unit_cost) / NULLIF(price, 0) AS margin_pct
        FROM products
    """).fetchdf()

    pid_to_idx = {int(pid): i for i, pid in enumerate(product_ids)}
    product_categories = np.full(num_products, "unknown", dtype=object)
    product_margins = np.zeros(num_products, dtype=np.float32)

    for _, row in product_df.iterrows():
        idx = pid_to_idx.get(int(row["product_id"]))
        if idx is not None:
            product_categories[idx] = str(row["category"])
            product_margins[idx] = float(row["margin_pct"] or 0)

    margin_boost = (1.0 + product_margins * config.margin_weight).reshape(1, -1)
    console.print(f"  {len(product_df)} products mapped, "
                  f"{len(product_df['category'].unique())} categories")

    # ── Materialize recency data ─────────────────────────────────────
    has_recency = con.execute("""
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_name = '_recent_purchases'
    """).fetchone()[0] > 0

    if has_recency and skip_recency_build:
        console.print("[yellow]Skipping recency materialization "
                      "(table exists, --skip-recency-build).[/yellow]")
    else:
        materialize_recency(con, config.recency_products)

    # ── Load lookup data ─────────────────────────────────────────────
    recency = load_recency(con)

    try:
        coupon_data = load_coupon_data(con)
    except Exception as e:
        console.print(f"[yellow]Could not load coupon data: {e}[/yellow]")
        coupon_data = {}

    con.close()

    # ── Device setup ─────────────────────────────────────────────────
    dev = "cpu"
    if device == "auto" and torch.backends.mps.is_available():
        dev = "mps"
    elif device != "auto":
        dev = device
    console.print(f"  Compute device: {dev}\n")

    prod_tensor = torch.from_numpy(product_embeddings).to(dev)

    # ── Chunked ranking with streaming parquet output ────────────────
    output_path = os.path.join(output_dir, "ranked_recommendations.parquet")
    writer = pq.ParquetWriter(output_path, OUTPUT_SCHEMA)

    total_chunks = (active_customers + config.chunk_size - 1) // config.chunk_size
    chunks_done = 0
    total_recs = 0
    t0 = time.time()

    console.print(f"[cyan]Ranking {active_customers:,} customers "
                  f"in {total_chunks} chunks...[/cyan]")

    for start in range(1, max_cid, config.chunk_size):
        end = min(start + config.chunk_size, max_cid)
        chunk_cids = np.arange(start, end)
        chunk_emb = customer_embeddings[start:end]

        # Compute raw affinity scores: (chunk, 256) @ (256, products)
        with torch.no_grad():
            chunk_tensor = torch.from_numpy(chunk_emb).to(dev)
            raw_scores = torch.mm(chunk_tensor, prod_tensor.T).cpu().numpy()

        # Apply business-logic ranking
        chunk_table = rank_chunk(
            raw_scores=raw_scores,
            customer_ids=chunk_cids,
            product_ids=product_ids,
            product_categories=product_categories,
            product_margins=product_margins,
            margin_boost=margin_boost,
            pid_to_idx=pid_to_idx,
            recency=recency,
            coupon_data=coupon_data,
            config=config,
        )

        writer.write_table(chunk_table)
        total_recs += len(chunk_table)

        chunks_done += 1
        if chunks_done % max(1, total_chunks // 20) == 0 or chunks_done == total_chunks:
            elapsed = time.time() - t0
            rate = chunks_done / elapsed if elapsed > 0 else 0
            eta = (total_chunks - chunks_done) / rate if rate > 0 else 0
            console.print(
                f"  Chunk {chunks_done}/{total_chunks} "
                f"({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)"
            )

    writer.close()
    elapsed = time.time() - t0

    # ── Summary ──────────────────────────────────────────────────────
    file_size = os.path.getsize(output_path)
    if file_size < 1024**3:
        size_str = f"{file_size / 1024**2:.1f} MB"
    else:
        size_str = f"{file_size / 1024**3:.2f} GB"

    console.print(f"\n[bold green]Ranking complete![/bold green]")
    console.print(f"  Output: {output_path}")
    console.print(f"  Recommendations: {total_recs:,} "
                  f"({active_customers:,} customers x up to {config.top_k} recs)")
    console.print(f"  File size: {size_str}")
    console.print(f"  Throughput: {elapsed:.1f}s "
                  f"({active_customers / elapsed:,.0f} customers/sec)")

    _print_sample(output_path)
    _print_summary(output_path, config)


if __name__ == "__main__":
    main()

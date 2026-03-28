"""Generate synthetic digital coupon clip events.

Creates a sparse table of coupon clips: ~10-15% of customers are active
clippers, each clipping 3-30 coupons over a 3-year window (2022-2024).
Only OTC products get digital coupons (no Rx).

Clip probability is weighted by product popularity_score so popular items
get clipped more often. Active clippers skew toward ages 25-55 (app-savvy).

Outputs partitioned Parquet files to data/synthetic/coupon_clips/.
"""

import multiprocessing
import os
import time
from pathlib import Path

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATE_START = np.datetime64("2022-01-01")
DATE_END = np.datetime64("2024-12-31")
DATE_RANGE_DAYS = int((DATE_END - DATE_START) / np.timedelta64(1, "D")) + 1

DISCOUNT_TYPES = ["percent_off", "dollar_off", "bogo"]
DISCOUNT_TYPE_PROBS = [0.50, 0.35, 0.15]

# Probability a customer is an active coupon clipper, by age band
CLIPPER_PROB_BY_AGE = {
    (18, 24): 0.08,
    (25, 34): 0.15,
    (35, 44): 0.18,
    (45, 54): 0.16,
    (55, 64): 0.10,
    (65, 74): 0.06,
    (75, 89): 0.03,
}

# How many coupons an active clipper clips (min, max) by age band
CLIPS_RANGE_BY_AGE = {
    (18, 24): (3, 12),
    (25, 34): (5, 20),
    (35, 44): (8, 30),
    (45, 54): (6, 25),
    (55, 64): (4, 18),
    (65, 74): (3, 10),
    (75, 89): (2, 8),
}

# Redemption rate by discount type
REDEMPTION_RATE = {
    "percent_off": 0.35,
    "dollar_off": 0.45,
    "bogo": 0.25,
}


def _age_band(age: int) -> tuple:
    for band in CLIPPER_PROB_BY_AGE:
        if band[0] <= age <= band[1]:
            return band
    return (75, 89)


# ---------------------------------------------------------------------------
# Shared product data (loaded once, passed to workers via initializer)
# ---------------------------------------------------------------------------

_product_ids = None
_product_weights = None
_product_prices = None


def _init_worker(pid_arr, pw_arr, pp_arr):
    global _product_ids, _product_weights, _product_prices
    _product_ids = pid_arr
    _product_weights = pw_arr
    _product_prices = pp_arr


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def _generate_batch(args: tuple) -> tuple:
    """Process a batch of customers and generate their coupon clips.

    Returns (output_path, num_clips).
    """
    batch_id, customer_file, output_dir, base_seed = args

    rng = np.random.default_rng(base_seed + batch_id)

    # Read customer batch
    ct = pq.read_table(customer_file, columns=["customer_id", "loyalty_number", "age"])
    customer_ids = ct.column("customer_id").to_numpy()
    loyalty_numbers = ct.column("loyalty_number").to_pylist()
    ages = ct.column("age").to_numpy()

    n_customers = len(customer_ids)

    # Determine which customers are active clippers
    clipper_probs = np.array([CLIPPER_PROB_BY_AGE[_age_band(a)] for a in ages])
    is_clipper = rng.random(n_customers) < clipper_probs

    clipper_indices = np.where(is_clipper)[0]
    if len(clipper_indices) == 0:
        return None, 0

    # For each clipper, determine how many clips
    all_clip_ids = []
    all_loyalty = []
    all_product_ids = []
    all_clip_dates = []
    all_exp_dates = []
    all_discount_types = []
    all_discount_values = []
    all_redeemed = []

    # Precompute product selection weights (normalized)
    pw = _product_weights / _product_weights.sum()

    for idx in clipper_indices:
        age = int(ages[idx])
        band = _age_band(age)
        lo, hi = CLIPS_RANGE_BY_AGE[band]
        n_clips = rng.integers(lo, hi + 1)

        # Pick products weighted by popularity
        chosen_products = rng.choice(len(_product_ids), size=n_clips, replace=True, p=pw)

        # Generate clip dates uniformly across the 3-year window
        day_offsets = rng.integers(0, DATE_RANGE_DAYS, size=n_clips)
        clip_dates = DATE_START + day_offsets.astype("timedelta64[D]")

        # Expiration: 7-30 days after clip
        exp_offsets = rng.integers(7, 31, size=n_clips)
        exp_dates = clip_dates + exp_offsets.astype("timedelta64[D]")

        # Discount types
        dtype_indices = rng.choice(len(DISCOUNT_TYPES), size=n_clips, p=DISCOUNT_TYPE_PROBS)
        dtypes = [DISCOUNT_TYPES[i] for i in dtype_indices]

        # Discount values based on type and product price
        d_values = np.empty(n_clips, dtype=np.float64)
        for i in range(n_clips):
            pid_idx = chosen_products[i]
            price = _product_prices[pid_idx]
            dt = dtypes[i]
            if dt == "percent_off":
                # 5%, 10%, 15%, 20%, 25%, 30%, 40% off
                d_values[i] = rng.choice([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
            elif dt == "dollar_off":
                # $0.50 to ~30% of price, rounded to $0.50
                max_off = max(0.50, round(price * 0.30 * 2) / 2)
                choices = np.arange(0.50, max_off + 0.25, 0.50)
                d_values[i] = rng.choice(choices)
            else:  # bogo
                d_values[i] = 1.0  # "buy one get one" — value=1 signals BOGO

        # Redemption
        redemption_probs = np.array([REDEMPTION_RATE[dt] for dt in dtypes])
        redeemed = rng.random(n_clips) < redemption_probs

        all_loyalty.extend([loyalty_numbers[idx]] * n_clips)
        all_product_ids.extend(_product_ids[chosen_products].tolist())
        all_clip_dates.extend(clip_dates.tolist())
        all_exp_dates.extend(exp_dates.tolist())
        all_discount_types.extend(dtypes)
        all_discount_values.extend(d_values.tolist())
        all_redeemed.extend(redeemed.tolist())

    total_clips = len(all_loyalty)
    if total_clips == 0:
        return None, 0

    # Clip IDs are assigned globally later — use batch-local sequential for now
    # We'll use batch_id * large_offset + local_index to avoid collisions
    offset = batch_id * 2_000_000  # plenty of room per batch
    clip_ids = np.arange(offset + 1, offset + 1 + total_clips, dtype=np.int64)

    table = pa.table(
        {
            "clip_id": pa.array(clip_ids, type=pa.int64()),
            "loyalty_number": pa.array(all_loyalty, type=pa.string()),
            "product_id": pa.array(all_product_ids, type=pa.int32()),
            "clip_date": pa.array(all_clip_dates, type=pa.date32()),
            "expiration_date": pa.array(all_exp_dates, type=pa.date32()),
            "discount_type": pa.array(all_discount_types, type=pa.string()),
            "discount_value": pa.array(all_discount_values, type=pa.float64()),
            "redeemed": pa.array(all_redeemed, type=pa.bool_()),
        }
    )

    out_path = os.path.join(output_dir, f"coupon_clips_{batch_id:05d}.parquet")
    pq.write_table(table, out_path, compression="snappy")

    return out_path, total_clips


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--workers", default=8, help="Number of parallel workers.")
@click.option("--customer-dir", default="data/synthetic/customers", help="Customer parquet dir.")
@click.option("--product-file", default="data/real/products.parquet", help="Product parquet file.")
@click.option("--output-dir", default="data/synthetic/coupon_clips", help="Output directory.")
@click.option("--seed", default=99, help="Base random seed.")
@click.option("--test", is_flag=True, help="Test mode: process only first customer batch.")
def main(
    workers: int,
    customer_dir: str,
    product_file: str,
    output_dir: str,
    seed: int,
    test: bool,
) -> None:
    """Generate synthetic digital coupon clip events."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load products (OTC only — no Rx coupons)
    products = pq.read_table(product_file, columns=["product_id", "is_rx", "popularity_score", "price"])
    otc_mask = pa.compute.equal(products.column("is_rx"), False)
    products = products.filter(otc_mask)

    product_ids = products.column("product_id").to_numpy().astype(np.int32)
    product_weights = products.column("popularity_score").to_numpy().astype(np.float64)
    product_prices = products.column("price").to_numpy().astype(np.float64)

    console.print(f"[bold]Loaded {len(product_ids):,} OTC products for coupon generation[/bold]")

    # Discover customer batch files
    cust_files = sorted(Path(customer_dir).glob("customers_*.parquet"))
    if not cust_files:
        console.print("[red]No customer parquet files found. Run gen_customers first.[/red]")
        return

    if test:
        cust_files = cust_files[:1]
        console.print("[cyan]TEST MODE: processing only first customer batch[/cyan]")

    num_batches = len(cust_files)
    batch_args = [
        (i, str(f), output_dir, seed)
        for i, f in enumerate(cust_files)
    ]

    console.print(
        f"[bold]Generating coupon clips → {output_dir}/[/bold]\n"
        f"  Workers: {workers} | Customer batches: {num_batches}"
    )

    t0 = time.perf_counter()
    total_clips = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating clips", total=num_batches)

        with multiprocessing.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(product_ids, product_weights, product_prices),
        ) as pool:
            for path, n_clips in pool.imap_unordered(_generate_batch, batch_args):
                total_clips += n_clips
                progress.advance(task)

    elapsed = time.perf_counter() - t0

    console.print(
        f"\n[bold green]Done![/bold green] {total_clips:,} coupon clips in {elapsed:.1f}s "
        f"({total_clips / elapsed:,.0f} clips/sec)"
    )

    _print_stats(output_dir)


def _print_stats(output_dir: str) -> None:
    """Read back all parquet files and print distribution stats."""
    from collections import Counter

    ds = pq.ParquetDataset(output_dir)
    table = ds.read()

    total = len(table)
    console.print(f"\n[bold]Total clips:[/bold] {total:,}")

    # Unique customers
    loyalty = table.column("loyalty_number").to_pylist()
    unique_customers = len(set(loyalty))
    console.print(f"[bold]Unique clippers:[/bold] {unique_customers:,}")
    console.print(f"[bold]Avg clips per clipper:[/bold] {total / max(unique_customers, 1):.1f}")

    # Discount type distribution
    dtypes = Counter(table.column("discount_type").to_pylist())
    console.print("\n[bold]Discount type distribution:[/bold]")
    for dt, cnt in dtypes.most_common():
        console.print(f"  {dt}: {cnt:,} ({cnt / total * 100:.1f}%)")

    # Redemption rate
    redeemed = sum(table.column("redeemed").to_pylist())
    console.print(f"\n[bold]Overall redemption rate:[/bold] {redeemed:,} / {total:,} ({redeemed / total * 100:.1f}%)")

    # Top 10 products by clip count
    prod_counts = Counter(table.column("product_id").to_pylist())
    console.print("\n[bold]Top 10 most-clipped products:[/bold]")
    for pid, cnt in prod_counts.most_common(10):
        console.print(f"  product_id {pid}: {cnt:,} clips")


if __name__ == "__main__":
    main()

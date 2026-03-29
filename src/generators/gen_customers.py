"""Generate synthetic customer profiles.

Creates 10M synthetic CVS customers with realistic demographics using
multiprocessing + Faker. Outputs partitioned Parquet files (snappy).

Target: 10M rows in under 15 minutes on M4 Max with 8 workers.
"""

import multiprocessing
import os
import time
from pathlib import Path

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from faker import Faker
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
# Distribution constants
# ---------------------------------------------------------------------------

# Age bins and their probabilities (pharmacy customer skew)
AGE_BINS = [(18, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 89)]
AGE_PROBS = [0.12, 0.18, 0.16, 0.18, 0.16, 0.12, 0.08]

# Gender distribution
GENDERS = ["M", "F", "NB"]
GENDER_PROBS = [0.45, 0.53, 0.02]

# Email domain distribution
EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "aol.com", "icloud.com", "comcast.net", "live.com",
]
EMAIL_DOMAIN_PROBS = [0.50, 0.18, 0.10, 0.08, 0.05, 0.05, 0.02, 0.02]

# State distribution proportional to CVS store density
STATES = [
    "CA", "FL", "TX", "NY", "OH", "PA", "IL", "MA", "NJ", "GA",
    "NC", "VA", "MI", "AZ", "MD", "IN", "TN", "MO", "CT", "SC",
    "MN", "WI", "CO", "AL", "KY", "LA", "OR", "OK", "NV", "IA",
    "RI", "MS", "AR", "UT", "KS", "NE", "NM", "NH", "WV", "ME",
    "HI", "ID", "DE", "MT", "VT", "SD", "ND", "WY", "AK", "DC",
]
_STATE_WEIGHTS = [
    120, 90, 80, 50, 50, 45, 40, 38, 35, 30,
    28, 27, 25, 22, 20, 18, 17, 15, 14, 13,
    12, 11, 10, 10, 9, 9, 8, 8, 7, 7,
    6, 6, 5, 5, 5, 4, 4, 4, 3, 3,
    3, 3, 3, 2, 2, 2, 1, 1, 1, 1,
]
_sw_total = sum(_STATE_WEIGHTS)
STATE_PROBS = [w / _sw_total for w in _STATE_WEIGHTS]

# Student probability by age
STUDENT_PROB_BY_AGE = {
    (18, 22): 0.65,
    (23, 25): 0.25,
    (26, 30): 0.05,
}
STUDENT_PROB_DEFAULT = 0.01


def _student_prob(age: int) -> float:
    for (lo, hi), prob in STUDENT_PROB_BY_AGE.items():
        if lo <= age <= hi:
            return prob
    return STUDENT_PROB_DEFAULT


# ---------------------------------------------------------------------------
# Batch generation (runs in worker process)
# ---------------------------------------------------------------------------


def _generate_batch(args: tuple) -> str:
    """Generate one batch of customers and write to Parquet. Returns file path."""
    batch_id, start_id, batch_size, output_dir, base_seed = args

    rng = np.random.default_rng(base_seed + batch_id)
    fake = Faker("en_US")
    Faker.seed(base_seed + batch_id)
    fake.seed_instance(base_seed + batch_id)

    n = batch_size
    customer_ids = np.arange(start_id, start_id + n, dtype=np.int32)

    # --- Vectorised draws ---------------------------------------------------

    # Ages: pick bin then uniform within bin
    bin_indices = rng.choice(len(AGE_BINS), size=n, p=AGE_PROBS)
    ages = np.empty(n, dtype=np.int8)
    for i, (lo, hi) in enumerate(AGE_BINS):
        mask = bin_indices == i
        ages[mask] = rng.integers(lo, hi + 1, size=mask.sum(), dtype=np.int8)

    # Genders
    gender_indices = rng.choice(len(GENDERS), size=n, p=GENDER_PROBS)
    genders = np.array(GENDERS)[gender_indices]

    # States
    state_indices = rng.choice(len(STATES), size=n, p=STATE_PROBS)
    states = np.array(STATES)[state_indices]

    # Email domains
    domain_indices = rng.choice(len(EMAIL_DOMAINS), size=n, p=EMAIL_DOMAIN_PROBS)
    domains = np.array(EMAIL_DOMAINS)[domain_indices]

    # Student status (vectorised by age)
    student_probs = np.vectorize(_student_prob)(ages)
    is_student = rng.random(n) < student_probs

    # --- Faker-based fields (names, addresses, zips, phones) ----------------

    first_names = []
    last_names = []
    addresses = []
    cities = []
    zip_codes = []
    phones = []

    for i in range(n):
        first_names.append(fake.first_name())
        last_names.append(fake.last_name())
        addresses.append(fake.street_address())
        cities.append(fake.city())
        st = states[i]
        try:
            zip_codes.append(fake.zipcode_in_state(state_abbr=st))
        except Exception:
            zip_codes.append(fake.zipcode())
        phones.append(fake.numerify("(###) ###-####"))

    # --- Derived fields -----------------------------------------------------

    loyalty_numbers = [f"EC{cid:010d}" for cid in customer_ids]

    emails = [
        f"{fn.lower()}.{ln.lower()}{cid % 10000}@{dom}"
        for fn, ln, cid, dom in zip(first_names, last_names, customer_ids, domains)
    ]

    # --- Build Arrow table and write ----------------------------------------

    table = pa.table(
        {
            "customer_id": pa.array(customer_ids, type=pa.int32()),
            "loyalty_number": pa.array(loyalty_numbers, type=pa.string()),
            "first_name": pa.array(first_names, type=pa.string()),
            "last_name": pa.array(last_names, type=pa.string()),
            "age": pa.array(ages, type=pa.int8()),
            "gender": pa.array(genders.tolist(), type=pa.string()),
            "address": pa.array(addresses, type=pa.string()),
            "city": pa.array(cities, type=pa.string()),
            "state": pa.array(states.tolist(), type=pa.string()),
            "zip_code": pa.array(zip_codes, type=pa.string()),
            "is_student": pa.array(is_student.tolist(), type=pa.bool_()),
            "email": pa.array(emails, type=pa.string()),
            "phone": pa.array(phones, type=pa.string()),
        }
    )

    out_path = os.path.join(output_dir, f"customers_{batch_id:05d}.parquet")
    pq.write_table(table, out_path, compression="snappy")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--count", default=10_000_000, help="Number of customers to generate.")
@click.option("--workers", default=8, help="Number of parallel workers.")
@click.option("--batch-size", default=100_000, help="Customers per batch/file.")
@click.option("--output-dir", default="data/synthetic/customers", help="Output directory.")
@click.option("--seed", default=42, help="Base random seed.")
@click.option("--test", is_flag=True, help="Test mode: generate only 10,000 customers.")
def main(
    count: int,
    workers: int,
    batch_size: int,
    output_dir: str,
    seed: int,
    test: bool,
) -> None:
    """Generate synthetic CVS customer profiles."""
    if test:
        count = 10_000
        batch_size = min(batch_size, 10_000)
        console.print("[cyan]TEST MODE: generating 10,000 customers[/cyan]")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    num_batches = (count + batch_size - 1) // batch_size
    # Adjust last batch size
    batch_args = []
    for b in range(num_batches):
        start_id = b * batch_size + 1
        this_batch = min(batch_size, count - b * batch_size)
        batch_args.append((b, start_id, this_batch, output_dir, seed))

    console.print(
        f"[bold]Generating {count:,} customers → {output_dir}/[/bold]\n"
        f"  Workers: {workers} | Batches: {num_batches} | Batch size: {batch_size:,}"
    )

    t0 = time.perf_counter()
    completed = 0

    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating", total=num_batches)

            with multiprocessing.Pool(processes=workers) as pool:
                for _path in pool.imap_unordered(_generate_batch, batch_args):
                    completed += 1
                    progress.advance(task)
    except KeyboardInterrupt:
        elapsed = time.perf_counter() - t0
        done_rows = completed * batch_size
        console.print(f"\n[yellow]Interrupted after {completed}/{num_batches} batches "
                      f"({done_rows:,} customers saved to {output_dir}/)[/yellow]")
        return

    elapsed = time.perf_counter() - t0
    rows_per_sec = count / elapsed

    # --- Print stats --------------------------------------------------------
    console.print(f"\n[bold green]Done![/bold green] {count:,} customers in {elapsed:.1f}s "
                  f"({rows_per_sec:,.0f} rows/sec)")

    _print_stats(output_dir)


def _print_stats(output_dir: str) -> None:
    """Read back all parquet files and print distribution stats."""
    import pyarrow.parquet as pq

    ds = pq.ParquetDataset(output_dir)
    table = ds.read(columns=["age", "gender", "state", "is_student"])

    ages = table.column("age").to_numpy()
    genders = table.column("gender").to_pylist()
    states_col = table.column("state").to_pylist()
    students = table.column("is_student").to_pylist()

    total = len(ages)

    # Gender distribution
    from collections import Counter
    gc = Counter(genders)
    console.print("\n[bold]Gender distribution:[/bold]")
    for g in ["M", "F", "NB"]:
        console.print(f"  {g}: {gc[g]:,} ({gc[g] / total * 100:.1f}%)")

    # Age distribution
    console.print("\n[bold]Age distribution:[/bold]")
    for lo, hi in AGE_BINS:
        cnt = int(((ages >= lo) & (ages <= hi)).sum())
        console.print(f"  {lo}-{hi}: {cnt:,} ({cnt / total * 100:.1f}%)")

    # Top 10 states
    sc = Counter(states_col)
    console.print("\n[bold]Top 10 states:[/bold]")
    for st, cnt in sc.most_common(10):
        console.print(f"  {st}: {cnt:,} ({cnt / total * 100:.1f}%)")

    # Student percentage
    student_count = sum(students)
    console.print(f"\n[bold]Students:[/bold] {student_count:,} ({student_count / total * 100:.1f}%)")


if __name__ == "__main__":
    main()

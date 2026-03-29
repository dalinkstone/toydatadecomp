"""Convert transaction .csv.zst files to Parquet format.

Processes one file at a time to stay within memory limits.
Parquet gives: true column pruning, row-group chunking, faster scans,
and dramatically lower memory usage for aggregation queries.

Usage:
    python scripts/csv_to_parquet.py
    python scripts/csv_to_parquet.py --delete-source
"""

import os
import time
from pathlib import Path

import click
import duckdb
from rich.console import Console

console = Console()


@click.command()
@click.option("--transactions-dir", default="data/synthetic/transactions",
              help="Transactions data dir.")
@click.option("--delete-source/--keep-source", default=False,
              help="Delete .csv.zst after conversion.")
def main(transactions_dir: str, delete_source: bool) -> None:
    """Convert transaction .csv.zst files to Parquet (one at a time)."""
    txn_dir = Path(transactions_dir)
    csv_files = sorted(txn_dir.glob("*.csv.zst"))

    if not csv_files:
        console.print("[red]No .csv.zst files found.[/red]")
        return

    console.print(f"[bold]Converting {len(csv_files)} files → Parquet[/bold]")
    console.print(f"  Source: {txn_dir}")
    console.print(f"  Delete source after: {delete_source}\n")

    t0_total = time.time()

    for i, csv_path in enumerate(csv_files):
        parquet_path = csv_path.with_suffix("").with_suffix(".parquet")
        console.print(f"[cyan][{i+1}/{len(csv_files)}] {csv_path.name} → {parquet_path.name}[/cyan]")

        if parquet_path.exists():
            console.print(f"  Already exists, skipping.")
            continue

        t0 = time.time()

        # Fresh connection per file to release all memory between files
        con = duckdb.connect()
        con.execute("SET memory_limit='32GB'")
        con.execute("SET threads=4")
        con.execute("SET preserve_insertion_order=false")

        con.execute(f"""
            COPY (
                SELECT * FROM read_csv('{csv_path}', ignore_errors=true)
            ) TO '{parquet_path}'
            (FORMAT PARQUET, CODEC 'ZSTD', ROW_GROUP_SIZE 122880)
        """)
        con.close()

        csv_size = csv_path.stat().st_size / (1024**3)
        pq_size = parquet_path.stat().st_size / (1024**3)
        elapsed = time.time() - t0

        console.print(f"  {csv_size:.1f} GB → {pq_size:.1f} GB ({elapsed:.0f}s)")

        if delete_source:
            csv_path.unlink()
            console.print(f"  Deleted {csv_path.name}")

    total = time.time() - t0_total
    console.print(f"\n[bold green]Conversion complete[/bold green] ({total:.0f}s)")

    # Summary
    parquet_files = sorted(txn_dir.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in parquet_files) / (1024**3)
    console.print(f"  {len(parquet_files)} Parquet files, {total_size:.1f} GB total")


if __name__ == "__main__":
    main()

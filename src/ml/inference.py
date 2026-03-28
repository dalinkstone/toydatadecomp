"""Run full inference: score 10M users × 10K products.

Loads the trained two-tower model, pre-computes all item embeddings
into the vecdb (C vector database), then for each user computes the
user embedding and retrieves top-K product recommendations via
SIMD-accelerated dot product search.

Output: recommendations table written back to DuckDB.
"""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--db-path", default="data/db/toydatadecomp.duckdb", help="DuckDB database path.")
@click.option("--model-path", default="data/db/model.pt", help="Trained model path.")
@click.option("--top-k", default=20, help="Number of recommendations per user.")
@click.option("--batch-size", default=1024, help="Inference batch size.")
def main(db_path: str, model_path: str, top_k: int, batch_size: int) -> None:
    """Run full inference (10M users × 10K products)."""
    console.print(f"[bold]Running inference[/bold]")
    console.print(f"  Model: {model_path}")
    console.print(f"  Top-K: {top_k}, Batch size: {batch_size}")
    # TODO: Implement inference pipeline
    # 1. Load model
    # 2. Compute all item embeddings → load into vecdb
    # 3. For each user batch: compute user embeddings → vecdb top-K
    # 4. Write recommendations to DuckDB
    console.print("[yellow]Not yet implemented.[/yellow]")


if __name__ == "__main__":
    main()

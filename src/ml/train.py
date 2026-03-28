"""Train the two-tower recommendation model.

Loads training data from DuckDB, constructs user-item pairs with
positive (purchased) and negative (not purchased) samples, and trains
the two-tower model using binary cross-entropy loss.

Training reads directly from DuckDB to avoid materializing the full
dataset in memory. Uses PyTorch DataLoader with custom DuckDB-backed
dataset for streaming mini-batches.
"""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--db-path", default="data/db/toydatadecomp.duckdb", help="DuckDB database path.")
@click.option("--epochs", default=5, help="Number of training epochs.")
@click.option("--batch-size", default=4096, help="Training batch size.")
@click.option("--lr", default=0.001, type=float, help="Learning rate.")
@click.option("--embedding-dim", default=64, help="Embedding dimension.")
@click.option("--neg-samples", default=4, help="Negative samples per positive.")
@click.option("--output", default="data/db/model.pt", help="Output model path.")
def main(db_path: str, epochs: int, batch_size: int, lr: float,
         embedding_dim: int, neg_samples: int, output: str) -> None:
    """Train the two-tower recommendation model."""
    console.print(f"[bold]Training two-tower model[/bold]")
    console.print(f"  DB: {db_path}")
    console.print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    console.print(f"  Embedding dim: {embedding_dim}, Neg samples: {neg_samples}")
    # TODO: Implement training loop
    console.print("[yellow]Not yet implemented.[/yellow]")


if __name__ == "__main__":
    main()

"""Two-tower neural network model for purchase prediction.

Defines the user tower and item tower architectures. Each tower maps
its input features to a dense embedding vector. Purchase probability
is computed as the dot product of user and item embeddings.

User tower input: customer_id, age, gender, zip_code, loyalty_tier,
                  purchase history embedding (aggregated).
Item tower input: product_id, category, subcategory, brand, price.

Both towers output a 64-dimensional embedding vector.
"""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--embedding-dim", default=64, help="Embedding dimension for both towers.")
@click.option("--hidden-dim", default=256, help="Hidden layer dimension.")
def main(embedding_dim: int, hidden_dim: int) -> None:
    """Print model architecture summary."""
    console.print(f"[bold]Two-Tower Model[/bold]")
    console.print(f"  Embedding dim: {embedding_dim}")
    console.print(f"  Hidden dim: {hidden_dim}")
    # TODO: Implement model definition with PyTorch
    console.print("[yellow]Not yet implemented.[/yellow]")


if __name__ == "__main__":
    main()

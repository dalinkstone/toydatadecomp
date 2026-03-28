"""Generate synthetic customer profiles.

Creates 10M synthetic customers with realistic demographics using Faker:
customer_id, first_name, last_name, email, age, gender, zip_code,
home_store_id, loyalty_tier, signup_date.

Outputs to data/synthetic/customers/ as CSV files.
"""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--count", default=10_000_000, help="Number of customers to generate.")
@click.option("--output-dir", default="data/synthetic/customers", help="Output directory.")
@click.option("--chunk-size", default=1_000_000, help="Rows per output file.")
@click.option("--seed", default=42, help="Random seed for reproducibility.")
def main(count: int, output_dir: str, chunk_size: int, seed: int) -> None:
    """Generate synthetic customer profiles."""
    console.print(f"[bold]Generating {count:,} customers → {output_dir}/[/bold]")
    # TODO: Implement customer generation with Faker
    console.print("[yellow]Not yet implemented.[/yellow]")


if __name__ == "__main__":
    main()

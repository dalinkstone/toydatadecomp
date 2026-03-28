"""Scrape CVS product catalog.

Collects product data including product ID, name, brand, category,
subcategory, price, and UPC for ~10,000 CVS products.
Outputs to data/real/products.csv.
"""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--output", default="data/real/products.csv", help="Output CSV path.")
@click.option("--limit", default=None, type=int, help="Max products to scrape (None=all).")
def main(output: str, limit: int | None) -> None:
    """Scrape CVS product catalog and save to CSV."""
    console.print(f"[bold]Scraping CVS product catalog → {output}[/bold]")
    # TODO: Implement product catalog scraping
    console.print("[yellow]Not yet implemented.[/yellow]")


if __name__ == "__main__":
    main()

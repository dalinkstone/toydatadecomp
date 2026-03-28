"""Scrape CVS store locations.

Collects store data including store ID, address, city, state, zip code,
latitude, longitude, and phone number for all ~9,000 CVS locations.
Outputs to data/real/stores.csv.
"""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--output", default="data/real/stores.csv", help="Output CSV path.")
@click.option("--limit", default=None, type=int, help="Max stores to scrape (None=all).")
def main(output: str, limit: int | None) -> None:
    """Scrape CVS store locations and save to CSV."""
    console.print(f"[bold]Scraping CVS store locations → {output}[/bold]")
    # TODO: Implement store location scraping
    console.print("[yellow]Not yet implemented.[/yellow]")


if __name__ == "__main__":
    main()

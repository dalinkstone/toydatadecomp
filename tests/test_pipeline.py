"""End-to-end pipeline tests.

Runs a mini version of the full pipeline with small data sizes
to verify all stages connect correctly: scrape → generate → load → train → infer.
"""

import click
from rich.console import Console

console = Console()


@click.command()
def main() -> None:
    """Run end-to-end pipeline tests."""
    console.print("[bold]Running pipeline integration tests...[/bold]")
    # TODO: Implement tests
    # - Generate 100 customers, 1000 transactions
    # - Load into in-memory DuckDB
    # - Train model for 1 epoch
    # - Run inference on small subset
    # - Verify output format
    console.print("[yellow]Not yet implemented.[/yellow]")


if __name__ == "__main__":
    main()

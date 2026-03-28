"""Convert .csv.zst files to Parquet format.

OPTIONAL: DuckDB reads .csv.zst natively with predicate pushdown,
so this conversion is not required for the pipeline to work.
Parquet offers faster analytical queries and better compression
for repeated reads.

Converts:
  data/synthetic/customers/*.csv.zst → *.parquet
  data/synthetic/transactions/*.csv.zst → *.parquet
"""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--customers-dir", default="data/synthetic/customers", help="Customers data dir.")
@click.option("--transactions-dir", default="data/synthetic/transactions", help="Transactions data dir.")
@click.option("--delete-source/--keep-source", default=False, help="Delete .csv.zst after conversion.")
def main(customers_dir: str, transactions_dir: str, delete_source: bool) -> None:
    """Convert .csv.zst files to Parquet."""
    console.print("[bold]Converting .csv.zst → Parquet[/bold]")
    console.print(f"  Customers: {customers_dir}")
    console.print(f"  Transactions: {transactions_dir}")
    console.print(f"  Delete source: {delete_source}")
    # TODO: Implement conversion using PyArrow or DuckDB COPY
    console.print("[yellow]Not yet implemented.[/yellow]")


if __name__ == "__main__":
    main()

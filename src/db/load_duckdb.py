"""Load all data into DuckDB.

Reads real data CSVs and synthetic data (.csv.zst or .parquet, whichever
exists) and loads them into a DuckDB database at data/db/toydatadecomp.duckdb.

DuckDB reads .csv.zst files natively with full predicate pushdown, so
Parquet conversion is not required.
"""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--db-path", default="data/db/toydatadecomp.duckdb", help="DuckDB database path.")
@click.option("--schema", default="src/db/schema.sql", help="SQL schema file.")
def main(db_path: str, schema: str) -> None:
    """Load all data into DuckDB."""
    console.print(f"[bold]Loading data into DuckDB → {db_path}[/bold]")
    # TODO: Implement data loading
    # - Execute schema.sql
    # - Load data/real/stores.csv into stores table
    # - Load data/real/products.csv into products table
    # - Load data/synthetic/customers/*.csv into customers table
    # - Load data/synthetic/transactions/*.csv.zst (or *.parquet) into transactions table
    console.print("[yellow]Not yet implemented.[/yellow]")


if __name__ == "__main__":
    main()

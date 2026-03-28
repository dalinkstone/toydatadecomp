"""CLI entry point for the toydatadecomp pipeline.

Provides a unified command-line interface with subcommands for each
pipeline stage: scrape, generate, load, train, infer, validate.

Usage:
    python src/cli.py scrape --stores
    python src/cli.py generate --customers --count 10000000
    python src/cli.py load
    python src/cli.py train --epochs 5
    python src/cli.py infer --top-k 20
    python src/cli.py validate
"""

import click
from rich.console import Console

console = Console()


@click.group()
def cli():
    """toydatadecomp — Retail Recommendation Engine CLI."""
    pass


@cli.command()
@click.option("--stores/--no-stores", default=True, help="Scrape store locations.")
@click.option("--products/--no-products", default=True, help="Scrape product catalog.")
def scrape(stores: bool, products: bool) -> None:
    """Scrape real CVS data (stores and products)."""
    if stores:
        from scrapers.scrape_stores import main as scrape_stores
        console.print("[bold]→ Scraping stores...[/bold]")
        scrape_stores(standalone_mode=False)
    if products:
        from scrapers.scrape_products import main as scrape_products
        console.print("[bold]→ Scraping products...[/bold]")
        scrape_products(standalone_mode=False)


@cli.command()
@click.option("--customers/--no-customers", default=True, help="Generate synthetic customers.")
@click.option("--count", default=10_000_000, help="Number of customers to generate.")
@click.option("--workers", default=8, help="Number of parallel workers.")
@click.option("--batch-size", default=100_000, help="Customers per batch/file.")
@click.option("--test", is_flag=True, help="Test mode: generate only 10,000 customers.")
@click.option("--coupon-clips/--no-coupon-clips", default=True, help="Generate coupon clips.")
def generate(customers: bool, count: int, workers: int, batch_size: int, test: bool,
             coupon_clips: bool) -> None:
    """Generate synthetic data (customers, coupon clips, transactions)."""
    if customers:
        from generators.gen_customers import main as gen_customers
        args = ["--count", str(count), "--workers", str(workers),
                "--batch-size", str(batch_size)]
        if test:
            args.append("--test")
        console.print(f"[bold]→ Generating customers...[/bold]")
        gen_customers(args, standalone_mode=False)
    if coupon_clips:
        from generators.gen_coupon_clips import main as gen_clips
        args = ["--workers", str(workers)]
        if test:
            args.append("--test")
        console.print(f"[bold]→ Generating coupon clips...[/bold]")
        gen_clips(args, standalone_mode=False)


@cli.command()
@click.option("--db-path", default="data/db/cvs_analytics.duckdb", help="DuckDB path.")
@click.option("--data-dir", default="data", help="Root data directory.")
@click.option("--materialize", is_flag=True, default=False,
              help="Materialize expensive views as tables.")
def load(db_path: str, data_dir: str, materialize: bool) -> None:
    """Load all data into DuckDB."""
    from db.load_duckdb import main as load_db
    console.print("[bold]→ Loading data into DuckDB...[/bold]")
    args = ["--db-path", db_path, "--data-dir", data_dir]
    if materialize:
        args.append("--materialize")
    load_db(args, standalone_mode=False)


@cli.command()
@click.option("--epochs", default=5, help="Training epochs.")
@click.option("--batch-size", default=4096, help="Batch size.")
@click.option("--lr", default=0.001, type=float, help="Learning rate.")
def train(epochs: int, batch_size: int, lr: float) -> None:
    """Train the two-tower recommendation model."""
    from ml.train import main as train_model
    console.print("[bold]→ Training model...[/bold]")
    train_model(["--epochs", str(epochs), "--batch-size", str(batch_size),
                 "--lr", str(lr)], standalone_mode=False)


@cli.command()
@click.option("--top-k", default=20, help="Recommendations per user.")
@click.option("--batch-size", default=1024, help="Inference batch size.")
def infer(top_k: int, batch_size: int) -> None:
    """Run full inference (10M × 10K scoring)."""
    from ml.inference import main as run_inference
    console.print("[bold]→ Running inference...[/bold]")
    run_inference(["--top-k", str(top_k), "--batch-size", str(batch_size)],
                  standalone_mode=False)


@cli.command()
@click.option("--db-path", default="data/db/cvs_analytics.duckdb", help="DuckDB path.")
def validate(db_path: str) -> None:
    """Run data validation checks."""
    from db.load_duckdb import validate as validate_db
    console.print("[bold]→ Running validation...[/bold]")
    validate_db(["--db-path", db_path], standalone_mode=False)


if __name__ == "__main__":
    cli()

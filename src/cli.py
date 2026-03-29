"""CLI entry point for the toydatadecomp pipeline.

Provides a unified command-line interface with subcommands for each
pipeline stage: scrape, generate, load, features, train, infer, validate.

Usage:
    python src/cli.py scrape --stores
    python src/cli.py generate --customers --count 10000000
    python src/cli.py load
    python src/cli.py features
    python src/cli.py train --epochs 5
    python src/cli.py infer --mode full-matrix
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
@click.option("--db-path", default="data/db/cvs_analytics.duckdb", help="DuckDB path.")
@click.option("--sample-pct", default=1.0, type=float,
              help="Percent of transactions to sample (1.0 = 1%%).")
@click.option("--skip-txn-scan", is_flag=True,
              help="Skip if feature tables already exist.")
def features(db_path: str, sample_pct: float, skip_txn_scan: bool) -> None:
    """Run feature engineering (materializes DuckDB tables)."""
    from ml.features import main as run_features
    console.print("[bold]→ Running feature engineering...[/bold]")
    args = ["--db-path", db_path, "--sample-pct", str(sample_pct)]
    if skip_txn_scan:
        args.append("--skip-txn-scan")
    run_features(args, standalone_mode=False)


@cli.command()
@click.option("--db-path", default="data/db/cvs_analytics.duckdb", help="DuckDB path.")
@click.option("--epochs", default=5, help="Training epochs.")
@click.option("--batch-size", default=8192, help="Batch size.")
@click.option("--lr", default=0.001, type=float, help="Learning rate.")
@click.option("--sample-pct", default=1.0, type=float,
              help="Percent of transactions to sample.")
@click.option("--device", default="auto",
              type=click.Choice(["auto", "mps", "cpu"]))
@click.option("--output-dir", default="data/model/", help="Model output directory.")
@click.option("--neg-samples", default=4, help="Negative samples per positive.")
@click.option("--margin-weight/--no-margin-weight", default=True,
              help="Use margin-weighted loss.")
@click.option("--skip-features", is_flag=True,
              help="Skip feature engineering (assume tables exist).")
def train(db_path: str, epochs: int, batch_size: int, lr: float,
          sample_pct: float, device: str, output_dir: str,
          neg_samples: int, margin_weight: bool, skip_features: bool) -> None:
    """Train the two-tower recommendation model."""
    from ml.train import main as train_model
    console.print("[bold]→ Training model...[/bold]")
    args = ["--db-path", db_path, "--epochs", str(epochs),
            "--batch-size", str(batch_size), "--lr", str(lr),
            "--sample-pct", str(sample_pct), "--device", device,
            "--output-dir", output_dir, "--neg-samples", str(neg_samples)]
    if margin_weight:
        args.append("--margin-weight")
    else:
        args.append("--no-margin-weight")
    if skip_features:
        args.append("--skip-features")
    train_model(args, standalone_mode=False)


@cli.command()
@click.option("--db-path", default="data/db/cvs_analytics.duckdb", help="DuckDB path.")
@click.option("--model-dir", default="data/model/", help="Model directory.")
@click.option("--output-dir", default="data/results/", help="Results output directory.")
@click.option("--mode", default="full-matrix",
              type=click.Choice(["full-matrix", "per-product", "per-customer"]))
@click.option("--top-k", default=100, help="Number of top products.")
@click.option("--chunk-size", default=100_000, help="Customers per inference chunk.")
@click.option("--device", default="auto",
              type=click.Choice(["auto", "mps", "cpu"]))
@click.option("--geographic/--no-geographic", default=True,
              help="Include state-level geographic grouping.")
def infer(db_path: str, model_dir: str, output_dir: str, mode: str,
          top_k: int, chunk_size: int, device: str, geographic: bool) -> None:
    """Run full inference (10M x 12K scoring)."""
    from ml.inference import main as run_inference
    console.print("[bold]→ Running inference...[/bold]")
    args = ["--db-path", db_path, "--model-dir", model_dir,
            "--output-dir", output_dir, "--mode", mode,
            "--top-k", str(top_k), "--chunk-size", str(chunk_size),
            "--device", device]
    if geographic:
        args.append("--geographic")
    else:
        args.append("--no-geographic")
    run_inference(args, standalone_mode=False)


@cli.command()
@click.option("--db-path", default="data/db/cvs_analytics.duckdb", help="DuckDB path.")
def validate(db_path: str) -> None:
    """Run data validation checks."""
    from db.load_duckdb import validate as validate_db
    console.print("[bold]→ Running validation...[/bold]")
    validate_db(["--db-path", db_path], standalone_mode=False)


if __name__ == "__main__":
    cli()

"""CLI entry point for the toydatadecomp pipeline.

Provides a unified command-line interface using Click groups with
subcommands for each pipeline stage.

Usage:
    python src/cli.py scrape stores
    python src/cli.py scrape products
    python src/cli.py generate customers
    python src/cli.py generate transactions
    python src/cli.py convert parquet
    python src/cli.py load
    python src/cli.py train
    python src/cli.py infer
    python src/cli.py validate
    python src/cli.py status
"""

import os
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════
# Main CLI group
# ═══════════════════════════════════════════════════════════════════════

@click.group()
def toydatadecomp():
    """toydatadecomp — Retail Recommendation Engine CLI."""
    pass


# ═══════════════════════════════════════════════════════════════════════
# scrape group
# ═══════════════════════════════════════════════════════════════════════

@toydatadecomp.group()
def scrape():
    """Scrape real CVS data (stores and products)."""
    pass


@scrape.command("stores")
@click.option("--output-dir", default="data/real", help="Output directory.")
@click.option("--delay", default=0.5, type=float, help="Delay between requests.")
@click.option("--max-retries", default=3, type=int, help="Max retries per request.")
@click.option("--dry-run", is_flag=True, help="Parse only, don't write files.")
def scrape_stores(output_dir, delay, max_retries, dry_run):
    """Scrape CVS store locations."""
    from scrapers.scrape_stores import main as _main
    console.print("[bold]→ Scraping stores...[/bold]")
    args = ["--output-dir", output_dir, "--delay", str(delay),
            "--max-retries", str(max_retries)]
    if dry_run:
        args.append("--dry-run")
    _main(args, standalone_mode=False)


@scrape.command("products")
@click.option("--mode", type=click.Choice(["build", "scrape"]), default="build",
              help="Mode: 'build' generates from knowledge base, 'scrape' tries cvs.com.")
@click.option("--output-dir", default="data/real", help="Output directory.")
@click.option("--count", default=10_000, type=int, help="Target OTC product count.")
@click.option("--rx-count", default=2_000, type=int, help="Target Rx product count.")
def scrape_products(mode, output_dir, count, rx_count):
    """Build/scrape CVS product catalog."""
    from scrapers.scrape_products import main as _main
    console.print(f"[bold]→ Building product catalog (mode={mode})...[/bold]")
    args = ["--mode", mode, "--output-dir", output_dir,
            "--count", str(count), "--rx-count", str(rx_count)]
    _main(args, standalone_mode=False)


# ═══════════════════════════════════════════════════════════════════════
# generate group
# ═══════════════════════════════════════════════════════════════════════

@toydatadecomp.group()
def generate():
    """Generate synthetic data."""
    pass


@generate.command("stores")
@click.option("--count", default=9000, help="Number of stores to generate.")
@click.option("--output-dir", default="data/real", help="Output directory.")
def generate_stores(count, output_dir):
    """Generate synthetic store locations (fallback if scraping fails)."""
    from generators.gen_stores import main as _main
    console.print("[bold]→ Generating synthetic stores...[/bold]")
    _main(["--count", str(count), "--output-dir", output_dir], standalone_mode=False)


@generate.command("customers")
@click.option("--count", default=10_000_000, help="Number of customers.")
@click.option("--workers", default=8, help="Parallel workers.")
@click.option("--batch-size", default=100_000, help="Customers per batch.")
@click.option("--test", is_flag=True, help="Test mode: 10,000 customers only.")
def generate_customers(count, workers, batch_size, test):
    """Generate synthetic customer profiles."""
    from generators.gen_customers import main as _main
    console.print("[bold]→ Generating customers...[/bold]")
    args = ["--count", str(count), "--workers", str(workers),
            "--batch-size", str(batch_size)]
    if test:
        args.append("--test")
    _main(args, standalone_mode=False)


@generate.command("transactions")
@click.option("--customers", default=10_000_000, type=int, help="Number of customers.")
@click.option("--txns-per-customer", default=1000, type=int, help="Txns per customer.")
@click.option("--products", default=12_000, type=int, help="Number of products.")
@click.option("--threads", default=0, type=int, help="Threads (0=auto).")
def generate_transactions(customers, txns_per_customer, products, threads):
    """Generate synthetic transactions via the C binary."""
    txn_gen = PROJECT_ROOT / "src" / "generators" / "txn_generator"
    if not txn_gen.exists():
        console.print("[red]txn_generator binary not found. Run: make compile-c[/red]")
        sys.exit(1)
    stores_csv = PROJECT_ROOT / "data" / "real" / "stores.csv"
    out_dir = PROJECT_ROOT / "data" / "synthetic" / "transactions"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(txn_gen), str(customers), str(txns_per_customer),
        str(products), str(threads), str(out_dir), str(stores_csv),
    ]
    console.print(f"[bold]→ Generating transactions...[/bold]")
    console.print(f"  Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ═══════════════════════════════════════════════════════════════════════
# convert group
# ═══════════════════════════════════════════════════════════════════════

@toydatadecomp.group()
def convert():
    """Convert data formats."""
    pass


@convert.command("parquet")
@click.option("--transactions-dir", default="data/synthetic/transactions",
              help="Transactions data directory.")
@click.option("--delete-source/--keep-source", default=False,
              help="Delete .csv.zst after conversion.")
def convert_parquet(transactions_dir, delete_source):
    """Convert transaction .csv.zst files to Parquet."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from csv_to_parquet import main as _main
    console.print("[bold]→ Converting to Parquet...[/bold]")
    args = ["--transactions-dir", transactions_dir]
    if delete_source:
        args.append("--delete-source")
    _main(args, standalone_mode=False)


# ═══════════════════════════════════════════════════════════════════════
# load
# ═══════════════════════════════════════════════════════════════════════

@toydatadecomp.command()
@click.option("--db-path", default="data/db/cvs_analytics.duckdb", help="DuckDB path.")
@click.option("--data-dir", default="data", help="Root data directory.")
@click.option("--materialize", is_flag=True, default=False,
              help="Materialize expensive views as tables.")
def load(db_path, data_dir, materialize):
    """Load all data into DuckDB."""
    from db.load_duckdb import main as _main
    console.print("[bold]→ Loading data into DuckDB...[/bold]")
    args = ["--db-path", db_path, "--data-dir", data_dir]
    if materialize:
        args.append("--materialize")
    _main(args, standalone_mode=False)


# ═══════════════════════════════════════════════════════════════════════
# train
# ═══════════════════════════════════════════════════════════════════════

@toydatadecomp.command()
@click.option("--db-path", default="data/db/cvs_analytics.duckdb")
@click.option("--epochs", default=5, help="Training epochs.")
@click.option("--batch-size", default=8192, help="Batch size.")
@click.option("--lr", default=0.001, type=float, help="Learning rate.")
@click.option("--sample-pct", default=1.0, type=float, help="Sample percent.")
@click.option("--device", default="auto", type=click.Choice(["auto", "mps", "cpu"]))
@click.option("--output-dir", default="data/model/", help="Model output directory.")
@click.option("--neg-samples", default=4, help="Negative samples per positive.")
@click.option("--margin-weight/--no-margin-weight", default=True)
@click.option("--skip-features", is_flag=True)
def train(db_path, epochs, batch_size, lr, sample_pct, device,
          output_dir, neg_samples, margin_weight, skip_features):
    """Train the two-tower recommendation model."""
    from ml.train import main as _main
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
    _main(args, standalone_mode=False)


# ═══════════════════════════════════════════════════════════════════════
# infer
# ═══════════════════════════════════════════════════════════════════════

@toydatadecomp.command()
@click.option("--db-path", default="data/db/cvs_analytics.duckdb")
@click.option("--model-dir", default="data/model/")
@click.option("--output-dir", default="data/results/")
@click.option("--mode", default="full-matrix",
              type=click.Choice(["full-matrix", "per-product", "per-customer"]))
@click.option("--top-k", default=100, help="Top products to output.")
@click.option("--chunk-size", default=100_000, help="Customers per chunk.")
@click.option("--device", default="auto", type=click.Choice(["auto", "mps", "cpu"]))
@click.option("--geographic/--no-geographic", default=True)
def infer(db_path, model_dir, output_dir, mode, top_k, chunk_size,
          device, geographic):
    """Run full inference (10M x 12K scoring)."""
    from ml.inference import main as _main
    console.print("[bold]→ Running inference...[/bold]")
    args = ["--db-path", db_path, "--model-dir", model_dir,
            "--output-dir", output_dir, "--mode", mode,
            "--top-k", str(top_k), "--chunk-size", str(chunk_size),
            "--device", device]
    if geographic:
        args.append("--geographic")
    else:
        args.append("--no-geographic")
    _main(args, standalone_mode=False)


# ═══════════════════════════════════════════════════════════════════════
# validate
# ═══════════════════════════════════════════════════════════════════════

@toydatadecomp.command()
@click.option("--db-path", default="data/db/cvs_analytics.duckdb")
def validate(db_path):
    """Run data validation checks."""
    from db.load_duckdb import validate as _validate
    console.print("[bold]→ Running validation...[/bold]")
    _validate(["--db-path", db_path], standalone_mode=False)


# ═══════════════════════════════════════════════════════════════════════
# status
# ═══════════════════════════════════════════════════════════════════════

def _file_size(path: Path) -> str:
    """Human-readable file size."""
    if not path.exists():
        return "—"
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _dir_size(dirpath: Path) -> str:
    """Human-readable total size of a directory."""
    if not dirpath.exists():
        return "—"
    total = sum(f.stat().st_size for f in dirpath.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} PB"


def _parquet_row_count(path: Path) -> str:
    """Row count from a single Parquet file or directory of Parquet files."""
    try:
        import pyarrow.parquet as pq
        if path.is_dir():
            files = sorted(path.glob("*.parquet"))
            if not files:
                return "—"
            total = sum(pq.read_metadata(f).num_rows for f in files)
            return f"{total:,}"
        elif path.exists() and path.suffix == ".parquet":
            return f"{pq.read_metadata(path).num_rows:,}"
    except Exception:
        pass
    return "—"


def _csv_zst_count(dirpath: Path) -> str:
    """Count .csv.zst files (not rows — too expensive to scan)."""
    if not dirpath.exists():
        return "—"
    zst = list(dirpath.glob("*.csv.zst"))
    pqt = list(dirpath.glob("*.parquet"))
    if pqt:
        return _parquet_row_count(dirpath)
    if zst:
        return f"{len(zst)} files"
    return "—"


def _duckdb_row_count(db_path: Path, table: str) -> str:
    """Row count from a DuckDB table."""
    if not db_path.exists():
        return "—"
    try:
        import duckdb
        con = duckdb.connect(str(db_path), read_only=True)
        result = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        con.close()
        return f"{result[0]:,}"
    except Exception:
        return "—"


@toydatadecomp.command()
def status():
    """Show what data exists, what's missing, sizes, and row counts."""
    data_dir = PROJECT_ROOT / "data"

    table = Table(title="toydatadecomp — Data Status", show_lines=True)
    table.add_column("Component", style="bold", min_width=20)
    table.add_column("Status", justify="center", min_width=6)
    table.add_column("Rows", justify="right", min_width=14)
    table.add_column("Size", justify="right", min_width=10)
    table.add_column("Path", style="dim", min_width=30)

    # Define components to check
    components = [
        {
            "name": "Stores",
            "path": data_dir / "real" / "stores.parquet",
            "type": "parquet_file",
        },
        {
            "name": "Products",
            "path": data_dir / "real" / "products.parquet",
            "type": "parquet_file",
        },
        {
            "name": "Customers",
            "path": data_dir / "synthetic" / "customers",
            "type": "parquet_dir",
        },
        {
            "name": "Coupon Clips",
            "path": data_dir / "synthetic" / "coupon_clips",
            "type": "parquet_dir",
        },
        {
            "name": "Transactions",
            "path": data_dir / "synthetic" / "transactions",
            "type": "txn_dir",
        },
        {
            "name": "DuckDB",
            "path": data_dir / "db" / "cvs_analytics.duckdb",
            "type": "duckdb",
        },
        {
            "name": "Model",
            "path": data_dir / "model",
            "type": "model_dir",
        },
        {
            "name": "Results",
            "path": data_dir / "results",
            "type": "results_dir",
        },
    ]

    for comp in components:
        name = comp["name"]
        path = comp["path"]
        ctype = comp["type"]
        rel_path = str(path.relative_to(PROJECT_ROOT))

        if ctype == "parquet_file":
            exists = path.exists()
            status_icon = "[green]✓[/green]" if exists else "[red]✗[/red]"
            rows = _parquet_row_count(path) if exists else "—"
            size = _file_size(path) if exists else "—"

        elif ctype == "parquet_dir":
            files = sorted(path.glob("*.parquet")) if path.exists() else []
            exists = len(files) > 0
            status_icon = "[green]✓[/green]" if exists else "[red]✗[/red]"
            rows = _parquet_row_count(path) if exists else "—"
            size = _dir_size(path) if exists else "—"
            if exists:
                rel_path += f"/ ({len(files)} files)"

        elif ctype == "txn_dir":
            has_zst = list(path.glob("*.csv.zst")) if path.exists() else []
            has_pqt = list(path.glob("*.parquet")) if path.exists() else []
            exists = len(has_zst) > 0 or len(has_pqt) > 0
            status_icon = "[green]✓[/green]" if exists else "[red]✗[/red]"
            rows = _csv_zst_count(path) if exists else "—"
            size = _dir_size(path) if exists else "—"
            n = len(has_zst) + len(has_pqt)
            fmt = []
            if has_zst:
                fmt.append(f"{len(has_zst)} .csv.zst")
            if has_pqt:
                fmt.append(f"{len(has_pqt)} .parquet")
            if fmt:
                rel_path += f"/ ({', '.join(fmt)})"

        elif ctype == "duckdb":
            exists = path.exists()
            status_icon = "[green]✓[/green]" if exists else "[red]✗[/red]"
            rows = "—"
            size = _file_size(path) if exists else "—"
            if exists:
                # Show table count
                try:
                    import duckdb
                    con = duckdb.connect(str(path), read_only=True)
                    tables = con.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema='main'").fetchall()
                    con.close()
                    rows = f"{len(tables)} tables"
                except Exception:
                    pass

        elif ctype in ("model_dir", "results_dir"):
            exists = path.exists() and any(path.iterdir()) if path.exists() else False
            status_icon = "[green]✓[/green]" if exists else "[red]✗[/red]"
            rows = "—"
            size = _dir_size(path) if exists else "—"
            if exists:
                files = list(path.rglob("*"))
                rows = f"{len([f for f in files if f.is_file()])} files"

        else:
            exists = path.exists()
            status_icon = "[green]✓[/green]" if exists else "[red]✗[/red]"
            rows = "—"
            size = "—"

        table.add_row(name, status_icon, rows, size, rel_path)

    console.print()
    console.print(table)
    console.print()


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    toydatadecomp()

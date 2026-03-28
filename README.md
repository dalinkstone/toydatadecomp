# toydatadecomp — Retail Recommendation Engine

A retail recommendation engine that combines real CVS store/product data with
large-scale synthetic data to train a two-tower neural network for purchase prediction.

**Target machine:** M4 Max MacBook Pro, 64GB RAM, 4TB SSD, macOS.

## Scale

| Entity       | Count  |
|--------------|--------|
| Stores       | ~9,000 |
| Products     | ~10,000|
| Customers    | 10M    |
| Transactions | 10B    |

## Architecture

```
scrape-stores ──┐
scrape-products ─┤
gen-customers ───┤──→ load-db ──→ train ──→ inference
gen-transactions ┘        ↑
                          │
              .csv.zst (native) or .parquet (optional)
```

### Pipeline Stages

1. **Scrape** — Collect real CVS store locations and product catalog
2. **Generate** — Create 10M synthetic customers (Python/Faker) and 10B transactions (C, multithreaded)
3. **Load** — Ingest all data into DuckDB (reads `.csv.zst` natively, no conversion needed)
4. **Train** — Train a two-tower neural network on purchase history
5. **Infer** — Score all 10M customers × 10K products for recommendations

### Disk Optimization

The C transaction generator writes zstd-compressed CSVs via `popen("zstd -", "w")`,
reducing the transaction data footprint from **~1.5TB** (raw CSV) to **~180GB** (.csv.zst).
DuckDB reads `.csv.zst` files natively with full predicate pushdown, so Parquet
conversion is optional (`make convert-parquet`).

## Two-Tower Model

The recommendation model uses a **two-tower architecture**:

- **User tower:** Encodes customer features (demographics, purchase history embeddings)
  into a dense user vector.
- **Item tower:** Encodes product features (category, price, brand) into a dense item vector.
- **Scoring:** Dot product of user and item vectors → purchase probability.

At inference time, item embeddings are pre-computed and stored in a custom C vector
database (`src/vecdb/`) that uses Apple's Accelerate framework for fast SIMD dot products.
For each of the 10M users, the model retrieves top-K product recommendations via
approximate nearest neighbor search.

## Quick Start

```bash
# Install everything (venv, Python deps, C compilation, zstd check)
make install

# Run the full pipeline end-to-end
make full-pipeline

# Or step by step:
make scrape-stores
make scrape-products
make gen-customers
make gen-transactions
make load-db
make train
make inference

# Optional: convert to Parquet (not required)
make convert-parquet

# Run tests
make test
```

## CLI

```bash
.venv/bin/python src/cli.py scrape --stores
.venv/bin/python src/cli.py generate --customers --count 10000000
.venv/bin/python src/cli.py load
.venv/bin/python src/cli.py train
.venv/bin/python src/cli.py infer
.venv/bin/python src/cli.py validate
```

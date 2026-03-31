# toydatadecomp

Retail recommendation engine for CVS front store revenue optimization. Trains a two-tower neural network on 10B synthetic transactions, estimates price elasticity per product, and runs a Monte Carlo simulation to project incremental revenue from personalized couponing. Validated against CVS 10-K financial benchmarks.

**Website:** [mysolution.works](https://mysolution.works)

## What It Does

1. **Scrapes** real CVS store locations and product catalog from cvs.com
2. **Generates** 10M customer profiles, 16M coupon clips, and 10B transactions (C binary, zstd-compressed)
3. **Loads** everything into DuckDB (transactions stay on disk as a VIEW over `.csv.zst` files)
4. **Trains** a two-tower neural network (PyTorch) on purchase history with margin-weighted loss
5. **Estimates** price elasticity per product via weighted least squares on coupon redemption data
6. **Identifies** breakout candidates from the long tail via cosine similarity in embedding space
7. **Scores** all 10M x 12K customer-product pairs and ranks personalized recommendations
8. **Simulates** 40 weeks of the recommend-purchase-retrain feedback loop (Monte Carlo, multiple runs)
9. **Validates** simulation output against CVS 10-K revenue, margins, and consumer behavior benchmarks

## Quick Start

```bash
make install          # venv, Python deps, C compilation, zstd check
make full-pipeline    # end-to-end: scrape -> generate -> load -> train -> infer -> validate
```

Or at 1/1000th scale (~2 min):

```bash
make demo
```

## Step by Step

```bash
make install
make scrape-products      # build 12K product catalog
make scrape-stores        # scrape CVS locations (or: make gen-stores)
make gen-customers        # 10M synthetic customers
make gen-transactions     # 10B transactions (~200GB .csv.zst)
make load-db              # load into DuckDB
make train                # train two-tower model
make inference            # score 10M x 12K
make rank                 # business-logic ranking layer
make simulate             # Monte Carlo simulation + validation + notebook
make status               # show pipeline status
```

## Simulation

```bash
make simulate                               # full scale (10M customers, 50 runs)
make simulate-demo                          # demo scale (10K customers, 10 runs)
make simulate EPOCHS=40 RUNS=75 SCALE=full  # custom
```

The `simulate` target runs the Monte Carlo simulation, then automatically runs revenue validation against 10-K benchmarks and re-executes the analysis notebook.

## Project Structure

```
src/
  cli.py                    Click CLI entry point
  scrapers/                 CVS store + product scraping
  generators/               Synthetic data (Python + C)
    txn_generator.c         10B transactions, multithreaded, zstd-compressed
  db/load_duckdb.py         DuckDB loader + analytical views
  ml/
    two_tower.py            Two-tower neural network (PyTorch)
    train.py                Training with margin-weighted BCE, negative sampling
    inference.py            Full-matrix 10M x 12K scoring
    features.py             Feature engineering from DuckDB
    product_tiers.py        Revenue-based 4-tier product classification
    elasticity.py           Price elasticity via weighted least squares
    breakout.py             Breakout candidate scoring (cosine similarity)
    validate_revenue.py     Simulation output vs 10-K benchmarks
  ranking/decision_engine.py  Business-logic ranking (recency, diversity, margin)
  simulation/
    monte_carlo.py          Monte Carlo orchestrator (40 epochs x N runs)
    vectorized_consumer.py  Consumer response model (calibrated sigmoid + fatigue)
  vecdb/                    C vector database (Apple Accelerate SIMD)
notebooks/
  analysis.ipynb            Simulation analysis (auto-updated by make simulate)
  campaign_analytics.ipynb  Campaign analytics dashboard
docs/
  index.html                Website (GitHub Pages)
data/
  real/                     Scraped stores + products
  synthetic/                Generated customers, transactions, coupon clips
  db/                       DuckDB analytics database
  model/                    Trained model, embeddings, tiers, elasticity
  results/                  Inference output + simulation results
```

## Dependencies

**Python:** click, rich, pyarrow, duckdb, torch, numpy, pandas, faker, seaborn, matplotlib

**System:** clang (Xcode), zstd (`brew install zstd`), macOS Accelerate framework

```bash
make test    # pytest tests/ -v
```

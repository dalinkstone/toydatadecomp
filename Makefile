# toydatadecomp — Retail Recommendation Engine Build System
# Target: M4 Max MacBook Pro, 64GB RAM, 4TB SSD, macOS
#
# Scale: 10M customers × 12K products × 9K stores → 10B transactions
# Model: Two-tower neural network for purchase prediction

SHELL := /bin/zsh
PYTHON := .venv/bin/python
PIP := .venv/bin/pip
CC := clang
CFLAGS := -O3 -march=native -Wall -Wextra -DACCELERATE_NEW_LAPACK

# Output binaries
TXN_GEN := src/generators/txn_generator
VECDB_LIB := src/vecdb/vecdb.o
VECDB_DYLIB := src/vecdb/vecdb.dylib
VECDB_TEST := src/vecdb/test_vecdb

.PHONY: install compile-c scrape-stores scrape-products gen-customers gen-transactions \
        convert-parquet load-db train inference test validate \
        full-pipeline full-pipeline-parquet clean

# --------------------------------------------------------------------------
# install: venv + deps + C compilation + zstd check
# --------------------------------------------------------------------------
install: .venv/bin/activate compile-c check-zstd
	@echo "✓ Installation complete."

.venv/bin/activate: requirements.txt
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch .venv/bin/activate

check-zstd:
	@command -v zstd >/dev/null 2>&1 || { \
		echo "ERROR: zstd not found on PATH."; \
		echo "Install with: brew install zstd"; \
		echo "The transaction generator uses zstd to compress output from ~1.5TB to ~180GB."; \
		exit 1; \
	}
	@echo "✓ zstd found: $$(command -v zstd)"

# --------------------------------------------------------------------------
# compile-c: build the C transaction generator and vecdb
# --------------------------------------------------------------------------
compile-c: $(TXN_GEN) $(VECDB_TEST) $(VECDB_DYLIB)

$(TXN_GEN): src/generators/txn_generator.c
	$(CC) $(CFLAGS) -o $@ $< -lm -lpthread

$(VECDB_LIB): src/vecdb/vecdb.c src/vecdb/vecdb.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(VECDB_DYLIB): src/vecdb/vecdb.c src/vecdb/vecdb.h
	$(CC) $(CFLAGS) -shared -o $@ $< -framework Accelerate

$(VECDB_TEST): src/vecdb/test_vecdb.c $(VECDB_LIB) src/vecdb/vecdb.h
	$(CC) $(CFLAGS) -o $@ $< $(VECDB_LIB) -framework Accelerate -lm

# --------------------------------------------------------------------------
# Data pipeline stages
# --------------------------------------------------------------------------
scrape-stores:
	$(PYTHON) src/scrapers/scrape_stores.py

scrape-products:
	$(PYTHON) src/scrapers/scrape_products.py

gen-customers:
	$(PYTHON) src/generators/gen_customers.py --count 10000000

gen-transactions: $(TXN_GEN) check-zstd
	./$(TXN_GEN) 10000000 1000 12000 0 data/synthetic/transactions data/real/stores.csv

convert-parquet:
	@echo "NOTE: This is optional. DuckDB reads .csv.zst natively with predicate pushdown."
	$(PYTHON) scripts/csv_to_parquet.py

load-db:
	$(PYTHON) src/db/load_duckdb.py

# --------------------------------------------------------------------------
# ML pipeline
# --------------------------------------------------------------------------
train:
	$(PYTHON) src/ml/train.py

inference:
	$(PYTHON) src/ml/inference.py

# --------------------------------------------------------------------------
# Testing & validation
# --------------------------------------------------------------------------
test:
	$(PYTHON) -m pytest tests/ -v

validate:
	$(PYTHON) src/cli.py validate

# --------------------------------------------------------------------------
# Full pipelines
# --------------------------------------------------------------------------
full-pipeline: install scrape-stores scrape-products gen-customers gen-transactions load-db train inference validate
	@echo "✓ Full pipeline complete (using .csv.zst directly)."

full-pipeline-parquet: install scrape-stores scrape-products gen-customers gen-transactions convert-parquet load-db train inference validate
	@echo "Removing .csv.zst files (Parquet conversion complete)..."
	find data/synthetic/transactions -name '*.csv.zst' -delete
	@echo "✓ Full pipeline complete (Parquet mode)."

# --------------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------------
clean:
	rm -rf data/real/*.csv data/real/*.json
	rm -rf data/synthetic/customers/*.csv data/synthetic/customers/*.csv.zst data/synthetic/customers/*.parquet
	rm -rf data/synthetic/transactions/*.csv data/synthetic/transactions/*.csv.zst data/synthetic/transactions/*.parquet
	rm -rf data/db/*.duckdb data/db/*.duckdb.wal
	rm -f $(TXN_GEN) $(VECDB_LIB) $(VECDB_TEST)
	@echo "✓ Cleaned all generated data and binaries."

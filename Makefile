# toydatadecomp — Retail Recommendation Engine Build System
# Target: M4 Max MacBook Pro, 64GB RAM, 4TB SSD, macOS
#
# Scale: 10M customers × 12K products × 9K stores → 10B transactions
# Model: Two-tower neural network for purchase prediction

SHELL := /bin/zsh
PYTHON := PYTHONPATH=src .venv/bin/python
PIP := .venv/bin/pip
CC := clang
CFLAGS := -O3 -march=native -Wall -Wextra -DACCELERATE_NEW_LAPACK

# Output binaries
TXN_GEN := src/generators/txn_generator
VECDB_LIB := src/vecdb/vecdb.o
VECDB_DYLIB := src/vecdb/vecdb.dylib
VECDB_TEST := src/vecdb/test_vecdb

.PHONY: install compile-c scrape-stores scrape-products gen-stores gen-customers \
        gen-transactions convert-parquet load-db train inference rank test validate \
        simulate simulate-demo full-pipeline full-pipeline-parquet demo status clean

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

gen-stores:
	@echo "Generating synthetic store locations (fallback)..."
	$(PYTHON) src/generators/gen_stores.py

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

rank:
	$(PYTHON) src/ranking/decision_engine.py

# --------------------------------------------------------------------------
# Monte Carlo Simulation
# --------------------------------------------------------------------------
EPOCHS ?= 250
RUNS ?= 75
RETRAIN_INTERVAL ?= 10
SCALE ?= full
CUSTOMERS ?= 0

simulate:
	$(PYTHON) src/cli.py simulate run --epochs $(EPOCHS) --runs $(RUNS) \
		--retrain-interval $(RETRAIN_INTERVAL) --scale $(SCALE) --customers $(CUSTOMERS)

simulate-demo:
	$(PYTHON) src/cli.py simulate run --epochs 50 --runs 10 --retrain-interval 10 --scale demo

# --------------------------------------------------------------------------
# Testing & validation
# --------------------------------------------------------------------------
test:
	$(PYTHON) -m pytest tests/ -v

validate:
	$(PYTHON) scripts/validate.py

status:
	$(PYTHON) src/cli.py status

# --------------------------------------------------------------------------
# Full pipelines
# --------------------------------------------------------------------------
#
# Order: install → scrape-products (build mode, always works)
#       → scrape-stores (best effort, falls back to gen-stores)
#       → gen-customers → compile-c → gen-transactions
#       → load-db → train → inference → rank → validate
#
# NOTE: convert-parquet is NOT in the default pipeline.
# DuckDB reads .csv.zst natively with predicate pushdown.
# Use full-pipeline-parquet for faster repeat queries.
#
full-pipeline:
	@SECONDS=0; \
	$(MAKE) install && \
	$(MAKE) scrape-products && \
	( $(MAKE) scrape-stores || $(MAKE) gen-stores ) && \
	$(MAKE) gen-customers && \
	$(MAKE) compile-c && \
	$(MAKE) gen-transactions && \
	$(MAKE) load-db && \
	$(MAKE) train && \
	$(MAKE) inference && \
	$(MAKE) rank && \
	$(MAKE) validate && \
	echo "" && \
	echo "════════════════════════════════════════════════════════" && \
	echo "✓ Full pipeline complete (using .csv.zst directly)." && \
	echo "  Elapsed: $${SECONDS}s" && \
	echo "════════════════════════════════════════════════════���═══" && \
	$(MAKE) status

full-pipeline-parquet:
	@SECONDS=0; \
	$(MAKE) install && \
	$(MAKE) scrape-products && \
	( $(MAKE) scrape-stores || $(MAKE) gen-stores ) && \
	$(MAKE) gen-customers && \
	$(MAKE) compile-c && \
	$(MAKE) gen-transactions && \
	$(MAKE) convert-parquet && \
	find data/synthetic/transactions -name '*.csv.zst' -delete && \
	$(MAKE) load-db && \
	$(MAKE) train && \
	$(MAKE) inference && \
	$(MAKE) rank && \
	$(MAKE) validate && \
	echo "" && \
	echo "════════════════════════════════════════════════════════" && \
	echo "✓ Full pipeline complete (Parquet mode)." && \
	echo "  Elapsed: $${SECONDS}s" && \
	echo "════════════════════════════════════════════════════════" && \
	$(MAKE) status

# --------------------------------------------------------------------------
# Demo: 1/1000th scale (~2 min total)
#   10K customers × 100 txns each = 1M transactions
# --------------------------------------------------------------------------
demo:
	@SECONDS=0; \
	echo "═══ Demo Pipeline (1/1000th scale) ═══" && \
	echo "  10K customers, 100 txns/customer, 1M total rows" && \
	echo "" && \
	$(MAKE) install && \
	$(MAKE) scrape-products && \
	( $(MAKE) scrape-stores || $(MAKE) gen-stores ) && \
	$(PYTHON) src/generators/gen_customers.py --count 10000 --test && \
	$(MAKE) compile-c && \
	./$(TXN_GEN) 10000 100 12000 0 data/synthetic/transactions data/real/stores.csv && \
	$(MAKE) load-db && \
	$(PYTHON) src/ml/train.py --epochs 2 --sample-pct 100.0 && \
	$(PYTHON) src/ml/inference.py && \
	$(PYTHON) src/ranking/decision_engine.py --demo && \
	$(MAKE) validate && \
	echo "" && \
	echo "════════════════════════════════════════════════════════" && \
	echo "✓ Demo pipeline complete." && \
	echo "  Elapsed: $${SECONDS}s" && \
	echo "════════════════════════════════════════════════════════" && \
	$(MAKE) status

# --------------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------------
clean:
	rm -rf data/real/*.csv data/real/*.json
	rm -rf data/synthetic/customers/*.csv data/synthetic/customers/*.csv.zst data/synthetic/customers/*.parquet
	rm -rf data/synthetic/coupon_clips/*.parquet
	rm -rf data/synthetic/transactions/*.csv data/synthetic/transactions/*.csv.zst data/synthetic/transactions/*.parquet
	rm -rf data/db/*.duckdb data/db/*.duckdb.wal
	rm -rf data/model/ data/results/
	rm -f $(TXN_GEN) $(VECDB_LIB) $(VECDB_TEST) $(VECDB_DYLIB)
	@echo "✓ Cleaned all generated data and binaries."

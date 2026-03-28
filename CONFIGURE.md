# CONFIGURE.md — Manual Setup Steps

This file lists everything you need to do by hand before and during the Claude Code prompts
in `PLAN.md`. These are things Claude Code can't do for you (installing system software,
buying data, etc.).

---

## Before You Start (One-Time Setup, ~10 minutes)

### 1. Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install System Dependencies

```bash
brew install python@3.12 duckdb zstd
```

The `zstd` package is critical — the transaction generator pipes its output through zstd
compression on-the-fly, reducing the disk footprint from ~1.5TB to ~180GB. Without it,
you'd need 1.5TB of free space for raw CSV files. DuckDB reads `.csv.zst` files natively.

You already have `clang` via Xcode Command Line Tools (ships with macOS). Verify:

```bash
clang --version    # Should show Apple clang 16+
python3 --version  # Should show 3.12+
duckdb --version   # Should show 1.x
zstd --version     # Should show 1.5+
```

### 3. Verify Your Hardware

Run this to confirm M4 Max capabilities:

```bash
sysctl -n hw.ncpu              # Should show 14-16 (performance + efficiency cores)
sysctl -n hw.memsize | awk '{print $1/1073741824 " GB"}'  # Should show 64 GB
```

PyTorch MPS (Metal Performance Shaders) is what gives you GPU acceleration on Apple Silicon.
It ships with PyTorch 2.0+ and works automatically — no CUDA, no drivers, nothing to install.

### 4. Create a Working Directory

```bash
mkdir -p ~/projects
cd ~/projects
```

This is where you'll run the Claude Code prompts. Claude Code will create `cvs-recsys/` here.

---

## During the Pipeline

### After Prompt 0 (Project Scaffold)

Verify the project structure was created correctly:

```bash
cd ~/projects/cvs-recsys
find . -type f | head -30
```

Then activate the virtual environment that `make install` created:

```bash
source .venv/bin/activate
```

You'll want this active for the rest of the session. If you open a new terminal, re-activate it.

### After Prompt 1 (Store Scraper)

The store scraper may or may not work depending on whether CVS is blocking automated requests
on that particular day. If it fails, you have three options:

**Option A — Fallback synthetic stores (free, immediate):**
The pipeline includes a `make gen-stores` fallback that generates ~9,000 realistic synthetic
CVS store locations. This is good enough for the ML pipeline. The synthetic stores use real
US city/state/zip combinations and realistic lat/lon coordinates.

**Option B — Buy the dataset ($50-100, 5 minutes):**
Several data vendors sell verified CVS store location data:

  - **ScrapeHero**: https://www.scrapehero.com/location-reports/CVS%20Pharmacy-USA/ (~$89)
    Download as CSV, rename to `data/real/stores.csv`, then run `make load-stores-csv`.

  - **AggData**: https://www.aggdata.com/aggdata/complete-list-cvs-store-locations-united-states (~$49)
    Same process — download, place in `data/real/`, load.

  - **LocationsCloud**: https://www.locationscloud.com/intelligence-reports/cvs-pharmacy-usa/
    Offers JSON, GeoJSON, and CSV formats.

After downloading, you may need to rename columns to match the expected schema. The loader
script will print clear errors if columns don't match and tell you what to rename.

**Option C — Manual scrape via browser (~30 minutes of clicking):**
Go to https://www.cvs.com/store-locator/cvs-pharmacy-locations, open browser DevTools
(Network tab), search for a zip code, and look for the XHR/Fetch response that returns
store JSON. Copy that URL pattern and update `scrape_stores.py` with the correct endpoint.

### After Prompt 4 (C Transaction Generator)

The C code compiles with clang on macOS. If you see warnings about `system()` return value,
those are safe to ignore. If you see errors about missing headers, make sure you have
Xcode Command Line Tools:

```bash
xcode-select --install
```

### After Prompt 7 (Vector DB)

The vector DB uses Apple's Accelerate framework for BLAS operations. This is built into macOS —
no installation needed. If you see linker errors about Accelerate, make sure the compile
command includes `-framework Accelerate`:

```bash
clang -O3 -march=native -framework Accelerate -o test_vecdb src/vecdb/test_vecdb.c src/vecdb/vecdb.c -lm
```

### After Prompt 8 (PyTorch Training)

PyTorch's MPS backend should be detected automatically. Verify:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
# Should print: True
```

If it prints False, you may need a newer PyTorch version:

```bash
pip install --upgrade torch torchvision
```

---

## Running the Full Pipeline

Once all code is written and tested (Prompts 0-10 complete), here's the execution plan:

### Quick Smoke Test (~2 minutes)

```bash
cd ~/projects/cvs-recsys
make demo
```

This runs everything at 1/1000th scale (10K customers, 1M transactions). If this succeeds,
you're ready for the full run.

### Full Pipeline (~90-120 minutes)

```bash
make full-pipeline 2>&1 | tee pipeline.log
```

Expected timeline on M4 Max 64GB:

| Phase                     | Time      | Output Size |
|---------------------------|-----------|-------------|
| Install + compile         | 2 min     | —           |
| Build product catalog     | 5 sec     | 1.8 MB      |
| Scrape/generate stores    | 1-10 min  | 2 MB        |
| Generate 10M customers    | 12-15 min | 600 MB      |
| Generate 10B transactions | 15-25 min | ~180 GB (.csv.zst) |
| Load into DuckDB          | 2-5 min   | ~1.2 GB     |
| Train two-tower model     | 15-25 min | ~50 MB      |
| Run inference             | 5-10 min  | ~200 MB     |
| Validation                | 2 min     | —           |
| **TOTAL**                 | **~60-90 min** | **~182 GB** |

Note: The old plan required a 20-30 minute Parquet conversion step and 1.5TB of disk space.
The zstd compression approach eliminates both — the C generator pipes output through zstd
on-the-fly, and DuckDB reads the .csv.zst files natively. If you want even faster queries,
you can optionally run `make convert-parquet` which adds ~25 minutes but shrinks the data
to ~150GB Parquet.

### Disk Space

The full pipeline needs approximately **200 GB of free disk space**. Here's how it breaks down:

The C transaction generator writes zstd-compressed CSV files (`.csv.zst`) on-the-fly using
popen() pipes, so the raw 1.5TB of CSV data never touches your disk. Instead, the compressed
output lands at roughly **~180GB**. DuckDB reads these files natively, so no conversion to
Parquet is required for the default pipeline. Combined with the customer data (~600MB), product
catalog (~2MB), and model artifacts (~250MB), the total footprint is about **182GB**.

If you optionally run `make convert-parquet` for faster queries, peak usage during conversion
will briefly reach ~330GB (180GB .csv.zst + 150GB Parquet), then drops to ~150GB after deleting
the .csv.zst files.

To check available space at any point:

```bash
df -h ~
```

### After the Pipeline Completes

The final output is a set of product recommendations. To explore them:

```bash
# Check what we built
make status

# Open DuckDB for interactive queries
duckdb data/db/cvs_analytics.duckdb

# Example queries inside DuckDB:
SELECT * FROM products WHERE category = 'Pain Relief & Fever' LIMIT 10;
SELECT * FROM customer_purchase_summary ORDER BY total_spend DESC LIMIT 20;
SELECT COUNT(*) FROM transactions;

# View recommendations
python3 -c "
import pyarrow.parquet as pq
df = pq.read_table('data/results/recommendations.parquet').to_pandas()
print(df.head(20))
"
```

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'torch'"**
Make sure the venv is activated: `source .venv/bin/activate`

**C compilation fails with "Accelerate/Accelerate.h not found"**
Run `xcode-select --install` to install Command Line Tools.

**DuckDB out of memory on large queries**
DuckDB uses up to 80% of RAM by default. For the 10B-row dataset, some aggregation queries
may need more than 64GB. Use LIMIT or WHERE clauses to filter, or configure DuckDB memory:
```sql
SET memory_limit = '50GB';
SET threads = 8;
```

**Transaction generator seems stuck**
It prints progress every 10K customers per thread. With 8 threads and 10M customers, that's
~1,250 progress lines per thread. If nothing appears for 60+ seconds, check disk space
(the compressed .csv.zst output needs ~180GB). Also verify zstd is working — if popen("zstd")
fails silently, the generator may appear to produce no output.

**"zstd: command not found" or empty .csv.zst files**
Run `brew install zstd` and verify with `which zstd`. The C generator falls back to gzip
if zstd isn't found, and to raw CSV as a last resort (but that requires ~1.5TB disk).

**PyTorch MPS errors during training**
Some PyTorch operations don't support MPS yet. If you get an MPS error, the training script
falls back to CPU automatically. Training on CPU with 14 performance cores is still fast
(~30 min instead of ~15 min).

**Parquet conversion is slow**
Make sure you're using all 8 cores: `python scripts/csv_to_parquet.py --workers 8`.
If a single file is especially large, it may appear stalled — check with `ls -la` that the
.parquet file is growing.

---

## Optional: Monitoring During the Pipeline

In a separate terminal, you can watch progress:

```bash
# Watch disk usage grow
watch -n 5 'du -sh ~/projects/cvs-recsys/data/*/* 2>/dev/null'

# Watch CPU usage (all cores should be busy during generation)
top -l 1 -s 0 | head -20

# Count compressed transaction files generated so far
ls -la ~/projects/cvs-recsys/data/synthetic/transactions/txns_*.csv.zst 2>/dev/null | wc -l

# Check total compressed size so far
du -sh ~/projects/cvs-recsys/data/synthetic/transactions/ 2>/dev/null
```

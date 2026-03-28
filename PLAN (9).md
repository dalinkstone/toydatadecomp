# PLAN.md — Claude Code Prompt Sequence

## How to Use This File

Each section below contains **one prompt** you paste into Claude Code. Run them in order.
Between some prompts you will need to do manual steps — those are marked with `⚠️ MANUAL STEP`
and the details are in `CONFIGURE.md`.

After all prompts are complete, you run `make full-pipeline` and go get a coffee. Everything
generates, loads, trains, and produces results. Total wall-clock time on M4 Max 64GB: **~60-90 minutes.**
Total disk usage: **~182GB** (not 1.5TB — the C generator compresses on-the-fly via zstd pipes).

---

## Prompt 0 — Project Scaffold

```
Create a new project called toydatadecomp in the current directory. This is a retail recommendation
engine that uses real CVS store/product data combined with 10M synthetic customers and 10B
synthetic transactions to train a two-tower neural network for purchase prediction.

Target machine: M4 Max MacBook Pro, 64GB RAM, 4TB SSD, macOS.

Create this exact directory structure with placeholder files:

toydatadecomp/
├── Makefile
├── requirements.txt
├── README.md
├── src/
│   ├── scrapers/
│   │   ├── __init__.py
│   │   ├── scrape_stores.py
│   │   └── scrape_products.py
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── gen_customers.py
│   │   └── txn_generator.c
│   ├── db/
│   │   ├── __init__.py
│   │   ├── schema.sql
│   │   └── load_duckdb.py
│   ├── vecdb/
│   │   ├── vecdb.h
│   │   ├── vecdb.c
│   │   └── test_vecdb.c
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── two_tower.py
│   │   ├── train.py
│   │   └── inference.py
│   └── cli.py
├── tests/
│   ├── test_generators.py
│   ├── test_vecdb.py
│   └── test_pipeline.py
├── scripts/
│   ├── csv_to_parquet.py
│   └── validate.py
└── data/
    ├── real/
    │   └── .gitkeep
    ├── synthetic/
    │   ├── customers/
    │   │   └── .gitkeep
    │   └── transactions/
    │       └── .gitkeep
    └── db/
        └── .gitkeep

For the Makefile, create a comprehensive build system with these targets:
- `make install` — creates a Python venv in .venv, installs requirements.txt, compiles C code, verifies zstd is available
- `make compile-c` — compiles txn_generator.c and vecdb.c with clang -O3 -march=native on macOS (use Accelerate framework for BLAS/SIMD)
- `make scrape-stores` — runs the store scraper
- `make scrape-products` — runs the product catalog builder
- `make gen-customers` — generates 10M synthetic customers
- `make gen-transactions` — runs the C transaction generator (10M customers × 1000 txns), outputs zstd-compressed CSVs (.csv.zst)
- `make convert-parquet` — OPTIONAL: converts .csv.zst to Parquet for faster queries (DuckDB reads .csv.zst natively, so this is not required)
- `make load-db` — loads everything into DuckDB (reads .csv.zst or .parquet, whichever exists)
- `make train` — trains the two-tower model
- `make inference` — runs full inference (10M × 10K scoring)
- `make test` — runs all tests
- `make validate` — runs data validation checks
- `make full-pipeline` — runs everything end-to-end in order (does NOT include convert-parquet — uses .csv.zst directly)
- `make full-pipeline-parquet` — same as full-pipeline but also converts to Parquet and deletes .csv.zst files afterward
- `make clean` — removes all generated data

All Python commands should use .venv/bin/python. The C compiler should be clang (macOS default).
For the transaction generator, link with -lm -lpthread. For vecdb, link with -framework Accelerate
for SIMD dot product operations.

IMPORTANT DISK OPTIMIZATION: The C transaction generator writes compressed output using zstd
via popen(). This reduces the transaction data footprint from ~1.5TB (raw CSV) to ~180GB
(zstd-compressed CSV). DuckDB reads .csv.zst files natively with full predicate pushdown,
so converting to Parquet is optional. The `make install` target should verify that `zstd`
is available on PATH and print an error with install instructions if not.

For requirements.txt include:
requests, beautifulsoup4, lxml, faker, pyarrow, duckdb, tqdm, torch, torchvision,
numpy, pandas, click, rich

For README.md write a concise project overview with the architecture diagram from the Makefile
targets, scale numbers (10M customers, 10K products, 9K stores, 10B transactions), and the
two-tower model explanation.

Every placeholder .py file should have a module docstring explaining its purpose and a
`if __name__ == "__main__"` block with argument parsing using click. Every placeholder .c file
should have a comment header explaining its purpose.

The cli.py should be a click CLI group with subcommands: scrape, generate, load, train, infer, validate.
Each subcommand delegates to the appropriate module.
```

---

## Prompt 1 — CVS Store Scraper

```
Implement src/scrapers/scrape_stores.py to collect real CVS Pharmacy store locations across
the United States.

Strategy (try in this order):
1. PRIMARY: Hit CVS's internal store search API. The store locator at cvs.com uses a backend
   API — inspect the endpoint by trying variations of:
   - https://www.cvs.com/rest/bean/cvs/store/model/StoreLocatorActor/findStoresByZipCode
   - https://www.cvs.com/Services/rest/store/findByZip?zipCode=XXXXX&radius=50&maxResults=50
   Send requests with a realistic User-Agent and Referer: https://www.cvs.com/store-locator/landing.
   Query ~150 well-distributed US zip codes with radius=50 miles to cover the entire country.
   Deduplicate by store number.

2. FALLBACK: Crawl https://www.cvs.com/store-locator/cvs-pharmacy-locations/{State}/{City}
   pages. Parse JSON-LD structured data (type: Pharmacy/Store/LocalBusiness) from <script>
   tags. Fall back to parsing HTML store cards if no JSON-LD.

3. LAST RESORT: Use a comprehensive list of ~200 zip codes (every US state, major metros,
   and rural areas) with the API, then supplement with the HTML crawl for any states with
   fewer than expected stores.

CVS has approximately 9,000 stores. We need at minimum 8,000 to be useful.

Output schema (save to data/real/stores.parquet AND data/real/stores.csv):
  store_id: string (CVS store number, e.g., "2345")
  name: string (always "CVS Pharmacy" unless MinuteClinic etc.)
  address: string
  city: string
  state: string (2-letter code)
  zip_code: string (5-digit)
  latitude: float64
  longitude: float64
  phone: string
  store_type: string ("pharmacy", "minuteclinic", "target_cvs")
  hours_mon_fri: string
  hours_sat: string
  hours_sun: string

Requirements:
- Polite crawling: 0.5s delay between requests, respect rate limits
- Retry logic: 3 attempts per request with exponential backoff
- Progress bar with tqdm showing stores found so far
- Deduplication by store_id
- Validation: reject stores with lat/lon of 0,0 or missing address
- Save intermediate results every 500 stores in case of crash (to data/real/stores_partial.json)
- Print summary at end: total stores, stores per state, any states with 0 stores
- CLI: click command with --output-dir, --delay, --max-retries options

If CVS blocks the requests or the API structure has changed, print a clear error message
suggesting the user check CONFIGURE.md for alternative data sources (like purchasing from
ScrapeHero or AggData for $50-100).

Test by running with --dry-run that fetches just 3 zip codes and prints results.
```

---

## Prompt 2 — CVS Product Catalog

```
Implement src/scrapers/scrape_products.py to build a realistic 10,000-SKU CVS product catalog.

This has two modes controlled by --mode flag:

MODE 1: --mode=build (default, always works)
Generate a comprehensive product catalog from embedded CVS product knowledge. This must be
extremely thorough and realistic. CVS carries products in these categories (include ALL of them):

Pain Relief & Fever, Cold/Flu/Allergy, Digestive Health, Vitamins & Supplements,
Skin Care, Hair Care, Oral Care, Deodorant, Shaving & Grooming, Cosmetics & Makeup,
Baby & Childcare, First Aid & Wound Care, Eye & Ear Care, Snacks & Beverages,
Household Essentials (paper, cleaning, batteries), Feminine Care, Sexual Health,
Foot Care, Diabetes & Blood Sugar, Greeting Cards & Gift Wrap, Photo & Electronics,
Sleep & Relaxation, Smoking Cessation, Pet Care, Seasonal Items

For each category, include the real brands that CVS actually carries (e.g., Tylenol, Advil,
CeraVe, Neutrogena, Pampers, Bounty, etc.) and realistic product names. CVS Health is their
store brand — it should appear in every category at 20-30% lower prices.

Product schema (save to data/real/products.parquet AND data/real/products.csv):
  product_id: int32 (1 to 10000)
  sku: string (realistic 12-digit UPC code, deterministically generated)
  name: string (e.g., "Tylenol Extra Strength Acetaminophen 500mg")
  brand: string
  category: string (one of the categories above)
  subcategory: string (e.g., "Tablets", "Liquid", "Cream")
  price: float64 (realistic CVS retail price)
  unit_cost: float64 (wholesale cost, typically 30-55% of retail)
  weight_oz: float64
  is_store_brand: bool
  is_rx: bool (false for all — OTC only)
  popularity_score: float64 (0.0-1.0, used as prior for transaction generation)
  avg_units_per_store_per_week: float64

The catalog should hit exactly 10,000 products by expanding across brands and size variants
(Travel Size, Family Size, Value Pack, etc.).

Use random.seed(42) for reproducibility. The popularity_score should follow a realistic
distribution: a few products are very popular (score > 0.8), most are moderate (0.2-0.6),
and a long tail of niche items (< 0.2).

MODE 2: --mode=scrape (best-effort, may get blocked)
Crawl https://www.cvs.com/shop/<category> pages. CVS's frontend is a Next.js app — look for
product data in __NEXT_DATA__ script tags, or in the rendered HTML product cards. Extract:
name, brand, price, SKU, category, image URL. Use playwright or requests+beautifulsoup.
Fall back to mode=build if scraping fails.

CLI: click command with --mode, --output-dir, --count (target product count, default 10000).
Print summary: total products, products per category, brand distribution, price distribution.
```

---

## Prompt 3 — Synthetic Customer Generator

```
Implement src/generators/gen_customers.py to generate 10,000,000 synthetic CVS customers.

This must be fast. Target: 10M rows in under 15 minutes on M4 Max with 8 workers.
Use multiprocessing.Pool with chunked generation (100K customers per batch).
Each worker gets its own Faker instance seeded deterministically (seed = 42 + batch_id).

Customer schema (output to data/synthetic/customers/ as partitioned Parquet files,
one file per 100K-customer batch):

  customer_id: int32 (1 to 10,000,000)
  loyalty_number: string (format "EC" + 10-digit zero-padded customer_id)
  first_name: string
  last_name: string
  age: int8 (18-89, distribution weighted toward 25-65 for pharmacy customers)
  gender: string ("M" 45%, "F" 53%, "NB" 2%)
  address: string (street address)
  city: string
  state: string (2-letter code, distributed proportional to CVS store density by state —
         CA gets ~12%, FL ~9%, TX ~8%, NY ~5%, OH ~5%, etc.)
  zip_code: string (valid zip for the assigned state, use Faker.zipcode_in_state)
  is_student: bool (correlates with age: 65% for 18-22, 25% for 23-25, 5% for 26-30, 1% else)
  email: string (deterministic: firstname.lastname{id%10000}@{domain})
  phone: string (format "(XXX) XXX-XXXX")

Email domain distribution: gmail.com 50%, yahoo.com 18%, outlook.com 10%, hotmail.com 8%,
aol.com 5%, icloud.com 5%, comcast.net 2%, live.com 2%.

Age distribution (pharmacy customer skew):
  18-24: 12%, 25-34: 18%, 35-44: 16%, 45-54: 18%, 55-64: 16%, 65-74: 12%, 75-89: 8%

Output: Parquet files with snappy compression to data/synthetic/customers/customers_NNNNN.parquet
(100 files of 100K rows each).

CLI: click command with --count (default 10M), --workers (default 8), --batch-size (default 100K),
--output-dir. Show progress bar and final stats: total time, rows/sec, gender distribution,
age distribution, top states, student percentage.

Include a --test mode that generates just 10,000 customers for validation.
```

---

## Prompt 4 — C Transaction Generator

```
Implement src/generators/txn_generator.c — a high-performance multithreaded transaction
generator in C that produces 10 billion rows in approximately 15-25 minutes on M4 Max.

This is the most performance-critical piece of the entire project. It must generate
10,000,000 customers × 1,000 transactions each = 10,000,000,000 rows.

Architecture:
- 8 pthreads, each handling a range of customer IDs
- Each thread writes to its own set of per-month COMPRESSED files (avoids contention)
- Output files: data/synthetic/transactions/txns_YYYY_MM_tNN.csv.zst (24 months × 8 threads = 192 files)
- CRITICAL DISK OPTIMIZATION: Instead of fopen(), use popen("zstd -1 -o filepath", "w") to write
  zstd-compressed CSV on-the-fly. This reduces output from ~1.5TB to ~180GB. zstd level 1 compresses
  at ~500MB/s per core, which is faster than SSD write bandwidth, so the generator actually runs
  FASTER with compression because it writes fewer bytes to disk. Each thread still uses fprintf()
  exactly the same way — the compression is transparent via the pipe.
  Implementation detail: open files with popen("zstd -1 -q -o <filepath>", "w") and close with
  pclose() instead of fclose(). The -q flag suppresses zstd's progress output. If zstd is not
  found on PATH, fall back to gzip: popen("gzip > <filepath>.gz", "w"). If neither is available,
  fall back to raw fopen() with a warning that disk usage will be ~1.5TB.
- Each thread uses 16MB write buffers (setvbuf) for I/O throughput — this still works with popen
- Use xoshiro256** PRNG (not rand()) for speed and quality — include the implementation inline

Transaction schema (CSV columns):
  transaction_id: int64 (globally unique, derived from customer range + counter)
  loyalty_number: string (EC + 10-digit customer_id)
  customer_id: int32
  store_id: int16 (1 to num_stores)
  product_id: int16 (1 to num_products)
  quantity: int8 (1, 2, or 3 — weighted 82%, 15%, 3%)
  unit_price: float (from product price table, initialized at startup)
  discount_pct: float (0.0 to 0.5)
  discount_amt: float (unit_price × quantity × discount_pct)
  subtotal: float (unit_price × quantity - discount_amt)
  tax_rate: float (state sales tax, looked up by customer's state index)
  tax_amt: float (subtotal × tax_rate)
  total: float (subtotal + tax_amt)
  date: string (YYYY-MM-DD, within 2024-01-01 to 2025-12-31)
  hour: int8 (7-21, weighted by shopping time distribution)

Realistic behavioral modeling:

1. PRODUCT POPULARITY: Zipf distribution (s=1.07). Pre-compute the CDF at startup for
   O(log n) sampling via binary search. A few products should dominate sales.

2. SEASONAL PATTERNS: Product IDs 0-999 (cold/flu) get 2.5x boost in Dec-Feb, 0.6x in
   Jun-Aug. IDs 1000-1499 (skin/sunscreen) get 1.8x boost Jun-Aug. IDs 6000-6999
   (greeting cards) spike 2.5x in December, 2x in February.

3. STORE ASSIGNMENT: Each customer has a "home store" (customer_id % num_stores + 1).
   90% of their transactions are at the home store, 10% at a "nearby" store (±50 IDs).

4. SHOPPING PATTERNS: Customers average ~250 shopping trips over 2 years (1 trip = ~4 items,
   250 trips × 4 = 1000 transactions). Trips are spaced 2-4 days apart with some randomness.
   Basket size follows a triangular distribution centered on 4 (min 1, max 12).

5. TIME OF DAY: Weekdays peak at 11am-1pm and 5-7pm. Weekends peak at 10am-1pm.

6. DISCOUNTS: 30% of items have no discount. 40% have 5-15% off. 20% have 15-30% off.
   10% have 30-50% off (BOGO / coupons).

7. TAX: Simplified state sales tax table (50 entries). Look up by customer_id % 50.

8. PRODUCT PRICES: Initialize a table of 10,000 prices at startup with realistic CVS
   distribution: 15% are $2-5, 30% are $5-10, 30% are $10-20, 17% are $20-35, 8% are $35-50.

Command line: ./txn_generator [num_customers] [txns_per_customer] [num_products] [num_stores] [output_dir]
Default: ./txn_generator 10000000 1000 10000 9000 data/synthetic/transactions

Progress reporting: Each thread prints progress every 10,000 customers to stderr.
Print final summary: total rows, elapsed time, throughput in M rows/sec.

IMPORTANT for macOS compilation:
- Use #include <stdint.h> for uint64_t
- Use pthread (macOS native, no -lpthread needed but include it for portability)
- Compile command: clang -O3 -march=native -o txn_generator txn_generator.c -lm -lpthread
- Make sure mkdir_cmd uses "mkdir -p" with system() or implement it with stat/mkdir

Also create a small test mode: when num_customers is < 10000, it should run fast and
verify output by printing 5 sample rows from the first output file after generation.
```

---

## Prompt 5 — CSV to Parquet Converter (OPTIONAL)

```
Implement scripts/csv_to_parquet.py to convert the zstd-compressed transaction CSV files
(.csv.zst) to Parquet format.

IMPORTANT: This step is OPTIONAL. DuckDB reads .csv.zst files natively, so the pipeline
works without this conversion. However, Parquet queries are 2-3x faster than compressed CSV
queries in DuckDB, so this is worth doing if you want faster analytical queries or if you
want to reclaim the ~180GB of .csv.zst space (Parquet will be ~150GB, so modest savings).

Approach:
- Use multiprocessing.Pool (8 workers) to convert files in parallel
- Each .csv.zst file gets converted to a same-name .parquet file
- Reading .csv.zst: PyArrow can read zstd-compressed CSV directly. Alternatively, decompress
  via subprocess: subprocess.Popen(["zstd", "-d", "-c", filepath], stdout=subprocess.PIPE)
  and feed the stdout pipe to PyArrow's csv.read_csv()
- Use PyArrow's csv.read_csv with explicit column types for optimal compression:
  - transaction_id: int64
  - loyalty_number: string (dictionary encoded — huge compression win since format is "ECXXXXXXXXXX")
  - customer_id: int32
  - store_id: int16
  - product_id: int16
  - quantity: int8
  - unit_price: float32
  - discount_pct: float32
  - discount_amt: float32
  - subtotal: float32
  - tax_rate: float32
  - tax_amt: float32
  - total: float32
  - date: string (dictionary encoded — only ~731 unique dates)
  - hour: int8

Write with snappy compression, row_group_size=1_000_000, and enable dictionary encoding
for loyalty_number and date columns.

After each file is converted, print: filename, row count, CSV size, Parquet size, compression ratio.
At the end, print totals: total rows, total CSV size, total Parquet size, overall ratio.

CLI: click command with --input-dir, --workers, --delete-zst (flag to remove .csv.zst files after conversion).
Print a note at the start that this step is optional since DuckDB reads .csv.zst natively.
```

---

## Prompt 6 — DuckDB Integration

```
Implement src/db/schema.sql and src/db/load_duckdb.py.

schema.sql should define the full analytical schema. load_duckdb.py loads all data into
DuckDB at data/db/cvs_analytics.duckdb.

Loading strategy:
- stores table: CREATE TABLE from data/real/stores.parquet (small, ~9K rows, load fully)
- products table: CREATE TABLE from data/real/products.parquet (small, 10K rows, load fully)
- customers table: CREATE TABLE from data/synthetic/customers/*.parquet (10M rows, ~600MB, load fully)
- transactions: CREATE VIEW over transaction files (10B rows, stays on disk)
  IMPORTANT: The loader must auto-detect whether Parquet or zstd-compressed CSV files exist.
  Check for .parquet files first, then .csv.zst, then .csv. Use the appropriate DuckDB reader:
    - Parquet: read_parquet('data/synthetic/transactions/*.parquet')
    - Zstd CSV: read_csv('data/synthetic/transactions/*.csv.zst')
    - Raw CSV:  read_csv('data/synthetic/transactions/*.csv')
  DuckDB reads all three formats natively with the same query interface. The VIEW abstraction
  means downstream code doesn't need to know which format is on disk.

The VIEW approach is crucial — DuckDB reads Parquet files directly without copying them into
the database file. This means the .duckdb file stays small (~1GB for customers + products)
while queries over 10B transaction rows work via columnar scanning with predicate pushdown.

Create these indexes on loaded tables:
- customers(customer_id)
- customers(state)
- products(product_id)
- products(category)

Create these analytical views:

1. customer_purchase_summary: Per-customer aggregation
   - customer_id, total_transactions, unique_products, unique_stores,
     total_spend, avg_transaction, first_purchase, last_purchase

2. product_monthly_sales: Per-product per-month aggregation
   - product_id, month, units_sold, revenue

3. store_performance: Per-store aggregation
   - store_id, unique_customers, total_transactions, total_revenue, avg_transaction_value

4. customer_product_matrix: Sparse co-occurrence matrix (for ML features)
   - customer_id, product_id, purchase_count, total_spent_on_product, last_purchased

5. product_cooccurrence: Products frequently bought together
   - product_a, product_b, co_purchase_count
   (This is computed from transactions where same customer_id + same date = same basket)

Note: Views 4 and 5 are expensive to materialize on 10B rows. Create them as VIEWs first,
and add a --materialize flag that creates them as actual tables (for when user has time to wait).

CLI: click command with --db-path, --data-dir, --materialize flag.
Print summary: table sizes, view list, sample queries with results.

Also create a validate subcommand in cli.py that runs these checks:
- Products table has exactly 10,000 rows
- Customers table has exactly 10,000,000 rows  
- Transaction count is approximately 10,000,000,000 (±1%)
- Every customer_id in transactions exists in customers
- Every product_id in transactions exists in products
- Date range is 2024-01-01 to 2025-12-31
- No negative totals
- Print a PASS/FAIL for each check
```

---

## Prompt 7 — Custom C Vector Database

```
Implement src/vecdb/vecdb.h, src/vecdb/vecdb.c, and src/vecdb/test_vecdb.c.

This is a purpose-built vector database in C for storing and querying the customer and product
embeddings from the two-tower model. It does NOT need to be a general-purpose vector DB —
it needs to do exactly two things extremely fast:

1. Store 10M customer embedding vectors (256 dimensions, float32) and 10K product vectors
2. For a given product vector, find the top-K most similar customer vectors (batch query)

This is essentially a matrix multiplication: score_matrix = customer_embeddings @ product_embedding.T
On M4 Max, we can use the Accelerate framework (Apple's BLAS) for this.

vecdb.h — Public API:
```c
#ifndef VECDB_H
#define VECDB_H

#include <stdint.h>

typedef struct VecDB VecDB;

// Create a new vector database
// capacity: max number of vectors, dims: embedding dimensions
VecDB* vecdb_create(uint32_t capacity, uint16_t dims);

// Free the database
void vecdb_destroy(VecDB* db);

// Add a vector (id must be unique, vec must be `dims` floats)
int vecdb_insert(VecDB* db, uint32_t id, const float* vec);

// Batch insert from a contiguous float array (n vectors × dims floats)
int vecdb_batch_insert(VecDB* db, const uint32_t* ids, const float* vecs, uint32_t n);

// Find top-k most similar vectors to query (cosine similarity)
// Results written to out_ids (top-k IDs) and out_scores (top-k scores)
int vecdb_query_topk(const VecDB* db, const float* query, uint32_t k,
                     uint32_t* out_ids, float* out_scores);

// Batch query: for each of n query vectors, find top-k similar
// out_ids is n×k, out_scores is n×k
int vecdb_batch_query_topk(const VecDB* db, const float* queries, uint32_t n,
                           uint32_t k, uint32_t* out_ids, float* out_scores);

// Save database to disk (binary format: header + raw float array)
int vecdb_save(const VecDB* db, const char* path);

// Load database from disk
VecDB* vecdb_load(const char* path);

// Get stats
uint32_t vecdb_count(const VecDB* db);
uint16_t vecdb_dims(const VecDB* db);

#endif
```

vecdb.c — Implementation:
- Store vectors in a contiguous float array (capacity × dims) for cache-friendly access
- For queries, compute dot products using Apple Accelerate's cblas_sgemv (matrix-vector)
  or cblas_sgemm (matrix-matrix for batch queries)
- Top-K selection: use a min-heap of size K, scan all dot products, keep top K
  - For K < 100 and N=10M, a simple partial sort is faster than a full sort
- L2-normalize all vectors on insert so dot product = cosine similarity
- Binary file format: [magic: 4 bytes "VCDB"] [version: uint32] [count: uint32]
  [dims: uint16] [padding: 6 bytes] [ids: count × uint32] [vecs: count × dims × float32]
- Thread safety: read-only queries are safe to parallelize. Inserts require a mutex.

For macOS/M4 Max optimization:
- #include <Accelerate/Accelerate.h>
- Use cblas_sgemv for single query (matrix × vector)
- Use cblas_sgemm for batch query (matrix × matrix, then extract top-K per row)
- Use vDSP_vdist or vDSP_dotpr for individual dot products if needed
- Compile: clang -O3 -march=native -framework Accelerate -o vecdb vecdb.c

test_vecdb.c:
- Create a DB with 100,000 vectors of 256 dims (random data)
- Insert all vectors
- Query top-10 for a random vector
- Verify results are sorted by descending similarity
- Benchmark: time 1000 individual queries and 1 batch query of 10K vectors
- Print: insert throughput (vectors/sec), query latency (ms), batch throughput
- Save to disk, reload, verify query results match
```

---

## Prompt 8 — Two-Tower PyTorch Model

```
Implement src/ml/two_tower.py, src/ml/train.py, and src/ml/inference.py.

This is a two-tower (dual encoder) recommendation model. One tower encodes customers into
a 256-dim embedding, the other encodes products. The dot product of a customer embedding
and product embedding predicts purchase probability.

two_tower.py — Model architecture:

class CustomerTower(nn.Module):
  Input features: customer_id (embedding lookup), age (normalized), gender (one-hot 3),
                  state (embedding lookup, 51 states), is_student (binary)
  Architecture: Embedding(10M, 64) for customer_id → concat with [age, gender_onehot, 
                state_embed(51, 16), is_student] → Linear(64+1+3+16+1, 256) → ReLU → 
                Linear(256, 256) → L2 normalize

class ProductTower(nn.Module):
  Input features: product_id (embedding lookup), category (embedding lookup, 25 categories),
                  brand (embedding lookup, ~150 brands), price (normalized), is_store_brand (binary),
                  popularity_score (float)
  Architecture: Embedding(10K, 64) for product_id → concat with [category_embed(25, 16),
                brand_embed(150, 16), price_norm, is_store_brand, popularity] →
                Linear(64+16+16+1+1+1, 256) → ReLU → Linear(256, 256) → L2 normalize

class TwoTowerModel(nn.Module):
  Combines both towers. Forward takes customer features + product features.
  Returns dot product (scalar) = predicted affinity score.
  Loss: Binary cross-entropy with logits. Positive examples are real transactions.
  Negative sampling: For each positive (customer, product) pair, sample 4 random products
  that the customer did NOT buy as negatives.

train.py — Training loop:

Data loading strategy (critical for 10B rows):
- Do NOT load all transactions into memory
- Use DuckDB to sample training data: SELECT customer_id, product_id FROM transactions
  USING SAMPLE 1% (100M rows). This fits in memory and is statistically sufficient.
- Or: use DuckDB's streaming result set to iterate in batches
- Build lookup tables for customer features and product features from DuckDB
- Training batch size: 8192
- Optimizer: Adam, lr=1e-3 with cosine annealing schedule
- Train for 5 epochs on the sampled data
- Use MPS (Metal Performance Shaders) device for M4 Max GPU acceleration:
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

Training pipeline:
1. Connect to DuckDB, load customer feature table (10M × features) — fits in 64GB RAM
2. Load product feature table (10K × features) — trivial
3. Sample 100M positive (customer_id, product_id) pairs from transactions
4. For each batch: construct positive + negative examples, forward both towers, compute loss
5. Save model checkpoints every epoch to data/model/
6. After training, extract all customer embeddings (10M × 256) and product embeddings (10K × 256)
7. Save embeddings as numpy arrays: data/model/customer_embeddings.npy, product_embeddings.npy
8. Print training metrics: loss per epoch, training time, final embedding stats

CLI: click command with --epochs, --batch-size, --lr, --sample-pct, --device (auto/mps/cpu),
     --db-path, --output-dir

inference.py — Full inference pipeline:

1. Load customer embeddings (10M × 256) and product embeddings (10K × 256)
2. Load them into the C vector database (call vecdb via ctypes or write a Python wrapper)
3. For each product (or a specified subset), find top-100 customers most likely to buy it
4. Output results to data/results/recommendations.parquet:
   product_id, customer_id, score, rank

For the full 10M × 10K computation:
- This is a matrix multiply: (10M × 256) @ (256 × 10K) = (10M × 10K) score matrix
- On M4 Max GPU via PyTorch MPS: should complete in seconds
- BUT the output matrix is 10M × 10K × 4 bytes = 400 GB — doesn't fit in memory
- Solution: chunk the computation. Process 100K customers at a time:
  - (100K × 256) @ (256 × 10K) = (100K × 10K) = 4 GB per chunk, fits in memory
  - Extract top-100 products per customer from each chunk
  - 100 chunks × a few seconds each = ~5-10 minutes total

Alternative: for the "which customers should we target for product X" use case,
use the vecdb: load 10M customer vectors, query with each product vector for top-K.

CLI: click command with --mode (full-matrix / per-product / per-customer),
     --top-k, --db-path, --output-dir, --chunk-size (default 100K)

Print final results: sample recommendations, coverage stats, timing breakdown.
```

---

## Prompt 9 — CLI and End-to-End Tests

```
Implement src/cli.py as the main entry point, and implement the test files.

src/cli.py should be a rich click CLI group called "toydatadecomp" with these subcommands:

  toydatadecomp scrape stores     — runs scrape_stores.py
  toydatadecomp scrape products   — runs scrape_products.py (default: build mode)
  toydatadecomp generate customers — runs gen_customers.py
  toydatadecomp generate transactions — runs txn_generator (the compiled C binary)
  toydatadecomp convert parquet   — runs csv_to_parquet.py
  toydatadecomp load              — runs load_duckdb.py
  toydatadecomp train             — runs train.py
  toydatadecomp infer             — runs inference.py
  toydatadecomp validate          — runs validation checks
  toydatadecomp status            — shows what data exists, what's missing, sizes, row counts

The "status" command is especially important — it should check for the existence of every
expected data file and print a nice rich table showing:
  Component | Status | Rows | Size | Path
  Stores    | ✓      | 9,021 | 2.1 MB | data/real/stores.parquet
  Products  | ✓      | 10,000 | 1.8 MB | data/real/products.parquet  
  Customers | ✗      | —     | —      | data/synthetic/customers/
  etc.

Use rich library for pretty console output (tables, progress bars, colored status).

tests/test_generators.py:
- Test customer generator with 1000 customers: verify schema, distributions, uniqueness
- Test product builder: verify 10,000 products, all categories present, prices reasonable
- Test that Parquet files are readable by PyArrow

tests/test_vecdb.py:
- Test insert + query correctness with small known vectors
- Test save/load roundtrip
- Test that top-K results are properly sorted
- Benchmark with 10K vectors

tests/test_pipeline.py:
- End-to-end mini test: generate 1000 customers, 100 products, 100K transactions
- Load into DuckDB, verify joins work
- Train model for 1 epoch on the tiny dataset
- Run inference, verify output shape
- This should complete in under 60 seconds

All tests use pytest. The Makefile's `make test` target should run pytest with verbose output.
```

---

## Prompt 10 — Final Integration and Polish

```
Review the entire toydatadecomp project and make sure everything is wired together correctly.

Specific things to check and fix:

1. The Makefile `full-pipeline` target should run these in order:
   make install
   make scrape-products  (build mode, always succeeds)
   make scrape-stores    (best effort, continues on failure)
   make gen-customers
   make compile-c
   make gen-transactions  (writes .csv.zst — ~180GB, NOT 1.5TB)
   make load-db           (reads .csv.zst directly, no Parquet conversion needed)
   make train
   make inference
   make validate

   Note: `make convert-parquet` is deliberately NOT in the default pipeline. DuckDB reads
   .csv.zst natively. If the user wants faster queries, they can run `make full-pipeline-parquet`
   instead, which includes the conversion step and deletes the .csv.zst files afterward.

2. If scrape-stores fails (CVS blocks us), the pipeline should continue with a generated
   set of ~9,000 synthetic store locations (realistic US addresses distributed by the same
   state weights used for customers). Add a `make gen-stores` fallback target.

3. The cli.py "status" command should work at any point in the pipeline and show exactly
   what's been completed and what's pending.

4. Add a `make demo` target that runs the full pipeline at 1/1000th scale:
   - 10K customers, 100 transactions each = 1M transaction rows
   - Takes ~2 minutes total
   - Useful for verifying everything works before the full run

5. Make sure all Python scripts handle interrupts gracefully (save partial progress).

6. Add a scripts/validate.py that does comprehensive data quality checks and prints a
   formatted report with PASS/FAIL for each check.

7. The final output of `make full-pipeline` should print:
   - Total elapsed time
   - Data sizes on disk
   - Sample recommendations: "Top 10 customers most likely to buy {random product}"
   - A summary of the model's performance metrics

8. Make sure the C transaction generator compiles on macOS with:
   clang -O3 -march=native -o txn_generator txn_generator.c -lm -lpthread

9. Make sure vecdb compiles on macOS with:
   clang -O3 -march=native -framework Accelerate -o test_vecdb test_vecdb.c vecdb.c -lm

10. All file paths should be relative to the project root, not absolute.
    Use Path(__file__).resolve().parent.parent patterns in Python.
```


-- toydatadecomp DuckDB analytical schema
-- Full schema definition for the retail recommendation engine
--
-- Tables loaded from files:
--   stores       - Real CVS store locations (~9K rows, from parquet)
--   products     - Real CVS product catalog (~12K rows, from parquet)
--   customers    - Synthetic customers (10M rows, from parquet)
--   coupon_clips - Synthetic digital coupon clips (from parquet)
--
-- View over files on disk:
--   transactions - Synthetic transactions (10B rows, VIEW with predicate pushdown)
--
-- load_duckdb.py creates tables via CTAS and auto-detects the transaction
-- file format (parquet > csv.zst > csv). This file documents the full schema
-- and provides the index + analytical view definitions executed post-load.

-- ============================================================
-- Table definitions (reference — actual creation is via CTAS)
-- ============================================================

-- Stores: ~9K CVS locations (store_id cast from VARCHAR to INTEGER for FK joins)
CREATE TABLE IF NOT EXISTS stores (
    store_id    INTEGER,
    name        VARCHAR,
    address     VARCHAR,
    city        VARCHAR,
    state       VARCHAR,
    zip_code    VARCHAR,
    latitude    DOUBLE,
    longitude   DOUBLE,
    phone       VARCHAR,
    store_type  VARCHAR,
    hours_mon_fri VARCHAR,
    hours_sat   VARCHAR,
    hours_sun   VARCHAR
);

-- Products: ~12K SKUs
CREATE TABLE IF NOT EXISTS products (
    product_id              INTEGER,
    sku                     VARCHAR,
    name                    VARCHAR,
    brand                   VARCHAR,
    category                VARCHAR,
    subcategory             VARCHAR,
    price                   DOUBLE,
    unit_cost               DOUBLE,
    weight_oz               DOUBLE,
    is_store_brand          BOOLEAN,
    is_rx                   BOOLEAN,
    popularity_score        DOUBLE,
    avg_units_per_store_per_week DOUBLE,
    rx_generic_name         VARCHAR,
    rx_therapeutic_class    VARCHAR,
    rx_days_supply          INTEGER
);

-- Customers: 10M synthetic
CREATE TABLE IF NOT EXISTS customers (
    customer_id    INTEGER,
    loyalty_number VARCHAR,
    first_name     VARCHAR,
    last_name      VARCHAR,
    age            TINYINT,
    gender         VARCHAR,
    address        VARCHAR,
    city           VARCHAR,
    state          VARCHAR,
    zip_code       VARCHAR,
    is_student     BOOLEAN,
    email          VARCHAR,
    phone          VARCHAR
);

-- Coupon clips: digital coupon engagement data
CREATE TABLE IF NOT EXISTS coupon_clips (
    clip_id         BIGINT,
    loyalty_number  VARCHAR,
    product_id      INTEGER,
    clip_date       DATE,
    expiration_date DATE,
    discount_type   VARCHAR,
    discount_value  DOUBLE,
    redeemed        BOOLEAN
);

-- Transactions: 10B rows, VIEW over files on disk (format auto-detected)
-- Example for parquet:
--   CREATE VIEW transactions AS
--   SELECT * FROM read_parquet('data/synthetic/transactions/*.parquet');
-- Example for zstd CSV:
--   CREATE VIEW transactions AS
--   SELECT * FROM read_csv('data/synthetic/transactions/*.csv.zst');
--
-- Transaction columns:
--   transaction_id BIGINT, loyalty_number VARCHAR, customer_id BIGINT,
--   store_id BIGINT, product_id BIGINT, quantity BIGINT,
--   unit_price DOUBLE, discount_pct DOUBLE, discount_amt DOUBLE,
--   subtotal DOUBLE, tax_rate DOUBLE, tax_amt DOUBLE, total DOUBLE,
--   date DATE, hour BIGINT

-- ============================================================
-- Indexes (on loaded tables)
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_customers_customer_id ON customers(customer_id);
CREATE INDEX IF NOT EXISTS idx_customers_state ON customers(state);
CREATE INDEX IF NOT EXISTS idx_products_product_id ON products(product_id);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_coupon_clips_loyalty ON coupon_clips(loyalty_number);
CREATE INDEX IF NOT EXISTS idx_coupon_clips_product ON coupon_clips(product_id);

-- ============================================================
-- Analytical Views
-- ============================================================

-- 1. Per-customer purchase summary
CREATE OR REPLACE VIEW customer_purchase_summary AS
SELECT
    customer_id,
    COUNT(*)                    AS total_transactions,
    COUNT(DISTINCT product_id)  AS unique_products,
    COUNT(DISTINCT store_id)    AS unique_stores,
    SUM(total)                  AS total_spend,
    AVG(total)                  AS avg_transaction,
    MIN(date)                   AS first_purchase,
    MAX(date)                   AS last_purchase
FROM transactions
GROUP BY customer_id;

-- 2. Per-product per-month sales
CREATE OR REPLACE VIEW product_monthly_sales AS
SELECT
    product_id,
    DATE_TRUNC('month', date)   AS month,
    SUM(quantity)               AS units_sold,
    SUM(total)                  AS revenue
FROM transactions
GROUP BY product_id, DATE_TRUNC('month', date);

-- 3. Per-store performance
CREATE OR REPLACE VIEW store_performance AS
SELECT
    store_id,
    COUNT(DISTINCT customer_id) AS unique_customers,
    COUNT(*)                    AS total_transactions,
    SUM(total)                  AS total_revenue,
    AVG(total)                  AS avg_transaction_value
FROM transactions
GROUP BY store_id;

-- 4. Customer-product sparse matrix (for ML features)
CREATE OR REPLACE VIEW customer_product_matrix AS
SELECT
    customer_id,
    product_id,
    COUNT(*)    AS purchase_count,
    SUM(total)  AS total_spent_on_product,
    MAX(date)   AS last_purchased
FROM transactions
GROUP BY customer_id, product_id;

-- 5. Product co-occurrence (same customer + same date = same basket)
CREATE OR REPLACE VIEW product_cooccurrence AS
SELECT
    t1.product_id AS product_a,
    t2.product_id AS product_b,
    COUNT(*)      AS co_purchase_count
FROM transactions t1
JOIN transactions t2
    ON  t1.customer_id = t2.customer_id
    AND t1.date        = t2.date
    AND t1.product_id  < t2.product_id
GROUP BY t1.product_id, t2.product_id;

-- toydatadecomp DuckDB schema
-- Stores real + synthetic data for the recommendation engine

-- Real data: CVS store locations
CREATE TABLE IF NOT EXISTS stores (
    store_id INTEGER PRIMARY KEY,
    address VARCHAR,
    city VARCHAR,
    state VARCHAR(2),
    zip_code VARCHAR(10),
    latitude DOUBLE,
    longitude DOUBLE,
    phone VARCHAR(20)
);

-- Real data: CVS product catalog
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    name VARCHAR,
    brand VARCHAR,
    category VARCHAR,
    subcategory VARCHAR,
    price DOUBLE,
    upc VARCHAR(20)
);

-- Synthetic data: 10M customers
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    first_name VARCHAR,
    last_name VARCHAR,
    email VARCHAR,
    age INTEGER,
    gender VARCHAR(1),
    zip_code VARCHAR(10),
    home_store_id INTEGER REFERENCES stores(store_id),
    loyalty_tier VARCHAR(10),
    signup_date DATE
);

-- Synthetic data: 10B transactions
CREATE TABLE IF NOT EXISTS transactions (
    txn_id BIGINT PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    store_id INTEGER REFERENCES stores(store_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER,
    unit_price DOUBLE,
    total_price DOUBLE,
    txn_date DATE,
    txn_time TIME,
    payment_method VARCHAR(20)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_txn_customer ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_txn_product ON transactions(product_id);
CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(txn_date);
CREATE INDEX IF NOT EXISTS idx_customer_store ON customers(home_store_id);

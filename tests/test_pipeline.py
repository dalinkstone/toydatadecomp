"""End-to-end mini pipeline test.

Generates small datasets, loads into DuckDB, trains for 1 epoch,
runs inference, and verifies output shape. Target: under 60 seconds.
"""

import os
import shutil
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch


@pytest.fixture(scope="module")
def pipeline_dir():
    """Create a temp directory for the entire mini pipeline."""
    tmpdir = tempfile.mkdtemp(prefix="test_pipeline_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def customers_path(pipeline_dir):
    """Generate 1000 mini customers."""
    from generators.gen_customers import _generate_batch

    out = pipeline_dir / "customers"
    out.mkdir()
    _generate_batch((0, 1, 1000, str(out), 42))
    return out


@pytest.fixture(scope="module")
def products_path(pipeline_dir):
    """Build 100 mini products."""
    out = pipeline_dir / "products"
    out.mkdir()

    rng = np.random.default_rng(42)
    categories = ["Pain Relief", "Cold & Flu", "Vitamins", "Beauty", "Oral Care",
                   "First Aid", "Digestive", "Allergy", "Sleep", "Wellness"]
    brands = ["CVS Health", "Tylenol", "Advil", "Benadryl", "Colgate",
              "Crest", "Tums", "Claritin", "ZzzQuil", "Nature Made"]

    rows = {
        "product_id": list(range(1, 101)),
        "category": [categories[i % len(categories)] for i in range(100)],
        "subcategory": ["General"] * 100,
        "brand": [brands[i % len(brands)] for i in range(100)],
        "name": [f"Product {i}" for i in range(1, 101)],
        "type_code": ["P30"] * 100,
        "price": rng.uniform(2.0, 50.0, 100).round(2).tolist(),
        "unit_cost": rng.uniform(1.0, 25.0, 100).round(2).tolist(),
        "is_store_brand": [i % 5 == 0 for i in range(100)],
        "is_rx": [False] * 100,
        "popularity_score": rng.uniform(0.1, 1.0, 100).round(3).tolist(),
    }

    table = pa.table(rows)
    pq.write_table(table, out / "products.parquet")
    # Also write CSV for txn compatibility
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "products.csv", index=False)
    return out


@pytest.fixture(scope="module")
def transactions_path(pipeline_dir, customers_path, products_path):
    """Generate 100K mini transactions as CSV."""
    out = pipeline_dir / "transactions"
    out.mkdir()

    rng = np.random.default_rng(99)
    n = 100_000

    customer_ids = rng.integers(1, 1001, size=n)
    product_ids = rng.integers(1, 101, size=n)
    store_ids = rng.integers(1, 10, size=n)
    quantities = rng.integers(1, 6, size=n)
    prices = rng.uniform(2.0, 50.0, n).round(2)
    totals = (quantities * prices).round(2)

    # Generate dates in 2024
    base_date = np.datetime64("2024-01-01")
    offsets = rng.integers(0, 365, size=n)
    dates = [str(base_date + np.timedelta64(int(d), "D")) for d in offsets]

    import pandas as pd
    df = pd.DataFrame({
        "transaction_id": range(1, n + 1),
        "customer_id": customer_ids,
        "product_id": product_ids,
        "store_id": store_ids,
        "quantity": quantities,
        "price": prices,
        "total": totals,
        "date": dates,
    })
    df.to_csv(out / "txns_test.csv", index=False)
    return out


@pytest.fixture(scope="module")
def duckdb_path(pipeline_dir, customers_path, products_path, transactions_path):
    """Load all mini data into DuckDB."""
    db_path = pipeline_dir / "test.duckdb"
    con = duckdb.connect(str(db_path))

    # Load stores (create minimal)
    con.execute("""
        CREATE TABLE stores AS
        SELECT * FROM (VALUES
            (1, 'Store 1', '123 Main St', 'Boston', 'MA', '02101', 42.36, -71.06, '555-0001', 'pharmacy'),
            (2, 'Store 2', '456 Oak Ave', 'New York', 'NY', '10001', 40.71, -74.01, '555-0002', 'pharmacy'),
            (3, 'Store 3', '789 Pine Rd', 'Chicago', 'IL', '60601', 41.88, -87.63, '555-0003', 'pharmacy')
        ) AS t(store_id, name, address, city, state, zip_code, lat, lng, phone, store_type)
    """)

    # Load products
    con.execute(f"""
        CREATE TABLE products AS
        SELECT * FROM read_parquet('{products_path}/products.parquet')
    """)

    # Load customers
    con.execute(f"""
        CREATE TABLE customers AS
        SELECT * FROM read_parquet('{customers_path}/*.parquet')
    """)

    # Load transactions
    con.execute(f"""
        CREATE TABLE transactions AS
        SELECT * FROM read_csv('{transactions_path}/txns_test.csv', auto_detect=true)
    """)

    con.close()
    return db_path


class TestMiniPipeline:
    """End-to-end mini test: 1000 customers, 100 products, 100K transactions."""

    def test_data_loaded(self, duckdb_path):
        con = duckdb.connect(str(duckdb_path), read_only=True)
        stores = con.execute("SELECT COUNT(*) FROM stores").fetchone()[0]
        products = con.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        customers = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        transactions = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        con.close()

        assert stores == 3
        assert products == 100
        assert customers == 1000
        assert transactions == 100_000

    def test_joins_work(self, duckdb_path):
        """Verify that foreign key joins produce results."""
        con = duckdb.connect(str(duckdb_path), read_only=True)

        # Customer-transaction join
        ct = con.execute("""
            SELECT COUNT(*) FROM transactions t
            JOIN customers c ON t.customer_id = c.customer_id
        """).fetchone()[0]
        assert ct > 0

        # Product-transaction join
        pt = con.execute("""
            SELECT COUNT(*) FROM transactions t
            JOIN products p ON t.product_id = p.product_id
        """).fetchone()[0]
        assert pt > 0

        con.close()

    def test_aggregations(self, duckdb_path):
        """Verify basic aggregation queries work."""
        con = duckdb.connect(str(duckdb_path), read_only=True)

        # Per-customer purchase summary
        result = con.execute("""
            SELECT customer_id, COUNT(*) AS txn_count, SUM(total) AS total_spend
            FROM transactions
            GROUP BY customer_id
            ORDER BY total_spend DESC
            LIMIT 5
        """).fetchall()
        assert len(result) == 5
        assert all(row[1] > 0 for row in result)

        con.close()


class TestMiniTrainInfer:
    """Train model for 1 epoch on tiny dataset, run inference."""

    def test_train_one_epoch(self, duckdb_path, pipeline_dir):
        """Train the two-tower model for 1 epoch on mini data."""
        from ml.two_tower import CustomerTower, ProductTower, TwoTowerModel

        con = duckdb.connect(str(duckdb_path), read_only=True)

        # Load product info
        products_df = con.execute("SELECT * FROM products").fetchdf()
        product_ids = products_df["product_id"].tolist()
        num_products = len(product_ids)

        # Build vocabs
        categories = sorted(products_df["category"].unique())
        brands = sorted(products_df["brand"].unique())
        category_vocab = {c: i for i, c in enumerate(categories)}
        brand_vocab = {b: i for i, b in enumerate(brands)}

        product_lookup = {}
        for _, row in products_df.iterrows():
            product_lookup[int(row["product_id"])] = {
                "category": row["category"],
                "brand": row["brand"],
                "price": float(row["price"]),
                "is_store_brand": bool(row["is_store_brand"]),
                "popularity_score": float(row["popularity_score"]),
                "margin_pct": 0.3,
                "coupon_clip_rate": 0.1,
                "coupon_redemption_rate": 0.05,
                "organic_purchase_ratio": 0.8,
            }

        # Load customer features (simplified)
        customers_df = con.execute("SELECT * FROM customers").fetchdf()
        max_cid = int(customers_df["customer_id"].max()) + 1

        gender_map = {"M": 0, "F": 1, "NB": 2}
        state_list = sorted(customers_df["state"].unique())
        state_map = {s: i for i, s in enumerate(state_list)}

        customer_features = {
            "age": np.zeros(max_cid, dtype=np.float32),
            "gender": np.zeros(max_cid, dtype=np.int32),
            "state": np.zeros(max_cid, dtype=np.int32),
            "is_student": np.zeros(max_cid, dtype=np.float32),
            "total_spend": np.zeros(max_cid, dtype=np.float32),
            "coupon_engagement_score": np.zeros(max_cid, dtype=np.float32),
            "coupon_redemption_rate": np.zeros(max_cid, dtype=np.float32),
            "avg_basket_size": np.zeros(max_cid, dtype=np.float32),
        }

        for _, row in customers_df.iterrows():
            cid = int(row["customer_id"])
            customer_features["age"][cid] = float(row["age"])
            customer_features["gender"][cid] = gender_map.get(row["gender"], 0)
            customer_features["state"][cid] = state_map.get(row["state"], 0)
            customer_features["is_student"][cid] = float(row["is_student"])

        # Add transaction-derived features
        txn_stats = con.execute("""
            SELECT customer_id, SUM(total) as total_spend, AVG(total) as avg_basket
            FROM transactions GROUP BY customer_id
        """).fetchdf()
        for _, row in txn_stats.iterrows():
            cid = int(row["customer_id"])
            if cid < max_cid:
                customer_features["total_spend"][cid] = float(row["total_spend"])
                customer_features["avg_basket_size"][cid] = float(row["avg_basket"])

        con.close()

        # Normalization stats
        norm_stats = {
            "age": (45.0, 15.0),
            "total_spend": (5000.0, 3000.0),
            "avg_basket_size": (50.0, 30.0),
            "price": (15.0, 10.0),
        }

        # Sample training pairs from transactions
        rng = np.random.default_rng(42)
        sample_size = min(5000, 100_000)
        con = duckdb.connect(str(duckdb_path), read_only=True)
        pairs = con.execute(f"""
            SELECT customer_id, product_id FROM transactions
            USING SAMPLE {sample_size}
        """).fetchdf()
        con.close()

        from ml.train import TransactionDataset, collate_fn

        dataset = TransactionDataset(
            customer_ids=pairs["customer_id"].values,
            product_ids=pairs["product_id"].values,
            customer_features=customer_features,
            product_lookup=product_lookup,
            brand_vocab=brand_vocab,
            category_vocab=category_vocab,
            norm_stats=norm_stats,
            num_products=num_products,
            neg_samples=2,
        )

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=256, shuffle=True,
            collate_fn=collate_fn, num_workers=0,
        )

        # Build model (small embeddings for test)
        customer_tower = CustomerTower(
            num_customers=max_cid, num_states=len(state_list) + 1,
            cust_embed_dim=16, state_embed_dim=8,
            hidden_dim=64, output_dim=64,
        )
        product_tower = ProductTower(
            num_products=max(product_ids) + 1,
            num_categories=len(categories) + 1,
            num_brands=len(brands) + 1,
            prod_embed_dim=16, cat_embed_dim=8, brand_embed_dim=8,
            hidden_dim=64, output_dim=64,
        )
        model = TwoTowerModel(customer_tower, product_tower)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train 1 epoch
        model.train()
        total_loss = 0.0
        batches = 0
        for cust_batch, pos_batch, neg_batches, pos_margin in loader:
            pos_scores, neg_scores = model(cust_batch, pos_batch, neg_batches)
            loss = TwoTowerModel.compute_loss(pos_scores, neg_scores, pos_margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        assert avg_loss > 0, "Loss should be positive"
        assert batches > 0, "Should have at least one batch"

        # Save model for inference test
        model_dir = pipeline_dir / "model"
        model_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_dir / "model.pt")

        # Store artifacts for inference test
        TestMiniTrainInfer._model = model
        TestMiniTrainInfer._customer_features = customer_features
        TestMiniTrainInfer._product_lookup = product_lookup
        TestMiniTrainInfer._brand_vocab = brand_vocab
        TestMiniTrainInfer._category_vocab = category_vocab
        TestMiniTrainInfer._norm_stats = norm_stats
        TestMiniTrainInfer._product_ids = product_ids
        TestMiniTrainInfer._max_cid = max_cid

    def test_inference_output_shape(self, pipeline_dir):
        """Run inference and verify output shape."""
        model = TestMiniTrainInfer._model
        product_lookup = TestMiniTrainInfer._product_lookup
        customer_features = TestMiniTrainInfer._customer_features
        brand_vocab = TestMiniTrainInfer._brand_vocab
        category_vocab = TestMiniTrainInfer._category_vocab
        norm_stats = TestMiniTrainInfer._norm_stats
        product_ids = TestMiniTrainInfer._product_ids
        max_cid = TestMiniTrainInfer._max_cid

        from ml.train import TransactionDataset

        model.eval()

        # Extract product embeddings
        dummy_ds = TransactionDataset(
            np.array([1]), np.array([1]),
            customer_features, product_lookup,
            brand_vocab, category_vocab, norm_stats,
            num_products=len(product_ids), neg_samples=0,
        )

        prod_feats_list = [dummy_ds._product_feats(pid) for pid in product_ids]
        prod_batch = {
            "product_id": torch.tensor([x["product_id"] for x in prod_feats_list], dtype=torch.long),
            "category_id": torch.tensor([x["category_id"] for x in prod_feats_list], dtype=torch.long),
            "brand_id": torch.tensor([x["brand_id"] for x in prod_feats_list], dtype=torch.long),
            "price": torch.tensor([x["price"] for x in prod_feats_list], dtype=torch.float32),
            "is_store_brand": torch.tensor([x["is_store_brand"] for x in prod_feats_list], dtype=torch.float32),
            "popularity": torch.tensor([x["popularity"] for x in prod_feats_list], dtype=torch.float32),
            "margin_pct": torch.tensor([x["margin_pct"] for x in prod_feats_list], dtype=torch.float32),
            "coupon_clip_rate": torch.tensor([x["coupon_clip_rate"] for x in prod_feats_list], dtype=torch.float32),
            "coupon_redemption_rate": torch.tensor([x["coupon_redemption_rate"] for x in prod_feats_list], dtype=torch.float32),
            "organic_purchase_ratio": torch.tensor([x["organic_purchase_ratio"] for x in prod_feats_list], dtype=torch.float32),
        }

        with torch.no_grad():
            prod_embeddings = model.product_tower(**prod_batch)

        assert prod_embeddings.shape == (len(product_ids), 64)

        # Extract customer embeddings for a small sample
        sample_cids = list(range(1, min(51, max_cid)))
        cust_feats_list = [dummy_ds._customer_feats(cid) for cid in sample_cids]
        cust_batch = {
            "customer_id": torch.tensor([x["customer_id"] for x in cust_feats_list], dtype=torch.long),
            "age": torch.tensor([x["age"] for x in cust_feats_list], dtype=torch.float32),
            "gender_onehot": torch.tensor([x["gender_onehot"] for x in cust_feats_list], dtype=torch.float32),
            "state_id": torch.tensor([x["state_id"] for x in cust_feats_list], dtype=torch.long),
            "is_student": torch.tensor([x["is_student"] for x in cust_feats_list], dtype=torch.float32),
            "total_spend": torch.tensor([x["total_spend"] for x in cust_feats_list], dtype=torch.float32),
            "coupon_engagement": torch.tensor([x["coupon_engagement"] for x in cust_feats_list], dtype=torch.float32),
            "coupon_redemption_rate": torch.tensor([x["coupon_redemption_rate"] for x in cust_feats_list], dtype=torch.float32),
            "avg_basket_size": torch.tensor([x["avg_basket_size"] for x in cust_feats_list], dtype=torch.float32),
        }

        with torch.no_grad():
            cust_embeddings = model.customer_tower(**cust_batch)

        assert cust_embeddings.shape == (len(sample_cids), 64)

        # Compute affinity matrix
        scores = cust_embeddings @ prod_embeddings.T
        assert scores.shape == (len(sample_cids), len(product_ids))

        # Scores should be in a reasonable range for normalized embeddings
        assert scores.min() >= -1.1
        assert scores.max() <= 1.1

"""Tests for synthetic data generators.

Validates customer generation output format, data distributions,
uniqueness, and product builder output.
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


# ---------------------------------------------------------------------------
# Customer generator tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def customer_dir():
    """Generate 1000 customers into a temp directory (once per module)."""
    from generators.gen_customers import _generate_batch

    tmpdir = tempfile.mkdtemp(prefix="test_customers_")
    # Generate 1000 customers in a single batch
    _generate_batch((0, 1, 1000, tmpdir, 42))
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def customer_table(customer_dir):
    """Read back the generated Parquet file as a PyArrow table."""
    files = sorted(Path(customer_dir).glob("*.parquet"))
    assert len(files) == 1
    return pq.read_table(files[0])


class TestCustomerGenerator:
    """Test customer generator with 1000 customers."""

    def test_row_count(self, customer_table):
        assert customer_table.num_rows == 1000

    def test_schema_columns(self, customer_table):
        expected = {
            "customer_id", "loyalty_number", "first_name", "last_name",
            "age", "gender", "address", "city", "state", "zip_code",
            "is_student", "email", "phone",
        }
        assert set(customer_table.column_names) == expected

    def test_customer_id_uniqueness(self, customer_table):
        ids = customer_table.column("customer_id").to_pylist()
        assert len(set(ids)) == len(ids)

    def test_loyalty_number_format(self, customer_table):
        for ln in customer_table.column("loyalty_number").to_pylist():
            assert ln.startswith("EC")
            assert len(ln) == 12  # EC + 10 digits

    def test_age_range(self, customer_table):
        ages = customer_table.column("age").to_numpy()
        assert ages.min() >= 18
        assert ages.max() <= 89

    def test_age_distribution(self, customer_table):
        """Verify age distribution roughly matches expected bins."""
        ages = customer_table.column("age").to_numpy()
        # Young adults (18-34) should be ~30% of population
        young = ((ages >= 18) & (ages <= 34)).sum() / len(ages)
        assert 0.15 < young < 0.50

    def test_gender_values(self, customer_table):
        genders = set(customer_table.column("gender").to_pylist())
        assert genders.issubset({"M", "F", "NB"})

    def test_gender_distribution(self, customer_table):
        genders = customer_table.column("gender").to_pylist()
        female_pct = genders.count("F") / len(genders)
        # Expected ~53% female, allow wide range for 1000 samples
        assert 0.35 < female_pct < 0.70

    def test_state_values(self, customer_table):
        states = set(customer_table.column("state").to_pylist())
        # All should be valid 2-letter US state codes
        for s in states:
            assert len(s) == 2
            assert s.isalpha()

    def test_email_uniqueness(self, customer_table):
        emails = customer_table.column("email").to_pylist()
        assert len(set(emails)) == len(emails)

    def test_email_format(self, customer_table):
        for email in customer_table.column("email").to_pylist():
            assert "@" in email
            local, domain = email.rsplit("@", 1)
            assert "." in domain

    def test_student_is_boolean(self, customer_table):
        vals = customer_table.column("is_student").to_pylist()
        assert all(isinstance(v, bool) for v in vals)

    def test_parquet_readable_by_pyarrow(self, customer_dir):
        """Verify Parquet files are readable."""
        files = sorted(Path(customer_dir).glob("*.parquet"))
        for f in files:
            t = pq.read_table(f)
            assert t.num_rows > 0


# ---------------------------------------------------------------------------
# Product builder tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def product_dir():
    """Build products into a temp directory."""
    from scrapers.scrape_products import main as build_products

    tmpdir = tempfile.mkdtemp(prefix="test_products_")
    # Use build mode with small counts for speed
    build_products(
        ["--mode", "build", "--output-dir", tmpdir,
         "--count", "10000", "--rx-count", "2000"],
        standalone_mode=False,
    )
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def product_table(product_dir):
    """Read back products.parquet."""
    path = Path(product_dir) / "products.parquet"
    assert path.exists(), f"products.parquet not found in {product_dir}"
    return pq.read_table(path)


class TestProductBuilder:
    """Test product catalog builder."""

    def test_product_count(self, product_table):
        # Build mode targets 10,000 OTC + 2,000 Rx = ~12,000
        assert product_table.num_rows >= 8_000

    def test_schema(self, product_table):
        required = {"product_id", "category", "brand", "name", "price"}
        assert required.issubset(set(product_table.column_names))

    def test_categories_present(self, product_table):
        categories = set(product_table.column("category").to_pylist())
        # Should have many distinct categories
        assert len(categories) >= 10

    def test_prices_reasonable(self, product_table):
        prices = product_table.column("price").to_pylist()
        for p in prices:
            assert 0.01 <= p <= 10_000.0, f"Unreasonable price: {p}"

    def test_product_id_uniqueness(self, product_table):
        ids = product_table.column("product_id").to_pylist()
        assert len(set(ids)) == len(ids)

    def test_brands_present(self, product_table):
        brands = set(product_table.column("brand").to_pylist())
        assert len(brands) >= 20

    def test_parquet_readable(self, product_dir):
        path = Path(product_dir) / "products.parquet"
        t = pq.read_table(path)
        assert t.num_rows > 0

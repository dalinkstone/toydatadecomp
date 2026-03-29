"""Tests for the VecDB Python wrapper over the C vector database.

Tests insert, query, save/load roundtrip, and benchmarks with 10K vectors.
"""

import os
import tempfile
import time

import numpy as np
import pytest

# Check if vecdb.dylib exists before importing
from pathlib import Path
VECDB_DYLIB = Path(__file__).resolve().parent.parent / "src" / "vecdb" / "vecdb.dylib"

pytestmark = pytest.mark.skipif(
    not VECDB_DYLIB.exists(),
    reason="vecdb.dylib not compiled — run: make compile-c"
)


@pytest.fixture
def vecdb():
    """Create a small VecDB instance."""
    from vecdb.vecdb_wrapper import VecDB
    return VecDB(capacity=1000, dims=32)


@pytest.fixture
def populated_vecdb():
    """Create a VecDB with 100 known vectors."""
    from vecdb.vecdb_wrapper import VecDB
    db = VecDB(capacity=200, dims=32)
    rng = np.random.default_rng(42)
    ids = np.arange(1, 101, dtype=np.uint32)
    vecs = rng.standard_normal((100, 32)).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    db.batch_insert(ids, vecs)
    return db, ids, vecs


class TestInsertQuery:
    """Test insert + query correctness with small known vectors."""

    def test_insert_and_count(self, vecdb):
        rng = np.random.default_rng(0)
        ids = np.array([1, 2, 3], dtype=np.uint32)
        vecs = rng.standard_normal((3, 32)).astype(np.float32)
        vecdb.batch_insert(ids, vecs)
        assert vecdb.count() == 3

    def test_query_returns_correct_nearest(self, populated_vecdb):
        db, ids, vecs = populated_vecdb
        # Query with vector 0 — should return itself as top result
        query = vecs[0:1].copy()
        result_ids, result_scores = db.batch_query_topk(query, k=5)
        # Top result should be id=1 (the first inserted vector)
        assert result_ids[0, 0] == 1
        # Score should be ~1.0 (self-similarity for normalized vectors)
        assert result_scores[0, 0] > 0.99

    def test_topk_sorted(self, populated_vecdb):
        """Top-K results should be sorted by descending score."""
        db, ids, vecs = populated_vecdb
        query = vecs[0:1].copy()
        result_ids, result_scores = db.batch_query_topk(query, k=10)
        scores = result_scores[0]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], \
                f"Not sorted: score[{i}]={scores[i]} < score[{i+1}]={scores[i+1]}"

    def test_batch_query(self, populated_vecdb):
        """Batch query should return correct shapes."""
        db, ids, vecs = populated_vecdb
        queries = vecs[:5].copy()
        result_ids, result_scores = db.batch_query_topk(queries, k=3)
        assert result_ids.shape == (5, 3)
        assert result_scores.shape == (5, 3)
        # Each query's top result should be itself
        for i in range(5):
            assert result_ids[i, 0] == ids[i]


class TestSaveLoad:
    """Test save/load roundtrip."""

    def test_save_load_roundtrip(self, populated_vecdb):
        db, ids, vecs = populated_vecdb
        with tempfile.NamedTemporaryFile(suffix=".vecdb", delete=False) as f:
            path = f.name
        try:
            db.save(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

            from vecdb.vecdb_wrapper import VecDB
            loaded = VecDB.load(path)
            assert loaded.count() == 100

            # Query should still return correct results
            query = vecs[0:1].copy()
            result_ids, result_scores = loaded.batch_query_topk(query, k=5)
            assert result_ids[0, 0] == 1
            assert result_scores[0, 0] > 0.99
        finally:
            os.unlink(path)

    def test_save_empty_db(self, vecdb):
        """Saving an empty DB should work."""
        with tempfile.NamedTemporaryFile(suffix=".vecdb", delete=False) as f:
            path = f.name
        try:
            vecdb.save(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)


class TestBenchmark:
    """Benchmark with 10K vectors."""

    def test_10k_insert_and_query(self):
        from vecdb.vecdb_wrapper import VecDB
        dims = 256
        n = 10_000
        k = 10

        db = VecDB(capacity=n + 100, dims=dims)
        rng = np.random.default_rng(123)

        # Generate and normalize 10K vectors
        vecs = rng.standard_normal((n, dims)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        ids = np.arange(1, n + 1, dtype=np.uint32)

        # Benchmark insert
        t0 = time.perf_counter()
        db.batch_insert(ids, vecs)
        insert_time = time.perf_counter() - t0
        assert db.count() == n
        print(f"\n  Insert 10K×{dims}d: {insert_time:.3f}s")

        # Benchmark query (100 queries)
        queries = vecs[:100].copy()
        t0 = time.perf_counter()
        result_ids, result_scores = db.batch_query_topk(queries, k=k)
        query_time = time.perf_counter() - t0
        print(f"  Query 100×top-{k}: {query_time:.3f}s")

        # Verify correctness: each query's top result is itself
        for i in range(100):
            assert result_ids[i, 0] == ids[i]
            assert result_scores[i, 0] > 0.99

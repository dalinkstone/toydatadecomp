"""Python ctypes wrapper for the vecdb C vector database.

Loads vecdb.dylib (compiled from vecdb.c with Apple Accelerate) and provides
a Pythonic API for batch insert and top-K cosine similarity queries.
"""

import ctypes
import numpy as np
from pathlib import Path


class VecDB:
    """Python interface to the C vector database."""

    def __init__(self, capacity: int, dims: int):
        self._lib = self._load_lib()
        self._setup_signatures()
        self._db = self._lib.vecdb_create(
            ctypes.c_uint32(capacity), ctypes.c_uint16(dims))
        if not self._db:
            raise RuntimeError("Failed to create VecDB")
        self.dims = dims

    @staticmethod
    def _load_lib():
        lib_path = Path(__file__).parent / "vecdb.dylib"
        if not lib_path.exists():
            raise FileNotFoundError(
                f"vecdb.dylib not found at {lib_path}. "
                "Run: clang -O3 -march=native -shared -o src/vecdb/vecdb.dylib "
                "src/vecdb/vecdb.c -framework Accelerate")
        return ctypes.CDLL(str(lib_path))

    def _setup_signatures(self):
        L = self._lib
        # vecdb_create
        L.vecdb_create.argtypes = [ctypes.c_uint32, ctypes.c_uint16]
        L.vecdb_create.restype = ctypes.c_void_p
        # vecdb_destroy
        L.vecdb_destroy.argtypes = [ctypes.c_void_p]
        L.vecdb_destroy.restype = None
        # vecdb_insert
        L.vecdb_insert.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        L.vecdb_insert.restype = ctypes.c_int
        # vecdb_batch_insert
        L.vecdb_batch_insert.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
        L.vecdb_batch_insert.restype = ctypes.c_int
        # vecdb_query_topk
        L.vecdb_query_topk.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_float)]
        L.vecdb_query_topk.restype = ctypes.c_int
        # vecdb_batch_query_topk
        L.vecdb_batch_query_topk.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_float)]
        L.vecdb_batch_query_topk.restype = ctypes.c_int
        # vecdb_save
        L.vecdb_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.vecdb_save.restype = ctypes.c_int
        # vecdb_load
        L.vecdb_load.argtypes = [ctypes.c_char_p]
        L.vecdb_load.restype = ctypes.c_void_p
        # vecdb_count
        L.vecdb_count.argtypes = [ctypes.c_void_p]
        L.vecdb_count.restype = ctypes.c_uint32
        # vecdb_dims
        L.vecdb_dims.argtypes = [ctypes.c_void_p]
        L.vecdb_dims.restype = ctypes.c_uint16

    def batch_insert(self, ids: np.ndarray, vecs: np.ndarray):
        """Insert batch of vectors. ids: (n,) uint32, vecs: (n, dims) float32."""
        ids = np.ascontiguousarray(ids, dtype=np.uint32)
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        assert vecs.shape[1] == self.dims
        n = len(ids)
        rc = self._lib.vecdb_batch_insert(
            self._db,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            vecs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint32(n))
        if rc != 0:
            raise RuntimeError(f"batch_insert failed: {rc}")

    def batch_query_topk(self, queries: np.ndarray, k: int):
        """Query top-k for each of n query vectors.

        Returns (ids: (n, k) uint32, scores: (n, k) float32).
        """
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        n = queries.shape[0]
        out_ids = np.zeros((n, k), dtype=np.uint32)
        out_scores = np.zeros((n, k), dtype=np.float32)
        rc = self._lib.vecdb_batch_query_topk(
            self._db,
            queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint32(n), ctypes.c_uint32(k),
            out_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if rc != 0:
            raise RuntimeError(f"batch_query_topk failed: {rc}")
        return out_ids, out_scores

    def count(self) -> int:
        return self._lib.vecdb_count(self._db)

    def save(self, path: str):
        rc = self._lib.vecdb_save(self._db, path.encode())
        if rc != 0:
            raise RuntimeError(f"save failed: {rc}")

    @classmethod
    def load(cls, path: str):
        instance = cls.__new__(cls)
        instance._lib = cls._load_lib()
        instance._setup_signatures(instance)
        instance._db = instance._lib.vecdb_load(path.encode())
        if not instance._db:
            raise RuntimeError(f"Failed to load VecDB from {path}")
        instance.dims = instance._lib.vecdb_dims(instance._db)
        return instance

    def __del__(self):
        if hasattr(self, '_db') and self._db:
            self._lib.vecdb_destroy(self._db)
            self._db = None

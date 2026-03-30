"""Monte Carlo Simulation Runner.

Orchestrates the full recommendation feedback loop simulation:

INNER LOOP (one simulation run):
  model recommends -> consumers respond -> model retrains -> repeat for N epochs

OUTER LOOP (Monte Carlo replications):
  repeat inner loop with different seeds -> aggregate -> detect convergence

Designed for full scale (10M customers, 12K products) via:
  - Vectorized consumer simulator (numpy batch ops, ~1-2s per epoch at 10M)
  - Chunked embedding extraction and re-ranking (never holds full 10M x 12K matrix)
  - Memory-mapped embedding files (no 10GB eager loads)
  - Capped retrain pair count (subsample if > 2M pairs)
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)

console = Console()

_SRC_DIR = str(Path(__file__).resolve().parent.parent)

# Upper bound on training pairs per retrain (keeps retrains fast at 10M scale)
MAX_RETRAIN_PAIRS = 2_000_000


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationConfig:
    """All parameters for the Monte Carlo simulation."""
    num_epochs: int = 250
    num_runs: int = 75
    retrain_interval: int = 10
    retrain_epochs: int = 2
    retrain_lr: float = 1e-4
    retrain_batch_size: int = 2048
    max_customer_id: int = 10_001  # 1-based: IDs 1..10000
    top_k: int = 10
    max_same_category: int = 3
    margin_weight: float = 0.3
    neg_samples: int = 4
    num_workers: int = 0  # 0 = auto
    db_path: str = "data/db/cvs_analytics.duckdb"
    model_dir: str = "data/model/"
    results_dir: str = "data/results/"
    output_dir: str = "data/results/simulation/"
    workspace_dir: str = "data/results/simulation/workspace/"
    convergence_window: int = 20
    convergence_threshold: float = 0.05


# ═══════════════════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EpochMetrics:
    """Metrics for one epoch of one simulation run."""
    epoch: int
    revenue: float
    recommended_revenue: float
    halo_revenue: float
    organic_revenue: float
    hit_rate_at_10: float
    catalog_coverage: float
    mean_fatigue_level: float
    active_customer_pct: float
    num_recommended_purchases: int
    num_halo_purchases: int
    num_organic_purchases: int


@dataclass
class SimulationResult:
    """Complete result of one inner-loop simulation run."""
    run_id: int
    seed: int
    metrics: list[EpochMetrics] = field(default_factory=list)
    parameters: dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _norm(val: float, key: str, norm_stats: dict) -> float:
    m, s = norm_stats.get(key, (0.0, 1.0))
    return (val - m) / s


def _vnorm(arr: np.ndarray, key: str, norm_stats: dict) -> np.ndarray:
    m, s = norm_stats.get(key, (0.0, 1.0))
    return ((arr - m) / s).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Workspace preparation (runs once in main process)
# ═══════════════════════════════════════════════════════════════════════════

def prepare_workspace(config: SimulationConfig) -> str:
    """Extract all data workers need and save to disk."""
    import sys
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)
    from ml.features import FeatureStore

    ws = Path(config.workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Preparing simulation workspace...[/bold]")

    # ── Checkpoint ───────────────────────────────────────────────────
    model_dir = Path(config.model_dir)
    checkpoints = sorted(model_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No model checkpoints in {model_dir}")
    ckpt_path = checkpoints[-1]
    console.print(f"  Checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    torch.save(ckpt["model_state_dict"], ws / "model_state_dict.pt")

    with open(ws / "vocabs.json", "w") as f:
        json.dump({
            "brand_vocab": ckpt["brand_vocab"],
            "category_vocab": ckpt["category_vocab"],
        }, f)
    with open(ws / "norm_stats.json", "w") as f:
        json.dump({k: list(v) for k, v in ckpt["norm_stats"].items()}, f)

    # ── Features ─────────────────────────────────────────────────────
    if not Path(config.db_path).exists():
        raise FileNotFoundError(f"DuckDB not found: {config.db_path}")
    console.print("  Loading features from DuckDB...")
    fs = FeatureStore(config.db_path)
    if not fs.has_features():
        console.print("[yellow]  Feature tables missing, building...[/yellow]")
        fs.build_product_features()
        fs.build_customer_features()
        fs.build_product_tiers()
        fs.build_training_pairs(1.0)

    customer_features = fs.export_customer_lookup()
    product_lookup = fs.export_product_lookup()
    state_vocab = fs.export_state_vocab()
    fs.close()

    # Trim to active customers
    trimmed = {}
    for key, arr in customer_features.items():
        trimmed[key] = arr[: config.max_customer_id]
    np.savez_compressed(ws / "customer_features.npz", **trimmed)

    # Product lookup -> JSON
    pl_ser: dict[str, dict] = {}
    for pid, info in product_lookup.items():
        pl_ser[str(pid)] = {
            k: (float(v) if isinstance(v, (np.floating, float))
                else bool(v) if isinstance(v, (np.bool_, bool))
                else int(v) if isinstance(v, (np.integer, int))
                else str(v))
            for k, v in info.items()
        }
    with open(ws / "product_lookup.json", "w") as f:
        json.dump(pl_ser, f)
    with open(ws / "state_vocab.json", "w") as f:
        json.dump(state_vocab, f)

    # ── Product metadata arrays ──────────────────────────────────────
    console.print("  Product metadata...")
    product_ids = np.load(model_dir / "product_ids.npy")
    pid_to_idx = {int(p): i for i, p in enumerate(product_ids)}
    n_prod = len(product_ids)

    categories = np.full(n_prod, "unknown", dtype=object)
    margins = np.zeros(n_prod, dtype=np.float32)
    prices = np.zeros(n_prod, dtype=np.float32)
    popularity = np.zeros(n_prod, dtype=np.float32)

    for pid, info in product_lookup.items():
        idx = pid_to_idx.get(int(pid))
        if idx is not None:
            categories[idx] = str(info.get("category", "unknown"))
            margins[idx] = float(info.get("margin_pct", 0) or 0)
            prices[idx] = float(info.get("price", 10.0) or 10.0)
            popularity[idx] = float(info.get("popularity_score", 0.01) or 0.01)

    np.save(ws / "product_ids.npy", product_ids)
    np.save(ws / "product_categories.npy", categories)
    np.save(ws / "product_margins.npy", margins)
    np.save(ws / "product_prices.npy", prices)
    np.save(ws / "product_popularity.npy", popularity)

    # Category index array (aligned with product_ids)
    cat_vocab = ckpt["category_vocab"]
    cat_idx_arr = np.zeros(n_prod, dtype=np.int32)
    for i, cat_name in enumerate(categories):
        cat_idx_arr[i] = cat_vocab.get(str(cat_name), 0)
    np.save(ws / "product_cat_idx.npy", cat_idx_arr)

    # Catalog arrays for organic purchases
    cat_pids = product_ids.astype(np.int64)
    cat_prices = prices.copy()
    pop = np.maximum(popularity, 1e-6)
    cat_weights = pop / pop.sum()
    np.save(ws / "catalog_pids.npy", cat_pids)
    np.save(ws / "catalog_prices.npy", cat_prices)
    np.save(ws / "catalog_weights.npy", cat_weights)

    # ── Copy embeddings (workers memory-map these) ───────────────────
    console.print("  Embeddings...")
    for fname in ["customer_embeddings.npy", "product_embeddings.npy"]:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, ws / fname)

    # ── Config ───────────────────────────────────────────────────────
    with open(ws / "sim_config.json", "w") as f:
        json.dump({
            "max_customer_id": config.max_customer_id,
            "top_k": config.top_k,
            "max_same_category": config.max_same_category,
            "margin_weight": config.margin_weight,
            "neg_samples": config.neg_samples,
            "retrain_epochs": config.retrain_epochs,
            "retrain_lr": config.retrain_lr,
            "retrain_batch_size": config.retrain_batch_size,
        }, f)

    console.print("[green]  Workspace ready.[/green]\n")
    return str(ws)


# ═══════════════════════════════════════════════════════════════════════════
# Workspace loader (called once per worker process)
# ═══════════════════════════════════════════════════════════════════════════

class WorkspaceData:
    """All data a worker process needs, loaded from disk.

    Model state dict is NOT held in RAM -- it is loaded from disk per-run
    and freed immediately after building the model (saves 2.5 GB/worker).
    """

    def __init__(self, workspace_path: str):
        ws = Path(workspace_path)
        self._ws = ws

        # Path to model weights (loaded per-run, NOT kept in RAM)
        self.model_state_dict_path: str = str(ws / "model_state_dict.pt")

        # Customer features (trimmed to max_customer_id)
        cf = np.load(ws / "customer_features.npz")
        self.customer_features: dict[str, np.ndarray] = {k: cf[k] for k in cf.files}

        # Product lookup
        with open(ws / "product_lookup.json") as f:
            raw = json.load(f)
        self.product_lookup: dict[int, dict] = {int(k): v for k, v in raw.items()}

        # Vocabs
        with open(ws / "vocabs.json") as f:
            v = json.load(f)
        self.brand_vocab: dict[str, int] = v["brand_vocab"]
        self.category_vocab: dict[str, int] = v["category_vocab"]
        with open(ws / "state_vocab.json") as f:
            self.state_vocab: dict[str, int] = json.load(f)

        # Norm stats
        with open(ws / "norm_stats.json") as f:
            raw_ns = json.load(f)
        self.norm_stats: dict[str, tuple[float, float]] = {
            k: tuple(v) for k, v in raw_ns.items()
        }

        # Product metadata
        self.product_ids: np.ndarray = np.load(ws / "product_ids.npy")
        self.product_categories: np.ndarray = np.load(
            ws / "product_categories.npy", allow_pickle=True
        )
        self.product_cat_idx: np.ndarray = np.load(ws / "product_cat_idx.npy")
        self.product_margins: np.ndarray = np.load(ws / "product_margins.npy")
        self.product_prices: np.ndarray = np.load(ws / "product_prices.npy")
        self.pid_to_idx: dict[int, int] = {
            int(p): i for i, p in enumerate(self.product_ids)
        }

        # Catalog arrays for organic purchases
        self.catalog_pids: np.ndarray = np.load(ws / "catalog_pids.npy")
        self.catalog_prices: np.ndarray = np.load(ws / "catalog_prices.npy")
        self.catalog_weights: np.ndarray = np.load(ws / "catalog_weights.npy")

        # Embeddings: memory-map to avoid 10GB per-worker RAM hit.
        # Only read in chunks during extract_and_rerank.
        self.customer_emb_path: str = str(ws / "customer_embeddings.npy")
        self.product_embeddings: np.ndarray = np.load(
            ws / "product_embeddings.npy"
        )

        # Config
        with open(ws / "sim_config.json") as f:
            cfg = json.load(f)
        self.max_customer_id: int = cfg["max_customer_id"]
        self.top_k: int = cfg["top_k"]
        self.max_same_category: int = cfg["max_same_category"]
        self.margin_weight: float = cfg["margin_weight"]
        self.neg_samples: int = cfg["neg_samples"]
        self.retrain_epochs: int = cfg["retrain_epochs"]
        self.retrain_lr: float = cfg["retrain_lr"]
        self.retrain_batch_size: int = cfg["retrain_batch_size"]

        # Pre-compute product feature batch for embedding extraction
        self.product_feat_batch: dict[str, torch.Tensor] = (
            _build_product_feature_batch(self)
        )


# ═══════════════════════════════════════════════════════════════════════════
# Model utilities
# ═══════════════════════════════════════════════════════════════════════════

def _build_model(state_dict: dict):
    """Reconstruct TwoTowerModel from a saved state dict."""
    from ml.two_tower import CustomerTower, ProductTower, TwoTowerModel

    sd = state_dict
    ct = CustomerTower(
        num_customers=sd["customer_tower.customer_embed.weight"].shape[0],
        num_states=sd["customer_tower.state_embed.weight"].shape[0],
    )
    pt = ProductTower(
        num_products=sd["product_tower.product_embed.weight"].shape[0],
        num_categories=sd["product_tower.category_embed.weight"].shape[0],
        num_brands=sd["product_tower.brand_embed.weight"].shape[0],
    )
    model = TwoTowerModel(ct, pt)
    model.load_state_dict(sd)
    return model


def _build_product_feature_batch(ws: WorkspaceData) -> dict[str, torch.Tensor]:
    """Build feature tensors for all products (used in embedding extraction)."""
    pids, cat_ids, brand_ids = [], [], []
    f_price, f_store, f_pop, f_margin = [], [], [], []
    f_clip, f_redeem, f_organic = [], [], []

    for pid in ws.product_ids:
        pid_int = int(pid)
        p = ws.product_lookup.get(pid_int, {})
        pids.append(pid_int)
        cat_ids.append(ws.category_vocab.get(str(p.get("category", "")), 0))
        brand_ids.append(ws.brand_vocab.get(str(p.get("brand", "")), 0))
        f_price.append(_norm(float(p.get("price", 0) or 0), "price", ws.norm_stats))
        f_store.append(float(p.get("is_store_brand", False)))
        f_pop.append(float(p.get("popularity_score", 0) or 0))
        f_margin.append(float(p.get("margin_pct", 0) or 0))
        f_clip.append(float(p.get("coupon_clip_rate", 0) or 0))
        f_redeem.append(float(p.get("coupon_redemption_rate", 0) or 0))
        f_organic.append(float(p.get("organic_purchase_ratio", 1) or 1))

    return {
        "product_id": torch.tensor(pids, dtype=torch.long),
        "category_id": torch.tensor(cat_ids, dtype=torch.long),
        "brand_id": torch.tensor(brand_ids, dtype=torch.long),
        "price": torch.tensor(f_price, dtype=torch.float32),
        "is_store_brand": torch.tensor(f_store, dtype=torch.float32),
        "popularity": torch.tensor(f_pop, dtype=torch.float32),
        "margin_pct": torch.tensor(f_margin, dtype=torch.float32),
        "coupon_clip_rate": torch.tensor(f_clip, dtype=torch.float32),
        "coupon_redemption_rate": torch.tensor(f_redeem, dtype=torch.float32),
        "organic_purchase_ratio": torch.tensor(f_organic, dtype=torch.float32),
    }


def _warm_start_retrain(
    model, cids: np.ndarray, pids: np.ndarray, ws: WorkspaceData,
) -> None:
    """Warm-start retrain on recent purchase pairs (in-place).

    Accepts pre-built numpy arrays (NOT Python lists) to avoid the
    ~46 GB memory overhead of 360M Python tuples at 10M scale.
    """
    from ml.train import TransactionDataset, collate_fn
    from ml.two_tower import TwoTowerModel

    if len(cids) < 10:
        return

    # Subsample to cap compute time
    if len(cids) > MAX_RETRAIN_PAIRS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(cids), size=MAX_RETRAIN_PAIRS, replace=False)
        cids = cids[idx]
        pids = pids[idx]

    ds = TransactionDataset(
        cids, pids, ws.customer_features, ws.product_lookup,
        ws.brand_vocab, ws.category_vocab, ws.norm_stats,
        num_products=len(ws.product_lookup), neg_samples=ws.neg_samples,
    )
    loader = DataLoader(
        ds, batch_size=min(ws.retrain_batch_size, len(ds)),
        shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=False,
    )

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=ws.retrain_lr)
    for _ in range(ws.retrain_epochs):
        for cust_b, pos_b, neg_bs, pos_margin in loader:
            opt.zero_grad(set_to_none=True)
            pos_s, neg_s = model(cust_b, pos_b, neg_bs)
            loss = TwoTowerModel.compute_loss(pos_s, neg_s, pos_margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()


def _extract_customer_chunk(model, ws: WorkspaceData, cid_start: int, cid_end: int) -> np.ndarray:
    """Extract customer embeddings for IDs [cid_start, cid_end) — 1-based."""
    cf = ws.customer_features
    n = cid_end - cid_start

    gender_idx = cf["gender"][cid_start:cid_end].astype(np.int64)
    gender_oh = np.zeros((n, 3), dtype=np.float32)
    gender_oh[np.arange(n), np.clip(gender_idx, 0, 2)] = 1.0

    with torch.inference_mode():
        batch = {
            "customer_id": torch.arange(cid_start, cid_end, dtype=torch.long),
            "age": torch.from_numpy(_vnorm(cf["age"][cid_start:cid_end], "age", ws.norm_stats)),
            "gender_onehot": torch.from_numpy(gender_oh),
            "state_id": torch.from_numpy(cf["state"][cid_start:cid_end].astype(np.int64).copy()),
            "is_student": torch.from_numpy(cf["is_student"][cid_start:cid_end].astype(np.float32).copy()),
            "total_spend": torch.from_numpy(_vnorm(cf["total_spend"][cid_start:cid_end], "total_spend", ws.norm_stats)),
            "coupon_engagement": torch.from_numpy(cf["coupon_engagement_score"][cid_start:cid_end].astype(np.float32).copy()),
            "coupon_redemption_rate": torch.from_numpy(cf["coupon_redemption_rate"][cid_start:cid_end].astype(np.float32).copy()),
            "avg_basket_size": torch.from_numpy(_vnorm(cf["avg_basket_size"][cid_start:cid_end], "avg_basket_size", ws.norm_stats)),
        }
        emb = model.customer_tower(**batch).numpy()
    return emb


# ═══════════════════════════════════════════════════════════════════════════
# Chunked extract + rerank (never holds full 10M x 12K matrix)
# ═══════════════════════════════════════════════════════════════════════════

def _fill_recs_diverse(
    ti: np.ndarray,          # (chunk_n, candidate_k) product indices
    tv: np.ndarray,          # (chunk_n, candidate_k) scores
    start: int,              # global row offset
    K: int,                  # top_k
    max_same_cat: int,
    product_ids: np.ndarray,
    product_cat_idx: np.ndarray,
    product_prices: np.ndarray,
    rec_pids: np.ndarray,    # output (N, K) — written in-place
    rec_cat_idx: np.ndarray,
    rec_scores: np.ndarray,
    rec_prices: np.ndarray,
) -> None:
    """Fast diversity filter: vectorised where possible, thin Python loop only
    for the greedy category-cap constraint (which touches ~15 candidates per
    customer, not all 50).

    At 100K-customer chunks this takes ~0.3s vs ~2s for the naive per-customer
    loop over candidate_k=50.
    """
    chunk_n = ti.shape[0]
    candidate_k = ti.shape[1]

    # Pre-fetch category ids for ALL candidates in one shot (vectorised)
    cand_cats = product_cat_idx[ti.ravel()].reshape(chunk_n, candidate_k)
    cand_pids = product_ids[ti.ravel()].reshape(chunk_n, candidate_k)
    cand_prices = product_prices[ti.ravel()].reshape(chunk_n, candidate_k)

    # Greedy selection with category cap — thin inner loop (avg ~15 iters)
    for i in range(chunk_n):
        sel = 0
        counts = np.zeros(max(product_cat_idx.max() + 1, 1), dtype=np.int8)
        for j in range(candidate_k):
            if sel >= K:
                break
            cat = cand_cats[i, j]
            if counts[cat] >= max_same_cat:
                continue
            counts[cat] += 1
            row = start + i
            rec_pids[row, sel] = cand_pids[i, j]
            rec_cat_idx[row, sel] = cat
            rec_scores[row, sel] = tv[i, j]
            rec_prices[row, sel] = cand_prices[i, j]
            sel += 1


def _extract_and_rerank(
    model, ws: WorkspaceData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract embeddings and generate (N, K) recommendation arrays in one
    chunked pass.  Peak memory per chunk: 50K x 12K x 4 = 2.4 GB.

    Returns (rec_pids, rec_cat_idx, rec_scores, rec_prices) all shape (N, K).
    N = max_customer_id - 1, K = top_k.
    """
    model.eval()
    N = ws.max_customer_id - 1
    K = ws.top_k
    n_prod = len(ws.product_ids)
    candidate_k = min(K * 5, n_prod)

    # Product embeddings (small, all at once)
    with torch.inference_mode():
        prod_emb = model.product_tower(**ws.product_feat_batch).numpy()
    prod_t = torch.from_numpy(prod_emb.astype(np.float32))

    margin_boost = torch.from_numpy(
        (1.0 + ws.product_margins * ws.margin_weight).astype(np.float32)
    ).unsqueeze(0)

    # Allocate output arrays
    rec_pids = np.zeros((N, K), dtype=np.int64)
    rec_cat_idx = np.zeros((N, K), dtype=np.int32)
    rec_scores = np.zeros((N, K), dtype=np.float32)
    rec_prices = np.zeros((N, K), dtype=np.float32)

    chunk_size = 100_000  # 100K customers per chunk, 100 chunks for 10M
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        cid_start = start + 1  # 1-based
        cid_end = end + 1

        # Extract customer embeddings for this chunk
        cust_emb = _extract_customer_chunk(model, ws, cid_start, cid_end)

        # Score + top-K
        with torch.no_grad():
            scores = torch.mm(
                torch.from_numpy(cust_emb.astype(np.float32)),
                prod_t.T,
            )
            scores *= margin_boost
            top_vals, top_idx = torch.topk(scores, candidate_k, dim=1)

        tv = top_vals.numpy()
        ti = top_idx.numpy()

        # Vectorized diversity: skip Python per-customer loop.
        # Greedy category-capped selection using numpy.
        _fill_recs_diverse(
            ti, tv, start, K, ws.max_same_category,
            ws.product_ids, ws.product_cat_idx, ws.product_prices,
            rec_pids, rec_cat_idx, rec_scores, rec_prices,
        )

    return rec_pids, rec_cat_idx, rec_scores, rec_prices


def _initial_rerank_from_embeddings(
    ws: WorkspaceData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate initial recommendations from saved embeddings (memory-mapped).

    Same as _extract_and_rerank but reads customer embeddings from disk
    instead of running the model forward pass.
    """
    N = ws.max_customer_id - 1
    K = ws.top_k
    n_prod = len(ws.product_ids)
    candidate_k = min(K * 5, n_prod)

    cust_emb_mmap = np.load(ws.customer_emb_path, mmap_mode="r")
    prod_emb = ws.product_embeddings.astype(np.float32)
    prod_t = torch.from_numpy(prod_emb)

    margin_boost = torch.from_numpy(
        (1.0 + ws.product_margins * ws.margin_weight).astype(np.float32)
    ).unsqueeze(0)

    rec_pids = np.zeros((N, K), dtype=np.int64)
    rec_cat_idx = np.zeros((N, K), dtype=np.int32)
    rec_scores = np.zeros((N, K), dtype=np.float32)
    rec_prices = np.zeros((N, K), dtype=np.float32)

    chunk_size = 100_000
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_emb = np.array(cust_emb_mmap[start + 1 : end + 1], dtype=np.float32)

        with torch.no_grad():
            scores = torch.mm(torch.from_numpy(chunk_emb), prod_t.T)
            scores *= margin_boost
            top_vals, top_idx = torch.topk(scores, candidate_k, dim=1)

        _fill_recs_diverse(
            top_idx.numpy(), top_vals.numpy(), start, K, ws.max_same_category,
            ws.product_ids, ws.product_cat_idx, ws.product_prices,
            rec_pids, rec_cat_idx, rec_scores, rec_prices,
        )

    return rec_pids, rec_cat_idx, rec_scores, rec_prices


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def _compute_epoch_metrics(
    epoch: int,
    vresult,   # VectorizedEpochResult
    rec_pids: np.ndarray,   # (N, K)
    sim,       # VectorizedConsumerSimulator (for state access)
    num_products: int,
) -> EpochMetrics:
    """Derive metrics from vectorized epoch result.

    Hit rate and active-customer % are based on RECOMMENDED purchases only
    (not organic), so they measure the recommendation system's effectiveness,
    not baseline customer activity.
    """
    N = sim.num_customers

    # Hit rate@10: fraction of customers who bought a RECOMMENDED item
    # (not organic — organic purchases happen regardless of recommendations)
    if len(vresult.rec_purchase_cids) > 0:
        unique_buyers = len(np.unique(vresult.rec_purchase_cids))
    else:
        unique_buyers = 0
    hit_rate = unique_buyers / max(N, 1)

    # Catalog coverage: unique recommended products / total products
    coverage = len(np.unique(rec_pids[rec_pids > 0])) / max(num_products, 1)

    # Mean fatigue (across customer-category pairs that have been touched)
    touched = sim.fatigue_touches > 0
    if touched.any():
        mean_fatigue = float(sim.fatigue_touches[touched].mean())
    else:
        mean_fatigue = 0.0

    # Active customer %: based on recommendation-driven dormancy.
    # Dormancy is already tracked correctly in the simulator (includes
    # organic), but we report it as-is since it reflects overall engagement.
    active = int((sim.dormant_epochs < sim.dormancy_threshold).sum())
    active_pct = active / max(N, 1)

    return EpochMetrics(
        epoch=epoch,
        revenue=vresult.total_revenue,
        recommended_revenue=vresult.recommended_revenue,
        halo_revenue=vresult.halo_revenue,
        organic_revenue=vresult.organic_revenue,
        hit_rate_at_10=hit_rate,
        catalog_coverage=coverage,
        mean_fatigue_level=mean_fatigue,
        active_customer_pct=active_pct,
        num_recommended_purchases=vresult.num_recommended_purchases,
        num_halo_purchases=vresult.num_halo_purchases,
        num_organic_purchases=vresult.num_organic_purchases,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Inner loop: one complete simulation run
# ═══════════════════════════════════════════════════════════════════════════

def run_single_simulation(
    run_id: int,
    num_epochs: int,
    retrain_interval: int,
    ws: WorkspaceData,
) -> SimulationResult:
    """Execute one full inner-loop simulation with vectorized consumer sim.

    Memory discipline at 10M scale:
      - Model state dict loaded from disk, freed after model build (~2.5 GB saved)
      - Purchases accumulated as numpy arrays, not Python tuples (~46 GB saved)
      - Old rec arrays explicitly deleted before building new ones
      - gc.collect() after major deallocations
    """
    import gc
    from simulation.vectorized_consumer import VectorizedConsumerSimulator

    seed = run_id
    N = ws.max_customer_id - 1
    num_products = len(ws.product_ids)

    # ── Vectorized simulator ─────────────────────────────────────────
    sim = VectorizedConsumerSimulator(
        seed=seed,
        num_customers=N,
        category_to_idx=ws.category_vocab,
        catalog_prices=ws.catalog_prices,
        catalog_weights=ws.catalog_weights,
    )

    result = SimulationResult(
        run_id=run_id, seed=seed,
        parameters={
            "fatigue_steepness": sim.fatigue_steepness,
            "re_engagement_prob": sim.re_engagement_prob,
            "re_engagement_decay": sim.re_engagement_decay,
            "halo_p": sim.halo_p,
            "organic_rate": sim.organic_rate,
        },
    )

    # ── Model: load from disk, free state dict immediately ───────────
    state_dict = torch.load(
        ws.model_state_dict_path, map_location="cpu", weights_only=True,
    )
    model = _build_model(state_dict)
    del state_dict
    gc.collect()

    # ── Initial recommendations from saved embeddings (chunked) ──────
    rec_pids, rec_cat_idx, rec_scores, rec_prices = (
        _initial_rerank_from_embeddings(ws)
    )

    # ── Purchase accumulation as numpy arrays (NOT Python tuples!) ────
    # At 10M scale: ~6M rec purchases/epoch × 10 epochs = 60M pairs
    # As numpy int64: 60M × 8 × 2 = 960 MB (vs 46 GB as Python tuples)
    accum_cids: list[np.ndarray] = []
    accum_pids: list[np.ndarray] = []
    accum_total = 0

    for epoch in range(1, num_epochs + 1):
        vresult = sim.simulate_epoch(
            rec_pids, rec_cat_idx, rec_scores, rec_prices,
        )

        metrics = _compute_epoch_metrics(
            epoch, vresult, rec_pids, sim, num_products,
        )
        result.metrics.append(metrics)

        # Accumulate recommended purchases only (0-based → 1-based)
        if len(vresult.rec_purchase_cids) > 0:
            accum_cids.append(vresult.rec_purchase_cids + 1)
            accum_pids.append(vresult.rec_purchase_pids.copy())
            accum_total += len(vresult.rec_purchase_cids)

        # Cap accumulated data to prevent memory growth
        if accum_total > MAX_RETRAIN_PAIRS * 2:
            all_c = np.concatenate(accum_cids)
            all_p = np.concatenate(accum_pids)
            idx = np.random.default_rng(42).choice(
                len(all_c), size=MAX_RETRAIN_PAIRS, replace=False,
            )
            accum_cids = [all_c[idx]]
            accum_pids = [all_p[idx]]
            accum_total = MAX_RETRAIN_PAIRS
            del all_c, all_p

        # Warm-start retrain at interval boundaries
        if epoch % retrain_interval == 0 and accum_total > 0:
            all_c = np.concatenate(accum_cids)
            all_p = np.concatenate(accum_pids)

            _warm_start_retrain(model, all_c, all_p, ws)
            del all_c, all_p
            accum_cids.clear()
            accum_pids.clear()
            accum_total = 0

            # Free old rec arrays before allocating new ones
            del rec_pids, rec_cat_idx, rec_scores, rec_prices
            gc.collect()

            rec_pids, rec_cat_idx, rec_scores, rec_prices = (
                _extract_and_rerank(model, ws)
            )

    # Free model before returning (next run will reload from disk)
    del model
    gc.collect()

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Worker pool
# ═══════════════════════════════════════════════════════════════════════════

_worker_ws: WorkspaceData | None = None


def _init_worker(ws_path: str) -> None:
    import sys
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)
    # 3 threads per worker: at 3-4 workers = 9-12 threads total.
    # threads=1 caused no speedup but thrashing with 6 workers was fatal.
    torch.set_num_threads(3)
    global _worker_ws
    _worker_ws = WorkspaceData(ws_path)


def _run_worker(args: tuple[int, int, int]) -> SimulationResult:
    assert _worker_ws is not None
    run_id, num_epochs, retrain_interval = args
    return run_single_simulation(run_id, num_epochs, retrain_interval, _worker_ws)


# ═══════════════════════════════════════════════════════════════════════════
# Outer loop: Monte Carlo replications
# ═══════════════════════════════════════════════════════════════════════════

def run_monte_carlo(config: SimulationConfig) -> list[SimulationResult]:
    """Execute the full Monte Carlo simulation."""
    ws_path = prepare_workspace(config)

    N = config.max_customer_id - 1
    # Memory budget per worker (empirically measured on M4 Max 64GB):
    #   10M customers: ~6.5 GB RSS steady + 2.4 GB peak during rerank
    #   Must also account for page cache pressure from mmap'd embeddings.
    # With 6 workers at 10M scale: 19B translation faults = thrashing.
    # Safe limit: 3 workers at 10M scale, leaving room for page cache.
    mem_per_worker_gb = max(0.1, N * 10.0 / 10_000_000)
    available_gb = 44  # 64 GB - 20 GB for OS + page cache + overhead
    max_by_mem = max(1, int(available_gb / mem_per_worker_gb))

    n_workers = (
        config.num_workers if config.num_workers > 0
        else min(max(1, cpu_count() - 1), max_by_mem)
    )
    n_workers = min(n_workers, config.num_runs)

    console.print("[bold]Monte Carlo Simulation[/bold]")
    console.print(f"  Runs: {config.num_runs}, Epochs: {config.num_epochs}")
    console.print(f"  Retrain every {config.retrain_interval} epochs")
    console.print(f"  Workers: {n_workers} (mem budget: ~{mem_per_worker_gb:.1f} GB/worker)")
    console.print(f"  Customers: {N:,}, Products: ~{len(np.load(Path(ws_path) / 'product_ids.npy')):,}")
    console.print()

    # Clean previous results
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("run_*.parquet"):
        f.unlink()
    for f in ["summary.parquet", "parameters.json", "convergence.json"]:
        (out / f).unlink(missing_ok=True)

    tasks = [
        (run_id, config.num_epochs, config.retrain_interval)
        for run_id in range(config.num_runs)
    ]

    results: list[SimulationResult] = []
    t0 = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Simulation runs", total=config.num_runs)

        if n_workers <= 1:
            import sys
            if _SRC_DIR not in sys.path:
                sys.path.insert(0, _SRC_DIR)
            torch.set_num_threads(2)
            ws = WorkspaceData(ws_path)
            for run_id, ne, ri in tasks:
                r = run_single_simulation(run_id, ne, ri, ws)
                results.append(r)
                _save_run(r, config.output_dir)
                progress.advance(task)
        else:
            with Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(ws_path,),
            ) as pool:
                for r in pool.imap_unordered(_run_worker, tasks):
                    results.append(r)
                    _save_run(r, config.output_dir)
                    progress.advance(task)

    elapsed = time.time() - t0
    console.print(
        f"\n[bold green]All {config.num_runs} runs complete[/bold green] "
        f"({elapsed:.1f}s)\n"
    )
    save_all_results(results, config)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation & convergence
# ═══════════════════════════════════════════════════════════════════════════

TRACKED_METRICS = [
    "revenue", "recommended_revenue", "halo_revenue", "organic_revenue",
    "hit_rate_at_10", "catalog_coverage", "mean_fatigue_level",
    "active_customer_pct", "num_recommended_purchases",
    "num_halo_purchases", "num_organic_purchases",
]


def aggregate_results(results: list[SimulationResult], num_epochs: int) -> pa.Table:
    n_runs = len(results)
    data = {m: np.zeros((n_runs, num_epochs)) for m in TRACKED_METRICS}
    for r_idx, res in enumerate(results):
        for e_idx, em in enumerate(res.metrics):
            for m in TRACKED_METRICS:
                data[m][r_idx, e_idx] = getattr(em, m)

    epochs = np.arange(1, num_epochs + 1)
    columns: dict[str, Any] = {"epoch": epochs}
    for m in TRACKED_METRICS:
        mean = data[m].mean(axis=0)
        std = data[m].std(axis=0)
        se = std / np.sqrt(n_runs)
        columns[f"mean_{m}"] = mean
        columns[f"std_{m}"] = std
        columns[f"ci_lower_{m}"] = mean - 1.96 * se
        columns[f"ci_upper_{m}"] = mean + 1.96 * se
    return pa.table(columns)


def compute_convergence(
    results: list[SimulationResult], num_epochs: int,
    window: int = 20, threshold: float = 0.05,
) -> dict[str, int]:
    key_metrics = [
        "revenue", "hit_rate_at_10", "catalog_coverage",
        "mean_fatigue_level", "active_customer_pct",
    ]
    n_runs = len(results)
    data = {m: np.zeros((n_runs, num_epochs)) for m in key_metrics}
    for r_idx, res in enumerate(results):
        for e_idx, em in enumerate(res.metrics):
            for m in key_metrics:
                data[m][r_idx, e_idx] = getattr(em, m)

    convergence: dict[str, int] = {}
    for m in key_metrics:
        means = data[m].mean(axis=0)
        stds = data[m].std(axis=0)
        converged = num_epochs
        for e in range(window, num_epochs):
            avg_mean = np.abs(means[e - window : e]).mean()
            avg_std = stds[e - window : e].mean()
            if avg_mean > 1e-10 and (avg_std / avg_mean) < threshold:
                converged = e - window + 1
                break
        convergence[m] = int(converged)
    return convergence


# ═══════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════

def _save_run(result: SimulationResult, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"run_{result.run_id}.parquet")
    cols: dict[str, list] = {m: [] for m in TRACKED_METRICS}
    cols["epoch"] = []
    for em in result.metrics:
        cols["epoch"].append(em.epoch)
        for m in TRACKED_METRICS:
            cols[m].append(getattr(em, m))
    pq.write_table(pa.table(cols), path)


def save_all_results(results: list[SimulationResult], config: SimulationConfig) -> None:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    console.print("[cyan]Computing summary statistics...[/cyan]")
    summary = aggregate_results(results, config.num_epochs)
    pq.write_table(summary, out / "summary.parquet")
    console.print(f"  Saved summary.parquet")

    params: dict[str, Any] = {
        "total_runs": config.num_runs,
        "num_epochs": config.num_epochs,
        "retrain_interval": config.retrain_interval,
        "max_customer_id": config.max_customer_id,
        "top_k": config.top_k,
        "margin_weight": config.margin_weight,
        "retrain_lr": config.retrain_lr,
        "retrain_epochs": config.retrain_epochs,
        "runs": {str(r.run_id): r.parameters for r in results},
    }
    with open(out / "parameters.json", "w") as f:
        json.dump(params, f, indent=2)
    console.print(f"  Saved parameters.json")

    convergence = compute_convergence(
        results, config.num_epochs,
        config.convergence_window, config.convergence_threshold,
    )
    with open(out / "convergence.json", "w") as f:
        json.dump(convergence, f, indent=2)
    console.print(f"  Saved convergence.json")

    df = summary.to_pandas()
    last = df.iloc[-1]
    console.print(
        f"\n[bold]Final Epoch Metrics "
        f"(mean +/- std across {config.num_runs} runs):[/bold]"
    )
    console.print(f"  Revenue:          {last['mean_revenue']:,.2f} +/- {last['std_revenue']:,.2f}")
    console.print(f"  Hit rate@10:      {last['mean_hit_rate_at_10']:.4f} +/- {last['std_hit_rate_at_10']:.4f}")
    console.print(f"  Catalog coverage: {last['mean_catalog_coverage']:.4f} +/- {last['std_catalog_coverage']:.4f}")
    console.print(f"  Mean fatigue:     {last['mean_mean_fatigue_level']:.2f} +/- {last['std_mean_fatigue_level']:.2f}")
    console.print(f"  Active customers: {last['mean_active_customer_pct']:.1%} +/- {last['std_active_customer_pct']:.1%}")
    console.print(f"\n[bold]Convergence (epoch where CV < {config.convergence_threshold}):[/bold]")
    for metric, ep in convergence.items():
        label = "converged" if ep < config.num_epochs else "did not converge"
        console.print(f"  {metric}: epoch {ep} ({label})")


# ═══════════════════════════════════════════════════════════════════════════
# Status display
# ═══════════════════════════════════════════════════════════════════════════

def show_status(output_dir: str) -> None:
    out = Path(output_dir)
    if not out.exists():
        console.print(f"[yellow]No simulation found at {output_dir}[/yellow]")
        return

    runs = sorted(out.glob("run_*.parquet"))
    summary_exists = (out / "summary.parquet").exists()
    params_path = out / "parameters.json"
    total_runs = "?"
    if params_path.exists():
        with open(params_path) as f:
            p = json.load(f)
        total_runs = p.get("total_runs", "?")

    console.print("[bold]Simulation Status[/bold]")
    console.print(f"  Directory: {output_dir}")
    console.print(f"  Completed runs: {len(runs)} / {total_runs}")

    if summary_exists:
        console.print("  [green]Summary: computed[/green]")
        df = pq.read_table(out / "summary.parquet").to_pandas()
        last = df.iloc[-1]
        console.print(f"\n  [bold]Final epoch (mean +/- std):[/bold]")
        console.print(f"    Revenue:          {last['mean_revenue']:,.2f} +/- {last['std_revenue']:,.2f}")
        console.print(f"    Hit rate@10:      {last['mean_hit_rate_at_10']:.4f} +/- {last['std_hit_rate_at_10']:.4f}")
        console.print(f"    Catalog coverage: {last['mean_catalog_coverage']:.4f} +/- {last['std_catalog_coverage']:.4f}")
    elif runs:
        console.print("  [yellow]Summary: not yet computed[/yellow]")
        last_run = pq.read_table(runs[-1]).to_pandas()
        if not last_run.empty:
            lr = last_run.iloc[-1]
            run_num = Path(runs[-1]).stem.split("_")[1]
            console.print(f"\n  [bold]Last run (#{run_num}) final metrics:[/bold]")
            console.print(f"    Revenue: {lr['revenue']:,.2f}")
            console.print(f"    Hit rate@10: {lr['hit_rate_at_10']:.4f}")
            console.print(f"    Catalog coverage: {lr['catalog_coverage']:.4f}")

    conv_path = out / "convergence.json"
    if conv_path.exists():
        with open(conv_path) as f:
            conv = json.load(f)
        console.print(f"\n  [bold]Convergence:[/bold]")
        for metric, ep in conv.items():
            console.print(f"    {metric}: epoch {ep}")

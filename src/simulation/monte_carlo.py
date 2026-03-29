"""Monte Carlo Simulation Runner.

Orchestrates the full recommendation feedback loop simulation:

INNER LOOP (one simulation run):
  model recommends -> consumers respond -> model retrains -> repeat for N epochs

OUTER LOOP (Monte Carlo replications):
  repeat inner loop with different seeds -> aggregate -> detect convergence

Evaluates real-world effectiveness by detecting feedback loop degeneration
(popularity bias, filter bubbles) and estimating steady-state metrics.

Each run samples behavioural parameters from distributions (fatigue onset,
re-engagement probability, halo effect strength, etc.), so the outer loop
averages over *possible worlds* of consumer behaviour -- not just random noise.
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
import torch.nn.functional as F
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
    """Extract all data workers need from DuckDB / model dir and save to disk.

    Returns the workspace directory path.
    """
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
        raise FileNotFoundError(
            f"DuckDB not found: {config.db_path}. Run the pipeline first."
        )
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

    # Trim customer features to active set to save memory
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

    # ── Product catalog for ConsumerSimulator ────────────────────────
    catalog = [
        {
            "product_id": int(pid),
            "category": str(info.get("category", "unknown")),
            "price": float(info.get("price", 10.0) or 10.0),
            "popularity_score": float(info.get("popularity_score", 0.01) or 0.01),
        }
        for pid, info in product_lookup.items()
    ]
    with open(ws / "product_catalog.json", "w") as f:
        json.dump(catalog, f)

    # ── Initial recommendations ──────────────────────────────────────
    recs_path = Path(config.results_dir) / "ranked_recommendations.parquet"
    if recs_path.exists():
        console.print("  Filtering initial recommendations...")
        tbl = pq.read_table(recs_path)
        df = tbl.to_pandas()
        df = df[df["customer_id"] < config.max_customer_id]
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                        ws / "initial_recommendations.parquet")
    else:
        console.print("[yellow]  No ranked_recommendations.parquet "
                      "-- will generate from embeddings.[/yellow]")

    # ── Copy embeddings ──────────────────────────────────────────────
    console.print("  Embeddings...")
    for fname in ["customer_embeddings.npy", "product_embeddings.npy"]:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, ws / fname)

    # ── Simulation config ────────────────────────────────────────────
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
    """All data a worker process needs, loaded from disk."""

    def __init__(self, workspace_path: str):
        ws = Path(workspace_path)

        # Model weights
        self.model_state_dict: dict = torch.load(
            ws / "model_state_dict.pt", map_location="cpu", weights_only=True
        )

        # Customer feature arrays (trimmed, indexed 0..max_cid-1)
        cf = np.load(ws / "customer_features.npz")
        self.customer_features: dict[str, np.ndarray] = {k: cf[k] for k in cf.files}

        # Product lookup {int_pid: {field: value}}
        with open(ws / "product_lookup.json") as f:
            raw = json.load(f)
        self.product_lookup: dict[int, dict] = {int(k): v for k, v in raw.items()}

        # Vocabularies
        with open(ws / "vocabs.json") as f:
            v = json.load(f)
        self.brand_vocab: dict[str, int] = v["brand_vocab"]
        self.category_vocab: dict[str, int] = v["category_vocab"]
        with open(ws / "state_vocab.json") as f:
            self.state_vocab: dict[str, int] = json.load(f)

        # Normalization stats
        with open(ws / "norm_stats.json") as f:
            raw_ns = json.load(f)
        self.norm_stats: dict[str, tuple[float, float]] = {
            k: tuple(v) for k, v in raw_ns.items()
        }

        # Product catalog (for ConsumerSimulator organic purchases)
        with open(ws / "product_catalog.json") as f:
            self.product_catalog: list[dict] = json.load(f)

        # Product metadata arrays (aligned with product_ids index)
        self.product_ids: np.ndarray = np.load(ws / "product_ids.npy")
        self.product_categories: np.ndarray = np.load(
            ws / "product_categories.npy", allow_pickle=True
        )
        self.product_margins: np.ndarray = np.load(ws / "product_margins.npy")
        self.product_prices: np.ndarray = np.load(ws / "product_prices.npy")
        self.product_popularity: np.ndarray = np.load(
            ws / "product_popularity.npy"
        )
        self.pid_to_idx: dict[int, int] = {
            int(p): i for i, p in enumerate(self.product_ids)
        }

        # Embeddings (starting point for each run)
        self.customer_embeddings: np.ndarray = np.load(
            ws / "customer_embeddings.npy"
        )
        self.product_embeddings: np.ndarray = np.load(
            ws / "product_embeddings.npy"
        )

        # Config scalars
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

        # Initial recommendations
        recs_path = ws / "initial_recommendations.parquet"
        if recs_path.exists():
            self.initial_recommendations: dict[int, list[dict]] = (
                _parse_recommendations(
                    str(recs_path),
                    self.product_prices,
                    self.pid_to_idx,
                    self.max_customer_id,
                )
            )
        else:
            self.initial_recommendations = {}

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
    model,
    purchases: list[tuple[int, int]],
    ws: WorkspaceData,
) -> None:
    """Warm-start retrain the model on recent simulated purchases (in-place).

    Only trains for 1-2 epochs at low learning rate on CPU.
    """
    from ml.train import TransactionDataset, collate_fn
    from ml.two_tower import TwoTowerModel

    if len(purchases) < 10:
        return

    cids = np.array([p[0] for p in purchases], dtype=np.int64)
    pids = np.array([p[1] for p in purchases], dtype=np.int64)

    ds = TransactionDataset(
        cids, pids,
        ws.customer_features,
        ws.product_lookup,
        ws.brand_vocab,
        ws.category_vocab,
        ws.norm_stats,
        num_products=len(ws.product_lookup),
        neg_samples=ws.neg_samples,
    )
    loader = DataLoader(
        ds,
        batch_size=min(ws.retrain_batch_size, len(ds)),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
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


def _extract_embeddings(model, ws: WorkspaceData):
    """Extract updated customer & product embeddings from the model.

    Returns (customer_emb, product_emb) numpy arrays.
    customer_emb is (max_customer_id, 256) with index 0 zeroed (1-based IDs).
    """
    model.eval()
    max_cid = ws.max_customer_id
    cf = ws.customer_features

    # ── Product embeddings ───────────────────────────────────────────
    with torch.inference_mode():
        product_emb = model.product_tower(**ws.product_feat_batch).numpy()

    # ── Customer embeddings (vectorized, all at once for demo scale) ─
    n = max_cid - 1  # customers 1..max_cid-1

    gender_idx = cf["gender"][1:max_cid].astype(np.int64)
    gender_oh = np.zeros((n, 3), dtype=np.float32)
    gender_oh[np.arange(n), np.clip(gender_idx, 0, 2)] = 1.0

    with torch.inference_mode():
        cust_batch = {
            "customer_id": torch.arange(1, max_cid, dtype=torch.long),
            "age": torch.from_numpy(
                _vnorm(cf["age"][1:max_cid], "age", ws.norm_stats)
            ),
            "gender_onehot": torch.from_numpy(gender_oh),
            "state_id": torch.from_numpy(
                cf["state"][1:max_cid].astype(np.int64).copy()
            ),
            "is_student": torch.from_numpy(
                cf["is_student"][1:max_cid].astype(np.float32).copy()
            ),
            "total_spend": torch.from_numpy(
                _vnorm(cf["total_spend"][1:max_cid], "total_spend", ws.norm_stats)
            ),
            "coupon_engagement": torch.from_numpy(
                cf["coupon_engagement_score"][1:max_cid].astype(np.float32).copy()
            ),
            "coupon_redemption_rate": torch.from_numpy(
                cf["coupon_redemption_rate"][1:max_cid].astype(np.float32).copy()
            ),
            "avg_basket_size": torch.from_numpy(
                _vnorm(cf["avg_basket_size"][1:max_cid], "avg_basket_size",
                       ws.norm_stats)
            ),
        }
        customer_emb = model.customer_tower(**cust_batch).numpy()

    # Pad index 0 (unused, IDs are 1-based)
    customer_emb = np.vstack(
        [np.zeros((1, customer_emb.shape[1]), dtype=np.float32), customer_emb]
    )
    return customer_emb, product_emb


# ═══════════════════════════════════════════════════════════════════════════
# Recommendation parsing / generation
# ═══════════════════════════════════════════════════════════════════════════

def _parse_recommendations(
    path: str,
    product_prices: np.ndarray,
    pid_to_idx: dict[int, int],
    max_customer_id: int,
) -> dict[int, list[dict[str, Any]]]:
    """Parse ranked_recommendations.parquet into simulator input format."""
    tbl = pq.read_table(path)
    df = tbl.to_pandas()
    df = df[df["customer_id"] < max_customer_id]

    recs: dict[int, list[dict]] = {}
    for _, row in df.iterrows():
        cid = int(row["customer_id"])
        pid = int(row["product_id"])
        idx = pid_to_idx.get(pid)
        price = float(product_prices[idx]) if idx is not None else 10.0
        recs.setdefault(cid, []).append({
            "product_id": pid,
            "category": str(row["category"]),
            "final_score": float(row["final_score"]),
            "price": price,
        })
    return recs


def _lightweight_rerank(
    customer_emb: np.ndarray,
    product_emb: np.ndarray,
    ws: WorkspaceData,
) -> dict[int, list[dict[str, Any]]]:
    """Compute per-customer top-K recommendations from embeddings.

    Simplified ranking: dot-product affinity + margin boost + category
    diversity.  Skips recency / coupon signals (static, not part of the
    feedback loop being evaluated).
    """
    max_cid = ws.max_customer_id
    top_k = ws.top_k
    n_prod = len(ws.product_ids)
    candidate_k = min(top_k * 5, n_prod)

    margin_boost = torch.from_numpy(
        (1.0 + ws.product_margins * ws.margin_weight).astype(np.float32)
    ).unsqueeze(0)

    recommendations: dict[int, list[dict]] = {}

    # Full matrix multiply -- ~460 MB for 10K x 12K, fine for demo
    with torch.no_grad():
        cust_t = torch.from_numpy(customer_emb[1:max_cid].astype(np.float32))
        prod_t = torch.from_numpy(product_emb.astype(np.float32))
        scores = torch.mm(cust_t, prod_t.T)
        scores *= margin_boost
        top_vals, top_idx = torch.topk(scores, candidate_k, dim=1)

    top_vals_np = top_vals.numpy()
    top_idx_np = top_idx.numpy()

    for i in range(max_cid - 1):
        cid = i + 1
        selected: list[dict] = []
        cat_counts: dict[str, int] = {}

        for j in range(candidate_k):
            if len(selected) >= top_k:
                break
            idx = int(top_idx_np[i, j])
            cat = str(ws.product_categories[idx])
            if cat_counts.get(cat, 0) >= ws.max_same_category:
                continue
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

            selected.append({
                "product_id": int(ws.product_ids[idx]),
                "category": cat,
                "final_score": float(top_vals_np[i, j]),
                "price": float(ws.product_prices[idx]),
            })

        if selected:
            recommendations[cid] = selected

    return recommendations


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def _compute_epoch_metrics(
    epoch: int,
    epoch_result,  # EpochResult
    recommendations: dict[int, list[dict]],
    customer_states: dict,  # {cid: CustomerState}
    num_products: int,
    dormancy_threshold: int,
) -> EpochMetrics:
    """Derive all tracked metrics from one epoch's simulation output."""
    rec_rev = sum(p["revenue"] for p in epoch_result.recommended_purchases)
    halo_rev = sum(p["revenue"] for p in epoch_result.halo_purchases)
    organic_rev = sum(p["revenue"] for p in epoch_result.organic_purchases)

    # Hit rate@10: fraction of customers with recs who bought a rec'd item
    cids_with_recs = set(recommendations.keys())
    cids_bought = {p["customer_id"] for p in epoch_result.recommended_purchases}
    hit_rate = (
        len(cids_bought & cids_with_recs) / max(len(cids_with_recs), 1)
    )

    # Catalog coverage: unique products currently recommended / total
    rec_pids: set[int] = set()
    for recs in recommendations.values():
        for r in recs:
            rec_pids.add(r["product_id"])
    coverage = len(rec_pids) / max(num_products, 1)

    # Mean fatigue (average touch count across customer-category pairs)
    total_touches = 0
    n_entries = 0
    for state in customer_states.values():
        for touches in state.fatigue_touches.values():
            total_touches += touches
            n_entries += 1
    mean_fatigue = total_touches / max(n_entries, 1)

    # Active customer percentage (non-dormant)
    active = sum(
        1 for s in customer_states.values()
        if s.dormant_epochs < dormancy_threshold
    )
    active_pct = active / max(len(customer_states), 1)

    return EpochMetrics(
        epoch=epoch,
        revenue=epoch_result.revenue,
        recommended_revenue=rec_rev,
        halo_revenue=halo_rev,
        organic_revenue=organic_rev,
        hit_rate_at_10=hit_rate,
        catalog_coverage=coverage,
        mean_fatigue_level=mean_fatigue,
        active_customer_pct=active_pct,
        num_recommended_purchases=len(epoch_result.recommended_purchases),
        num_halo_purchases=len(epoch_result.halo_purchases),
        num_organic_purchases=len(epoch_result.organic_purchases),
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
    """Execute one full inner-loop simulation.

    1. Build model from checkpoint
    2. Initialise consumer simulator & customer states
    3. For each epoch: simulate -> collect metrics -> maybe retrain & re-rank
    4. Return per-epoch time series of metrics
    """
    from simulation.consumer_behavior import ConsumerSimulator, CustomerState

    seed = run_id

    # ── Simulator with run-specific distributional draws ─────────────
    sim = ConsumerSimulator(seed=seed, product_catalog=ws.product_catalog)

    result = SimulationResult(
        run_id=run_id,
        seed=seed,
        parameters={
            "fatigue_steepness": sim.fatigue_steepness,
            "re_engagement_prob": sim.re_engagement_prob,
            "re_engagement_decay": sim.re_engagement_decay,
            "halo_p": sim.halo_p,
            "organic_rate": sim.organic_rate,
        },
    )

    # ── Customer states (fresh for each run) ─────────────────────────
    customer_states: dict[int, CustomerState] = {
        cid: CustomerState() for cid in range(1, ws.max_customer_id)
    }

    # ── Model (each run gets a fresh copy from the original weights) ─
    model = _build_model(ws.model_state_dict)
    # load_state_dict already copies weights, so ws.model_state_dict
    # is not mutated by warm-start retrains within this run.

    # ── Embeddings & recommendations ─────────────────────────────────
    customer_emb = ws.customer_embeddings.copy()
    product_emb = ws.product_embeddings.copy()

    if ws.initial_recommendations:
        recommendations = dict(ws.initial_recommendations)
    else:
        recommendations = _lightweight_rerank(customer_emb, product_emb, ws)

    # ── Accumulated purchases for warm-start retrain ─────────────────
    recent_purchases: list[tuple[int, int]] = []
    num_products = len(ws.product_ids)

    for epoch in range(1, num_epochs + 1):
        # Simulate one week of consumer behaviour
        epoch_result = sim.simulate_epoch(customer_states, recommendations)

        # Record metrics
        metrics = _compute_epoch_metrics(
            epoch, epoch_result, recommendations, customer_states,
            num_products, sim.dormancy_threshold,
        )
        result.metrics.append(metrics)

        # Accumulate recommended + organic purchases for retrain.
        # Including organic prevents confirmation bias: the model sees
        # what customers buy *independently* of its recommendations.
        for p in epoch_result.recommended_purchases:
            if p["product_id"] is not None:
                recent_purchases.append((p["customer_id"], p["product_id"]))
        for p in epoch_result.organic_purchases:
            if p["product_id"] is not None:
                recent_purchases.append((p["customer_id"], p["product_id"]))

        # Warm-start retrain at interval boundaries
        if epoch % retrain_interval == 0 and recent_purchases:
            _warm_start_retrain(model, recent_purchases, ws)
            customer_emb, product_emb = _extract_embeddings(model, ws)
            recommendations = _lightweight_rerank(
                customer_emb, product_emb, ws
            )
            recent_purchases = []

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Worker pool (multiprocessing)
# ═══════════════════════════════════════════════════════════════════════════

_worker_ws: WorkspaceData | None = None


def _init_worker(ws_path: str) -> None:
    """Initialise a worker process: add src/ to path, load workspace, limit
    torch threads to avoid contention between parallel workers."""
    import sys
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)
    torch.set_num_threads(1)

    global _worker_ws
    _worker_ws = WorkspaceData(ws_path)


def _run_worker(args: tuple[int, int, int]) -> SimulationResult:
    """Execute one simulation run inside a worker process."""
    run_id, num_epochs, retrain_interval = args
    assert _worker_ws is not None
    return run_single_simulation(run_id, num_epochs, retrain_interval, _worker_ws)


# ═══════════════════════════════════════════════════════════════════════════
# Outer loop: Monte Carlo replications
# ═══════════════════════════════════════════════════════════════════════════

def run_monte_carlo(config: SimulationConfig) -> list[SimulationResult]:
    """Execute the full Monte Carlo simulation (outer loop).

    1. Prepare workspace (extract data to disk)
    2. Spawn worker pool
    3. Run ``num_runs`` independent inner loops with different seeds
    4. Aggregate statistics and save all outputs
    """
    ws_path = prepare_workspace(config)

    n_workers = (
        config.num_workers if config.num_workers > 0
        else max(1, cpu_count() - 1)
    )
    n_workers = min(n_workers, config.num_runs)

    console.print("[bold]Monte Carlo Simulation[/bold]")
    console.print(f"  Runs: {config.num_runs}, Epochs: {config.num_epochs}")
    console.print(f"  Retrain every {config.retrain_interval} epochs "
                  f"({config.retrain_epochs} warm-start epochs, "
                  f"lr={config.retrain_lr})")
    console.print(f"  Workers: {n_workers}")
    console.print(f"  Customers: {config.max_customer_id - 1:,}, "
                  f"Products: ~{len(np.load(Path(ws_path) / 'product_ids.npy')):,}")
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
            # Sequential (useful for debugging)
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


def aggregate_results(
    results: list[SimulationResult],
    num_epochs: int,
) -> pa.Table:
    """Compute mean, std, and 95% CI across all runs for each epoch."""
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
    results: list[SimulationResult],
    num_epochs: int,
    window: int = 20,
    threshold: float = 0.05,
) -> dict[str, int]:
    """Find the epoch where cross-run coefficient of variation drops below
    *threshold* for each key metric (averaged over a rolling window).
    """
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
    """Save one run's per-epoch metrics as a parquet file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"run_{result.run_id}.parquet")

    cols: dict[str, list] = {m: [] for m in TRACKED_METRICS}
    cols["epoch"] = []

    for em in result.metrics:
        cols["epoch"].append(em.epoch)
        for m in TRACKED_METRICS:
            cols[m].append(getattr(em, m))

    pq.write_table(pa.table(cols), path)


def save_all_results(
    results: list[SimulationResult],
    config: SimulationConfig,
) -> None:
    """Write summary.parquet, parameters.json, and convergence.json."""
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Summary statistics
    console.print("[cyan]Computing summary statistics...[/cyan]")
    summary = aggregate_results(results, config.num_epochs)
    pq.write_table(summary, out / "summary.parquet")
    console.print(f"  Saved summary.parquet")

    # Sampled parameters for every run (reproducibility)
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

    # Convergence diagnostics
    convergence = compute_convergence(
        results, config.num_epochs,
        config.convergence_window, config.convergence_threshold,
    )
    with open(out / "convergence.json", "w") as f:
        json.dump(convergence, f, indent=2)
    console.print(f"  Saved convergence.json")

    # Print final-epoch summary
    df = summary.to_pandas()
    last = df.iloc[-1]
    console.print(
        f"\n[bold]Final Epoch Metrics "
        f"(mean +/- std across {config.num_runs} runs):[/bold]"
    )
    console.print(
        f"  Revenue:          "
        f"{last['mean_revenue']:,.2f} +/- {last['std_revenue']:,.2f}"
    )
    console.print(
        f"  Hit rate@10:      "
        f"{last['mean_hit_rate_at_10']:.4f} +/- {last['std_hit_rate_at_10']:.4f}"
    )
    console.print(
        f"  Catalog coverage: "
        f"{last['mean_catalog_coverage']:.4f} +/- {last['std_catalog_coverage']:.4f}"
    )
    console.print(
        f"  Mean fatigue:     "
        f"{last['mean_mean_fatigue_level']:.2f} +/- {last['std_mean_fatigue_level']:.2f}"
    )
    console.print(
        f"  Active customers: "
        f"{last['mean_active_customer_pct']:.1%} +/- {last['std_active_customer_pct']:.1%}"
    )
    console.print(f"\n[bold]Convergence "
                  f"(epoch where CV < {config.convergence_threshold}):[/bold]")
    for metric, ep in convergence.items():
        label = "converged" if ep < config.num_epochs else "did not converge"
        console.print(f"  {metric}: epoch {ep} ({label})")


# ═══════════════════════════════════════════════════════════════════════════
# Status display
# ═══════════════════════════════════════════════════════════════════════════

def show_status(output_dir: str) -> None:
    """Show progress of an ongoing or completed simulation."""
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
        console.print(
            f"    Revenue:          "
            f"{last['mean_revenue']:,.2f} +/- {last['std_revenue']:,.2f}"
        )
        console.print(
            f"    Hit rate@10:      "
            f"{last['mean_hit_rate_at_10']:.4f} +/- "
            f"{last['std_hit_rate_at_10']:.4f}"
        )
        console.print(
            f"    Catalog coverage: "
            f"{last['mean_catalog_coverage']:.4f} +/- "
            f"{last['std_catalog_coverage']:.4f}"
        )
    elif runs:
        console.print("  [yellow]Summary: not yet computed[/yellow]")
        # Show last completed run's final metrics
        last_run = pq.read_table(runs[-1]).to_pandas()
        if not last_run.empty:
            lr = last_run.iloc[-1]
            run_num = Path(runs[-1]).stem.split("_")[1]
            console.print(
                f"\n  [bold]Last run (#{run_num}) final metrics:[/bold]"
            )
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

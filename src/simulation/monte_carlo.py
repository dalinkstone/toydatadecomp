"""Monte Carlo Simulation Runner (Phase 5 — Revenue-Calibrated Tiered Simulation).

Orchestrates the full recommendation feedback loop with tiered strategies:

INNER LOOP (one simulation run):
  generate tiered offers -> consumers respond -> retrain model -> repeat

OUTER LOOP (Monte Carlo replications):
  repeat inner loop with different seeds -> aggregate -> detect convergence

Key Phase 5 features:
  - Tiered recommendation strategy (Tier 1-4)
  - Revenue calibrated to CVS 10-K front store numbers (~$57.1M/week)
  - Weekly epochs matching CVS promotional cadence
  - Three-level fatigue model (per-product, per-category, global)
  - 70/30 retrain blend (original + simulation data)
  - Breakout candidate tracking and tier migration
  - 10% exploration slots for long-tail discovery
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

from simulation.vectorized_consumer import (
    WEEKLY_REVENUE_TARGET,
    AVG_REVENUE_PER_VISIT,
    MAX_COUPON_OFFERS,
)

console = Console()

_SRC_DIR = str(Path(__file__).resolve().parent.parent)

# Upper bound on training pairs per retrain
MAX_RETRAIN_PAIRS = 2_000_000


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationConfig:
    """All parameters for the Monte Carlo simulation."""
    num_epochs: int = 40
    num_runs: int = 50
    retrain_interval: int = 10
    retrain_epochs: int = 2
    retrain_lr: float = 1e-4
    retrain_batch_size: int = 2048
    max_customer_id: int = 10_001  # 1-based: IDs 1..10000
    top_k: int = MAX_COUPON_OFFERS  # coupon offer slots
    max_same_category: int = 3
    margin_weight: float = 0.3
    neg_samples: int = 4
    num_workers: int = 0  # 0 = auto
    db_path: str = "data/db/cvs_analytics.duckdb"
    model_dir: str = "data/model/"
    results_dir: str = "data/results/"
    output_dir: str = "data/results/simulation/"
    workspace_dir: str = "data/results/simulation/workspace/"
    convergence_window: int = 10
    convergence_threshold: float = 0.05


# ═══════════════════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EpochMetrics:
    """Metrics for one epoch of one simulation run."""
    epoch: int
    total_revenue: float
    recommended_revenue: float
    organic_revenue: float
    discount_cost: float
    net_revenue: float
    hit_rate_at_5: float
    hit_rate_at_2: float
    catalog_coverage: float
    tier_migration_count: int
    active_customer_pct: float
    mean_coupons_per_customer: float
    breakout_success_count: int


@dataclass
class SimulationResult:
    """Complete result of one inner-loop simulation run."""
    run_id: int
    seed: int
    metrics: list[EpochMetrics] = field(default_factory=list)
    parameters: dict[str, float] = field(default_factory=dict)
    tier_transitions: list[dict] = field(default_factory=list)
    breakout_promotions: dict[int, int] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Breakout tracker
# ═══════════════════════════════════════════════════════════════════════════

class BreakoutTracker:
    """Track Tier 4 breakout candidates for promotion to Tier 2.

    A product is promoted after sustaining Tier 2-level purchase volume
    for 4 consecutive epochs.
    """

    def __init__(self, breakout_pids: np.ndarray, volume_threshold: int):
        self.pids = set(int(p) for p in breakout_pids)
        self.threshold = max(volume_threshold, 1)
        self.consecutive: dict[int, int] = {pid: 0 for pid in self.pids}
        self.promoted: dict[int, int] = {}  # pid -> epoch promoted

    def update_epoch(
        self, epoch: int, purchase_counts: dict[int, int]
    ) -> list[int]:
        """Update tracker.  Returns list of newly promoted product IDs."""
        newly_promoted: list[int] = []
        for pid in self.pids:
            if pid in self.promoted:
                continue
            if purchase_counts.get(pid, 0) >= self.threshold:
                self.consecutive[pid] = self.consecutive.get(pid, 0) + 1
            else:
                self.consecutive[pid] = 0
            if self.consecutive[pid] >= 4:
                self.promoted[pid] = epoch
                newly_promoted.append(pid)
        return newly_promoted


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
            "tier_vocab": ckpt.get("tier_vocab", {}),
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
    elasticity_lookup = fs.export_elasticity_lookup()
    state_vocab = fs.export_state_vocab()
    fs.close()

    # Elasticity lookup -> JSON
    el_ser = {str(k): v for k, v in elasticity_lookup.items()}
    with open(ws / "elasticity_lookup.json", "w") as f:
        json.dump(el_ser, f)

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

    # ── Product tier assignments ─────────────────────────────────────
    console.print("  Product tiers...")
    tier_path = model_dir / "product_tiers.parquet"
    product_tier_array = np.full(n_prod, 4, dtype=np.int8)
    if tier_path.exists():
        # Read only needed columns to avoid duplicate-name schema errors
        pf = pq.ParquetFile(tier_path)
        tier_df = pf.read().select(["product_id", "tier"]).to_pandas()
        for _, row in tier_df.iterrows():
            idx = pid_to_idx.get(int(row["product_id"]))
            if idx is not None:
                product_tier_array[idx] = int(row["tier"])
    else:
        console.print("[yellow]  No product_tiers.parquet, using revenue fallback[/yellow]")
        rev = prices * popularity
        rev_rank = np.argsort(-rev)
        for rank, idx in enumerate(rev_rank):
            if rank < 30:
                product_tier_array[idx] = 1
            elif rank < 2000:
                product_tier_array[idx] = 2
            elif rank < 6000:
                product_tier_array[idx] = 3
    np.save(ws / "product_tier_array.npy", product_tier_array)

    # Tier 1 data
    t1_mask = product_tier_array == 1
    np.save(ws / "tier1_pids.npy", product_ids[t1_mask].astype(np.int64))
    np.save(ws / "tier1_prices.npy", prices[t1_mask])

    # Tier 3 average price
    t3_mask = product_tier_array == 3
    t3_avg = float(prices[t3_mask].mean()) if t3_mask.any() else 8.0
    with open(ws / "tier3_avg_price.json", "w") as f:
        json.dump({"tier3_avg_price": t3_avg}, f)

    # ── Optimal discounts per product (from elasticity) ──────────────
    product_optimal_discounts = np.full(n_prod, 0.10, dtype=np.float32)
    for pid_str, edata in el_ser.items():
        idx = pid_to_idx.get(int(pid_str))
        if idx is not None:
            product_optimal_discounts[idx] = float(
                edata.get("optimal_discount", 0.10)
            )
    np.save(ws / "product_optimal_discounts.npy", product_optimal_discounts)

    # ── Breakout candidates ──────────────────────────────────────────
    console.print("  Breakout candidates...")
    breakout_path = model_dir / "breakout_candidates.parquet"
    if breakout_path.exists():
        bk_df = pq.read_table(breakout_path).to_pandas()
        bk_pids = bk_df["product_id"].to_numpy().astype(np.int64)
        bk_discs = bk_df["estimated_discount_to_break_in"].to_numpy().astype(
            np.float32
        )
    else:
        console.print("[yellow]  No breakout_candidates.parquet[/yellow]")
        t4_idx = np.where(product_tier_array == 4)[0]
        bk_pids = product_ids[t4_idx[:50]].astype(np.int64) if len(t4_idx) > 0 else np.array([], dtype=np.int64)
        bk_discs = np.full(len(bk_pids), 0.15, dtype=np.float32)
    np.save(ws / "breakout_pids.npy", bk_pids)
    np.save(ws / "breakout_discounts.npy", bk_discs)

    # Per-product breakout discount lookup (indexed by product array position)
    breakout_disc_by_pidx = np.full(n_prod, 0.15, dtype=np.float32)
    for i, pid in enumerate(bk_pids):
        idx = pid_to_idx.get(int(pid))
        if idx is not None:
            breakout_disc_by_pidx[idx] = bk_discs[i]
    np.save(ws / "breakout_disc_by_pidx.npy", breakout_disc_by_pidx)

    # ── Visit probabilities (revenue-calibrated) ─────────────────────
    console.print("  Visit probabilities...")
    N = config.max_customer_id - 1
    total_txn = trimmed.get("total_transactions",
                            np.ones(config.max_customer_id, dtype=np.float32))
    basket = trimmed.get("avg_basket_size",
                         np.full(config.max_customer_id, 3.0, dtype=np.float32))
    # Use 1-based portion (skip index 0)
    txn_1 = total_txn[1:config.max_customer_id].astype(np.float32)
    basket_1 = np.maximum(
        basket[1:config.max_customer_id].astype(np.float32), 1.0
    )
    weekly_visits = (txn_1 / basket_1) / 52.0
    raw_prob = np.clip(weekly_visits, 0.01, 0.95)

    # Estimate actual per-visit revenue from product prices and tier mix
    from simulation.vectorized_consumer import (
        TIER1_ITEMS_PER_VISIT, TIER3_ITEMS_PER_VISIT,
    )
    t1_avg = float(prices[t1_mask].mean()) if t1_mask.any() else 10.0
    t2_avg = float(prices[product_tier_array == 2].mean()) if (product_tier_array == 2).any() else 10.0
    t3_avg_safe = t3_avg if t3_mask.any() else 8.0
    t4_avg = float(prices[product_tier_array == 4].mean()) if (product_tier_array == 4).any() else 10.0
    # Estimate: Tier1 items + Tier3 items + ~1 coupon conversion + halo
    estimated_per_visit = (
        TIER1_ITEMS_PER_VISIT * t1_avg * 0.95   # Tier 1 with avg 5% discount
        + TIER3_ITEMS_PER_VISIT * t3_avg_safe    # Tier 3 organic
        + 1.2 * t2_avg * 0.90                    # ~1.2 Tier 2 conversions
        + 0.3 * min(t4_avg, t2_avg * 2) * 0.85   # ~0.3 Tier 4 conversions (cap price)
    )
    estimated_per_visit = max(estimated_per_visit, 10.0)
    console.print(f"  Estimated per-visit revenue: ${estimated_per_visit:,.2f}")

    # Calibrate to revenue target (scaled by customer count)
    base_fraction = N / 10_000_000
    scaled_target = WEEKLY_REVENUE_TARGET * base_fraction
    target_active = scaled_target / estimated_per_visit
    scale = target_active / max(float(raw_prob.sum()), 1.0)
    visit_probs = np.clip(raw_prob * scale, 0.005, 0.95).astype(np.float32)
    np.save(ws / "visit_probs.npy", visit_probs)
    console.print(
        f"  Expected active/epoch: {visit_probs.sum():,.0f} "
        f"({visit_probs.mean():.1%} mean prob)"
    )

    # ── Price sensitivity (0-based, N elements) ──────────────────────
    ps = trimmed.get(
        "price_sensitivity_bucket",
        np.full(config.max_customer_id, 2, dtype=np.int8),
    )
    np.save(ws / "price_sensitivity.npy",
            ps[1:config.max_customer_id].astype(np.int8))

    # ── Sample original training pairs for 70/30 retrain blend ───────
    console.print("  Sampling original training pairs...")
    try:
        import duckdb
        con = duckdb.connect(config.db_path, read_only=True)
        orig = con.execute("""
            SELECT customer_id::BIGINT AS customer_id,
                   product_id::BIGINT AS product_id
            FROM transactions
            USING SAMPLE 0.01 PERCENT (system, 42)
            LIMIT 1500000
        """).fetchnumpy()
        con.close()
        np.save(ws / "original_train_cids.npy", orig["customer_id"])
        np.save(ws / "original_train_pids.npy", orig["product_id"])
        console.print(f"  {len(orig['customer_id']):,} original training pairs")
    except Exception as e:
        console.print(f"[yellow]  Could not sample training pairs: {e}[/yellow]")
        np.save(ws / "original_train_cids.npy", np.array([], dtype=np.int64))
        np.save(ws / "original_train_pids.npy", np.array([], dtype=np.int64))

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
    and freed immediately after building the model.
    """

    def __init__(self, workspace_path: str):
        ws = Path(workspace_path)
        self._ws = ws

        self.model_state_dict_path: str = str(ws / "model_state_dict.pt")

        # Customer features
        cf = np.load(ws / "customer_features.npz")
        self.customer_features: dict[str, np.ndarray] = {
            k: cf[k] for k in cf.files
        }

        # Product lookup
        with open(ws / "product_lookup.json") as f:
            raw = json.load(f)
        self.product_lookup: dict[int, dict] = {
            int(k): v for k, v in raw.items()
        }

        # Vocabs
        with open(ws / "vocabs.json") as f:
            v = json.load(f)
        self.brand_vocab: dict[str, int] = v["brand_vocab"]
        self.category_vocab: dict[str, int] = v["category_vocab"]
        self.tier_vocab: dict[str, int] = v.get("tier_vocab", {})

        # Elasticity lookup
        el_path = ws / "elasticity_lookup.json"
        if el_path.exists():
            with open(el_path) as f:
                raw_el = json.load(f)
            self.elasticity_lookup: dict[int, dict] = {
                int(k): v for k, v in raw_el.items()
            }
        else:
            self.elasticity_lookup = {}
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

        # Catalog arrays
        self.catalog_pids: np.ndarray = np.load(ws / "catalog_pids.npy")
        self.catalog_prices: np.ndarray = np.load(ws / "catalog_prices.npy")
        self.catalog_weights: np.ndarray = np.load(ws / "catalog_weights.npy")

        # Embeddings
        self.customer_emb_path: str = str(ws / "customer_embeddings.npy")
        self.product_embeddings: np.ndarray = np.load(
            ws / "product_embeddings.npy"
        )

        # Tier data
        self.product_tier_array: np.ndarray = np.load(
            ws / "product_tier_array.npy"
        )
        self.tier1_pids: np.ndarray = np.load(ws / "tier1_pids.npy")
        self.tier1_prices: np.ndarray = np.load(ws / "tier1_prices.npy")
        with open(ws / "tier3_avg_price.json") as f:
            self.tier3_avg_price: float = json.load(f)["tier3_avg_price"]
        self.product_optimal_discounts: np.ndarray = np.load(
            ws / "product_optimal_discounts.npy"
        )

        # Breakout data
        self.breakout_pids: np.ndarray = np.load(ws / "breakout_pids.npy")
        self.breakout_discounts: np.ndarray = np.load(
            ws / "breakout_discounts.npy"
        )
        self.breakout_disc_by_pidx: np.ndarray = np.load(
            ws / "breakout_disc_by_pidx.npy"
        )

        # Visit probabilities & price sensitivity
        self.visit_probs: np.ndarray = np.load(ws / "visit_probs.npy")
        self.price_sensitivity: np.ndarray = np.load(
            ws / "price_sensitivity.npy"
        )

        # Original training pairs (for 70/30 blend)
        self.original_train_cids: np.ndarray = np.load(
            ws / "original_train_cids.npy"
        )
        self.original_train_pids: np.ndarray = np.load(
            ws / "original_train_pids.npy"
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

        # Pre-compute product feature batch
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
        num_tiers=sd["product_tower.tier_embed.weight"].shape[0],
    )
    model = TwoTowerModel(ct, pt)
    model.load_state_dict(sd)
    return model


def _build_product_feature_batch(ws: WorkspaceData) -> dict[str, torch.Tensor]:
    """Build feature tensors for all products."""
    from ml.features import TIER_VOCAB

    pids, cat_ids, brand_ids, tier_ids = [], [], [], []
    f_price, f_store, f_pop, f_margin = [], [], [], []
    f_clip, f_redeem, f_organic = [], [], []
    f_elast, f_optdisc, f_disc_offer = [], [], []

    el = getattr(ws, "elasticity_lookup", {})

    for pid in ws.product_ids:
        pid_int = int(pid)
        p = ws.product_lookup.get(pid_int, {})
        e = el.get(pid_int, {})
        pids.append(pid_int)
        cat_ids.append(ws.category_vocab.get(str(p.get("category", "")), 0))
        brand_ids.append(ws.brand_vocab.get(str(p.get("brand", "")), 0))
        tier_ids.append(TIER_VOCAB.get(str(p.get("tier", "")), 0))
        f_price.append(_norm(float(p.get("price", 0) or 0), "price", ws.norm_stats))
        f_store.append(float(p.get("is_store_brand", False)))
        f_pop.append(float(p.get("popularity_score", 0) or 0))
        f_margin.append(float(p.get("margin_pct", 0) or 0))
        f_clip.append(float(p.get("coupon_clip_rate", 0) or 0))
        f_redeem.append(float(p.get("coupon_redemption_rate", 0) or 0))
        f_organic.append(float(p.get("organic_purchase_ratio", 1) or 1))
        f_elast.append(float(e.get("elasticity_beta", 0)))
        f_optdisc.append(float(e.get("optimal_discount", 0)))
        f_disc_offer.append(0.0)

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
        "tier_id": torch.tensor(tier_ids, dtype=torch.long),
        "elasticity_beta": torch.tensor(f_elast, dtype=torch.float32),
        "optimal_discount": torch.tensor(f_optdisc, dtype=torch.float32),
        "discount_offer": torch.tensor(f_disc_offer, dtype=torch.float32),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Retraining
# ═══════════════════════════════════════════════════════════════════════════

def _warm_start_retrain(
    model, cids: np.ndarray, pids: np.ndarray, ws: WorkspaceData,
) -> None:
    """Warm-start retrain on purchase pairs (in-place)."""
    from ml.train import CouponResponseDataset, collate_fn
    from ml.two_tower import TwoTowerModel

    if len(cids) < 10:
        return

    if len(cids) > MAX_RETRAIN_PAIRS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(cids), size=MAX_RETRAIN_PAIRS, replace=False)
        cids = cids[idx]
        pids = pids[idx]

    n = len(cids)
    el = getattr(ws, "elasticity_lookup", {})
    tv = getattr(ws, "tier_vocab", {})

    ds = CouponResponseDataset(
        cids, pids,
        np.zeros(n, dtype=np.float32),
        np.ones(n, dtype=np.float32),
        np.ones(n, dtype=np.float32),
        ws.customer_features, ws.product_lookup, el, tv,
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
        for cust_b, pos_b, neg_bs, labels, weights, pos_margin in loader:
            opt.zero_grad(set_to_none=True)
            pos_s, neg_s = model(cust_b, pos_b, neg_bs)
            loss = TwoTowerModel.compute_loss(
                pos_s, neg_s, labels, weights, pos_margin
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()


def _blended_retrain(
    model, sim_cids: np.ndarray, sim_pids: np.ndarray, ws: WorkspaceData,
) -> None:
    """70% original training data + 30% simulation data retrain."""
    orig_c = ws.original_train_cids
    orig_p = ws.original_train_pids

    if len(orig_c) == 0:
        # No original pairs available — use sim-only
        _warm_start_retrain(model, sim_cids, sim_pids, ws)
        return

    # Filter original pairs to customers within the active range
    max_cid = ws.max_customer_id
    valid = (orig_c > 0) & (orig_c < max_cid)
    orig_c = orig_c[valid]
    orig_p = orig_p[valid]
    if len(orig_c) == 0:
        _warm_start_retrain(model, sim_cids, sim_pids, ws)
        return

    n_total = min(MAX_RETRAIN_PAIRS, len(sim_cids) + len(orig_c))
    n_orig = int(n_total * 0.7)
    n_sim = n_total - n_orig

    rng = np.random.default_rng(42)
    if len(orig_c) > n_orig:
        idx = rng.choice(len(orig_c), n_orig, replace=False)
        orig_c = orig_c[idx]
        orig_p = orig_p[idx]
    if len(sim_cids) > n_sim:
        idx = rng.choice(len(sim_cids), n_sim, replace=False)
        sim_cids = sim_cids[idx]
        sim_pids = sim_pids[idx]

    all_c = np.concatenate([orig_c, sim_cids])
    all_p = np.concatenate([orig_p, sim_pids])
    _warm_start_retrain(model, all_c, all_p, ws)


# ═══════════════════════════════════════════════════════════════════════════
# Customer embedding extraction
# ═══════════════════════════════════════════════════════════════════════════

def _extract_customer_chunk(
    model, ws: WorkspaceData, cid_start: int, cid_end: int,
) -> np.ndarray:
    """Extract customer embeddings for IDs [cid_start, cid_end) — 1-based."""
    cf = ws.customer_features
    n = cid_end - cid_start

    gender_idx = cf["gender"][cid_start:cid_end].astype(np.int64)
    gender_oh = np.zeros((n, 3), dtype=np.float32)
    gender_oh[np.arange(n), np.clip(gender_idx, 0, 2)] = 1.0

    with torch.inference_mode():
        batch = {
            "customer_id": torch.arange(cid_start, cid_end, dtype=torch.long),
            "age": torch.from_numpy(
                _vnorm(cf["age"][cid_start:cid_end], "age", ws.norm_stats)
            ),
            "gender_onehot": torch.from_numpy(gender_oh),
            "state_id": torch.from_numpy(
                cf["state"][cid_start:cid_end].astype(np.int64).copy()
            ),
            "is_student": torch.from_numpy(
                cf["is_student"][cid_start:cid_end].astype(np.float32).copy()
            ),
            "total_spend": torch.from_numpy(
                _vnorm(cf["total_spend"][cid_start:cid_end],
                       "total_spend", ws.norm_stats)
            ),
            "coupon_engagement": torch.from_numpy(
                cf["coupon_engagement_score"][cid_start:cid_end]
                .astype(np.float32).copy()
            ),
            "coupon_redemption_rate": torch.from_numpy(
                cf["coupon_redemption_rate"][cid_start:cid_end]
                .astype(np.float32).copy()
            ),
            "avg_basket_size": torch.from_numpy(
                _vnorm(cf["avg_basket_size"][cid_start:cid_end],
                       "avg_basket_size", ws.norm_stats)
            ),
            "price_sensitivity_bucket": torch.from_numpy(
                cf["price_sensitivity_bucket"][cid_start:cid_end]
                .astype(np.int64).copy()
            ),
        }
        emb = model.customer_tower(**batch).numpy()
    return emb


# ═══════════════════════════════════════════════════════════════════════════
# Tiered offer generation
# ═══════════════════════════════════════════════════════════════════════════

def _score_chunk_offers(
    cust_emb: np.ndarray,
    start: int,
    cn: int,
    tier2_idx: np.ndarray,
    tier2_emb_t: torch.Tensor | None,
    tier2_boost: torch.Tensor | None,
    breakout_idx: np.ndarray,
    breakout_emb_t: torch.Tensor | None,
    tier4_all_idx: np.ndarray,
    ws: WorkspaceData,
    exp_rng: np.random.Generator,
    # Output arrays (written in-place):
    offer_pids: np.ndarray,
    offer_scores: np.ndarray,
    offer_prices: np.ndarray,
    offer_discounts: np.ndarray,
    offer_tiers: np.ndarray,
    offer_cat_idx: np.ndarray,
) -> None:
    """Score one chunk of customers against tier-specific products and
    fill the corresponding rows of the output offer arrays."""
    cust_t = torch.from_numpy(cust_emb.astype(np.float32))

    # ── Tier 2: top-5 ──
    if tier2_emb_t is not None and len(tier2_idx) > 0:
        with torch.no_grad():
            s = torch.mm(cust_t, tier2_emb_t.T)
            if tier2_boost is not None:
                s *= tier2_boost
            n_t2 = min(5, s.shape[1])
            vals, idx = torch.topk(s, n_t2, dim=1)
        for j in range(n_t2):
            pidx = tier2_idx[idx[:, j].numpy()]
            offer_pids[start:start + cn, j] = ws.product_ids[pidx]
            offer_scores[start:start + cn, j] = vals[:, j].numpy()
            offer_prices[start:start + cn, j] = ws.product_prices[pidx]
            offer_discounts[start:start + cn, j] = (
                ws.product_optimal_discounts[pidx]
            )
            offer_tiers[start:start + cn, j] = 2
            offer_cat_idx[start:start + cn, j] = ws.product_cat_idx[pidx]

    # ── Tier 4 breakout: top-2 ──
    if breakout_emb_t is not None and len(breakout_idx) > 0:
        with torch.no_grad():
            s = torch.mm(cust_t, breakout_emb_t.T)
            n_t4 = min(2, s.shape[1])
            vals, idx = torch.topk(s, n_t4, dim=1)
        for j in range(n_t4):
            pidx = breakout_idx[idx[:, j].numpy()]
            slot = 5 + j
            offer_pids[start:start + cn, slot] = ws.product_ids[pidx]
            offer_scores[start:start + cn, slot] = vals[:, j].numpy()
            offer_prices[start:start + cn, slot] = ws.product_prices[pidx]
            offer_discounts[start:start + cn, slot] = (
                ws.breakout_disc_by_pidx[pidx]
            )
            offer_tiers[start:start + cn, slot] = 4
            offer_cat_idx[start:start + cn, slot] = ws.product_cat_idx[pidx]

    # ── Exploration slot (slot 7) ──
    if len(tier4_all_idx) > 0:
        exp_idx = tier4_all_idx[
            exp_rng.integers(0, len(tier4_all_idx), size=cn)
        ]
        offer_pids[start:start + cn, 7] = ws.product_ids[exp_idx]
        offer_scores[start:start + cn, 7] = 0.30
        offer_prices[start:start + cn, 7] = ws.product_prices[exp_idx]
        offer_discounts[start:start + cn, 7] = exp_rng.uniform(
            0.05, 0.50, size=cn
        ).astype(np.float32)
        offer_tiers[start:start + cn, 7] = 4
        offer_cat_idx[start:start + cn, 7] = ws.product_cat_idx[exp_idx]


def _alloc_offer_arrays(N: int, K: int):
    """Allocate the six (N, K) offer arrays."""
    return (
        np.zeros((N, K), dtype=np.int64),    # pids
        np.zeros((N, K), dtype=np.float32),  # scores
        np.zeros((N, K), dtype=np.float32),  # prices
        np.zeros((N, K), dtype=np.float32),  # discounts
        np.zeros((N, K), dtype=np.int8),     # tiers
        np.zeros((N, K), dtype=np.int32),    # cat_idx
    )


def _prep_tier_subsets(ws: WorkspaceData, tier_array: np.ndarray, prod_emb: np.ndarray):
    """Prepare tier-specific index arrays and embedding tensors."""
    tier2_idx = np.where(tier_array == 2)[0]
    tier4_all_idx = np.where(tier_array == 4)[0]

    # Breakout subset of Tier 4
    breakout_idx = np.array(
        [ws.pid_to_idx.get(int(p), -1) for p in ws.breakout_pids],
        dtype=np.int64,
    )
    breakout_idx = breakout_idx[breakout_idx >= 0]

    tier2_emb_t = (
        torch.from_numpy(prod_emb[tier2_idx])
        if len(tier2_idx) > 0 else None
    )
    breakout_emb_t = (
        torch.from_numpy(prod_emb[breakout_idx])
        if len(breakout_idx) > 0 else None
    )
    tier2_boost = None
    if tier2_emb_t is not None:
        tier2_boost = torch.from_numpy(
            (1.0 + ws.product_margins[tier2_idx] * ws.margin_weight)
            .astype(np.float32)
        ).unsqueeze(0)

    return tier2_idx, tier2_emb_t, tier2_boost, breakout_idx, breakout_emb_t, tier4_all_idx


def _initial_tiered_offers(ws: WorkspaceData):
    """Build initial tiered offers from saved embeddings."""
    N = ws.max_customer_id - 1
    K = MAX_COUPON_OFFERS
    prod_emb = ws.product_embeddings.astype(np.float32)

    (tier2_idx, tier2_emb_t, tier2_boost,
     breakout_idx, breakout_emb_t, tier4_all_idx) = _prep_tier_subsets(
        ws, ws.product_tier_array, prod_emb
    )

    arrays = _alloc_offer_arrays(N, K)
    cust_emb_mmap = np.load(ws.customer_emb_path, mmap_mode="r")
    exp_rng = np.random.default_rng(42)

    chunk = 100_000
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        cn = end - start
        chunk_emb = np.array(
            cust_emb_mmap[start + 1: end + 1], dtype=np.float32
        )
        _score_chunk_offers(
            chunk_emb, start, cn,
            tier2_idx, tier2_emb_t, tier2_boost,
            breakout_idx, breakout_emb_t, tier4_all_idx,
            ws, exp_rng, *arrays,
        )

    return arrays


def _extract_and_generate_tiered_offers(
    model, ws: WorkspaceData, tier_array: np.ndarray, retrain_count: int,
):
    """Build tiered offers from freshly extracted embeddings (post-retrain)."""
    model.eval()
    N = ws.max_customer_id - 1
    K = MAX_COUPON_OFFERS

    with torch.inference_mode():
        prod_emb = (
            model.product_tower(**ws.product_feat_batch).numpy()
            .astype(np.float32)
        )

    (tier2_idx, tier2_emb_t, tier2_boost,
     breakout_idx, breakout_emb_t, tier4_all_idx) = _prep_tier_subsets(
        ws, tier_array, prod_emb
    )

    arrays = _alloc_offer_arrays(N, K)
    exp_rng = np.random.default_rng(42 + retrain_count)

    chunk = 100_000
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        cn = end - start
        chunk_emb = _extract_customer_chunk(model, ws, start + 1, end + 1)
        _score_chunk_offers(
            chunk_emb, start, cn,
            tier2_idx, tier2_emb_t, tier2_boost,
            breakout_idx, breakout_emb_t, tier4_all_idx,
            ws, exp_rng, *arrays,
        )

    return arrays


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def _compute_epoch_metrics(
    epoch: int,
    vresult,  # TieredEpochResult
    sim,      # TieredConsumerSimulator
    num_products: int,
    tier_migration_count: int = 0,
    breakout_success_count: int = 0,
) -> EpochMetrics:
    N = sim.num_customers
    active_pct = vresult.active_customers / max(N, 1)
    coverage = len(vresult.unique_pids_purchased) / max(num_products, 1)
    mean_coupons = (
        vresult.total_coupons_offered / max(vresult.active_customers, 1)
    )
    hit5 = (
        vresult.tier2_conversions / max(vresult.tier2_offers, 1)
    )
    hit2 = (
        vresult.tier4_conversions / max(vresult.tier4_offers, 1)
    )

    return EpochMetrics(
        epoch=epoch,
        total_revenue=vresult.total_revenue,
        recommended_revenue=vresult.recommended_revenue,
        organic_revenue=vresult.organic_revenue,
        discount_cost=vresult.discount_cost,
        net_revenue=vresult.net_revenue,
        hit_rate_at_5=hit5,
        hit_rate_at_2=hit2,
        catalog_coverage=coverage,
        tier_migration_count=tier_migration_count,
        active_customer_pct=active_pct,
        mean_coupons_per_customer=mean_coupons,
        breakout_success_count=breakout_success_count,
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
    """Execute one full inner-loop tiered simulation."""
    import gc
    from simulation.vectorized_consumer import TieredConsumerSimulator

    seed = run_id
    N = ws.max_customer_id - 1
    num_products = len(ws.product_ids)

    # ── Tiered simulator ─────────────────────────────────────────────
    sim = TieredConsumerSimulator(
        seed=seed,
        num_customers=N,
        visit_probs=ws.visit_probs,
        price_sensitivity=ws.price_sensitivity,
        tier1_prices=ws.tier1_prices,
        tier1_pids=ws.tier1_pids,
        tier3_avg_price=ws.tier3_avg_price,
        category_to_idx=ws.category_vocab,
    )

    result = SimulationResult(
        run_id=run_id, seed=seed,
        parameters={
            "fatigue_steepness": sim.fatigue_steepness,
            "halo_p": sim.halo_p,
        },
    )

    # ── Model ────────────────────────────────────────────────────────
    state_dict = torch.load(
        ws.model_state_dict_path, map_location="cpu", weights_only=True,
    )
    model = _build_model(state_dict)
    del state_dict
    gc.collect()

    # ── Initial offers from saved embeddings ─────────────────────────
    tier_array = ws.product_tier_array.copy()
    offers = _initial_tiered_offers(ws)

    # ── Breakout tracker ─────────────────────────────────────────────
    # Threshold: rough estimate — refine after first epoch
    bk_threshold = max(1, N // 10000)
    breakout = BreakoutTracker(ws.breakout_pids, bk_threshold)
    tier_transitions: list[dict] = []
    cumulative_breakout_count = 0

    # ── Accumulation ─────────────────────────────────────────────────
    accum_cids: list[np.ndarray] = []
    accum_pids: list[np.ndarray] = []
    accum_total = 0
    retrain_count = 0

    for epoch in range(1, num_epochs + 1):
        vresult = sim.simulate_epoch(*offers)

        # Per-product purchase counts for breakout tracking
        epoch_counts: dict[int, int] = {}
        if len(vresult.rec_purchase_pids) > 0:
            u_pids, u_counts = np.unique(
                vresult.rec_purchase_pids, return_counts=True
            )
            epoch_counts = dict(
                zip(u_pids.astype(int), u_counts.astype(int))
            )

        # Calibrate breakout threshold after first epoch using Tier 2
        if epoch == 1 and epoch_counts:
            tier2_prod_set = set(
                int(ws.product_ids[i])
                for i in np.where(tier_array == 2)[0]
            )
            t2_volumes = [
                epoch_counts.get(pid, 0) for pid in tier2_prod_set
            ]
            if t2_volumes:
                bk_threshold = max(1, int(np.median(t2_volumes)))
                breakout.threshold = bk_threshold

        # Update breakout tracker
        newly_promoted = breakout.update_epoch(epoch, epoch_counts)
        cumulative_breakout_count += len(newly_promoted)

        # Record tier transitions for newly promoted products
        for pid in newly_promoted:
            pidx = ws.pid_to_idx.get(pid)
            if pidx is not None and tier_array[pidx] == 4:
                tier_array[pidx] = 2
                tier_transitions.append({
                    "product_id": pid,
                    "epoch": epoch,
                    "old_tier": 4,
                    "new_tier": 2,
                    "reason": "breakout_promotion",
                })

        # Metrics
        tier_mig_this = len(newly_promoted)
        metrics = _compute_epoch_metrics(
            epoch, vresult, sim, num_products,
            tier_migration_count=tier_mig_this,
            breakout_success_count=cumulative_breakout_count,
        )
        result.metrics.append(metrics)

        # Accumulate purchase pairs
        if len(vresult.rec_purchase_cids) > 0:
            accum_cids.append(vresult.rec_purchase_cids + 1)  # 0 -> 1 based
            accum_pids.append(vresult.rec_purchase_pids.copy())
            accum_total += len(vresult.rec_purchase_cids)

        # Cap accumulation
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

        # ── Retrain ──────────────────────────────────────────────────
        if epoch % retrain_interval == 0 and accum_total > 0:
            retrain_count += 1
            all_c = np.concatenate(accum_cids)
            all_p = np.concatenate(accum_pids)

            _blended_retrain(model, all_c, all_p, ws)
            del all_c, all_p
            accum_cids.clear()
            accum_pids.clear()
            accum_total = 0

            # Free old offers before allocating new ones
            del offers
            gc.collect()

            offers = _extract_and_generate_tiered_offers(
                model, ws, tier_array, retrain_count,
            )
            sim.reset_streaks()

    del model
    gc.collect()

    result.tier_transitions = tier_transitions
    result.breakout_promotions = breakout.promoted

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Worker pool
# ═══════════════════════════════════════════════════════════════════════════

_worker_ws: WorkspaceData | None = None


def _init_worker(ws_path: str) -> None:
    import sys
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)
    torch.set_num_threads(3)
    global _worker_ws
    _worker_ws = WorkspaceData(ws_path)


def _run_worker(args: tuple[int, int, int]) -> SimulationResult:
    assert _worker_ws is not None
    run_id, num_epochs, retrain_interval = args
    return run_single_simulation(
        run_id, num_epochs, retrain_interval, _worker_ws,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Outer loop: Monte Carlo replications
# ═══════════════════════════════════════════════════════════════════════════

def run_monte_carlo(config: SimulationConfig) -> list[SimulationResult]:
    """Execute the full Monte Carlo simulation."""
    ws_path = prepare_workspace(config)

    N = config.max_customer_id - 1
    mem_per_worker_gb = max(0.1, N * 10.0 / 10_000_000)
    available_gb = 44
    max_by_mem = max(1, int(available_gb / mem_per_worker_gb))

    n_workers = (
        config.num_workers if config.num_workers > 0
        else min(max(1, cpu_count() - 1), max_by_mem)
    )
    n_workers = min(n_workers, config.num_runs)

    console.print("[bold]Monte Carlo Simulation (Phase 5 — Tiered)[/bold]")
    console.print(f"  Runs: {config.num_runs}, Epochs: {config.num_epochs}")
    console.print(f"  Retrain every {config.retrain_interval} epochs")
    console.print(
        f"  Workers: {n_workers} "
        f"(mem budget: ~{mem_per_worker_gb:.1f} GB/worker)"
    )
    console.print(
        f"  Customers: {N:,}, Products: "
        f"~{len(np.load(Path(ws_path) / 'product_ids.npy')):,}"
    )
    console.print()

    # Clean previous results
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("run_*.parquet"):
        f.unlink()
    for fname in [
        "summary.parquet", "parameters.json", "convergence.json",
        "tier_transitions.parquet", "breakout_results.parquet",
    ]:
        (out / fname).unlink(missing_ok=True)

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
    "total_revenue", "recommended_revenue", "organic_revenue",
    "discount_cost", "net_revenue",
    "hit_rate_at_5", "hit_rate_at_2",
    "catalog_coverage", "tier_migration_count",
    "active_customer_pct", "mean_coupons_per_customer",
    "breakout_success_count",
]

CONVERGENCE_METRICS = ["net_revenue", "hit_rate_at_5", "catalog_coverage"]


def aggregate_results(
    results: list[SimulationResult], num_epochs: int,
) -> pa.Table:
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
    window: int = 10,
    threshold: float = 0.05,
) -> dict[str, int]:
    """Find first epoch where CV = std/mean across runs stays below threshold
    for an entire rolling window."""
    n_runs = len(results)
    data = {m: np.zeros((n_runs, num_epochs)) for m in CONVERGENCE_METRICS}
    for r_idx, res in enumerate(results):
        for e_idx, em in enumerate(res.metrics):
            for m in CONVERGENCE_METRICS:
                data[m][r_idx, e_idx] = getattr(em, m)

    convergence: dict[str, int] = {}
    for m in CONVERGENCE_METRICS:
        means = data[m].mean(axis=0)
        stds = data[m].std(axis=0)
        converged = num_epochs
        for e in range(window, num_epochs):
            window_means = np.abs(means[e - window: e])
            window_stds = stds[e - window: e]
            with np.errstate(divide="ignore", invalid="ignore"):
                cv = np.where(
                    window_means > 1e-10,
                    window_stds / window_means,
                    0.0,
                )
            if cv.max() < threshold:
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


def save_all_results(
    results: list[SimulationResult], config: SimulationConfig,
) -> None:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Summary ──────────────────────────────────────────────────────
    console.print("[cyan]Computing summary statistics...[/cyan]")
    summary = aggregate_results(results, config.num_epochs)
    pq.write_table(summary, out / "summary.parquet")
    console.print("  Saved summary.parquet")

    # ── Parameters ───────────────────────────────────────────────────
    params: dict[str, Any] = {
        "total_runs": config.num_runs,
        "num_epochs": config.num_epochs,
        "retrain_interval": config.retrain_interval,
        "max_customer_id": config.max_customer_id,
        "top_k": config.top_k,
        "margin_weight": config.margin_weight,
        "retrain_lr": config.retrain_lr,
        "retrain_epochs": config.retrain_epochs,
        "weekly_revenue_target": WEEKLY_REVENUE_TARGET,
        "avg_revenue_per_visit": AVG_REVENUE_PER_VISIT,
        "runs": {str(r.run_id): r.parameters for r in results},
    }
    with open(out / "parameters.json", "w") as f:
        json.dump(params, f, indent=2)
    console.print("  Saved parameters.json")

    # ── Convergence ──────────────────────────────────────────────────
    convergence = compute_convergence(
        results, config.num_epochs,
        config.convergence_window, config.convergence_threshold,
    )
    with open(out / "convergence.json", "w") as f:
        json.dump(convergence, f, indent=2)
    console.print("  Saved convergence.json")

    # ── Tier transitions ─────────────────────────────────────────────
    all_trans: list[dict] = []
    for r in results:
        for t in r.tier_transitions:
            all_trans.append({"run_id": r.run_id, **t})
    if all_trans:
        tt = pa.table({
            "run_id": [t["run_id"] for t in all_trans],
            "product_id": [t["product_id"] for t in all_trans],
            "epoch": [t["epoch"] for t in all_trans],
            "old_tier": [t["old_tier"] for t in all_trans],
            "new_tier": [t["new_tier"] for t in all_trans],
            "reason": [t["reason"] for t in all_trans],
        })
        pq.write_table(tt, out / "tier_transitions.parquet")
        console.print(
            f"  Saved tier_transitions.parquet ({len(all_trans)} transitions)"
        )
    else:
        # Write empty file
        pq.write_table(
            pa.table({
                "run_id": pa.array([], type=pa.int64()),
                "product_id": pa.array([], type=pa.int64()),
                "epoch": pa.array([], type=pa.int64()),
                "old_tier": pa.array([], type=pa.int64()),
                "new_tier": pa.array([], type=pa.int64()),
                "reason": pa.array([], type=pa.string()),
            }),
            out / "tier_transitions.parquet",
        )
        console.print("  Saved tier_transitions.parquet (0 transitions)")

    # ── Breakout results ─────────────────────────────────────────────
    all_bk: list[dict] = []
    for r in results:
        for pid, ep in r.breakout_promotions.items():
            all_bk.append({
                "run_id": r.run_id,
                "product_id": pid,
                "promoted_epoch": ep,
            })
    if all_bk:
        br = pa.table({
            "run_id": [b["run_id"] for b in all_bk],
            "product_id": [b["product_id"] for b in all_bk],
            "promoted_epoch": [b["promoted_epoch"] for b in all_bk],
        })
        pq.write_table(br, out / "breakout_results.parquet")
        console.print(
            f"  Saved breakout_results.parquet ({len(all_bk)} promotions)"
        )
    else:
        pq.write_table(
            pa.table({
                "run_id": pa.array([], type=pa.int64()),
                "product_id": pa.array([], type=pa.int64()),
                "promoted_epoch": pa.array([], type=pa.int64()),
            }),
            out / "breakout_results.parquet",
        )
        console.print("  Saved breakout_results.parquet (0 promotions)")

    # ── Display final metrics ────────────────────────────────────────
    df = summary.to_pandas()
    last = df.iloc[-1]
    console.print(
        f"\n[bold]Final Epoch Metrics "
        f"(mean +/- std across {config.num_runs} runs):[/bold]"
    )
    console.print(
        f"  Net Revenue:      ${last['mean_net_revenue']:,.2f} "
        f"+/- ${last['std_net_revenue']:,.2f}"
    )
    console.print(
        f"  Total Revenue:    ${last['mean_total_revenue']:,.2f} "
        f"+/- ${last['std_total_revenue']:,.2f}"
    )
    console.print(
        f"  Discount Cost:    ${last['mean_discount_cost']:,.2f} "
        f"+/- ${last['std_discount_cost']:,.2f}"
    )
    console.print(
        f"  Hit rate@5 (T2):  {last['mean_hit_rate_at_5']:.4f} "
        f"+/- {last['std_hit_rate_at_5']:.4f}"
    )
    console.print(
        f"  Hit rate@2 (T4):  {last['mean_hit_rate_at_2']:.4f} "
        f"+/- {last['std_hit_rate_at_2']:.4f}"
    )
    console.print(
        f"  Catalog coverage: {last['mean_catalog_coverage']:.4f} "
        f"+/- {last['std_catalog_coverage']:.4f}"
    )
    console.print(
        f"  Active customers: {last['mean_active_customer_pct']:.1%} "
        f"+/- {last['std_active_customer_pct']:.1%}"
    )
    console.print(
        f"  Coupons/customer: {last['mean_mean_coupons_per_customer']:.2f} "
        f"+/- {last['std_mean_coupons_per_customer']:.2f}"
    )
    console.print(
        f"\n[bold]Convergence "
        f"(epoch where CV < {config.convergence_threshold}):[/bold]"
    )
    for metric, ep in convergence.items():
        label = (
            "converged" if ep < config.num_epochs else "did not converge"
        )
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
        console.print(
            f"    Net Revenue:    ${last['mean_net_revenue']:,.2f} "
            f"+/- ${last['std_net_revenue']:,.2f}"
        )
        console.print(
            f"    Hit rate@5:     {last['mean_hit_rate_at_5']:.4f} "
            f"+/- {last['std_hit_rate_at_5']:.4f}"
        )
        console.print(
            f"    Catalog cover:  {last['mean_catalog_coverage']:.4f} "
            f"+/- {last['std_catalog_coverage']:.4f}"
        )
    elif runs:
        console.print("  [yellow]Summary: not yet computed[/yellow]")
        last_run = pq.read_table(runs[-1]).to_pandas()
        if not last_run.empty:
            lr = last_run.iloc[-1]
            run_num = Path(runs[-1]).stem.split("_")[1]
            console.print(f"\n  [bold]Last run (#{run_num}) final:[/bold]")
            console.print(f"    Net Revenue: ${lr['net_revenue']:,.2f}")
            console.print(f"    Hit rate@5:  {lr['hit_rate_at_5']:.4f}")
            console.print(f"    Catalog:     {lr['catalog_coverage']:.4f}")

    conv_path = out / "convergence.json"
    if conv_path.exists():
        with open(conv_path) as f:
            conv = json.load(f)
        console.print(f"\n  [bold]Convergence:[/bold]")
        for metric, ep in conv.items():
            console.print(f"    {metric}: epoch {ep}")

    # Breakout summary
    bk_path = out / "breakout_results.parquet"
    if bk_path.exists():
        bk_df = pq.read_table(bk_path).to_pandas()
        n_bk = len(bk_df)
        if n_bk > 0:
            n_unique = bk_df["product_id"].nunique()
            console.print(
                f"\n  [bold]Breakouts:[/bold] {n_bk} promotions "
                f"({n_unique} unique products across runs)"
            )

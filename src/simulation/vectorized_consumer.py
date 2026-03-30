"""Vectorized Consumer Behavior Simulator.

Numpy-vectorized version of ConsumerSimulator for large-scale (10M+ customer)
Monte Carlo simulation.  All behavioral dynamics are identical to the original
per-customer simulator in ``consumer_behavior.py``, but every operation is
batched over numpy arrays so that a 10M-customer epoch completes in ~1-2 seconds.

The five behavioral dynamics are preserved exactly:

1. **Recommendation Fatigue** (inverse sigmoid, per customer-category)
2. **Re-engagement After Dormancy** (burst + diminished onset reset)
3. **Cross-Category Halo Effect** (adjacency-based related-category purchase)
4. **Seasonal Demand Modifier** (month-dependent category multipliers)
5. **Concurrent Organic Purchases** (Poisson-distributed, popularity-weighted)

All distributional parameters are sampled once per seed, identical to
``ConsumerSimulator``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.consumer_behavior import (
    CATEGORY_ADJACENCY,
    seasonal_multiplier,
)


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VectorizedEpochResult:
    """Compact result of one vectorized epoch over all customers."""

    total_revenue: float
    recommended_revenue: float
    halo_revenue: float
    organic_revenue: float

    # Flat arrays of (customer_index, product_id) for retrain data.
    # customer_index is 0-based; caller adds 1 for 1-based customer_id.
    rec_purchase_cids: np.ndarray   # int64
    rec_purchase_pids: np.ndarray   # int64
    organic_purchase_cids: np.ndarray
    organic_purchase_pids: np.ndarray

    num_customers_who_purchased: int
    num_recommended_purchases: int
    num_halo_purchases: int
    num_organic_purchases: int


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized simulator
# ═══════════════════════════════════════════════════════════════════════════

class VectorizedConsumerSimulator:
    """Numpy-vectorized consumer behavior simulator.

    Parameters
    ----------
    seed : int
        Pins all distributional draws for one Monte Carlo run.
    num_customers : int
        Number of customers (array dimension).
    category_to_idx : dict[str, int]
        Maps category name -> integer index (1-based from model vocab,
        0 = unknown).
    catalog_pids, catalog_prices, catalog_weights : np.ndarray
        Product catalog arrays for organic purchases (aligned by index).
    dormancy_threshold : int
        Consecutive purchase-free epochs before a customer is dormant.
    """

    def __init__(
        self,
        seed: int,
        num_customers: int,
        category_to_idx: dict[str, int],
        catalog_pids: np.ndarray,
        catalog_prices: np.ndarray,
        catalog_weights: np.ndarray,
        dormancy_threshold: int = 4,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.num_customers = num_customers
        self.dormancy_threshold = dormancy_threshold

        num_categories = max(category_to_idx.values()) + 1 if category_to_idx else 1
        self.num_categories = num_categories

        # ── Distributional parameters (same draws as ConsumerSimulator) ──
        self.fatigue_steepness: float = float(self.rng.uniform(0.4, 1.2))
        self.re_engagement_prob: float = float(self.rng.beta(3, 17))
        self.re_engagement_decay: float = float(self.rng.uniform(0.5, 0.8))
        self.halo_p: float = float(self.rng.beta(2, 18))
        self.organic_rate: float = float(self.rng.gamma(shape=2.0, scale=1.5))

        # ── Mutable state arrays (persist across epochs) ─────────────
        N, C = num_customers, num_categories
        self.fatigue_touches = np.zeros((N, C), dtype=np.int32)
        self.fatigue_onset = np.full((N, C), -1.0, dtype=np.float32)
        self.dormant_epochs = np.zeros(N, dtype=np.int32)
        self.re_engagement_count = np.zeros(N, dtype=np.int32)

        # ── Precomputed seasonal table (C, 12) ──────────────────────
        self.seasonal_table = np.ones((C, 12), dtype=np.float32)
        for cat_name, cat_idx in category_to_idx.items():
            if 0 <= cat_idx < C:
                for month in range(1, 13):
                    self.seasonal_table[cat_idx, month - 1] = seasonal_multiplier(
                        cat_name, month
                    )

        # ── Precomputed adjacency (C, max_adj) ──────────────────────
        max_adj = max((len(v) for v in CATEGORY_ADJACENCY.values()), default=1)
        max_adj = max(max_adj, 1)
        self.adjacency = np.full((C, max_adj), -1, dtype=np.int32)
        self.adj_counts = np.zeros(C, dtype=np.int32)

        for cat_name, related in CATEGORY_ADJACENCY.items():
            cat_idx = category_to_idx.get(cat_name, -1)
            if cat_idx < 0 or cat_idx >= C:
                continue
            count = 0
            for j, rel_name in enumerate(related):
                rel_idx = category_to_idx.get(rel_name, -1)
                if rel_idx >= 0 and j < max_adj:
                    self.adjacency[cat_idx, j] = rel_idx
                    count += 1
            self.adj_counts[cat_idx] = count

        # Category average prices (updated from recommendations each epoch)
        self.category_avg_prices = np.full(C, 10.0, dtype=np.float32)

        # ── Organic purchase catalog ─────────────────────────────────
        self.catalog_pids = catalog_pids
        self.catalog_prices = catalog_prices
        self.catalog_weights = catalog_weights

        self._epoch: int = 0

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def simulate_epoch(
        self,
        rec_pids: np.ndarray,      # (N, K) int32  — product IDs
        rec_cat_idx: np.ndarray,   # (N, K) int32  — category indices
        rec_scores: np.ndarray,    # (N, K) float32 — recommendation scores
        rec_prices: np.ndarray,    # (N, K) float32 — product prices
    ) -> VectorizedEpochResult:
        """Simulate one epoch (~1 week) for all customers.

        All inputs are aligned: row *i* is customer index *i* (0-based).
        """
        month_idx = (self._epoch // 4) % 12   # 0-based month
        self._epoch += 1

        N = self.num_customers
        K = rec_pids.shape[1]

        # Reusable index vector for advanced indexing: (N, 1)
        ci = np.arange(N, dtype=np.int64)[:, np.newaxis]

        # ── Update category avg prices from current recs ─────────────
        for c in range(self.num_categories):
            mask = rec_cat_idx == c
            if mask.any():
                self.category_avg_prices[c] = float(rec_prices[mask].mean())

        recommended_revenue = 0.0
        rec_cid_parts: list[np.ndarray] = []
        rec_pid_parts: list[np.ndarray] = []
        customer_purchased = np.zeros(N, dtype=np.bool_)

        # ── 1. Re-engagement ─────────────────────────────────────────
        is_dormant = self.dormant_epochs >= self.dormancy_threshold
        n_dormant = int(is_dormant.sum())

        if n_dormant > 0:
            re_rolls = self.rng.random(N)
            re_engaged = is_dormant & (re_rolls < self.re_engagement_prob)
            n_re = int(re_engaged.sum())

            if n_re > 0:
                re_idx = np.where(re_engaged)[0]
                burst_counts = np.minimum(
                    self.rng.integers(1, 4, size=n_re), K
                ).astype(np.int32)

                # Vectorized burst: pick random rec indices per customer
                max_burst = 3
                all_picks = self.rng.integers(0, max(K, 1), size=(n_re, max_burst))
                burst_mask = np.arange(max_burst) < burst_counts[:, np.newaxis]

                flat_cust = np.repeat(re_idx, burst_counts)
                flat_rec = all_picks[burst_mask]

                if len(flat_cust) > 0:
                    b_pids = rec_pids[flat_cust, flat_rec]
                    b_cats = rec_cat_idx[flat_cust, flat_rec]
                    b_seasonal = self.seasonal_table[b_cats, month_idx]
                    b_rev = rec_prices[flat_cust, flat_rec] * b_seasonal

                    recommended_revenue += float(b_rev.sum())
                    rec_cid_parts.append(flat_cust.astype(np.int64))
                    rec_pid_parts.append(b_pids.astype(np.int64))
                    customer_purchased[flat_cust] = True

                # Reset fatigue for re-engaged
                self.fatigue_touches[re_engaged] = 0
                init_mask = self.fatigue_onset[re_engaged] >= 0
                updated = self.fatigue_onset[re_engaged].copy()
                updated[init_mask] *= self.re_engagement_decay
                self.fatigue_onset[re_engaged] = updated
                self.re_engagement_count[re_engaged] += 1
                self.dormant_epochs[re_engaged] = 0

        # ── 2. Initialize fatigue onset for new exposures ────────────
        flat_ci = np.broadcast_to(ci, (N, K)).ravel()
        flat_cat = rec_cat_idx.ravel()
        uninit = self.fatigue_onset[flat_ci, flat_cat] < 0

        if uninit.any():
            u_ci = flat_ci[uninit]
            u_cat = flat_cat[uninit]
            # Deduplicate (customer, category) pairs
            pair_keys = u_ci.astype(np.int64) * self.num_categories + u_cat
            _, uniq_idx = np.unique(pair_keys, return_index=True)
            uc, ucat = u_ci[uniq_idx], u_cat[uniq_idx]
            new_onset = np.maximum(
                1.0, self.rng.normal(8.0, 2.5, size=len(uc))
            ).astype(np.float32)
            self.fatigue_onset[uc, ucat] = new_onset

        # ── 3. Fatigue-adjusted purchase probabilities ───────────────
        base_p = np.clip(rec_scores, 0.0, 1.0).astype(np.float32)
        base_p *= self.seasonal_table[rec_cat_idx, month_idx]
        np.minimum(base_p, 1.0, out=base_p)

        touches = self.fatigue_touches[ci, rec_cat_idx].astype(np.float32)
        onset = self.fatigue_onset[ci, rec_cat_idx]

        farg = np.clip(
            (touches - onset) * self.fatigue_steepness, -500.0, 500.0
        )
        fatigue = 1.0 / (1.0 + np.exp(-farg))
        p = base_p * (1.0 - fatigue)

        # ── 4. Purchase decisions ────────────────────────────────────
        rolls = self.rng.random((N, K), dtype=np.float32)
        purchased = rolls < p

        # Increment touch counters for ALL recs (recommendation received)
        np.add.at(
            self.fatigue_touches,
            (np.broadcast_to(ci, (N, K)).ravel(), flat_cat),
            1,
        )

        if purchased.any():
            p_cust, p_k = np.where(purchased)
            p_pids = rec_pids[p_cust, p_k]
            recommended_revenue += float(rec_prices[p_cust, p_k].sum())
            rec_cid_parts.append(p_cust.astype(np.int64))
            rec_pid_parts.append(p_pids.astype(np.int64))
            customer_purchased[p_cust] = True

        # ── 5. Halo effect ───────────────────────────────────────────
        halo_rolls = self.rng.random((N, K), dtype=np.float32)
        halo_fired = halo_rolls < self.halo_p
        n_halo = 0
        halo_revenue = 0.0

        if halo_fired.any():
            h_cust, h_k = np.where(halo_fired)
            h_cats = rec_cat_idx[h_cust, h_k]
            h_adj_n = self.adj_counts[h_cats]
            has_adj = h_adj_n > 0

            if has_adj.any():
                v_cats = h_cats[has_adj]
                v_counts = h_adj_n[has_adj]
                rand_idx = (
                    self.rng.random(len(v_cats)) * v_counts
                ).astype(np.int32)
                target_cats = self.adjacency[v_cats, rand_idx]
                valid = target_cats >= 0

                if valid.any():
                    final_cats = target_cats[valid]
                    halo_revenue = float(
                        self.category_avg_prices[final_cats].sum()
                    )
                    n_halo = int(valid.sum())
                    customer_purchased[h_cust[has_adj][valid]] = True

        # ── 6. Organic purchases ─────────────────────────────────────
        o_cids, o_pids, o_rev = self._organic_purchases()
        if len(o_cids) > 0:
            customer_purchased[o_cids] = True

        # ── 7. Dormancy tracking ─────────────────────────────────────
        self.dormant_epochs[customer_purchased] = 0
        self.dormant_epochs[~customer_purchased] += 1

        # ── Assemble result ──────────────────────────────────────────
        if rec_cid_parts:
            all_rec_cids = np.concatenate(rec_cid_parts)
            all_rec_pids = np.concatenate(rec_pid_parts)
        else:
            all_rec_cids = np.array([], dtype=np.int64)
            all_rec_pids = np.array([], dtype=np.int64)

        total_rev = recommended_revenue + halo_revenue + o_rev

        return VectorizedEpochResult(
            total_revenue=total_rev,
            recommended_revenue=recommended_revenue,
            halo_revenue=halo_revenue,
            organic_revenue=o_rev,
            rec_purchase_cids=all_rec_cids,
            rec_purchase_pids=all_rec_pids,
            organic_purchase_cids=o_cids,
            organic_purchase_pids=o_pids,
            num_customers_who_purchased=int(customer_purchased.sum()),
            num_recommended_purchases=len(all_rec_cids),
            num_halo_purchases=n_halo,
            num_organic_purchases=len(o_cids),
        )

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _organic_purchases(self):
        """Generate organic purchases for all customers (vectorized)."""
        N = self.num_customers
        if len(self.catalog_pids) == 0:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                0.0,
            )

        counts = self.rng.poisson(self.organic_rate, size=N).astype(np.int32)
        total = int(counts.sum())
        if total == 0:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                0.0,
            )

        indices = self.rng.choice(
            len(self.catalog_pids),
            size=total,
            replace=True,
            p=self.catalog_weights,
        )
        pids = self.catalog_pids[indices]
        revenue = float(self.catalog_prices[indices].sum())
        cids = np.repeat(np.arange(N, dtype=np.int64), counts)

        return cids, pids, revenue

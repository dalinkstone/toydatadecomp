"""Vectorized Consumer Behavior Simulator.

Numpy-vectorized version of ConsumerSimulator for large-scale (10M+ customer)
Monte Carlo simulation.  Processes customers in chunks of 1M to keep working
memory under ~400MB regardless of total customer count.

All five behavioral dynamics from ``consumer_behavior.py`` are preserved:
  1. Recommendation Fatigue (inverse sigmoid, per customer-category)
  2. Re-engagement After Dormancy (burst + diminished onset reset)
  3. Cross-Category Halo Effect (adjacency-based related-category purchase)
  4. Seasonal Demand Modifier (month-dependent category multipliers)
  5. Concurrent Organic Purchases (Poisson count, approximate revenue)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.consumer_behavior import (
    CATEGORY_ADJACENCY,
    seasonal_multiplier,
)


# ═══════════════════════════════════════════════════════════════════════════
# Result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VectorizedEpochResult:
    """Compact result of one epoch.  Only recommended-purchase arrays are
    materialised; organic counts use scalar aggregates to avoid 30M-element
    temporary arrays at 10M scale."""

    total_revenue: float
    recommended_revenue: float
    halo_revenue: float
    organic_revenue: float

    # Recommended purchase pairs (0-based customer index, product id).
    # Caller adds 1 for 1-based customer_id.
    rec_purchase_cids: np.ndarray   # int64
    rec_purchase_pids: np.ndarray   # int64

    num_customers_who_purchased: int
    num_recommended_purchases: int
    num_halo_purchases: int
    num_organic_purchases: int


# ═══════════════════════════════════════════════════════════════════════════
# Simulator
# ═══════════════════════════════════════════════════════════════════════════

# Customer chunk size for working-memory control (~40 MB working per chunk).
_CUST_CHUNK = 1_000_000


class VectorizedConsumerSimulator:
    """Numpy-vectorized consumer behaviour simulator.

    Parameters
    ----------
    seed : int
        Pins all distributional draws for one Monte Carlo run.
    num_customers : int
        Total customers (array dimension).
    category_to_idx : dict[str, int]
        Category name -> integer index (1-based, 0 = unknown).
    catalog_prices, catalog_weights : np.ndarray
        Product catalog arrays for organic revenue approximation.
    dormancy_threshold : int
        Consecutive purchase-free epochs before dormancy (default 4).
    """

    def __init__(
        self,
        seed: int,
        num_customers: int,
        category_to_idx: dict[str, int],
        catalog_prices: np.ndarray,
        catalog_weights: np.ndarray,
        dormancy_threshold: int = 4,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.num_customers = num_customers
        self.dormancy_threshold = dormancy_threshold

        num_cat = max(category_to_idx.values()) + 1 if category_to_idx else 1
        self.num_categories = num_cat

        # ── Distributional parameters (identical to ConsumerSimulator) ──
        self.fatigue_steepness: float = float(self.rng.uniform(0.4, 1.2))
        self.re_engagement_prob: float = float(self.rng.beta(3, 17))
        self.re_engagement_decay: float = float(self.rng.uniform(0.5, 0.8))
        self.halo_p: float = float(self.rng.beta(2, 18))
        self.organic_rate: float = float(self.rng.gamma(shape=2.0, scale=1.5))

        # ── State arrays ─────────────────────────────────────────────
        N, C = num_customers, num_cat
        self.fatigue_touches = np.zeros((N, C), dtype=np.int32)
        self.fatigue_onset = np.full((N, C), -1.0, dtype=np.float32)
        self.dormant_epochs = np.zeros(N, dtype=np.int32)
        self.re_engagement_count = np.zeros(N, dtype=np.int32)

        # ── Precomputed tables ───────────────────────────────────────
        self.seasonal_table = np.ones((C, 12), dtype=np.float32)
        for cat_name, cat_idx in category_to_idx.items():
            if 0 <= cat_idx < C:
                for month in range(1, 13):
                    self.seasonal_table[cat_idx, month - 1] = seasonal_multiplier(
                        cat_name, month
                    )

        max_adj = max((len(v) for v in CATEGORY_ADJACENCY.values()), default=1)
        max_adj = max(max_adj, 1)
        self.adjacency = np.full((C, max_adj), -1, dtype=np.int32)
        self.adj_counts = np.zeros(C, dtype=np.int32)
        for cat_name, related in CATEGORY_ADJACENCY.items():
            ci = category_to_idx.get(cat_name, -1)
            if ci < 0 or ci >= C:
                continue
            cnt = 0
            for j, rel in enumerate(related):
                ri = category_to_idx.get(rel, -1)
                if ri >= 0 and j < max_adj:
                    self.adjacency[ci, j] = ri
                    cnt += 1
            self.adj_counts[ci] = cnt

        self.category_avg_prices = np.full(C, 10.0, dtype=np.float32)

        # Organic: approximate revenue = count * weighted-average price
        self._organic_avg_price: float = float(
            (catalog_prices * catalog_weights).sum()
        ) if len(catalog_prices) > 0 else 10.0

        self._epoch: int = 0

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def simulate_epoch(
        self,
        rec_pids: np.ndarray,      # (N, K) int64
        rec_cat_idx: np.ndarray,   # (N, K) int32
        rec_scores: np.ndarray,    # (N, K) float32
        rec_prices: np.ndarray,    # (N, K) float32
    ) -> VectorizedEpochResult:
        """Simulate one epoch for all customers, processed in chunks."""
        month_idx = (self._epoch // 4) % 12
        self._epoch += 1

        N = self.num_customers
        K = rec_pids.shape[1]

        # ── Update category average prices (lightweight scan) ────────
        for c in range(self.num_categories):
            mask = rec_cat_idx == c
            if mask.any():
                self.category_avg_prices[c] = float(rec_prices[mask].mean())

        # ── Accumulators ─────────────────────────────────────────────
        total_rec_rev = 0.0
        total_halo_rev = 0.0
        total_halo_count = 0
        rec_cid_parts: list[np.ndarray] = []
        rec_pid_parts: list[np.ndarray] = []
        customer_purchased = np.zeros(N, dtype=np.bool_)

        # ── Process customers in chunks ──────────────────────────────
        for start in range(0, N, _CUST_CHUNK):
            end = min(start + _CUST_CHUNK, N)
            cn = end - start

            # Views into persistent state (no copy)
            ct = self.fatigue_touches[start:end]
            co = self.fatigue_onset[start:end]
            cd = self.dormant_epochs[start:end]
            cr = self.re_engagement_count[start:end]

            # Views into rec arrays (no copy)
            cp = rec_pids[start:end]
            cc = rec_cat_idx[start:end]
            cs = rec_scores[start:end]
            cx = rec_prices[start:end]

            ci = np.arange(cn, dtype=np.int32)[:, np.newaxis]  # (cn, 1)

            # ── Re-engagement ────────────────────────────────────────
            is_dormant = cd >= self.dormancy_threshold
            if is_dormant.any():
                re_rolls = self.rng.random(cn)
                re_engaged = is_dormant & (re_rolls < self.re_engagement_prob)
                n_re = int(re_engaged.sum())

                if n_re > 0:
                    re_loc = np.where(re_engaged)[0]
                    burst_n = np.minimum(
                        self.rng.integers(1, 4, size=n_re), K
                    ).astype(np.int32)

                    picks = self.rng.integers(0, max(K, 1), size=(n_re, 3))
                    bmask = np.arange(3) < burst_n[:, np.newaxis]
                    flat_loc = np.repeat(re_loc, burst_n)
                    flat_k = picks[bmask]

                    if len(flat_loc) > 0:
                        b_cats = cc[flat_loc, flat_k]
                        b_seas = self.seasonal_table[b_cats, month_idx]
                        total_rec_rev += float(
                            (cx[flat_loc, flat_k] * b_seas).sum()
                        )
                        rec_cid_parts.append(
                            (flat_loc + start).astype(np.int64)
                        )
                        rec_pid_parts.append(
                            cp[flat_loc, flat_k].astype(np.int64)
                        )
                        customer_purchased[flat_loc + start] = True

                    ct[re_engaged] = 0
                    upd = co[re_engaged].copy()
                    upd[upd >= 0] *= self.re_engagement_decay
                    co[re_engaged] = upd
                    cr[re_engaged] += 1
                    cd[re_engaged] = 0

            # ── Fatigue onset init (loop over K, avoids 100M flat) ───
            for k in range(K):
                cats_k = cc[:, k]
                uninit = co[np.arange(cn), cats_k] < 0
                if uninit.any():
                    n_new = int(uninit.sum())
                    co[uninit, cats_k[uninit]] = np.maximum(
                        1.0, self.rng.normal(8.0, 2.5, size=n_new)
                    ).astype(np.float32)

            # ── Purchase probabilities ───────────────────────────────
            base_p = np.clip(cs, 0.0, 1.0).astype(np.float32)
            base_p *= self.seasonal_table[cc, month_idx]
            np.minimum(base_p, 1.0, out=base_p)

            touches = ct[ci, cc].astype(np.float32)
            onset = co[ci, cc]

            farg = np.clip(
                (touches - onset) * self.fatigue_steepness, -500, 500
            )
            p = base_p / (1.0 + np.exp(-farg))
            # p = base_p * (1 - sigmoid) = base_p * sigmoid(-farg)
            # Rewritten: p = base_p * 1/(1+exp(farg)) = base_p / (1+exp(-(-farg)))
            # Actually let me fix this. sigmoid(x) = 1/(1+exp(-x)).
            # fatigue = sigmoid((touches-onset)*steepness) = 1/(1+exp(-farg))
            # p = base_p * (1 - fatigue) = base_p * exp(-farg)/(1+exp(-farg))
            #   = base_p / (1 + exp(farg))
            p = base_p / (1.0 + np.exp(farg))

            # ── Purchase decisions ───────────────────────────────────
            rolls = self.rng.random((cn, K), dtype=np.float32)
            purchased = rolls < p

            # Increment fatigue (K-loop avoids huge flat index array)
            for k in range(K):
                np.add.at(ct, (np.arange(cn), cc[:, k]), 1)

            if purchased.any():
                p_loc, p_k = np.where(purchased)
                total_rec_rev += float(cx[p_loc, p_k].sum())
                rec_cid_parts.append((p_loc + start).astype(np.int64))
                rec_pid_parts.append(cp[p_loc, p_k].astype(np.int64))
                customer_purchased[p_loc + start] = True

            # ── Halo ─────────────────────────────────────────────────
            halo_rolls = self.rng.random((cn, K), dtype=np.float32)
            halo_fired = halo_rolls < self.halo_p

            if halo_fired.any():
                h_loc, h_k = np.where(halo_fired)
                h_cats = cc[h_loc, h_k]
                h_adj_n = self.adj_counts[h_cats]
                has_adj = h_adj_n > 0

                if has_adj.any():
                    v_cats = h_cats[has_adj]
                    v_n = h_adj_n[has_adj]
                    ri = (self.rng.random(len(v_cats)) * v_n).astype(np.int32)
                    tgt = self.adjacency[v_cats, ri]
                    valid = tgt >= 0
                    if valid.any():
                        total_halo_rev += float(
                            self.category_avg_prices[tgt[valid]].sum()
                        )
                        total_halo_count += int(valid.sum())
                        customer_purchased[
                            h_loc[has_adj][valid] + start
                        ] = True

            # Free chunk working arrays
            del base_p, touches, onset, farg, p, rolls, purchased
            del halo_rolls, halo_fired

        # ── Organic purchases (approximate revenue, no large temps) ──
        org_total, org_rev, org_mask = self._organic_purchases()
        customer_purchased |= org_mask

        # ── Dormancy ─────────────────────────────────────────────────
        self.dormant_epochs[customer_purchased] = 0
        self.dormant_epochs[~customer_purchased] += 1

        # ── Assemble result ──────────────────────────────────────────
        if rec_cid_parts:
            all_cids = np.concatenate(rec_cid_parts)
            all_pids = np.concatenate(rec_pid_parts)
        else:
            all_cids = np.array([], dtype=np.int64)
            all_pids = np.array([], dtype=np.int64)

        return VectorizedEpochResult(
            total_revenue=total_rec_rev + total_halo_rev + org_rev,
            recommended_revenue=total_rec_rev,
            halo_revenue=total_halo_rev,
            organic_revenue=org_rev,
            rec_purchase_cids=all_cids,
            rec_purchase_pids=all_pids,
            num_customers_who_purchased=int(customer_purchased.sum()),
            num_recommended_purchases=len(all_cids),
            num_halo_purchases=total_halo_count,
            num_organic_purchases=org_total,
        )

    # ──────────────────────────────────────────────────────────────────
    def _organic_purchases(self) -> tuple[int, float, np.ndarray]:
        """Return (count, revenue, customer_mask) without materialising
        per-purchase arrays.  Revenue = count * weighted-average price."""
        counts = self.rng.poisson(self.organic_rate, size=self.num_customers)
        total = int(counts.sum())
        mask = counts > 0
        revenue = total * self._organic_avg_price
        return total, revenue, mask

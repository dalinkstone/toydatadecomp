"""Tiered Consumer Behavior Simulator (Phase 5).

Revenue-calibrated, tier-aware Monte Carlo simulation for 10M+ customers.
Weekly epochs matching CVS's promotional cadence.

Revenue calibration:
  CVS front store ~$22B/year across 74M ExtraCare members.
  10M simulated = 13.5% of base -> target ~$2.97B/year -> ~$57.1M/week.

Tiered recommendation strategy per epoch:
  Tier 1 (Core Revenue Drivers): always in basket, optimal discount by customer
  Tier 2 (Discount-Responsive): top-5 personalized coupon offers
  Tier 3 (Organic Sellers): probabilistic basket inclusion, no discount
  Tier 4 (Breakout Candidates): 1-2 offers at estimated discount

Three-level fatigue model:
  Per-product: 3 consecutive offers w/o purchase -> 4-week cooldown
  Per-category: 3+ offers in same category/epoch -> 50% probability reduction
  Global: max 8 coupon offers per customer per epoch
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.consumer_behavior import CATEGORY_ADJACENCY, seasonal_multiplier


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Revenue calibration (10M customers -> $57.1M/week)
WEEKLY_REVENUE_TARGET = 57_100_000
AVG_REVENUE_PER_VISIT = 35.0

# Tier 1: core revenue drivers (always bought by visitors)
TIER1_ITEMS_PER_VISIT = 2.5  # Poisson lambda

# Tier 1 discount by price_sensitivity_bucket (0=insensitive .. 4=very)
TIER1_DISCOUNT_SCHEDULE = np.array(
    [0.00, 0.03, 0.07, 0.10, 0.15], dtype=np.float32
)

# Tier 3: organic sellers (no discount, probabilistic)
TIER3_ITEMS_PER_VISIT = 1.2  # Poisson lambda

# Fatigue
PRODUCT_FATIGUE_STREAK = 3   # consecutive offers w/o purchase before cooldown
PRODUCT_COOLDOWN_WEEKS = 4   # weeks product is suppressed after fatigue
CATEGORY_FATIGUE_OFFERS = 3  # max same-category offers before penalty
CATEGORY_FATIGUE_PENALTY = 0.50  # probability multiplier for excess offers
MAX_COUPON_OFFERS = 8        # global cap per customer per epoch

# Chunked processing
_CUST_CHUNK = 1_000_000


# ═══════════════════════════════════════════════════════════════════════════════
# Result
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TieredEpochResult:
    """Comprehensive metrics from one epoch of tiered simulation."""

    # Revenue
    total_revenue: float
    recommended_revenue: float   # Tier 2 + Tier 4 coupon conversions
    organic_revenue: float       # Tier 1 at-price + Tier 3 passive
    discount_cost: float         # total discount dollars given away
    net_revenue: float           # total_revenue - discount_cost
    incremental_revenue: float   # coupon revenue that would NOT have happened organically
    cannibalized_revenue: float  # coupon revenue that would have happened anyway

    # Per-tier
    tier1_revenue: float
    tier1_discount_cost: float
    tier2_offers: int
    tier2_conversions: int
    tier2_revenue: float
    tier2_discount_cost: float
    tier3_revenue: float
    tier4_offers: int
    tier4_conversions: int
    tier4_revenue: float
    tier4_discount_cost: float
    halo_revenue: float

    # Retraining data (0-based customer index)
    rec_purchase_cids: np.ndarray
    rec_purchase_pids: np.ndarray

    # Metrics
    active_customers: int
    total_coupons_offered: int
    unique_pids_purchased: np.ndarray


# ═══════════════════════════════════════════════════════════════════════════════
# Internal accumulator (avoids passing dozens of variables across chunks)
# ═══════════════════════════════════════════════════════════════════════════════

class _Acc:
    __slots__ = (
        "tier1_rev", "tier1_disc",
        "tier2_offers", "tier2_conv", "tier2_rev", "tier2_disc",
        "tier3_rev",
        "tier4_offers", "tier4_conv", "tier4_rev", "tier4_disc",
        "halo_rev", "active", "coupons",
        "incr_rev", "cannibal_rev",
    )

    def __init__(self) -> None:
        self.tier1_rev = 0.0
        self.tier1_disc = 0.0
        self.tier2_offers = 0
        self.tier2_conv = 0
        self.tier2_rev = 0.0
        self.tier2_disc = 0.0
        self.tier3_rev = 0.0
        self.tier4_offers = 0
        self.tier4_conv = 0
        self.tier4_rev = 0.0
        self.tier4_disc = 0.0
        self.halo_rev = 0.0
        self.active = 0
        self.coupons = 0
        self.incr_rev = 0.0
        self.cannibal_rev = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Simulator
# ═══════════════════════════════════════════════════════════════════════════════

class TieredConsumerSimulator:
    """Tier-aware consumer simulator with revenue calibration.

    Parameters
    ----------
    seed : int
        Pins all distributional draws for one MC run.
    num_customers : int
        Total customer count (0-based indexing internally).
    visit_probs : ndarray (N,) float32
        Per-customer weekly visit probability, pre-calibrated to revenue target.
    price_sensitivity : ndarray (N,) int8
        Price sensitivity bucket 0-4 per customer.
    tier1_prices : ndarray (n_t1,) float32
        Tier 1 product prices.
    tier1_pids : ndarray (n_t1,) int64
        Tier 1 product IDs (for catalog coverage).
    tier3_avg_price : float
        Mean Tier 3 product price.
    category_to_idx : dict
        Category name -> integer index.
    """

    def __init__(
        self,
        seed: int,
        num_customers: int,
        visit_probs: np.ndarray,
        price_sensitivity: np.ndarray,
        tier1_prices: np.ndarray,
        tier1_pids: np.ndarray,
        tier3_avg_price: float,
        category_to_idx: dict[str, int],
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.num_customers = num_customers
        self.visit_probs = visit_probs.astype(np.float32)
        self.price_sensitivity = price_sensitivity.astype(np.int8)

        # Tier 1
        self.tier1_prices = tier1_prices.astype(np.float32)
        self.tier1_pids = tier1_pids.astype(np.int64)
        self.n_tier1 = len(tier1_prices)
        self.tier1_avg_price = (
            float(tier1_prices.mean()) if self.n_tier1 > 0 else 8.0
        )

        # Tier 3
        self.tier3_avg_price = tier3_avg_price

        # Categories & seasonal
        num_cat = max(category_to_idx.values()) + 1 if category_to_idx else 1
        self.num_categories = num_cat
        self.seasonal_table = np.ones((num_cat, 12), dtype=np.float32)
        for cat_name, cat_idx in category_to_idx.items():
            if 0 <= cat_idx < num_cat:
                for month in range(1, 13):
                    self.seasonal_table[cat_idx, month - 1] = seasonal_multiplier(
                        cat_name, month
                    )

        # Halo effect
        self.halo_p: float = float(self.rng.beta(2, 18))
        max_adj = max((len(v) for v in CATEGORY_ADJACENCY.values()), default=1)
        max_adj = max(max_adj, 1)
        self.adjacency = np.full((num_cat, max_adj), -1, dtype=np.int32)
        self.adj_counts = np.zeros(num_cat, dtype=np.int32)
        for cat_name, related in CATEGORY_ADJACENCY.items():
            ci = category_to_idx.get(cat_name, -1)
            if ci < 0 or ci >= num_cat:
                continue
            cnt = 0
            for j, rel in enumerate(related):
                ri = category_to_idx.get(rel, -1)
                if ri >= 0 and j < max_adj:
                    self.adjacency[ci, j] = ri
                    cnt += 1
            self.adj_counts[ci] = cnt
        self.category_avg_prices = np.full(num_cat, 10.0, dtype=np.float32)

        # Distributional parameters for this run
        self.fatigue_steepness: float = float(self.rng.uniform(0.4, 1.2))

        # Per-product fatigue state (slot-based tracking)
        N = num_customers
        self.prev_offer_pids = np.zeros((N, MAX_COUPON_OFFERS), dtype=np.int64)
        self.slot_streak = np.zeros((N, MAX_COUPON_OFFERS), dtype=np.int8)
        self.slot_cooldown = np.zeros((N, MAX_COUPON_OFFERS), dtype=np.int8)

        self._epoch: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def simulate_epoch(
        self,
        offer_pids: np.ndarray,            # (N, 8) int64
        offer_scores: np.ndarray,          # (N, 8) float32
        offer_prices: np.ndarray,          # (N, 8) float32
        offer_discounts: np.ndarray,       # (N, 8) float32
        offer_tiers: np.ndarray,           # (N, 8) int8  (2 or 4)
        offer_cat_idx: np.ndarray,         # (N, 8) int32
        offer_organic_ratios: np.ndarray,  # (N, 8) float32
    ) -> TieredEpochResult:
        """Simulate one weekly epoch for all customers, processed in chunks.

        The offer arrays contain up to 8 coupon offers per customer
        (slots 0-4 Tier 2, 5-6 Tier 4 breakout, 7 exploration).
        Tier 1 and Tier 3 revenue is computed internally.
        """
        month_idx = (self._epoch // 4) % 12
        self._epoch += 1

        N = self.num_customers
        K = MAX_COUPON_OFFERS
        seasonal_mean = float(self.seasonal_table[:, month_idx].mean())

        acc = _Acc()
        rec_cid_parts: list[np.ndarray] = []
        rec_pid_parts: list[np.ndarray] = []
        purchased_pid_parts: list[np.ndarray] = []

        # Tier 1 PIDs always count toward catalog coverage
        if self.n_tier1 > 0:
            purchased_pid_parts.append(self.tier1_pids.copy())

        for start in range(0, N, _CUST_CHUNK):
            end = min(start + _CUST_CHUNK, N)
            cn = end - start

            # ── Which customers visit this week ─────────────────────
            visit_rolls = self.rng.random(cn, dtype=np.float32)
            active = visit_rolls < self.visit_probs[start:end]
            n_active = int(active.sum())
            acc.active += n_active

            # Decrement cooldowns for ALL customers in this chunk
            cd = self.slot_cooldown[start:end]
            cd[cd > 0] -= 1

            if n_active == 0:
                continue

            active_idx = np.where(active)[0]

            # ── Tier 1 revenue (aggregate) ──────────────────────────
            t1_count = self.rng.poisson(TIER1_ITEMS_PER_VISIT, size=n_active)
            np.minimum(t1_count, max(self.n_tier1, 1), out=t1_count)

            sens = self.price_sensitivity[start:end][active_idx]
            disc_r = TIER1_DISCOUNT_SCHEDULE[np.clip(sens, 0, 4)]

            t1_rev = (t1_count * self.tier1_avg_price
                      * (1.0 - disc_r) * seasonal_mean)
            t1_disc = (t1_count * self.tier1_avg_price
                       * disc_r * seasonal_mean)
            acc.tier1_rev += float(t1_rev.sum())
            acc.tier1_disc += float(t1_disc.sum())

            # ── Tier 3 organic revenue ──────────────────────────────
            t3_count = self.rng.poisson(TIER3_ITEMS_PER_VISIT, size=n_active)
            acc.tier3_rev += float(
                (t3_count * self.tier3_avg_price * seasonal_mean).sum()
            )

            # ── Coupon offers (Tier 2 + Tier 4) ────────────────────
            a_pids = offer_pids[start:end][active_idx]
            a_scores = offer_scores[start:end][active_idx]
            a_prices = offer_prices[start:end][active_idx]
            a_discs = offer_discounts[start:end][active_idx]
            a_tiers = offer_tiers[start:end][active_idx]
            a_cats = offer_cat_idx[start:end][active_idx]
            a_organic = offer_organic_ratios[start:end][active_idx]

            # -- Per-product fatigue: cooldown filter --
            a_cool = self.slot_cooldown[start:end][active_idx]
            cooled = a_cool > 0
            valid = (a_pids > 0) & ~cooled

            # -- Per-product fatigue: streak tracking --
            a_prev = self.prev_offer_pids[start:end][active_idx]
            a_streak = self.slot_streak[start:end][active_idx].copy()
            same = (a_pids == a_prev) & (a_pids > 0)
            a_streak = np.where(
                same, a_streak + 1,
                np.where(a_pids > 0, np.int8(1), np.int8(0)),
            )
            entering_cooldown = a_streak >= PRODUCT_FATIGUE_STREAK
            valid &= ~entering_cooldown

            # -- Per-category fatigue (within this epoch) --
            cat_count = np.zeros((n_active, self.num_categories), dtype=np.int8)
            cat_penalty = np.ones((n_active, K), dtype=np.float32)
            for j in range(K):
                cats_j = a_cats[:, j]
                slot_ok = valid[:, j]
                curr = cat_count[np.arange(n_active), cats_j]
                excess = slot_ok & (curr >= CATEGORY_FATIGUE_OFFERS)
                cat_penalty[excess, j] = CATEGORY_FATIGUE_PENALTY
                ok_idx = np.where(slot_ok)[0]
                if len(ok_idx) > 0:
                    np.add.at(cat_count, (ok_idx, cats_j[ok_idx]), 1)

            # -- Purchase probabilities --
            # Calibrated sigmoid: score 0.3->~3%, 0.5->~10%, 0.7->~18%
            base_p = (
                0.25 / (1.0 + np.exp(-7.0 * (a_scores - 0.45)))
            ).astype(np.float32)

            # Seasonal modifier per slot
            for j in range(K):
                base_p[:, j] *= self.seasonal_table[a_cats[:, j], month_idx]

            base_p *= cat_penalty
            base_p[~valid] = 0.0
            np.minimum(base_p, 1.0, out=base_p)

            # -- Purchase decisions --
            rolls = self.rng.random((n_active, K), dtype=np.float32)
            purchased = rolls < base_p

            acc.coupons += int(valid.sum())

            # -- Tier 2 --
            is_t2 = a_tiers == 2
            t2_v = valid & is_t2
            t2_b = purchased & t2_v
            acc.tier2_offers += int(t2_v.sum())
            acc.tier2_conv += int(t2_b.sum())
            if t2_b.any():
                loc = np.where(t2_b)
                rev = a_prices[loc] * (1.0 - a_discs[loc])
                acc.tier2_rev += float(rev.sum())
                acc.tier2_disc += float((a_prices[loc] * a_discs[loc]).sum())
                org = a_organic[loc]
                acc.cannibal_rev += float((rev * org).sum())
                acc.incr_rev += float((rev * (1.0 - org)).sum())

            # -- Tier 4 --
            is_t4 = a_tiers == 4
            t4_v = valid & is_t4
            t4_b = purchased & t4_v
            acc.tier4_offers += int(t4_v.sum())
            acc.tier4_conv += int(t4_b.sum())
            if t4_b.any():
                loc = np.where(t4_b)
                rev = a_prices[loc] * (1.0 - a_discs[loc])
                acc.tier4_rev += float(rev.sum())
                acc.tier4_disc += float((a_prices[loc] * a_discs[loc]).sum())
                org = a_organic[loc]
                acc.cannibal_rev += float((rev * org).sum())
                acc.incr_rev += float((rev * (1.0 - org)).sum())

            # -- Record purchase pairs for retraining --
            if purchased.any():
                p_rows, p_cols = np.where(purchased)
                global_cids = (active_idx[p_rows] + start).astype(np.int64)
                bought_pids = a_pids[p_rows, p_cols].astype(np.int64)
                rec_cid_parts.append(global_cids)
                rec_pid_parts.append(bought_pids)
                purchased_pid_parts.append(bought_pids)

            # -- Halo effect (on coupon purchases) --
            if purchased.any():
                p_rows, p_cols = np.where(purchased)
                h_cats = a_cats[p_rows, p_cols]
                halo_rolls = self.rng.random(len(p_rows), dtype=np.float32)
                halo_fired = halo_rolls < self.halo_p
                if halo_fired.any():
                    h_cats_f = h_cats[halo_fired]
                    h_adj_n = self.adj_counts[h_cats_f]
                    has_adj = h_adj_n > 0
                    if has_adj.any():
                        v_cats = h_cats_f[has_adj]
                        v_n = h_adj_n[has_adj]
                        ri = (self.rng.random(len(v_cats)) * v_n).astype(
                            np.int32
                        )
                        tgt = self.adjacency[v_cats, ri]
                        ok = tgt >= 0
                        if ok.any():
                            acc.halo_rev += float(
                                self.category_avg_prices[tgt[ok]].sum()
                            )

            # -- Update per-product fatigue state --
            a_streak[purchased] = 0  # reset streak on purchase
            new_cool = a_cool.copy()
            new_cool[entering_cooldown] = PRODUCT_COOLDOWN_WEEKS

            # Write back to persistent arrays
            act_mask = np.zeros(cn, dtype=np.bool_)
            act_mask[active_idx] = True
            self.prev_offer_pids[start:end][act_mask] = a_pids
            self.slot_streak[start:end][act_mask] = a_streak.astype(np.int8)
            self.slot_cooldown[start:end][act_mask] = new_cool

            # Update category average prices from current offers
            for c in range(self.num_categories):
                mask = (a_cats == c) & valid
                if mask.any():
                    self.category_avg_prices[c] = float(a_prices[mask].mean())

            del (a_pids, a_scores, a_prices, a_discs, a_tiers, a_cats,
                 a_organic, base_p, rolls, purchased, valid, cooled)

        # ── Assemble result ──────────────────────────────────────────────
        all_cids = (
            np.concatenate(rec_cid_parts)
            if rec_cid_parts
            else np.array([], dtype=np.int64)
        )
        all_pids = (
            np.concatenate(rec_pid_parts)
            if rec_pid_parts
            else np.array([], dtype=np.int64)
        )
        unique_pids = (
            np.unique(np.concatenate(purchased_pid_parts))
            if purchased_pid_parts
            else np.array([], dtype=np.int64)
        )

        rec_rev = acc.tier2_rev + acc.tier4_rev
        org_rev = acc.tier1_rev + acc.tier3_rev
        disc_cost = acc.tier1_disc + acc.tier2_disc + acc.tier4_disc
        total_rev = rec_rev + org_rev + acc.halo_rev

        return TieredEpochResult(
            total_revenue=total_rev,
            recommended_revenue=rec_rev,
            organic_revenue=org_rev,
            discount_cost=disc_cost,
            net_revenue=total_rev - disc_cost,
            incremental_revenue=acc.incr_rev,
            cannibalized_revenue=acc.cannibal_rev,
            tier1_revenue=acc.tier1_rev,
            tier1_discount_cost=acc.tier1_disc,
            tier2_offers=acc.tier2_offers,
            tier2_conversions=acc.tier2_conv,
            tier2_revenue=acc.tier2_rev,
            tier2_discount_cost=acc.tier2_disc,
            tier3_revenue=acc.tier3_rev,
            tier4_offers=acc.tier4_offers,
            tier4_conversions=acc.tier4_conv,
            tier4_revenue=acc.tier4_rev,
            tier4_discount_cost=acc.tier4_disc,
            halo_revenue=acc.halo_rev,
            rec_purchase_cids=all_cids,
            rec_purchase_pids=all_pids,
            active_customers=acc.active,
            total_coupons_offered=acc.coupons,
            unique_pids_purchased=unique_pids,
        )

    # ──────────────────────────────────────────────────────────────────────
    def reset_streaks(self) -> None:
        """Reset per-product streak counts.  Call when offers change at
        retrain boundaries.  Cooldowns persist across retrains."""
        self.prev_offer_pids[:] = 0
        self.slot_streak[:] = 0

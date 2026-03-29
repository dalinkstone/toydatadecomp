"""Consumer Behavior Simulator for Monte Carlo evaluation of recommendations.

This module models how a synthetic customer responds to a recommendation over
time.  It is the core behavioral engine behind the Monte Carlo simulation that
evaluates the ranking layer's real-world effectiveness.

Every behavioral parameter is drawn from a **distribution**, not a point
estimate.  A single ``seed`` passed to ``ConsumerSimulator.__init__`` pins all
distributional draws for one Monte Carlo run, so different seeds explore
different "possible worlds" while remaining internally reproducible.

Behavioral dynamics modeled
----------------------------

1. **Recommendation Fatigue** (inverse sigmoid)

   Each customer maintains a per-category fatigue counter that increments every
   time they receive a recommendation in that category.  Purchase probability
   decays as:

       p = base_p * (1 - sigmoid((touches - onset) * steepness))

   where ``onset ~ Normal(mu=8, sigma=2.5)`` controls when fatigue kicks in
   and ``steepness ~ Uniform(0.4, 1.2)`` controls how sharply it drops.
   ``base_p`` is the ranked recommendation score rescaled to [0, 1].

2. **Re-engagement After Dormancy**

   A customer becomes dormant after ``dormancy_threshold`` consecutive epochs
   with no purchases.  Each dormant epoch, a re-engagement event fires with
   probability drawn from ``Beta(3, 17)`` (~15% mean).

   Re-engagement produces a burst of 1-3 purchases.  The fatigue curve then
   **resets** but with a shorter onset:

       new_onset = previous_onset * decay_factor
       decay_factor ~ Uniform(0.5, 0.8)

   This encodes diminishing returns across re-engagement cycles.

3. **Cross-Category Halo Effect**

   When a customer interacts with a recommendation (even without buying),
   there is a probability ``halo_p ~ Beta(2, 18)`` (~10% mean) that they
   purchase from a *related* category instead.  Relatedness is defined by a
   static adjacency map derived from the CVS product taxonomy (e.g.,
   Oral Care -> Hair Care / Skin Care).

   Halo purchases represent revenue the system influenced but that the
   attribution model would not directly credit to a recommendation.

4. **Seasonal Demand Modifier**

   Product categories carry month-dependent multipliers.  Cold medicine peaks
   in months 11-2, sunscreen peaks in months 5-8, etc.  The multiplier
   scales ``base_p`` before the fatigue curve is applied, so seasonal lift
   interacts multiplicatively with fatigue.

   Each epoch advances the simulation clock by approximately one week.

5. **Concurrent Organic Purchases**

   Independently of recommendations, every customer makes organic purchases
   at a baseline rate ``organic_rate ~ Gamma(shape=2, scale=1.5)`` per epoch.
   Products are drawn from the full catalog weighted by ``popularity_score``.
   Organic revenue counts toward totals but is **not** attributed to the
   recommendation engine.

Distributional assumptions summary
------------------------------------

=========================  ===================================
Parameter                  Distribution
=========================  ===================================
fatigue onset              Normal(mu=8, sigma=2.5)
fatigue steepness          Uniform(0.4, 1.2)
re-engagement probability  Beta(3, 17)
re-engagement decay_factor Uniform(0.5, 0.8)
halo probability           Beta(2, 18)
organic purchase rate      Gamma(shape=2, scale=1.5)
=========================  ===================================

Usage
-----
::

    sim = ConsumerSimulator(seed=42)
    result = sim.simulate_epoch(customer_states, recommendations)
    print(result.revenue)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Category adjacency map
# ═══════════════════════════════════════════════════════════════════════════

#: Maps each product category to a list of "related" categories for the
#: cross-category halo effect.  Derived from the CVS product taxonomy used
#: by the generators (see ``src/generators/scrape_products.py``).
CATEGORY_ADJACENCY: dict[str, list[str]] = {
    "Pain Relief & Fever":      ["Cold/Flu/Allergy", "First Aid & Wound Care",
                                 "Vitamins & Supplements"],
    "Cold/Flu/Allergy":         ["Pain Relief & Fever", "Vitamins & Supplements",
                                 "Eye & Ear Care"],
    "Digestive Health":         ["Vitamins & Supplements", "Snacks & Beverages"],
    "Vitamins & Supplements":   ["Digestive Health", "Pain Relief & Fever",
                                 "Cold/Flu/Allergy"],
    "Skin Care":                ["Cosmetics & Makeup", "Hair Care",
                                 "First Aid & Wound Care"],
    "Hair Care":                ["Skin Care", "Shaving & Grooming",
                                 "Oral Care"],
    "Oral Care":                ["Hair Care", "Skin Care"],
    "Deodorant":                ["Shaving & Grooming", "Skin Care"],
    "Shaving & Grooming":       ["Deodorant", "Hair Care", "Skin Care"],
    "Cosmetics & Makeup":       ["Skin Care", "Hair Care"],
    "Baby & Childcare":         ["Feminine Care", "Household Essentials",
                                 "Skin Care"],
    "First Aid & Wound Care":   ["Pain Relief & Fever", "Skin Care"],
    "Eye & Ear Care":           ["Cold/Flu/Allergy", "Vitamins & Supplements"],
    "Snacks & Beverages":       ["Household Essentials", "Digestive Health"],
    "Household Essentials":     ["Snacks & Beverages", "Baby & Childcare"],
    "Feminine Care":            ["Baby & Childcare", "Skin Care"],
    "Greeting Cards & Gift Wrap": ["Photo & Electronics"],
    "Photo & Electronics":      ["Greeting Cards & Gift Wrap",
                                 "Household Essentials"],
    "Smoking Cessation":        ["Oral Care"],
    "Sexual Health":            ["Feminine Care", "Skin Care"],
    "Foot Care":                ["First Aid & Wound Care", "Pain Relief & Fever",
                                 "Skin Care"],
}


# ═══════════════════════════════════════════════════════════════════════════
# Seasonal demand multipliers
# ═══════════════════════════════════════════════════════════════════════════

def _build_seasonal_table() -> dict[tuple[str, int], float]:
    """Build ``(category, month) -> multiplier`` table.

    Months are 1-indexed (January = 1).  Any ``(category, month)`` pair not
    present in the table defaults to ``1.0``.
    """
    table: dict[tuple[str, int], float] = {}

    # Cold/Flu peaks Nov-Feb, depressed Jun-Aug
    for m in (11, 12, 1, 2):
        table[("Cold/Flu/Allergy", m)] = 2.5
    for m in (6, 7, 8):
        table[("Cold/Flu/Allergy", m)] = 0.6
    for m in (3, 10):
        table[("Cold/Flu/Allergy", m)] = 1.4

    # Pain Relief slight winter bump
    for m in (11, 12, 1, 2):
        table[("Pain Relief & Fever", m)] = 1.3

    # Skin Care / Sunscreen peaks summer
    for m in (5, 6, 7, 8):
        table[("Skin Care", m)] = 1.8
    for m in (12, 1, 2):
        table[("Skin Care", m)] = 0.7

    # Vitamins slight cold-season lift
    for m in (10, 11, 12, 1, 2):
        table[("Vitamins & Supplements", m)] = 1.3

    # Greeting Cards spike in Dec and Feb (holidays, Valentine's)
    table[("Greeting Cards & Gift Wrap", 12)] = 2.5
    table[("Greeting Cards & Gift Wrap", 2)] = 2.0
    table[("Greeting Cards & Gift Wrap", 5)] = 1.5  # Mother's Day

    # Digestive Health mild holiday bump
    for m in (11, 12):
        table[("Digestive Health", m)] = 1.3

    # Snacks & Beverages summer + holiday
    for m in (6, 7):
        table[("Snacks & Beverages", m)] = 1.4
    for m in (11, 12):
        table[("Snacks & Beverages", m)] = 1.3

    # Baby & Childcare slight spring lift
    for m in (3, 4, 5):
        table[("Baby & Childcare", m)] = 1.2

    return table


SEASONAL_TABLE: dict[tuple[str, int], float] = _build_seasonal_table()


def seasonal_multiplier(category: str, month: int) -> float:
    """Return the seasonal demand multiplier for *category* in *month*.

    Returns ``1.0`` for any ``(category, month)`` pair not explicitly listed
    in ``SEASONAL_TABLE``.
    """
    return SEASONAL_TABLE.get((category, month), 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Sigmoid helper
# ═══════════════════════════════════════════════════════════════════════════

def _sigmoid(x: float) -> float:
    """Standard logistic sigmoid, clamped to avoid overflow."""
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EpochResult:
    """Outcome of one simulation epoch.

    Attributes
    ----------
    recommended_purchases : list[dict]
        Purchases directly attributable to a recommendation.  Each dict
        contains ``customer_id``, ``product_id``, ``category``, ``revenue``.
    halo_purchases : list[dict]
        Cross-category halo purchases influenced (but not directly caused)
        by a recommendation.  Same dict structure.
    organic_purchases : list[dict]
        Purchases that occurred independently of recommendations.
    fatigue_states : dict[int, dict[str, int]]
        Updated ``{customer_id: {category: touch_count}}`` after this epoch.
    revenue : float
        Total revenue across all purchase types in this epoch.
    """
    recommended_purchases: list[dict[str, Any]] = field(default_factory=list)
    halo_purchases: list[dict[str, Any]] = field(default_factory=list)
    organic_purchases: list[dict[str, Any]] = field(default_factory=list)
    fatigue_states: dict[int, dict[str, int]] = field(default_factory=dict)
    revenue: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Customer state
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CustomerState:
    """Mutable per-customer state carried across epochs.

    Attributes
    ----------
    fatigue_touches : dict[str, int]
        Number of recommendation touches per category.
    fatigue_onset : dict[str, float]
        Current fatigue onset threshold per category (shrinks on re-engagement).
    dormant_epochs : int
        Consecutive epochs with zero purchases (of any kind).
    re_engagement_count : int
        How many times this customer has been re-engaged.
    """
    fatigue_touches: dict[str, int] = field(default_factory=dict)
    fatigue_onset: dict[str, float] = field(default_factory=dict)
    dormant_epochs: int = 0
    re_engagement_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Main simulator
# ═══════════════════════════════════════════════════════════════════════════

class ConsumerSimulator:
    """Monte Carlo consumer behavior simulator.

    Each instance represents one "possible world" — a single draw of all
    distributional parameters.  Create multiple instances with different
    seeds to explore the parameter space.

    Parameters
    ----------
    seed : int
        Random seed that determines every distributional draw for this run.
        Two simulators with the same seed will produce identical results
        given identical inputs.
    dormancy_threshold : int
        Number of consecutive purchase-free epochs before a customer is
        considered dormant (default 4, i.e. ~1 month of inactivity).
    product_catalog : list[dict] | None
        Full product catalog for organic purchase draws.  Each dict must
        contain ``product_id``, ``category``, ``price``, ``popularity_score``.
        If ``None``, organic purchases are skipped.
    """

    def __init__(
        self,
        seed: int,
        dormancy_threshold: int = 4,
        product_catalog: list[dict[str, Any]] | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.dormancy_threshold = dormancy_threshold

        # ── Sample global distributional parameters for this run ─────
        self.fatigue_onset_mu: float = 8.0
        self.fatigue_onset_sigma: float = 2.5
        self.fatigue_steepness: float = float(self.rng.uniform(0.4, 1.2))

        self.re_engagement_prob: float = float(self.rng.beta(3, 17))
        self.re_engagement_decay: float = float(self.rng.uniform(0.5, 0.8))

        self.halo_p: float = float(self.rng.beta(2, 18))

        self.organic_rate: float = float(self.rng.gamma(shape=2.0, scale=1.5))

        # ── Pre-compute organic catalog weights ──────────────────────
        self._catalog: list[dict[str, Any]] = product_catalog or []
        if self._catalog:
            pops = np.array(
                [p.get("popularity_score", 0.01) for p in self._catalog],
                dtype=np.float64,
            )
            pops = np.maximum(pops, 1e-6)
            self._catalog_weights = pops / pops.sum()
        else:
            self._catalog_weights = np.array([], dtype=np.float64)

        # ── Epoch counter (for seasonal month calculation) ───────────
        self._epoch: int = 0

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _current_month(self) -> int:
        """Map the current epoch to a 1-indexed month (1 = January).

        Each epoch represents approximately 1 week, so 4 epochs ≈ 1 month.
        The simulation starts in January of a hypothetical year.
        """
        return (self._epoch // 4) % 12 + 1

    def _sample_onset(self) -> float:
        """Sample a fatigue onset value for a new category exposure.

        Returns
        -------
        float
            Onset drawn from ``Normal(mu=8, sigma=2.5)``, floored at 1.0 to
            avoid degenerate immediate-fatigue scenarios.
        """
        return max(1.0, float(self.rng.normal(self.fatigue_onset_mu,
                                              self.fatigue_onset_sigma)))

    def _purchase_probability(
        self, base_p: float, touches: int, onset: float
    ) -> float:
        """Compute fatigue-adjusted purchase probability.

        Parameters
        ----------
        base_p : float
            Base purchase probability in [0, 1] derived from the ranked
            recommendation score (``final_score`` from the decision engine).
        touches : int
            Number of previous recommendation touches in this category.
        onset : float
            Current fatigue onset threshold for this customer-category pair.

        Returns
        -------
        float
            Adjusted probability: ``base_p * (1 - sigmoid((touches - onset) * steepness))``.
        """
        fatigue = _sigmoid((touches - onset) * self.fatigue_steepness)
        return base_p * (1.0 - fatigue)

    def _try_halo(
        self, category: str, customer_id: int, price_lookup: dict[str, float]
    ) -> dict[str, Any] | None:
        """Attempt a cross-category halo purchase.

        Parameters
        ----------
        category : str
            The category the customer just interacted with.
        customer_id : int
            The customer identifier.
        price_lookup : dict[str, float]
            Mapping from category to an average price, used to assign revenue
            to halo purchases.

        Returns
        -------
        dict or None
            A purchase dict if the halo fires, else ``None``.
        """
        if self.rng.random() >= self.halo_p:
            return None

        related = CATEGORY_ADJACENCY.get(category)
        if not related:
            return None

        halo_cat = related[int(self.rng.integers(len(related)))]
        price = price_lookup.get(halo_cat, 10.0)

        return {
            "customer_id": customer_id,
            "product_id": None,  # halo — specific product unattributed
            "category": halo_cat,
            "revenue": float(price),
        }

    def _organic_purchases(
        self, customer_id: int
    ) -> list[dict[str, Any]]:
        """Generate organic (non-recommendation-driven) purchases.

        The count is drawn from ``Poisson(organic_rate)`` each epoch.

        Returns
        -------
        list[dict]
            Organic purchase dicts.
        """
        if not self._catalog:
            return []

        count = int(self.rng.poisson(self.organic_rate))
        if count == 0:
            return []

        indices = self.rng.choice(
            len(self._catalog), size=count, replace=True, p=self._catalog_weights
        )
        purchases = []
        for idx in indices:
            prod = self._catalog[idx]
            purchases.append({
                "customer_id": customer_id,
                "product_id": prod["product_id"],
                "category": prod["category"],
                "revenue": float(prod.get("price", 10.0)),
            })
        return purchases

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def simulate_epoch(
        self,
        customer_states: dict[int, CustomerState],
        recommendations: dict[int, list[dict[str, Any]]],
    ) -> EpochResult:
        """Simulate one epoch (~1 week) of customer behavior.

        Parameters
        ----------
        customer_states : dict[int, CustomerState]
            Mutable state for every customer in the simulation, keyed by
            ``customer_id``.  Updated **in-place** during this call.
        recommendations : dict[int, list[dict]]
            Recommendations produced by the decision engine for this epoch.
            Keyed by ``customer_id``.  Each recommendation dict must have:

            - ``product_id`` (int)
            - ``category`` (str)
            - ``final_score`` (float): ranked score, used as base purchase
              probability after rescaling to [0, 1]
            - ``price`` (float): product price for revenue calculation

        Returns
        -------
        EpochResult
            Aggregated purchases, updated fatigue states, and total revenue.
        """
        month = self._current_month()
        self._epoch += 1

        result = EpochResult()

        # Build a quick category -> avg price lookup for halo purchases
        all_prices: dict[str, list[float]] = {}
        for recs in recommendations.values():
            for rec in recs:
                all_prices.setdefault(rec["category"], []).append(
                    rec.get("price", 10.0)
                )
        cat_avg_price = {
            cat: sum(ps) / len(ps) for cat, ps in all_prices.items()
        }

        all_customer_ids = set(customer_states.keys()) | set(recommendations.keys())

        for cid in all_customer_ids:
            state = customer_states.get(cid)
            if state is None:
                state = CustomerState()
                customer_states[cid] = state

            made_purchase = False
            recs = recommendations.get(cid, [])

            # ── Re-engagement check ──────────────────────────────────
            if state.dormant_epochs >= self.dormancy_threshold:
                if self.rng.random() < self.re_engagement_prob:
                    # Burst of 1-3 purchases from their recommendation set
                    burst_count = int(self.rng.integers(1, 4))
                    burst_pool = recs if recs else []
                    for _ in range(min(burst_count, max(len(burst_pool), 1))):
                        if burst_pool:
                            pick = burst_pool[
                                int(self.rng.integers(len(burst_pool)))
                            ]
                            s_mult = seasonal_multiplier(
                                pick["category"], month
                            )
                            revenue = float(pick.get("price", 10.0)) * s_mult
                            result.recommended_purchases.append({
                                "customer_id": cid,
                                "product_id": pick["product_id"],
                                "category": pick["category"],
                                "revenue": revenue,
                            })
                            result.revenue += revenue
                            made_purchase = True

                    # Reset fatigue with diminished onset
                    for cat in list(state.fatigue_touches.keys()):
                        state.fatigue_touches[cat] = 0
                        prev_onset = state.fatigue_onset.get(
                            cat, self._sample_onset()
                        )
                        state.fatigue_onset[cat] = (
                            prev_onset * self.re_engagement_decay
                        )
                    state.re_engagement_count += 1
                    state.dormant_epochs = 0

            # ── Process each recommendation ──────────────────────────
            for rec in recs:
                cat = rec["category"]
                score = rec["final_score"]
                price = rec.get("price", 10.0)

                # Rescale score to [0, 1] — scores from the decision engine
                # are raw dot-product affinities; clamp to valid probability.
                base_p = max(0.0, min(1.0, float(score)))

                # Apply seasonal modifier
                base_p *= seasonal_multiplier(cat, month)
                base_p = min(base_p, 1.0)

                # Initialize onset for new category exposures
                if cat not in state.fatigue_onset:
                    state.fatigue_onset[cat] = self._sample_onset()

                touches = state.fatigue_touches.get(cat, 0)
                onset = state.fatigue_onset[cat]

                p = self._purchase_probability(base_p, touches, onset)

                # Increment touch counter (recommendation received)
                state.fatigue_touches[cat] = touches + 1

                # Purchase decision
                if self.rng.random() < p:
                    result.recommended_purchases.append({
                        "customer_id": cid,
                        "product_id": rec["product_id"],
                        "category": cat,
                        "revenue": float(price),
                    })
                    result.revenue += float(price)
                    made_purchase = True

                # Halo effect (fires on interaction regardless of purchase)
                halo = self._try_halo(cat, cid, cat_avg_price)
                if halo is not None:
                    result.halo_purchases.append(halo)
                    result.revenue += halo["revenue"]
                    made_purchase = True

            # ── Organic purchases ────────────────────────────────────
            organic = self._organic_purchases(cid)
            if organic:
                result.organic_purchases.extend(organic)
                for op in organic:
                    result.revenue += op["revenue"]
                made_purchase = True

            # ── Dormancy tracking ────────────────────────────────────
            if made_purchase:
                state.dormant_epochs = 0
            else:
                state.dormant_epochs += 1

            # ── Snapshot fatigue for output ───────────────────────────
            result.fatigue_states[cid] = dict(state.fatigue_touches)

        return result

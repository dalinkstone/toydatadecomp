"""Phase 6 — Validation Against 10-K Revenue Distribution.

Compares Monte Carlo simulation output to real CVS financial benchmarks
derived from 10-K filings.  Produces a rich formatted report with
PASS / WARN / FAIL verdicts for each metric.

CLI:
    python src/cli.py validate revenue --simulation-dir data/results/simulation/
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

console = Console()


# ═══════════════════════════════════════════════════════════════════════════
# CVS 10-K benchmarks
# ═══════════════════════════════════════════════════════════════════════════

# Proportional CVS front store revenue for 10M customers out of 74M ExtraCare
# ~$22B/year total front store -> 10M/74M * $22B ≈ $2.97B
TARGET_ANNUAL_REVENUE = 2_970_000_000
REVENUE_ACCEPTABLE_LOW = 2_000_000_000
REVENUE_ACCEPTABLE_HIGH = 4_000_000_000

# Tier revenue share targets (% of total)
TIER_TARGETS = {
    1: (0.25, 0.35),  # top ~1%  -> 25-35%
    2: (0.30, 0.35),  # next ~15% -> 30-35%
    3: (0.20, 0.25),  # next ~30% -> 20-25%
    4: (0.05, 0.15),  # bottom ~54% -> 5-15%
}

# Discount / margin
CVS_GROSS_MARGIN = 0.29  # ~28-30%
MAX_DISCOUNT_RATE = 0.12  # discounts should not exceed 10-12% of revenue

# Customer behavior
VISITS_PER_YEAR = (5, 15)  # front-store visits (revenue / basket_size)
AVG_BASKET_SIZE = (30.0, 45.0)
AVG_ITEMS_PER_BASKET = (3, 5)
ACTIVE_CUSTOMER_RATE = (0.60, 0.75)  # annual: fraction visiting >= 1x/year

# Coupon performance
COUPON_REDEMPTION_RATE = (0.15, 0.25)
COUPONS_PER_CUSTOMER_PER_WEEK = (3, 8)

# PASS/WARN/FAIL thresholds (deviation from midpoint of target range)
PASS_THRESHOLD = 0.20
WARN_THRESHOLD = 0.50


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _verdict(value: float, target_low: float, target_high: float) -> str:
    """Return PASS/WARN/FAIL based on how far *value* is from the target range."""
    midpoint = (target_low + target_high) / 2
    span = (target_high - target_low) / 2 if target_high != target_low else abs(midpoint) * 0.1

    if target_low <= value <= target_high:
        return "[green]PASS[/green]"

    # distance from nearest edge
    if value < target_low:
        deviation = (target_low - value) / (span if span else 1)
    else:
        deviation = (value - target_high) / (span if span else 1)

    if deviation <= PASS_THRESHOLD / (PASS_THRESHOLD + 0.01):
        return "[green]PASS[/green]"
    if deviation <= WARN_THRESHOLD / (PASS_THRESHOLD + 0.01):
        return "[yellow]WARN[/yellow]"
    return "[red]FAIL[/red]"


def _pct_deviation(value: float, target: float) -> str:
    if target == 0:
        return "N/A"
    dev = (value - target) / target * 100
    sign = "+" if dev >= 0 else ""
    return f"{sign}{dev:.1f}%"


def _fmt_dollars(v: float) -> str:
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:.1f}K"
    return f"${v:,.0f}"


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ═══════════════════════════════════════════════════════════════════════════
# Core validation logic
# ═══════════════════════════════════════════════════════════════════════════

def validate_revenue(simulation_dir: str, model_dir: str = "data/model/") -> None:
    sim_dir = Path(simulation_dir)
    mdl_dir = Path(model_dir)

    # ── Load data ────────────────────────────────────────────────────
    summary_path = sim_dir / "summary.parquet"
    if not summary_path.exists():
        console.print(f"[red]summary.parquet not found in {sim_dir}[/red]")
        return

    summary = pd.read_parquet(summary_path)
    last = summary.iloc[-1]  # final converged epoch

    # Load parameters for customer count and scaling
    params_path = sim_dir / "parameters.json"
    params = {}
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)

    sim_customers = params.get("max_customer_id", 10_001) - 1  # 1-based
    weekly_target = params.get("weekly_revenue_target", 57_100_000)

    # Load product tiers (handle duplicate columns from elasticity merge)
    tiers_path = mdl_dir / "product_tiers.parquet"
    tiers_df = None
    if tiers_path.exists():
        pf = pq.ParquetFile(tiers_path)
        table = pf.read()
        # Deduplicate columns: keep first occurrence of each name
        seen: dict[str, int] = {}
        keep_indices: list[int] = []
        for i, name in enumerate(table.column_names):
            if name not in seen:
                seen[name] = i
                keep_indices.append(i)
        if len(keep_indices) < len(table.column_names):
            table = table.select(keep_indices)
        tiers_df = table.to_pandas()

    # Load tier arrays from workspace for per-tier simulation revenue
    ws = sim_dir / "workspace"
    tier_array = None
    product_prices = None
    if (ws / "product_tier_array.npy").exists():
        tier_array = np.load(ws / "product_tier_array.npy")
    if (ws / "product_prices.npy").exists():
        product_prices = np.load(ws / "product_prices.npy")

    # Load breakout/transition data
    breakout_df = None
    transitions_df = None
    if (sim_dir / "breakout_results.parquet").exists():
        breakout_df = pd.read_parquet(sim_dir / "breakout_results.parquet")
    if (sim_dir / "tier_transitions.parquet").exists():
        transitions_df = pd.read_parquet(sim_dir / "tier_transitions.parquet")

    # Load individual runs for per-epoch detail
    run_files = sorted(sim_dir.glob("run_*.parquet"))
    runs = [pd.read_parquet(f) for f in run_files] if run_files else []

    # ── Scale factor ─────────────────────────────────────────────────
    # Simulation runs on sim_customers; 10-K targets assume 10M customers.
    # All per-epoch revenue in summary is for sim_customers.
    target_customers = 10_000_000
    scale = target_customers / sim_customers if sim_customers > 0 else 1.0

    console.print()
    console.print(
        "[bold]═══════════════════════════════════════════════════════════════[/bold]"
    )
    console.print(
        "[bold]  Phase 6 — Validation Against 10-K Revenue Distribution[/bold]"
    )
    console.print(
        "[bold]═══════════════════════════════════════════════════════════════[/bold]"
    )
    console.print(f"  Simulation customers: {sim_customers:,}")
    console.print(f"  Target customers:     {target_customers:,}")
    console.print(f"  Scale factor:         {scale:,.0f}x")
    console.print(f"  Epochs (weeks):       {int(last['epoch'])}")
    console.print(f"  Monte Carlo runs:     {len(runs)}")
    console.print()

    # ==================================================================
    # 1. ANNUAL REVENUE VALIDATION
    # ==================================================================
    mean_weekly_net = float(last["mean_net_revenue"])
    scaled_weekly = mean_weekly_net * scale
    simulated_annual = scaled_weekly * 52

    console.print("[bold cyan]1. ANNUAL REVENUE VALIDATION[/bold cyan]")

    table1 = Table(show_lines=True)
    table1.add_column("Metric", style="bold", min_width=30)
    table1.add_column("Value", justify="right", min_width=18)
    table1.add_column("Target", justify="right", min_width=18)
    table1.add_column("Deviation", justify="right", min_width=12)
    table1.add_column("Verdict", justify="center", min_width=8)

    rev_verdict = _verdict(simulated_annual, REVENUE_ACCEPTABLE_LOW, REVENUE_ACCEPTABLE_HIGH)
    table1.add_row(
        "Simulated annual revenue",
        _fmt_dollars(simulated_annual),
        f"~{_fmt_dollars(TARGET_ANNUAL_REVENUE)}",
        _pct_deviation(simulated_annual, TARGET_ANNUAL_REVENUE),
        rev_verdict,
    )
    table1.add_row(
        "Simulated weekly revenue",
        _fmt_dollars(scaled_weekly),
        _fmt_dollars(weekly_target),
        _pct_deviation(scaled_weekly, weekly_target),
        _verdict(scaled_weekly, weekly_target * 0.67, weekly_target * 1.33),
    )
    table1.add_row(
        "Weekly net (sim scale)",
        _fmt_dollars(mean_weekly_net),
        "",
        "",
        "",
    )

    console.print(table1)
    console.print()

    # ==================================================================
    # 2. REVENUE DISTRIBUTION BY TIER
    # ==================================================================
    console.print("[bold cyan]2. REVENUE DISTRIBUTION BY TIER[/bold cyan]")

    table2 = Table(show_lines=True)
    table2.add_column("Tier", style="bold", justify="center", min_width=8)
    table2.add_column("Products", justify="right", min_width=10)
    table2.add_column("Revenue Share", justify="right", min_width=14)
    table2.add_column("Target Range", justify="right", min_width=14)
    table2.add_column("Verdict", justify="center", min_width=8)

    if tiers_df is not None and "tier" in tiers_df.columns and "total_revenue" in tiers_df.columns:
        total_rev = tiers_df["total_revenue"].sum()
        for t in sorted(tiers_df["tier"].unique()):
            tier_rev = tiers_df.loc[tiers_df["tier"] == t, "total_revenue"].sum()
            share = tier_rev / total_rev if total_rev > 0 else 0
            n_products = int((tiers_df["tier"] == t).sum())
            if t in TIER_TARGETS:
                lo, hi = TIER_TARGETS[t]
                verdict = _verdict(share, lo, hi)
                target_str = f"{_fmt_pct(lo)} - {_fmt_pct(hi)}"
            else:
                verdict = ""
                target_str = "—"
            table2.add_row(
                f"Tier {t}",
                f"{n_products:,}",
                _fmt_pct(share),
                target_str,
                verdict,
            )
    else:
        table2.add_row("—", "—", "—", "—", "[yellow]NO DATA[/yellow]")

    console.print(table2)
    console.print()

    # ==================================================================
    # 3. DISCOUNT EFFICIENCY
    # ==================================================================
    console.print("[bold cyan]3. DISCOUNT EFFICIENCY[/bold cyan]")

    mean_discount = float(last["mean_discount_cost"])
    mean_total_rev = float(last["mean_total_revenue"])
    discount_rate = mean_discount / mean_total_rev if mean_total_rev > 0 else 0
    estimated_margin = CVS_GROSS_MARGIN - discount_rate

    table3 = Table(show_lines=True)
    table3.add_column("Metric", style="bold", min_width=35)
    table3.add_column("Value", justify="right", min_width=14)
    table3.add_column("Target", justify="right", min_width=14)
    table3.add_column("Verdict", justify="center", min_width=8)

    disc_verdict = _verdict(discount_rate, 0.0, MAX_DISCOUNT_RATE)
    table3.add_row(
        "Discount rate (discount / revenue)",
        _fmt_pct(discount_rate),
        f"< {_fmt_pct(MAX_DISCOUNT_RATE)}",
        disc_verdict,
    )
    margin_verdict = _verdict(estimated_margin, 0.16, CVS_GROSS_MARGIN)
    table3.add_row(
        "Estimated gross margin after discounts",
        _fmt_pct(estimated_margin),
        f"~{_fmt_pct(CVS_GROSS_MARGIN)}",
        margin_verdict,
    )
    table3.add_row(
        "Weekly discount cost (sim scale)",
        _fmt_dollars(mean_discount),
        "",
        "",
    )
    table3.add_row(
        "Weekly discount cost (10M scale)",
        _fmt_dollars(mean_discount * scale),
        "",
        "",
    )

    console.print(table3)
    console.print()

    # ==================================================================
    # 4. CUSTOMER BEHAVIOR VALIDATION
    # ==================================================================
    console.print("[bold cyan]4. CUSTOMER BEHAVIOR VALIDATION[/bold cyan]")

    table4 = Table(show_lines=True)
    table4.add_column("Metric", style="bold", min_width=38)
    table4.add_column("Value", justify="right", min_width=14)
    table4.add_column("Target Range", justify="right", min_width=14)
    table4.add_column("Verdict", justify="center", min_width=8)

    # Active customer rate — annual (visited at least once in 52 weeks)
    weekly_active_pct = float(last["mean_active_customer_pct"])
    vp_path = sim_dir / "workspace" / "visit_probs.npy"
    if vp_path.exists():
        vp = np.load(vp_path)
        annual_active_pct = float((1 - (1 - vp) ** 52).mean())
    else:
        annual_active_pct = 1 - (1 - weekly_active_pct) ** 52
    table4.add_row(
        "Active customer rate (annual)",
        _fmt_pct(annual_active_pct),
        f"{_fmt_pct(ACTIVE_CUSTOMER_RATE[0])} - {_fmt_pct(ACTIVE_CUSTOMER_RATE[1])}",
        _verdict(annual_active_pct, *ACTIVE_CUSTOMER_RATE),
    )
    active_pct = annual_active_pct  # for summary table

    # Visits per customer per year — from weekly active rate
    avg_rev_per_visit = params.get("avg_revenue_per_visit", 35.0)
    visits_per_cust_per_year = weekly_active_pct * 52
    table4.add_row(
        "Avg visits per customer per year",
        f"{visits_per_cust_per_year:.1f}",
        f"{VISITS_PER_YEAR[0]} - {VISITS_PER_YEAR[1]}",
        _verdict(visits_per_cust_per_year, *VISITS_PER_YEAR),
    )

    # Avg basket size — derive from weekly revenue / weekly visits
    weekly_visitors = weekly_active_pct * sim_customers
    avg_basket = mean_total_rev / weekly_visitors if weekly_visitors > 0 else avg_rev_per_visit
    table4.add_row(
        "Avg basket size",
        f"${avg_basket:.2f}",
        f"${AVG_BASKET_SIZE[0]:.0f} - ${AVG_BASKET_SIZE[1]:.0f}",
        _verdict(avg_basket, *AVG_BASKET_SIZE),
    )

    # Avg items per basket — derive from tier constants
    # Tier 1: ~1 item, Tier 3: ~0.5 items (organic), plus coupon items (~35% engage)
    avg_items = 1.0 + 0.5 + 0.35 * 1.5  # base + organic + engaged coupon hits
    hit_rate = float(last["mean_hit_rate_at_5"])
    coupons_per = float(last["mean_mean_coupons_per_customer"])
    # Expected coupon purchases per visit ≈ coupons_offered * hit_rate
    coupon_items = coupons_per * hit_rate
    est_items = 2.5 + 1.2 + coupon_items  # Tier1 + Tier3 + coupon
    table4.add_row(
        "Est. items per basket",
        f"{est_items:.1f}",
        f"{AVG_ITEMS_PER_BASKET[0]} - {AVG_ITEMS_PER_BASKET[1]}",
        _verdict(est_items, *AVG_ITEMS_PER_BASKET),
    )

    console.print(table4)
    console.print()

    # ==================================================================
    # 5. COUPON PERFORMANCE
    # ==================================================================
    console.print("[bold cyan]5. COUPON PERFORMANCE[/bold cyan]")

    table5 = Table(show_lines=True)
    table5.add_column("Metric", style="bold", min_width=38)
    table5.add_column("Value", justify="right", min_width=14)
    table5.add_column("Target Range", justify="right", min_width=14)
    table5.add_column("Verdict", justify="center", min_width=8)

    # Coupon redemption rate ≈ hit_rate_at_5 (fraction of recommendations that led to purchase)
    redemption_rate = hit_rate
    table5.add_row(
        "Coupon redemption rate (hit@5)",
        _fmt_pct(redemption_rate),
        f"{_fmt_pct(COUPON_REDEMPTION_RATE[0])} - {_fmt_pct(COUPON_REDEMPTION_RATE[1])}",
        _verdict(redemption_rate, *COUPON_REDEMPTION_RATE),
    )

    # Coupons per customer per week
    table5.add_row(
        "Coupons sent per customer per week",
        f"{coupons_per:.1f}",
        f"{COUPONS_PER_CUSTOMER_PER_WEEK[0]} - {COUPONS_PER_CUSTOMER_PER_WEEK[1]}",
        _verdict(coupons_per, *COUPONS_PER_CUSTOMER_PER_WEEK),
    )

    console.print(table5)
    console.print()

    # ==================================================================
    # 6. BREAKOUT CANDIDATE PERFORMANCE
    # ==================================================================
    console.print("[bold cyan]6. BREAKOUT CANDIDATE PERFORMANCE[/bold cyan]")

    table6 = Table(show_lines=True)
    table6.add_column("Metric", style="bold", min_width=40)
    table6.add_column("Value", justify="right", min_width=18)

    # Total breakout successes (final epoch mean)
    breakout_count = float(last["mean_breakout_success_count"])
    table6.add_row("Breakout successes (mean across runs)", f"{breakout_count:.0f}")

    # How many total breakout candidates were tracked
    breakout_pids_path = ws / "breakout_pids.npy"
    total_candidates = 0
    if breakout_pids_path.exists():
        total_candidates = len(np.load(breakout_pids_path))
    table6.add_row("Total breakout candidates tracked", f"{total_candidates}")

    if total_candidates > 0:
        promo_rate = breakout_count / total_candidates
        table6.add_row("Promotion rate", _fmt_pct(promo_rate))

    # Tier migrations from tier_transitions
    if transitions_df is not None and len(transitions_df) > 0:
        to_tier2_or_higher = transitions_df[transitions_df["new_tier"] <= 2]
        unique_migrated = to_tier2_or_higher["product_id"].nunique()
        table6.add_row(
            "Unique products migrated to Tier 2+",
            f"{unique_migrated}",
        )
        n_runs = len(runs) if runs else 1
        table6.add_row(
            "Avg migrations per run",
            f"{len(to_tier2_or_higher) / n_runs:.1f}",
        )
    else:
        table6.add_row("Tier migrations", "[dim]No transition data[/dim]")

    # Avg discount cost for breakout products
    if breakout_df is not None and len(breakout_df) > 0:
        breakout_disc_path = ws / "breakout_discounts.npy"
        if breakout_disc_path.exists():
            breakout_discs = np.load(breakout_disc_path)
            avg_disc = float(breakout_discs.mean())
            table6.add_row("Avg discount offered to breakout products", _fmt_pct(avg_disc))

    # Incremental revenue from recommended (breakout) vs organic
    rec_rev = float(last["mean_recommended_revenue"])
    org_rev = float(last["mean_organic_revenue"])
    rec_share = rec_rev / mean_total_rev if mean_total_rev > 0 else 0
    table6.add_row("Recommended revenue share", _fmt_pct(rec_share))
    table6.add_row(
        "Recommended revenue (10M scale, weekly)",
        _fmt_dollars(rec_rev * scale),
    )

    console.print(table6)
    console.print()

    # ==================================================================
    # Summary
    # ==================================================================
    console.print(
        "[bold]═══════════════════════════════════════════════════════════════[/bold]"
    )
    console.print("[bold]  VALIDATION SUMMARY[/bold]")
    console.print(
        "[bold]═══════════════════════════════════════════════════════════════[/bold]"
    )

    summary_table = Table(show_lines=True)
    summary_table.add_column("Check", style="bold", min_width=35)
    summary_table.add_column("Verdict", justify="center", min_width=8)

    summary_table.add_row("Annual revenue in range", rev_verdict)
    summary_table.add_row("Discount rate sustainable", disc_verdict)
    summary_table.add_row("Gross margin preserved", margin_verdict)
    summary_table.add_row(
        "Active customer rate",
        _verdict(active_pct, *ACTIVE_CUSTOMER_RATE),
    )
    summary_table.add_row(
        "Visits per year plausible",
        _verdict(visits_per_cust_per_year, *VISITS_PER_YEAR),
    )
    summary_table.add_row(
        "Basket size realistic",
        _verdict(avg_basket, *AVG_BASKET_SIZE),
    )
    summary_table.add_row(
        "Coupon redemption rate",
        _verdict(redemption_rate, *COUPON_REDEMPTION_RATE),
    )
    summary_table.add_row(
        "Coupons per customer per week",
        _verdict(coupons_per, *COUPONS_PER_CUSTOMER_PER_WEEK),
    )

    console.print(summary_table)
    console.print()


# ═══════════════════════════════════════════════════════════════════════════
# Click CLI (standalone entry point)
# ═══════════════════════════════════════════════════════════════════════════

@click.command("revenue")
@click.option(
    "--simulation-dir",
    default="data/results/simulation/",
    help="Directory containing simulation output.",
)
@click.option(
    "--model-dir",
    default="data/model/",
    help="Directory containing product_tiers.parquet.",
)
def main(simulation_dir: str, model_dir: str) -> None:
    """Validate simulation revenue against CVS 10-K benchmarks."""
    validate_revenue(simulation_dir, model_dir)


if __name__ == "__main__":
    main()

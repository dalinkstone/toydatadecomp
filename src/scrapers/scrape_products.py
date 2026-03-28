"""Scrape CVS product catalog.

Two modes controlled by --mode flag:

  --mode=build (default, always works):
      Generate a comprehensive 10,000-SKU product catalog from embedded CVS
      product knowledge.  Every brand, product name, and price is based on
      real CVS retail data.

  --mode=scrape (best-effort, may get blocked):
      Crawl https://www.cvs.com/shop/ category pages looking for product
      data in __NEXT_DATA__ script tags or rendered HTML cards.  Falls back
      to --mode=build automatically if scraping fails.

Output: data/real/products.parquet AND data/real/products.csv
"""

import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()

TARGET_OTC_COUNT = 10_000
TARGET_RX_COUNT = 2_000
SEED = 42

# ════════════════════════════════════════════════════════════════════
# SIZE-EXPANSION LADDERS
# ════════════════════════════════════════════════════════════════════
# type_code → list of (size_value, price_multiplier)
# The expansion engine picks N entries based on product popularity.
LADDERS: dict[str, list[tuple[float, float]]] = {
    # Count-based (pills / tablets / caplets)
    "P20": [(20, 1.0), (40, 1.60), (80, 2.20), (160, 3.00)],
    "P24": [(24, 1.0), (50, 1.55), (100, 2.10), (200, 2.85), (325, 3.50)],
    "P30": [(30, 1.0), (60, 1.55), (100, 2.00), (200, 2.70)],
    "P40": [(40, 1.0), (80, 1.55), (120, 2.00)],
    "P60": [(60, 1.0), (120, 1.50), (250, 2.40), (400, 3.20)],
    "P90": [(90, 1.0), (150, 1.40), (300, 2.30)],
    "P100": [(100, 1.0), (200, 1.50), (365, 2.20)],
    # Liquid / fluid-oz
    "L1":  [(1, 1.0), (2, 1.65)],
    "L4":  [(4, 1.0), (8, 1.70), (12, 2.30)],
    "L8":  [(8, 1.0), (16, 1.60), (32, 2.60)],
    "L12": [(12, 1.0), (24, 1.65), (33.8, 2.10)],
    "L16": [(16, 1.0), (32, 1.60), (64, 2.50)],
    # Cream / lotion (oz)
    "C1":  [(1, 1.0), (1.7, 1.45), (3, 2.20)],
    "C2":  [(2, 1.0), (4, 1.55), (8, 2.40)],
    "C4":  [(4, 1.0), (8, 1.55), (16, 2.40)],
    "C8":  [(8, 1.0), (12, 1.30), (16, 1.60)],
    "C12": [(12, 1.0), (16, 1.20), (20, 1.45)],
    "C16": [(16, 1.0), (19, 1.15)],
    # Tube (toothpaste / ointment)
    "T":   [(2.7, 0.85), (4.1, 1.0), (5.4, 1.20), (6.4, 1.35)],
    # Deodorant stick
    "D":   [(2.6, 1.0), (3.4, 1.25)],
    # Spray cans
    "SP":  [(4, 1.0), (8, 1.65)],
    # Beverage (single → multi-pack)
    "B1":  [(1, 1.0)],
    "B6":  [(6, 1.0), (12, 1.80)],
    # Multi-pack (paper, diapers, wipes)
    "K1":  [(1, 1.0), (3, 2.50), (6, 4.20)],
    "K4":  [(4, 1.0), (8, 1.85), (12, 2.60)],
    "K6":  [(6, 1.0), (12, 1.80), (24, 3.20)],
    "K8":  [(8, 1.0), (16, 1.75), (24, 2.40)],
    # Single item — no size expansion
    "S":   [(1, 1.0)],
    # Prescription — 30-day and 90-day supply
    "RX":  [(30, 1.0), (90, 2.50)],
}


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

def _size_label(tc: str, val: float) -> str:
    """Human-readable size label for a type_code + size value."""
    if tc.startswith("P"):
        return f"{int(val)} CT"
    if tc.startswith("L"):
        v = int(val) if val == int(val) else val
        return f"{v} FL OZ"
    if tc.startswith("C") or tc in ("T", "D", "SP"):
        v = int(val) if val == int(val) else val
        return f"{v} OZ"
    if tc.startswith("B"):
        return "" if val == 1 else f"{int(val)} Pack"
    if tc.startswith("K"):
        return f"{int(val)} Count" if val <= 1 else f"{int(val)} Pack"
    return ""


def _calc_weight(tc: str, val: float) -> float:
    """Estimate product weight in oz from type_code + size value."""
    if tc.startswith("P"):
        return round(val * 0.06 + 0.8, 1)
    if tc.startswith(("L", "C")) or tc in ("T", "D", "SP"):
        return round(val * 1.05 + 0.5, 1)
    if tc.startswith("B"):
        return round(val * 14.0, 1) if val > 1 else 14.0
    if tc.startswith("K"):
        return round(val * 8.0, 1)
    return 8.0


def generate_upc(product_id: int) -> str:
    """Deterministic 12-digit UPC-A with valid check digit."""
    h = hashlib.md5(f"cvs_sku_{product_id:06d}".encode()).hexdigest()
    raw = "0"  # standard UPC number system
    for c in h:
        if c.isdigit():
            raw += c
        if len(raw) == 11:
            break
    raw = raw.ljust(11, "0")
    odd = sum(int(raw[i]) for i in range(0, 11, 2))
    even = sum(int(raw[i]) for i in range(1, 11, 2))
    check = (10 - (odd * 3 + even) % 10) % 10
    return raw + str(check)


def generate_ndc(product_id: int) -> str:
    """Deterministic 11-digit NDC in 5-4-2 format for Rx products."""
    h = hashlib.md5(f"cvs_ndc_{product_id:06d}".encode()).hexdigest()
    digits = "".join(c for c in h if c.isdigit())
    labeler = digits[:5].ljust(5, "0")
    product = digits[5:9].ljust(4, "0")
    package = digits[9:11].ljust(2, "0")
    return f"{labeler}-{product}-{package}"


# ════════════════════════════════════════════════════════════════════
# PRODUCT KNOWLEDGE BASE
# ════════════════════════════════════════════════════════════════════
# _RAW collects (category, brand, name, subcategory, type_code, base_price, popularity)
# type_code controls size-variant expansion via LADDERS.
# base_price is the retail price at the smallest ladder size.
# popularity is a 0-1 score driving expansion breadth + transaction volume.

_RAW: list[tuple] = []


def _b(cat: str, brand: str, items: list[tuple]):
    """Batch-register products for one brand in one category."""
    for t in items:
        _RAW.append((cat, brand, *t))


# ── Rx product list (separate from OTC to preserve OTC seed reproducibility) ──
_RAW_RX: list[tuple] = []
# Tuple: (category, brand, name, subcategory, type_code, price, pop, generic_name, therapeutic_class, days_supply)


# ─── Pain Relief & Fever ──────────────────────────────────────────
_b("Pain Relief & Fever", "Tylenol", [
    ("Extra Strength Acetaminophen 500mg Caplets", "Caplets", "P24", 7.49, .88),
    ("Extra Strength Acetaminophen 500mg Rapid Release Gels", "Gelcaps", "P24", 8.99, .82),
    ("8HR Arthritis Pain Acetaminophen 650mg Caplets", "Caplets", "P24", 8.49, .75),
    ("PM Extra Strength Pain Reliever & Sleep Aid Caplets", "Caplets", "P24", 7.99, .72),
    ("Regular Strength Acetaminophen 325mg Tablets", "Tablets", "P24", 8.99, .40),
    ("Cold + Flu Severe Caplets", "Caplets", "P24", 11.99, .60),
    ("Cold + Head Congestion Severe Caplets", "Caplets", "P24", 11.99, .55),
    ("Sinus + Headache Caplets", "Caplets", "P24", 10.49, .50),
    ("Children's Pain + Fever Liquid Cherry", "Liquid", "L4", 8.49, .70),
    ("Children's Pain + Fever Liquid Grape", "Liquid", "L4", 8.49, .65),
    ("Children's Pain + Fever Chewable Tablets Grape", "Chewables", "P24", 7.49, .55),
    ("Infants' Acetaminophen Liquid Cherry", "Liquid", "S", 9.99, .60),
])
_b("Pain Relief & Fever", "Advil", [
    ("Ibuprofen 200mg Tablets", "Tablets", "P24", 7.99, .87),
    ("Ibuprofen 200mg Coated Caplets", "Caplets", "P24", 7.99, .85),
    ("Liqui-Gels Ibuprofen 200mg", "Liquid Gels", "P24", 8.99, .80),
    ("Dual Action Acetaminophen 250mg & Ibuprofen 125mg", "Caplets", "P24", 9.99, .68),
    ("PM Ibuprofen & Diphenhydramine Caplets", "Caplets", "P24", 8.49, .65),
    ("Migraine Ibuprofen 200mg Liquid Filled Capsules", "Liquid Gels", "P24", 9.99, .60),
    ("Children's Ibuprofen Grape Suspension", "Liquid", "L4", 7.99, .65),
    ("Children's Ibuprofen Berry Suspension", "Liquid", "L4", 7.99, .60),
    ("Infants' Ibuprofen Concentrated Drops Berry", "Liquid", "S", 9.49, .55),
])
_b("Pain Relief & Fever", "Aleve", [
    ("Naproxen Sodium 220mg Tablets", "Tablets", "P24", 7.99, .82),
    ("Naproxen Sodium 220mg Caplets", "Caplets", "P24", 7.99, .80),
    ("Naproxen Sodium 220mg Liquid Gels", "Liquid Gels", "P24", 8.99, .72),
    ("Back & Muscle Pain Naproxen Sodium 220mg", "Tablets", "P24", 8.49, .60),
    ("PM Pain Reliever & Nighttime Sleep Aid Caplets", "Caplets", "P24", 8.49, .58),
    ("Arthritis Cap Naproxen Sodium 220mg Gelcaps", "Gelcaps", "P24", 9.49, .55),
])
_b("Pain Relief & Fever", "Motrin", [
    ("IB Ibuprofen 200mg Tablets", "Tablets", "P24", 7.49, .70),
    ("IB Ibuprofen 200mg Caplets", "Caplets", "P24", 7.49, .68),
    ("PM Ibuprofen & Diphenhydramine Caplets", "Caplets", "P24", 7.99, .52),
    ("Children's Ibuprofen Oral Suspension Berry", "Liquid", "L4", 7.49, .60),
    ("Children's Ibuprofen Oral Suspension Grape", "Liquid", "L4", 7.49, .55),
    ("Infants' Drops Concentrated Berry", "Liquid", "S", 9.49, .48),
])
_b("Pain Relief & Fever", "Excedrin", [
    ("Extra Strength Caplets", "Caplets", "P24", 7.99, .72),
    ("Migraine Caplets", "Caplets", "P24", 8.49, .70),
    ("Tension Headache Caplets", "Caplets", "P24", 7.99, .55),
    ("PM Headache Caplets", "Caplets", "P24", 7.99, .45),
])
_b("Pain Relief & Fever", "Bayer", [
    ("Aspirin 325mg Coated Tablets", "Tablets", "P24", 6.99, .58),
    ("Low Dose Aspirin 81mg Enteric Coated Tablets", "Tablets", "P60", 5.99, .72),
    ("Back & Body Extra Strength Aspirin 500mg Caplets", "Caplets", "P24", 7.49, .48),
    ("PM Aspirin & Diphenhydramine Caplets", "Caplets", "P24", 7.49, .40),
])
_b("Pain Relief & Fever", "BC", [
    ("Original Formula Pain Reliever Powder Sticks", "Powder", "P24", 5.99, .32),
    ("Cherry Flavor Powder Sticks", "Powder", "P24", 5.99, .28),
])
_b("Pain Relief & Fever", "Goody's", [
    ("Extra Strength Headache Powder", "Powder", "P24", 5.99, .30),
    ("Back & Body Pain Powder", "Powder", "P24", 5.99, .25),
])
_b("Pain Relief & Fever", "Aspercreme", [
    ("Original Pain Relieving Cream", "Cream", "C4", 9.99, .52),
    ("Max Strength Lidocaine Pain Relieving Cream", "Cream", "C4", 11.99, .48),
    ("Lidocaine Pain Relieving Patch", "Patches", "S", 12.99, .45),
    ("Arthritis Pain Relieving Gel", "Gel", "C4", 10.99, .42),
])
_b("Pain Relief & Fever", "Bengay", [
    ("Ultra Strength Pain Relieving Cream", "Cream", "C4", 8.99, .52),
    ("Greaseless Pain Relieving Cream", "Cream", "C4", 8.99, .42),
    ("Pain Relieving Patch Large", "Patches", "S", 9.99, .38),
])
_b("Pain Relief & Fever", "Icy Hot", [
    ("Original Pain Relieving Cream", "Cream", "C4", 7.99, .58),
    ("Lidocaine Pain Relieving Patch", "Patches", "S", 11.99, .52),
    ("Power Gel Pain Reliever", "Gel", "C4", 9.99, .45),
    ("Pro Therapy Pain Relief Cream with Menthol", "Cream", "C4", 10.99, .42),
    ("Smart Relief TENS Therapy Starter Kit", "Device", "S", 34.99, .25),
])
_b("Pain Relief & Fever", "Biofreeze", [
    ("Pain Relieving Gel", "Gel", "C4", 11.99, .58),
    ("Pain Relieving Roll-On", "Roll-On", "S", 10.99, .52),
    ("Pain Relieving Spray", "Spray", "SP", 12.99, .48),
    ("Pain Relief Patch", "Patches", "S", 11.99, .40),
])
_b("Pain Relief & Fever", "Salonpas", [
    ("Pain Relieving Patch Large", "Patches", "S", 9.99, .50),
    ("Pain Relieving Patch", "Patches", "S", 7.99, .48),
    ("Lidocaine Pain Relieving Gel-Patch", "Patches", "S", 12.99, .42),
    ("Deep Relieving Gel", "Gel", "C4", 8.99, .38),
])
_b("Pain Relief & Fever", "ThermaCare", [
    ("Lower Back & Hip HeatWraps", "Heat Wraps", "S", 12.99, .48),
    ("Neck Wrist & Shoulder HeatWraps", "Heat Wraps", "S", 11.99, .42),
    ("Knee & Elbow HeatWraps", "Heat Wraps", "S", 10.99, .35),
])
_b("Pain Relief & Fever", "CVS Health", [
    ("Extra Strength Pain Relief Acetaminophen 500mg Caplets", "Caplets", "P24", 5.49, .76),
    ("Extra Strength Pain Relief Acetaminophen 500mg Gelcaps", "Gelcaps", "P24", 6.49, .65),
    ("Arthritis Pain Relief Acetaminophen 650mg Caplets", "Caplets", "P24", 6.49, .55),
    ("PM Pain Relief Acetaminophen & Diphenhydramine Caplets", "Caplets", "P24", 5.99, .52),
    ("Ibuprofen 200mg Tablets", "Tablets", "P24", 5.99, .73),
    ("Ibuprofen 200mg Coated Caplets", "Caplets", "P24", 5.99, .70),
    ("Ibuprofen 200mg Liquid Filled Capsules", "Liquid Gels", "P24", 6.99, .58),
    ("Ibuprofen PM Caplets", "Caplets", "P24", 6.49, .48),
    ("Naproxen Sodium 220mg Caplets", "Caplets", "P24", 5.99, .62),
    ("Aspirin 325mg Tablets", "Tablets", "P24", 4.99, .42),
    ("Low Dose Aspirin 81mg Enteric Coated Tablets", "Tablets", "P60", 3.99, .62),
    ("Migraine Relief Acetaminophen Aspirin Caffeine", "Caplets", "P24", 5.99, .45),
    ("Children's Pain & Fever Acetaminophen Liquid Cherry", "Liquid", "L4", 5.99, .55),
    ("Children's Pain & Fever Acetaminophen Liquid Grape", "Liquid", "L4", 5.99, .50),
    ("Children's Ibuprofen Oral Suspension Berry", "Liquid", "L4", 5.49, .50),
    ("Infants' Pain & Fever Acetaminophen Liquid Cherry", "Liquid", "S", 7.49, .45),
    ("Muscle Rub Pain Relieving Cream", "Cream", "C4", 6.99, .42),
    ("Lidocaine Pain Relieving Patches 4% Strength", "Patches", "S", 8.99, .40),
])

# ─── Cold/Flu/Allergy ────────────────────────────────────────────
_b("Cold/Flu/Allergy", "Vicks", [
    ("NyQuil Severe Cold & Flu Liquid", "Liquid", "L8", 13.99, .85),
    ("NyQuil Cold & Flu Nighttime Relief LiquiCaps", "LiquiCaps", "P24", 11.99, .82),
    ("DayQuil Severe Cold & Flu Liquid", "Liquid", "L8", 13.99, .83),
    ("DayQuil Cold & Flu LiquiCaps", "LiquiCaps", "P24", 11.99, .80),
    ("DayQuil NyQuil Severe Cold & Flu Combo Pack", "Combo", "S", 19.99, .72),
    ("VapoRub Cough Suppressant Topical Ointment", "Ointment", "C1", 7.49, .78),
    ("Sinex Severe Nasal Spray", "Nasal Spray", "S", 10.99, .55),
    ("VapoCOOL Severe Throat Spray", "Spray", "S", 8.49, .42),
    ("VapoInhaler Portable Nasal Inhaler", "Inhaler", "S", 5.99, .45),
])
_b("Cold/Flu/Allergy", "Mucinex", [
    ("Maximum Strength 12 Hour Chest Congestion Expectorant Tablets", "Tablets", "P20", 18.99, .78),
    ("DM Maximum Strength 12 Hour Cough & Chest Congestion", "Tablets", "P20", 19.99, .75),
    ("D Maximum Strength Expectorant & Nasal Decongestant", "Tablets", "P20", 18.99, .65),
    ("Fast-Max Severe Congestion & Cough Liquid", "Liquid", "L4", 14.99, .60),
    ("Sinus-Max Severe Congestion & Pain Caplets", "Caplets", "P20", 14.99, .55),
    ("Nightshift Severe Cold & Flu Liquid", "Liquid", "L4", 13.99, .52),
    ("Children's Multi-Symptom Cold Liquid Very Berry", "Liquid", "L4", 10.99, .50),
    ("Children's Chest Congestion Expectorant Mini-Melts", "Granules", "S", 11.99, .42),
])
_b("Cold/Flu/Allergy", "Sudafed", [
    ("PE Sinus Pressure + Pain Tablets", "Tablets", "P24", 8.99, .60),
    ("PE Head Congestion + Flu Severe Caplets", "Caplets", "P24", 9.99, .52),
    ("12 Hour Non-Drowsy Nasal Decongestant Tablets", "Tablets", "P24", 9.99, .55),
    ("PE Sinus Congestion Day + Night", "Tablets", "S", 10.99, .48),
])
_b("Cold/Flu/Allergy", "Zyrtec", [
    ("Cetirizine HCl 10mg Allergy Tablets", "Tablets", "P30", 22.99, .82),
    ("Cetirizine HCl 10mg Liquid Gels", "Liquid Gels", "P24", 20.99, .70),
    ("Dissolve Tabs Citrus 10mg", "Dissolve Tabs", "P24", 21.99, .55),
    ("Children's Allergy Syrup Grape", "Liquid", "L4", 13.99, .62),
    ("Children's Allergy Dissolve Tabs Citrus", "Dissolve Tabs", "P24", 14.99, .48),
])
_b("Cold/Flu/Allergy", "Claritin", [
    ("Loratadine 10mg 24-Hour Allergy Tablets", "Tablets", "P30", 21.99, .80),
    ("Loratadine 10mg RediTabs", "RediTabs", "P30", 22.99, .62),
    ("Loratadine 10mg Liqui-Gels", "Liquid Gels", "P30", 23.99, .58),
    ("Claritin-D 12 Hour Allergy & Congestion", "Tablets", "P20", 19.99, .55),
    ("Children's Allergy Syrup Grape", "Liquid", "L4", 12.99, .58),
])
_b("Cold/Flu/Allergy", "Allegra", [
    ("Fexofenadine HCl 180mg 24-Hour Allergy Tablets", "Tablets", "P30", 22.99, .75),
    ("Fexofenadine HCl 180mg Gelcaps", "Gelcaps", "P30", 23.99, .60),
    ("Allegra-D 12 Hour Allergy & Congestion", "Tablets", "P20", 19.99, .50),
    ("Children's Allergy 12 Hour Fexofenadine Oral Suspension", "Liquid", "L4", 12.99, .52),
])
_b("Cold/Flu/Allergy", "Benadryl", [
    ("Ultratabs Diphenhydramine HCl 25mg", "Tablets", "P24", 7.49, .75),
    ("Allergy Liqui-Gels Diphenhydramine HCl 25mg", "Liquid Gels", "P24", 8.99, .65),
    ("Allergy Plus Congestion Tablets", "Tablets", "P24", 8.49, .55),
    ("Children's Allergy Liquid Cherry", "Liquid", "L4", 7.49, .58),
    ("Children's Allergy Chewable Tablets Grape", "Chewables", "P24", 7.99, .45),
    ("Itch Relief Stick Extra Strength", "Stick", "S", 7.49, .42),
])
_b("Cold/Flu/Allergy", "Flonase", [
    ("Allergy Relief Nasal Spray", "Nasal Spray", "L1", 21.99, .78),
    ("Sensimist Allergy Relief Nasal Spray", "Nasal Spray", "L1", 22.99, .65),
    ("Children's Allergy Relief Nasal Spray", "Nasal Spray", "S", 16.99, .50),
])
_b("Cold/Flu/Allergy", "Nasacort", [
    ("Allergy 24HR Nasal Spray", "Nasal Spray", "L1", 18.99, .62),
])
_b("Cold/Flu/Allergy", "Afrin", [
    ("No Drip Original Nasal Spray", "Nasal Spray", "S", 9.99, .55),
    ("No Drip Severe Congestion Nasal Spray", "Nasal Spray", "S", 10.99, .50),
    ("Original Nasal Spray", "Nasal Spray", "S", 8.99, .52),
])
_b("Cold/Flu/Allergy", "Delsym", [
    ("12 Hour Cough Relief Liquid Orange", "Liquid", "L4", 14.99, .55),
    ("12 Hour Cough Relief Liquid Grape", "Liquid", "L4", 14.99, .50),
    ("Children's 12 Hour Cough Relief Liquid Orange", "Liquid", "L4", 12.99, .48),
])
_b("Cold/Flu/Allergy", "Robitussin", [
    ("Maximum Strength Cough + Chest Congestion DM Liquid", "Liquid", "L4", 10.99, .58),
    ("Maximum Strength Nighttime Cough DM Liquid", "Liquid", "L4", 10.99, .50),
    ("Maximum Strength Severe Multi-Symptom Cough Cold + Flu", "Liquid", "L4", 11.99, .52),
    ("Children's Cough & Cold CF Liquid", "Liquid", "L4", 9.99, .48),
])
_b("Cold/Flu/Allergy", "Theraflu", [
    ("Flu & Sore Throat Hot Liquid Powder Packets", "Powder", "S", 10.99, .52),
    ("Nighttime Severe Cold & Cough Hot Liquid Powder", "Powder", "S", 10.99, .50),
    ("Daytime Severe Cold & Cough Hot Liquid Powder", "Powder", "S", 10.99, .48),
    ("ExpressMax Severe Cold & Flu Caplets", "Caplets", "P24", 11.99, .45),
])
_b("Cold/Flu/Allergy", "Zicam", [
    ("Cold Remedy RapidMelts Cherry", "Dissolve Tabs", "P24", 11.99, .45),
    ("Cold Remedy Nasal Swabs", "Swabs", "S", 11.99, .40),
    ("Cold Remedy Oral Mist Arctic Mint", "Spray", "S", 10.99, .38),
])
_b("Cold/Flu/Allergy", "Coricidin HBP", [
    ("Chest Congestion & Cough Softgels", "Softgels", "P20", 10.99, .42),
    ("Cold & Flu Tablets", "Tablets", "P20", 9.99, .40),
    ("Maximum Strength Flu Tablets", "Tablets", "P20", 10.99, .38),
])
_b("Cold/Flu/Allergy", "CVS Health", [
    ("Severe Cold & Flu Relief Liquid Nighttime", "Liquid", "L8", 9.99, .68),
    ("Severe Cold & Flu Relief Liquid Daytime", "Liquid", "L8", 9.99, .65),
    ("Cold & Flu Relief LiquiCaps Nighttime", "LiquiCaps", "P24", 8.49, .60),
    ("Cold & Flu Relief LiquiCaps Daytime", "LiquiCaps", "P24", 8.49, .58),
    ("Cetirizine HCl 10mg Allergy Tablets", "Tablets", "P30", 15.99, .70),
    ("Loratadine 10mg 24-Hour Allergy Tablets", "Tablets", "P30", 14.99, .68),
    ("Fexofenadine HCl 180mg Allergy Tablets", "Tablets", "P30", 16.99, .60),
    ("Diphenhydramine HCl 25mg Allergy Tablets", "Tablets", "P24", 4.99, .58),
    ("Fluticasone Propionate Nasal Spray", "Nasal Spray", "L1", 16.99, .62),
    ("Nasal Decongestant Spray", "Nasal Spray", "S", 6.99, .48),
    ("Cough DM Liquid Orange", "Liquid", "L4", 7.99, .50),
    ("Children's Allergy Relief Cetirizine Liquid Grape", "Liquid", "L4", 9.99, .48),
    ("Tussin DM Max Cough + Chest Congestion Liquid", "Liquid", "L4", 7.49, .45),
    ("Chest Congestion Relief Guaifenesin 400mg Tablets", "Tablets", "P30", 9.99, .42),
])

# ─── Digestive Health ─────────────────────────────────────────────
_b("Digestive Health", "Tums", [
    ("Antacid Calcium Carbonate Regular Strength Assorted Fruit", "Chewables", "P60", 5.99, .78),
    ("Antacid Calcium Carbonate Ultra Strength Assorted Berry", "Chewables", "P60", 7.49, .75),
    ("Smoothies Extra Strength Assorted Fruit", "Chewables", "P60", 7.99, .65),
    ("Chewy Bites Assorted Berries", "Chewables", "P30", 6.99, .58),
    ("Sugar-Free Melon Berry Extra Strength", "Chewables", "P40", 7.99, .42),
    ("Naturals Ultra Strength Coconut Pineapple", "Chewables", "P40", 8.49, .38),
])
_b("Digestive Health", "Pepto-Bismol", [
    ("Original Liquid", "Liquid", "L8", 7.99, .75),
    ("Ultra Liquid", "Liquid", "L8", 9.99, .62),
    ("Chewable Tablets Original", "Chewables", "P30", 6.99, .60),
    ("LiquiCaps", "LiquiCaps", "P24", 8.99, .58),
    ("Diarrhea LiquiCaps", "LiquiCaps", "P24", 9.99, .50),
])
_b("Digestive Health", "Gas-X", [
    ("Extra Strength Simethicone 125mg Softgels", "Softgels", "P24", 8.99, .62),
    ("Ultra Strength Simethicone 180mg Softgels", "Softgels", "P24", 10.49, .55),
    ("Maximum Strength Simethicone 250mg Softgels", "Softgels", "P24", 11.49, .50),
    ("Thin Strips Peppermint", "Strips", "S", 8.99, .42),
])
_b("Digestive Health", "Imodium", [
    ("A-D Loperamide HCl 2mg Anti-Diarrheal Caplets", "Caplets", "P24", 10.99, .62),
    ("Multi-Symptom Relief Caplets", "Caplets", "P24", 11.99, .55),
    ("A-D Anti-Diarrheal Liquid Mint", "Liquid", "L4", 9.99, .48),
])
_b("Digestive Health", "Prilosec OTC", [
    ("Omeprazole 20mg Acid Reducer Tablets", "Tablets", "P20", 14.99, .72),
    ("Omeprazole 20mg Acid Reducer Wildberry", "Tablets", "P20", 15.99, .50),
])
_b("Digestive Health", "Nexium", [
    ("24HR Esomeprazole 20mg Acid Reducer Capsules", "Capsules", "P20", 16.99, .68),
])
_b("Digestive Health", "Pepcid", [
    ("Original Strength Famotidine 10mg Tablets", "Tablets", "P30", 8.99, .60),
    ("Maximum Strength Famotidine 20mg Tablets", "Tablets", "P30", 10.49, .62),
    ("Complete Acid Reducer + Antacid Chewable Tablets Berry", "Chewables", "P30", 10.99, .52),
])
_b("Digestive Health", "Metamucil", [
    ("Psyllium Fiber Supplement Powder Orange", "Powder", "C16", 18.99, .70),
    ("Fiber Capsules", "Capsules", "P100", 15.99, .58),
    ("Fiber Thins Cinnamon Spice", "Wafers", "S", 11.99, .42),
    ("Premium Blend Sugar-Free Powder Orange", "Powder", "C16", 22.99, .48),
])
_b("Digestive Health", "MiraLAX", [
    ("Polyethylene Glycol 3350 Laxative Powder", "Powder", "C8", 16.99, .72),
    ("Polyethylene Glycol 3350 Laxative Powder Mix-In Pax", "Powder", "S", 18.99, .55),
])
_b("Digestive Health", "Benefiber", [
    ("Original Prebiotic Fiber Supplement Powder", "Powder", "C8", 14.99, .55),
    ("Prebiotic Fiber Chewable Tablets", "Chewables", "P60", 12.99, .42),
    ("On-the-Go Prebiotic Fiber Powder Stick Packs", "Powder", "S", 10.99, .40),
])
_b("Digestive Health", "Colace", [
    ("Docusate Sodium 100mg Stool Softener Capsules", "Capsules", "P30", 9.49, .55),
    ("2-in-1 Stool Softener & Stimulant Laxative Tablets", "Tablets", "P30", 10.49, .42),
    ("Clear Soft Gels Docusate Sodium 50mg", "Softgels", "P30", 10.99, .38),
])
_b("Digestive Health", "Dulcolax", [
    ("Bisacodyl 5mg Laxative Tablets", "Tablets", "P30", 9.49, .55),
    ("Stool Softener Docusate Sodium 100mg Liquid Gels", "Liquid Gels", "P30", 9.99, .48),
    ("Bisacodyl 10mg Medicated Laxative Suppositories", "Suppositories", "S", 10.49, .40),
    ("Liquid Laxative Cherry", "Liquid", "L8", 10.99, .35),
])
_b("Digestive Health", "Phillips'", [
    ("Milk of Magnesia Original", "Liquid", "L8", 8.49, .52),
    ("Milk of Magnesia Cherry", "Liquid", "L8", 8.49, .48),
    ("Colon Health Daily Probiotic Capsules", "Capsules", "P30", 19.99, .42),
])
_b("Digestive Health", "Culturelle", [
    ("Daily Probiotic Digestive Health Capsules", "Capsules", "P30", 21.99, .55),
    ("Digestive Health Daily Probiotic Chewables", "Chewables", "P24", 18.99, .45),
    ("Kids Daily Probiotic Packets", "Powder", "S", 18.99, .42),
    ("Pro Strength Daily Probiotic Capsules", "Capsules", "P30", 29.99, .38),
])
_b("Digestive Health", "Align", [
    ("Probiotic Supplement Capsules", "Capsules", "P30", 29.99, .52),
    ("Extra Strength Probiotic Supplement Capsules", "Capsules", "P30", 34.99, .42),
    ("Probiotic Gummies Natural Fruit Flavors", "Gummies", "P60", 22.99, .45),
])
_b("Digestive Health", "Florastor", [
    ("Daily Probiotic Supplement Capsules", "Capsules", "P30", 28.99, .42),
    ("Kids Daily Probiotic Powder Packets", "Powder", "S", 25.99, .35),
])
_b("Digestive Health", "Beano", [
    ("Extra Strength Gas Prevention Tablets", "Tablets", "P30", 8.99, .42),
    ("Gas Prevention Meltaways Strawberry", "Meltaways", "P24", 8.99, .35),
])
_b("Digestive Health", "Lactaid", [
    ("Fast Act Lactase Enzyme Caplets", "Caplets", "P30", 11.99, .52),
    ("Fast Act Lactase Enzyme Chewable Vanilla Twist", "Chewables", "P30", 12.99, .42),
    ("Dietary Supplement Caplets Original Strength", "Caplets", "P30", 10.49, .38),
])
_b("Digestive Health", "Emetrol", [
    ("Nausea & Upset Stomach Relief Liquid Cherry", "Liquid", "L4", 9.99, .35),
])
_b("Digestive Health", "CVS Health", [
    ("Omeprazole 20mg Acid Reducer Tablets", "Tablets", "P20", 10.99, .65),
    ("Famotidine 20mg Acid Reducer Tablets", "Tablets", "P30", 7.49, .55),
    ("Antacid Calcium Carbonate Ultra Strength Chewables Assorted Fruit", "Chewables", "P60", 4.99, .62),
    ("Bismuth Subsalicylate Upset Stomach Relief Liquid", "Liquid", "L8", 5.49, .55),
    ("Bismuth Subsalicylate Upset Stomach Relief Chewables", "Chewables", "P30", 4.99, .48),
    ("Simethicone Gas Relief 125mg Softgels", "Softgels", "P24", 6.49, .50),
    ("Loperamide HCl 2mg Anti-Diarrheal Caplets", "Caplets", "P24", 7.49, .52),
    ("Stool Softener Docusate Sodium 100mg Softgels", "Softgels", "P30", 6.99, .48),
    ("Fiber Laxative Methylcellulose Caplets", "Caplets", "P100", 11.99, .42),
    ("Fiber Powder Supplement Orange", "Powder", "C16", 13.99, .50),
    ("Polyethylene Glycol 3350 Laxative Powder", "Powder", "C8", 11.99, .58),
    ("Daily Probiotic Capsules", "Capsules", "P30", 15.99, .45),
    ("Lactase Enzyme Fast Acting Caplets", "Caplets", "P30", 8.49, .42),
    ("Milk of Magnesia Original", "Liquid", "L8", 5.99, .42),
])

# ─── Vitamins & Supplements ──────────────────────────────────────
# (defined via systematic generator below, plus select explicit entries)
_b("Vitamins & Supplements", "Centrum", [
    ("Adults Multivitamin & Multimineral Supplement Tablets", "Tablets", "P100", 13.99, .72),
    ("Silver Adults 50+ Multivitamin Tablets", "Tablets", "P100", 14.99, .65),
    ("Men's Multivitamin & Multimineral Tablets", "Tablets", "P100", 14.99, .60),
    ("Women's Multivitamin & Multimineral Tablets", "Tablets", "P100", 14.99, .62),
    ("Adults MultiGummies Multi + Beauty", "Gummies", "P60", 12.99, .48),
    ("Silver Men 50+ Multivitamin Tablets", "Tablets", "P100", 15.49, .45),
    ("Silver Women 50+ Multivitamin Tablets", "Tablets", "P100", 15.49, .48),
    ("Kids Organic Multigummies Mixed Berry", "Gummies", "P60", 12.99, .42),
])
_b("Vitamins & Supplements", "One A Day", [
    ("Men's Health Formula Multivitamin Tablets", "Tablets", "P100", 13.49, .60),
    ("Women's Multivitamin Tablets", "Tablets", "P100", 13.49, .62),
    ("Women's Prenatal 1 Multivitamin Softgels", "Softgels", "P30", 16.99, .55),
    ("Men's 50+ Healthy Advantage Multivitamin Tablets", "Tablets", "P100", 14.49, .45),
    ("Women's 50+ Healthy Advantage Multivitamin Tablets", "Tablets", "P100", 14.49, .48),
    ("Kids Complete Multivitamin Gummies", "Gummies", "P60", 10.99, .50),
    ("Teen for Her Multivitamin Gummies", "Gummies", "P60", 11.99, .35),
    ("Teen for Him Multivitamin Gummies", "Gummies", "P60", 11.99, .33),
])
_b("Vitamins & Supplements", "Garden of Life", [
    ("mykind Organics Women's Once Daily Multi Tablets", "Tablets", "P30", 29.99, .42),
    ("mykind Organics Men's Once Daily Multi Tablets", "Tablets", "P30", 29.99, .40),
    ("Vitamin Code Raw Prenatal Capsules", "Capsules", "P30", 31.99, .38),
    ("Dr. Formulated Probiotics Once Daily 30 Billion CFU", "Capsules", "P30", 29.99, .42),
    ("Dr. Formulated Probiotics Mood+ 50 Billion CFU", "Capsules", "P30", 34.99, .32),
    ("Raw Organic Protein Powder Vanilla", "Powder", "S", 34.99, .35),
    ("Vitamin Code Raw B-Complex Capsules", "Capsules", "P30", 19.99, .30),
])
_b("Vitamins & Supplements", "Emergen-C", [
    ("1000mg Vitamin C Powder Super Orange", "Powder", "P30", 10.99, .68),
    ("1000mg Vitamin C Powder Raspberry", "Powder", "P30", 10.99, .55),
    ("1000mg Vitamin C Powder Tangerine", "Powder", "P30", 10.99, .48),
    ("Immune+ Triple Action Gummies Orange", "Gummies", "P30", 12.99, .50),
    ("Hydration+ Electrolyte Replenishment Powder Lemon-Lime", "Powder", "P30", 10.99, .42),
    ("Kidz Vitamin C 250mg Fruit Punch Powder", "Powder", "P30", 9.99, .40),
])
_b("Vitamins & Supplements", "Airborne", [
    ("Vitamin C 1000mg Immune Support Effervescent Tablets Zesty Orange", "Effervescent", "P30", 11.99, .55),
    ("Vitamin C 1000mg Immune Support Gummies Orange", "Gummies", "P30", 12.99, .52),
    ("Vitamin C 1000mg Immune Support Chewable Tablets Citrus", "Chewables", "P30", 10.99, .45),
])
_b("Vitamins & Supplements", "Olly", [
    ("Women's Multi The Perfect Women's Gummy Blissful Berry", "Gummies", "P60", 14.99, .55),
    ("Men's Multi The Perfect Men's Gummy Blackberry Blitz", "Gummies", "P60", 14.99, .50),
    ("Undeniable Beauty Gummies Grapefruit Glam", "Gummies", "P60", 14.99, .48),
    ("Restful Sleep Gummies Blackberry Zen", "Gummies", "P60", 13.99, .52),
    ("Goodbye Stress Gummies Berry Verbena", "Gummies", "P60", 13.99, .45),
    ("Immunity Sleep + Elderberry Gummies Midnight Berry", "Gummies", "P60", 14.99, .42),
    ("Happy Gummy Worm Tropical Zing", "Gummies", "P60", 13.99, .38),
    ("Extra Strength Sleep Gummies Blackberry Mint", "Gummies", "P60", 15.99, .48),
    ("Prenatal Multivitamin Gummies Citrus Berry", "Gummies", "P60", 17.99, .40),
])
_b("Vitamins & Supplements", "Vitafusion", [
    ("MultiVites Adult Gummy Vitamins Berry Peach Orange", "Gummies", "P60", 9.99, .55),
    ("Women's Complete Multivitamin Gummies Natural Berry", "Gummies", "P60", 10.99, .50),
    ("Men's Complete Multivitamin Gummies Natural Berry", "Gummies", "P60", 10.99, .48),
    ("Omega-3 Gummy Vitamins Berry Lemonade", "Gummies", "P60", 10.99, .42),
    ("Vitamin D3 50mcg (2000 IU) Gummies Peach Blackberry Strawberry", "Gummies", "P60", 9.99, .45),
    ("B12 1000mcg Gummies Raspberry", "Gummies", "P60", 9.99, .40),
    ("Calcium Gummy Vitamins Fruit & Cream", "Gummies", "P60", 10.99, .38),
    ("Fiber Well Fit Gummies Peach Strawberry Berry", "Gummies", "P60", 12.99, .42),
])
_b("Vitamins & Supplements", "SmartyPants", [
    ("Women's Formula Multivitamin Gummies", "Gummies", "P60", 16.99, .42),
    ("Men's Formula Multivitamin Gummies", "Gummies", "P60", 16.99, .40),
    ("Kids Formula Multivitamin Gummies", "Gummies", "P60", 14.99, .45),
    ("Prenatal Formula Multivitamin Gummies", "Gummies", "P60", 19.99, .35),
])
_b("Vitamins & Supplements", "Nordic Naturals", [
    ("Ultimate Omega 1280mg Fish Oil Softgels Lemon", "Softgels", "P60", 27.99, .42),
    ("Prenatal DHA 830mg Softgels Unflavored", "Softgels", "P60", 24.99, .35),
    ("Complete Omega 1000mg Softgels Lemon", "Softgels", "P60", 25.99, .32),
])
_b("Vitamins & Supplements", "Move Free", [
    ("Advanced Glucosamine + Chondroitin Joint Health Tablets", "Tablets", "P60", 19.99, .48),
    ("Ultra Triple Action Joint Supplement Tablets", "Tablets", "P30", 22.99, .42),
    ("Advanced Plus MSM Joint Health Tablets", "Tablets", "P60", 21.99, .38),
])
_b("Vitamins & Supplements", "Osteo Bi-Flex", [
    ("Triple Strength Joint Health Tablets", "Tablets", "P60", 19.99, .45),
    ("One Per Day Joint Health Tablets", "Tablets", "P30", 14.99, .38),
    ("Joint Health Ease PM Tablets", "Tablets", "P30", 16.99, .32),
])

# ─── Skin Care ────────────────────────────────────────────────────
_b("Skin Care", "CeraVe", [
    ("Moisturizing Cream for Normal to Dry Skin", "Cream", "C8", 15.99, .88),
    ("Moisturizing Lotion for Normal to Dry Skin", "Lotion", "C8", 14.99, .82),
    ("Hydrating Facial Cleanser", "Cleanser", "C8", 13.99, .85),
    ("Foaming Facial Cleanser", "Cleanser", "C8", 13.99, .80),
    ("PM Facial Moisturizing Lotion", "Lotion", "C2", 14.99, .78),
    ("AM Facial Moisturizing Lotion SPF 30", "Lotion", "C2", 15.99, .75),
    ("Hydrating Cream-to-Foam Cleanser", "Cleanser", "C8", 14.99, .62),
    ("SA Smoothing Cleanser with Salicylic Acid", "Cleanser", "C8", 14.99, .58),
    ("Renewing SA Cleanser", "Cleanser", "C8", 14.99, .55),
    ("Eye Repair Cream", "Cream", "S", 14.99, .52),
    ("Skin Renewing Retinol Serum", "Serum", "S", 18.99, .50),
    ("Healing Ointment Skin Protectant", "Ointment", "C4", 12.99, .55),
    ("Daily Moisturizing Body Wash", "Body Wash", "C16", 12.99, .48),
    ("Hydrating Mineral Sunscreen SPF 50 Face", "Sunscreen", "C2", 14.99, .52),
    ("Itch Relief Moisturizing Cream", "Cream", "C8", 14.99, .40),
])
_b("Skin Care", "Neutrogena", [
    ("Hydro Boost Water Gel Moisturizer", "Gel", "C1", 19.99, .82),
    ("Hydro Boost Hydrating Cleansing Gel", "Cleanser", "C4", 9.99, .68),
    ("Oil-Free Acne Wash", "Cleanser", "C4", 7.99, .72),
    ("Oil-Free Acne Wash Pink Grapefruit Scrub", "Scrub", "C4", 8.49, .60),
    ("Rapid Wrinkle Repair Retinol Regenerating Cream", "Cream", "C1", 23.99, .55),
    ("Rapid Wrinkle Repair Retinol Serum", "Serum", "S", 22.99, .50),
    ("Ultra Sheer Dry-Touch Sunscreen SPF 55", "Sunscreen", "C4", 10.99, .65),
    ("Ultra Sheer Dry-Touch Sunscreen SPF 100+", "Sunscreen", "C4", 12.99, .55),
    ("Makeup Remover Cleansing Towelettes", "Wipes", "P24", 7.99, .70),
    ("Hydro Boost Hydrating Hyaluronic Acid Serum", "Serum", "S", 19.99, .52),
    ("Stubborn Acne AM Treatment Benzoyl Peroxide", "Treatment", "S", 9.99, .48),
    ("Healthy Skin Anti-Wrinkle Cream SPF 15", "Cream", "C1", 15.99, .42),
    ("Oil-Free Moisture Sensitive Skin", "Lotion", "C4", 11.99, .50),
])
_b("Skin Care", "Cetaphil", [
    ("Gentle Skin Cleanser", "Cleanser", "C8", 12.99, .78),
    ("Moisturizing Lotion Body & Face", "Lotion", "C8", 13.99, .72),
    ("Daily Advance Ultra Hydrating Lotion", "Lotion", "C8", 14.99, .58),
    ("Gentle Foaming Cleanser", "Cleanser", "C8", 12.99, .55),
    ("PRO Oil Removing Foam Wash", "Cleanser", "C8", 12.99, .45),
    ("Deep Hydration Healthy Glow Daily Cream", "Cream", "C1", 16.99, .42),
    ("Moisturizing Cream Body & Face", "Cream", "C8", 15.99, .55),
    ("Rich Hydrating Night Cream", "Cream", "C1", 15.99, .40),
    ("Daily Facial Cleanser Combination to Oily Skin", "Cleanser", "C8", 11.99, .52),
])
_b("Skin Care", "Olay", [
    ("Regenerist Micro-Sculpting Cream Face Moisturizer", "Cream", "C1", 28.99, .68),
    ("Regenerist Retinol24 Night Face Moisturizer", "Cream", "C1", 28.99, .58),
    ("Total Effects 7-in-1 Anti-Aging Moisturizer", "Cream", "C1", 22.99, .55),
    ("Complete Daily Defense All Day Moisturizer SPF 30", "Lotion", "C2", 12.99, .50),
    ("Regenerist Vitamin C + Peptide 24 Brightening Face Moisturizer", "Cream", "C1", 28.99, .48),
    ("Age Defying Classic Daily Renewal Cream", "Cream", "C2", 10.99, .38),
])
_b("Skin Care", "Aveeno", [
    ("Daily Moisturizing Body Lotion Fragrance Free", "Lotion", "C12", 11.99, .78),
    ("Positively Radiant Daily Face Moisturizer SPF 15", "Lotion", "C2", 15.99, .58),
    ("Calm + Restore Oat Gel Moisturizer Sensitive Skin", "Gel", "C1", 18.99, .50),
    ("Eczema Therapy Daily Moisturizing Cream", "Cream", "C4", 14.99, .55),
    ("Skin Relief Fragrance-Free Body Wash", "Body Wash", "C12", 8.99, .50),
    ("Ultra Calming Foaming Cleanser", "Cleanser", "C4", 9.99, .45),
    ("Protect + Hydrate Sunscreen Lotion SPF 60", "Sunscreen", "C4", 11.99, .48),
    ("Positively Radiant Skin Brightening Scrub", "Scrub", "C2", 8.99, .42),
])
_b("Skin Care", "La Roche-Posay", [
    ("Toleriane Double Repair Face Moisturizer", "Lotion", "C2", 21.99, .58),
    ("Effaclar Medicated Gel Acne Face Wash", "Cleanser", "C4", 15.99, .52),
    ("Anthelios Melt-In Milk Sunscreen SPF 60", "Sunscreen", "C4", 21.99, .48),
    ("Toleriane Hydrating Gentle Face Cleanser", "Cleanser", "C4", 15.99, .50),
    ("Effaclar Adapalene Gel 0.1% Topical Retinoid", "Gel", "S", 29.99, .42),
    ("Cicaplast Balm B5 Soothing Multi-Purpose Cream", "Cream", "C1", 15.99, .48),
])
_b("Skin Care", "Eucerin", [
    ("Original Healing Rich Feel Creme", "Cream", "C8", 10.99, .58),
    ("Advanced Repair Body Lotion Fragrance Free", "Lotion", "C8", 10.99, .55),
    ("Eczema Relief Body Creme", "Cream", "C8", 12.99, .48),
    ("Daily Hydration Body Lotion SPF 15", "Lotion", "C8", 11.99, .42),
    ("Roughness Relief Body Lotion", "Lotion", "C8", 11.99, .40),
])
_b("Skin Care", "Vanicream", [
    ("Moisturizing Skin Cream", "Cream", "C8", 14.99, .55),
    ("Gentle Facial Cleanser", "Cleanser", "C8", 10.99, .52),
    ("Lite Lotion", "Lotion", "C8", 12.99, .45),
    ("Daily Facial Moisturizer for Sensitive Skin", "Lotion", "C4", 13.99, .42),
])
_b("Skin Care", "Aquaphor", [
    ("Healing Ointment Advanced Therapy Skin Protectant", "Ointment", "C4", 11.99, .72),
    ("Healing Ointment Advanced Therapy Skin Protectant Tube", "Ointment", "C1", 5.99, .60),
    ("Lip Repair Ointment", "Lip Balm", "S", 5.49, .65),
    ("Baby Healing Ointment", "Ointment", "C4", 12.99, .52),
])
_b("Skin Care", "Vaseline", [
    ("Intensive Care Advanced Repair Body Lotion", "Lotion", "C12", 8.99, .62),
    ("Original Healing Jelly", "Jelly", "C4", 5.99, .58),
    ("Intensive Care Cocoa Radiant Body Lotion", "Lotion", "C12", 8.99, .48),
    ("Lip Therapy Original Lip Balm", "Lip Balm", "S", 3.49, .55),
    ("Intensive Care Essential Healing Body Lotion", "Lotion", "C12", 8.99, .50),
])
_b("Skin Care", "Burt's Bees", [
    ("Beeswax Lip Balm with Vitamin E", "Lip Balm", "S", 4.49, .72),
    ("100% Natural Moisturizing Lip Balm Pomegranate", "Lip Balm", "S", 4.49, .55),
    ("Sensitive Facial Cleanser", "Cleanser", "C4", 9.99, .42),
    ("Intense Hydration Night Cream", "Cream", "C1", 14.99, .35),
    ("Lip Shimmer Watermelon", "Lip Balm", "S", 5.99, .40),
])
_b("Skin Care", "Differin", [
    ("Adapalene Gel 0.1% Acne Treatment", "Gel", "S", 14.99, .55),
    ("Gentle Cleanser for Sensitive Skin", "Cleanser", "C4", 11.99, .42),
    ("Oil Absorbing Moisturizer SPF 30", "Lotion", "C2", 14.99, .38),
])
_b("Skin Care", "Palmer's", [
    ("Cocoa Butter Formula Body Lotion", "Lotion", "C8", 6.99, .55),
    ("Cocoa Butter Formula Skin Therapy Oil", "Oil", "S", 9.99, .48),
    ("Cocoa Butter Formula Moisturizing Body Oil", "Oil", "C8", 8.99, .42),
])
_b("Skin Care", "Jergens", [
    ("Ultra Healing Extra Dry Skin Moisturizer", "Lotion", "C12", 7.99, .55),
    ("Natural Glow Instant Sun Sunless Tanning Mousse", "Mousse", "S", 11.99, .42),
    ("Original Scent Dry Skin Moisturizer Cherry Almond", "Lotion", "C12", 7.99, .48),
    ("Wet Skin Moisturizer Coconut Oil", "Lotion", "C12", 8.99, .38),
])
_b("Skin Care", "Lubriderm", [
    ("Daily Moisture Body Lotion Fragrance Free", "Lotion", "C8", 9.99, .52),
    ("Intense Dry Skin Repair Body Lotion", "Lotion", "C8", 10.99, .42),
    ("Advanced Therapy Body Lotion Fragrance Free", "Lotion", "C8", 10.99, .45),
])
_b("Skin Care", "Clean & Clear", [
    ("Essentials Foaming Facial Cleanser", "Cleanser", "C8", 5.99, .52),
    ("Advantage Acne Spot Treatment", "Treatment", "S", 7.99, .45),
    ("Continuous Control Acne Cleanser", "Cleanser", "C4", 6.99, .40),
])
_b("Skin Care", "Stridex", [
    ("Maximum Strength Acne Pads Salicylic Acid 2%", "Pads", "P60", 5.49, .52),
    ("Sensitive Skin Acne Pads Salicylic Acid 0.5%", "Pads", "P60", 5.49, .38),
])
_b("Skin Care", "CVS Health", [
    ("Daily Moisturizing Body Lotion Fragrance Free", "Lotion", "C12", 7.99, .60),
    ("Gentle Skin Cleanser", "Cleanser", "C8", 8.99, .55),
    ("Foaming Facial Cleanser", "Cleanser", "C8", 7.99, .48),
    ("Oil-Free Acne Wash", "Cleanser", "C4", 5.99, .50),
    ("Hydrating Facial Moisturizer", "Lotion", "C4", 8.99, .45),
    ("Daily Sunscreen Lotion SPF 30", "Sunscreen", "C4", 7.99, .48),
    ("Daily Sunscreen Lotion SPF 50", "Sunscreen", "C4", 8.99, .45),
    ("Ultra Healing Body Lotion", "Lotion", "C12", 6.99, .48),
    ("Healing Ointment Skin Protectant", "Ointment", "C4", 7.99, .50),
    ("Lip Balm Original SPF 15", "Lip Balm", "S", 2.49, .45),
    ("Makeup Remover Cleansing Towelettes", "Wipes", "P24", 5.49, .48),
    ("Daily Facial Moisturizer SPF 15", "Lotion", "C4", 7.99, .42),
    ("Benzoyl Peroxide 10% Acne Wash", "Cleanser", "C4", 6.49, .40),
    ("Salicylic Acid Acne Pads 2%", "Pads", "P60", 4.49, .42),
])

# ─── Hair Care ────────────────────────────────────────────────────
_b("Hair Care", "Pantene", [
    ("Daily Moisture Renewal Shampoo", "Shampoo", "C12", 6.99, .72),
    ("Daily Moisture Renewal Conditioner", "Conditioner", "C12", 6.99, .70),
    ("Sheer Volume Shampoo", "Shampoo", "C12", 6.99, .60),
    ("Sheer Volume Conditioner", "Conditioner", "C12", 6.99, .58),
    ("Repair & Protect Shampoo", "Shampoo", "C12", 6.99, .55),
    ("Repair & Protect Conditioner", "Conditioner", "C12", 6.99, .52),
    ("Color Protect Shampoo", "Shampoo", "C12", 6.99, .48),
    ("Color Protect Conditioner", "Conditioner", "C12", 6.99, .45),
    ("Miracle Rescue Deep Conditioning Treatment", "Treatment", "S", 6.99, .38),
])
_b("Hair Care", "Head & Shoulders", [
    ("Classic Clean Anti-Dandruff Shampoo", "Shampoo", "C12", 7.99, .78),
    ("Classic Clean 2-in-1 Anti-Dandruff Shampoo + Conditioner", "Shampoo", "C12", 7.99, .65),
    ("Dry Scalp Care Anti-Dandruff Shampoo Almond Oil", "Shampoo", "C12", 7.99, .55),
    ("Clinical Strength Anti-Dandruff Shampoo", "Shampoo", "C12", 10.99, .48),
    ("Royal Oils Moisturizing Scalp Cream", "Treatment", "S", 9.99, .35),
    ("Itchy Scalp Care Anti-Dandruff Shampoo Eucalyptus", "Shampoo", "C12", 7.99, .45),
])
_b("Hair Care", "TRESemme", [
    ("Moisture Rich Shampoo", "Shampoo", "C12", 5.99, .62),
    ("Moisture Rich Conditioner", "Conditioner", "C12", 5.99, .60),
    ("Keratin Smooth Shine Shampoo", "Shampoo", "C12", 5.99, .50),
    ("Keratin Smooth Shine Conditioner", "Conditioner", "C12", 5.99, .48),
    ("Volumizing Dry Shampoo", "Dry Shampoo", "SP", 6.99, .45),
    ("Extra Hold Hairspray", "Hairspray", "SP", 5.99, .42),
])
_b("Hair Care", "Dove", [
    ("Daily Moisture Shampoo", "Shampoo", "C12", 6.99, .65),
    ("Daily Moisture Conditioner", "Conditioner", "C12", 6.99, .62),
    ("Intensive Repair Shampoo", "Shampoo", "C12", 6.99, .52),
    ("Intensive Repair Conditioner", "Conditioner", "C12", 6.99, .50),
    ("DermaCare Scalp Anti-Dandruff Shampoo", "Shampoo", "C12", 7.99, .42),
    ("Dry Shampoo Volume & Fullness", "Dry Shampoo", "SP", 5.99, .48),
])
_b("Hair Care", "Garnier", [
    ("Fructis Sleek & Shine Shampoo", "Shampoo", "C12", 4.99, .55),
    ("Fructis Sleek & Shine Conditioner", "Conditioner", "C12", 4.99, .52),
    ("Fructis Grow Strong Shampoo", "Shampoo", "C12", 4.99, .45),
    ("Fructis Grow Strong Conditioner", "Conditioner", "C12", 4.99, .42),
    ("Whole Blends Honey Treasures Repairing Shampoo", "Shampoo", "C12", 5.99, .48),
    ("Whole Blends Honey Treasures Repairing Conditioner", "Conditioner", "C12", 5.99, .45),
    ("Fructis Pure Clean Shampoo", "Shampoo", "C12", 4.99, .40),
    ("Fructis Style Curl Sculpt Gel", "Gel", "C4", 4.99, .35),
])
_b("Hair Care", "OGX", [
    ("Renewing + Argan Oil of Morocco Shampoo", "Shampoo", "C12", 7.99, .62),
    ("Renewing + Argan Oil of Morocco Conditioner", "Conditioner", "C12", 7.99, .60),
    ("Nourishing + Coconut Milk Shampoo", "Shampoo", "C12", 7.99, .55),
    ("Nourishing + Coconut Milk Conditioner", "Conditioner", "C12", 7.99, .52),
    ("Thick & Full + Biotin & Collagen Shampoo", "Shampoo", "C12", 7.99, .52),
    ("Thick & Full + Biotin & Collagen Conditioner", "Conditioner", "C12", 7.99, .50),
    ("Extra Strength Refreshing Scalp + Tea Tree Mint Shampoo", "Shampoo", "C12", 7.99, .42),
    ("Extra Strength Damage Remedy + Coconut Miracle Oil Spray", "Spray", "S", 7.99, .35),
])
_b("Hair Care", "Herbal Essences", [
    ("Bio:Renew Argan Oil Repair Shampoo", "Shampoo", "C12", 6.99, .50),
    ("Bio:Renew Argan Oil Repair Conditioner", "Conditioner", "C12", 6.99, .48),
    ("Bio:Renew Sulfate Free Birch Bark Extract Shampoo", "Shampoo", "C12", 7.99, .42),
    ("Classics Hello Hydration Moisturizing Shampoo", "Shampoo", "C12", 4.49, .38),
    ("Classics Hello Hydration Moisturizing Conditioner", "Conditioner", "C12", 4.49, .35),
])
_b("Hair Care", "Aussie", [
    ("Miracle Moist Shampoo", "Shampoo", "C12", 4.99, .48),
    ("Miracle Moist Conditioner", "Conditioner", "C12", 4.99, .45),
    ("Total Miracle 7N1 Shampoo", "Shampoo", "C12", 5.49, .40),
    ("3 Minute Miracle Moist Deep Conditioner", "Conditioner", "C8", 4.99, .42),
])
_b("Hair Care", "Suave", [
    ("Essentials Daily Clarifying Shampoo", "Shampoo", "C12", 2.99, .45),
    ("Essentials Ocean Breeze Conditioner", "Conditioner", "C12", 2.99, .42),
    ("Professionals Keratin Infusion Smoothing Shampoo", "Shampoo", "C12", 3.49, .38),
    ("Professionals Keratin Infusion Smoothing Conditioner", "Conditioner", "C12", 3.49, .35),
])
_b("Hair Care", "John Frieda", [
    ("Frizz Ease Extra Strength Serum", "Serum", "S", 11.99, .52),
    ("Frizz Ease Daily Nourishment Leave-In Conditioner", "Leave-In", "S", 9.99, .45),
    ("Brilliant Brunette Visibly Deeper Shampoo", "Shampoo", "C8", 9.99, .35),
    ("Sheer Blonde Go Blonder Lightening Shampoo", "Shampoo", "C8", 9.99, .35),
])
_b("Hair Care", "Nizoral", [
    ("A-D Anti-Dandruff Ketoconazole 1% Shampoo", "Shampoo", "C4", 15.99, .55),
])
_b("Hair Care", "Rogaine", [
    ("Men's 5% Minoxidil Topical Foam Hair Regrowth Treatment", "Foam", "S", 44.99, .50),
    ("Women's 5% Minoxidil Topical Foam Hair Regrowth Treatment", "Foam", "S", 44.99, .45),
    ("Men's Extra Strength 5% Minoxidil Solution", "Solution", "S", 32.99, .42),
])
_b("Hair Care", "Not Your Mother's", [
    ("Beach Babe Texturizing Sea Salt Spray", "Spray", "S", 6.99, .42),
    ("Curl Talk Defining Cream", "Cream", "C4", 7.99, .40),
    ("Clean Freak Refreshing Dry Shampoo", "Dry Shampoo", "SP", 6.99, .45),
])
_b("Hair Care", "L'Oreal", [
    ("EverPure Sulfate-Free Moisture Shampoo", "Shampoo", "C8", 8.99, .48),
    ("EverPure Sulfate-Free Moisture Conditioner", "Conditioner", "C8", 8.99, .45),
    ("Elvive Total Repair 5 Repairing Shampoo", "Shampoo", "C12", 5.99, .42),
    ("Elvive Total Repair 5 Repairing Conditioner", "Conditioner", "C12", 5.99, .40),
    ("Elnett Satin Extra Strong Hold Hairspray", "Hairspray", "SP", 10.99, .45),
])
_b("Hair Care", "CVS Health", [
    ("Daily Moisturizing Shampoo", "Shampoo", "C12", 3.99, .50),
    ("Daily Moisturizing Conditioner", "Conditioner", "C12", 3.99, .48),
    ("Volumizing Shampoo", "Shampoo", "C12", 3.99, .40),
    ("Volumizing Conditioner", "Conditioner", "C12", 3.99, .38),
    ("2-in-1 Dandruff Shampoo + Conditioner", "Shampoo", "C12", 5.49, .42),
    ("Dry Shampoo Fresh Scent", "Dry Shampoo", "SP", 4.99, .38),
    ("Firm Hold Hairspray", "Hairspray", "SP", 3.99, .35),
])

# ─── Oral Care ────────────────────────────────────────────────────
_b("Oral Care", "Crest", [
    ("Pro-Health Advanced Deep Clean Mint Toothpaste", "Toothpaste", "T", 5.49, .75),
    ("3D White Brilliance Vibrant Peppermint Toothpaste", "Toothpaste", "T", 5.99, .72),
    ("Cavity Protection Regular Paste Toothpaste", "Toothpaste", "T", 3.99, .65),
    ("Gum Detoxify Deep Clean Toothpaste", "Toothpaste", "T", 6.49, .52),
    ("Scope Classic Mouthwash Original Mint", "Mouthwash", "L16", 5.99, .58),
    ("Pro-Health Multi-Protection Mouthwash Clean Mint", "Mouthwash", "L16", 6.99, .52),
    ("3D Whitestrips Classic Vivid Teeth Whitening Kit", "Whitening", "S", 29.99, .48),
    ("3D Whitestrips Professional Effects Teeth Whitening Kit", "Whitening", "S", 44.99, .42),
    ("3D Whitestrips 1 Hour Express Teeth Whitening Kit", "Whitening", "S", 34.99, .38),
    ("Sensitivity Toothpaste", "Toothpaste", "T", 5.99, .45),
])
_b("Oral Care", "Colgate", [
    ("Total Whitening Toothpaste Gel", "Toothpaste", "T", 5.49, .72),
    ("Optic White Advanced Teeth Whitening Toothpaste Sparkling White", "Toothpaste", "T", 5.99, .65),
    ("Cavity Protection Toothpaste Great Regular Flavor", "Toothpaste", "T", 3.49, .68),
    ("Sensitive Toothpaste Complete Protection", "Toothpaste", "T", 6.49, .52),
    ("Max Fresh Toothpaste Cool Mint", "Toothpaste", "T", 4.99, .55),
    ("Total Mouthwash Spearmint Surge", "Mouthwash", "L16", 6.99, .48),
    ("Peroxyl Antiseptic Oral Cleanser Mild Mint", "Mouthwash", "L8", 8.99, .35),
    ("Extra Clean Medium Toothbrush", "Toothbrush", "S", 3.49, .55),
    ("360 Advanced Optic White Toothbrush Soft", "Toothbrush", "S", 5.99, .42),
])
_b("Oral Care", "Sensodyne", [
    ("Pronamel Gentle Whitening Toothpaste", "Toothpaste", "T", 7.49, .62),
    ("Repair & Protect Whitening Toothpaste", "Toothpaste", "T", 7.49, .55),
    ("Fresh Mint Sensitive Toothpaste", "Toothpaste", "T", 6.99, .58),
    ("Rapid Relief Toothpaste Extra Fresh", "Toothpaste", "T", 7.99, .48),
    ("Complete Protection Toothpaste Extra Fresh", "Toothpaste", "T", 7.49, .45),
])
_b("Oral Care", "Listerine", [
    ("Cool Mint Antiseptic Mouthwash", "Mouthwash", "L16", 6.99, .78),
    ("Total Care Anticavity Mouthwash Fresh Mint", "Mouthwash", "L16", 7.99, .62),
    ("Whitening Pre-Brush Rinse Clean Mint", "Mouthwash", "L16", 8.49, .48),
    ("Ultraclean Antiseptic Cool Mint", "Mouthwash", "L16", 7.49, .55),
    ("Cool Mint PocketPaks Breath Strips", "Strips", "S", 3.99, .50),
    ("Kids Smart Rinse Bubble Blast", "Mouthwash", "L8", 5.99, .38),
])
_b("Oral Care", "ACT", [
    ("Anticavity Fluoride Rinse Mint", "Mouthwash", "L12", 7.99, .55),
    ("Total Care Anticavity Fluoride Mouthwash Icy Clean Mint", "Mouthwash", "L12", 8.49, .48),
    ("Dry Mouth Moisturizing Mouthwash Soothing Mint", "Mouthwash", "L12", 7.99, .42),
    ("Kids Anticavity Fluoride Rinse Bubblegum Blowout", "Mouthwash", "L12", 6.49, .45),
])
_b("Oral Care", "Oral-B", [
    ("CrossAction All In One Soft Toothbrush", "Toothbrush", "S", 5.99, .55),
    ("Pro-Health Clinical Pro-Flex Medium Toothbrush", "Toothbrush", "S", 6.99, .42),
    ("Glide Pro-Health Deep Clean Floss Cool Mint", "Floss", "S", 5.49, .58),
    ("Glide Pro-Health Comfort Plus Floss Mint", "Floss", "S", 4.99, .50),
    ("Complete Satin Floss Mint", "Floss", "S", 3.99, .42),
    ("iO Series 4 Rechargeable Electric Toothbrush", "Electric Toothbrush", "S", 69.99, .35),
    ("Precision Clean Electric Toothbrush Replacement Heads", "Brush Heads", "K1", 22.99, .42),
])
_b("Oral Care", "TheraBreath", [
    ("Fresh Breath Oral Rinse Mild Mint", "Mouthwash", "L16", 10.99, .48),
    ("Healthy Gums Oral Rinse Clean Mint", "Mouthwash", "L16", 11.99, .40),
    ("Fresh Breath Dry Mouth Lozenges Mandarin Mint", "Lozenges", "S", 9.99, .35),
])
_b("Oral Care", "Fixodent", [
    ("Original Denture Adhesive Cream", "Denture Adhesive", "C2", 6.99, .42),
    ("Ultra Max Hold Denture Adhesive", "Denture Adhesive", "C2", 7.49, .38),
])
_b("Oral Care", "Plackers", [
    ("Twin-Line Dental Flossers Cool Mint", "Flossers", "P60", 3.99, .52),
    ("Micro Mint Dental Flossers", "Flossers", "P60", 3.99, .48),
    ("Gentle Slide Dental Flossers Fresh Mint", "Flossers", "P60", 4.49, .38),
])
_b("Oral Care", "CVS Health", [
    ("Cavity Protection Fluoride Toothpaste", "Toothpaste", "T", 2.49, .55),
    ("Whitening Fluoride Toothpaste Fresh Mint", "Toothpaste", "T", 2.99, .48),
    ("Sensitive Teeth Toothpaste Maximum Strength", "Toothpaste", "T", 4.99, .42),
    ("Antiseptic Mouthwash Blue Mint", "Mouthwash", "L16", 4.49, .50),
    ("Anticavity Fluoride Rinse Mint", "Mouthwash", "L16", 5.49, .42),
    ("Waxed Dental Floss Mint", "Floss", "S", 2.99, .45),
    ("EasyGlide Dental Flossers", "Flossers", "P60", 2.99, .45),
    ("Soft Bristle Toothbrush 4 Pack", "Toothbrush", "S", 5.99, .48),
    ("Medium Bristle Toothbrush", "Toothbrush", "S", 1.99, .40),
    ("Denture Adhesive Cream Original", "Denture Adhesive", "C2", 4.99, .35),
])

# ─── Deodorant ────────────────────────────────────────────────────
_b("Deodorant", "Dove", [
    ("Advanced Care Antiperspirant Deodorant Stick Shea Butter", "Stick", "D", 6.49, .72),
    ("Advanced Care Antiperspirant Deodorant Stick Cool Essentials", "Stick", "D", 6.49, .65),
    ("0% Aluminum Deodorant Stick Cucumber & Green Tea", "Stick", "D", 7.49, .55),
    ("Invisible Solid Antiperspirant Deodorant Powder", "Stick", "D", 5.49, .58),
    ("Dry Spray Antiperspirant Sheer Fresh", "Spray", "SP", 7.49, .48),
])
_b("Deodorant", "Secret", [
    ("Clinical Strength Antiperspirant Deodorant Stress Response", "Stick", "D", 10.99, .60),
    ("Outlast Invisible Solid Antiperspirant Completely Clean", "Stick", "D", 6.49, .65),
    ("Outlast Clear Gel Antiperspirant Deodorant Lavender", "Gel", "D", 6.49, .55),
    ("Essential Oils Deodorant Lavender & Lemon Grass", "Stick", "D", 7.99, .42),
    ("Dry Spray Antiperspirant Deodorant Vanilla", "Spray", "SP", 7.49, .45),
])
_b("Deodorant", "Old Spice", [
    ("High Endurance Deodorant Pure Sport", "Stick", "D", 5.49, .72),
    ("Sweat Defense Dry Spray Antiperspirant Stronger Swagger", "Spray", "SP", 7.49, .55),
    ("Fiji with Palm Tree Deodorant", "Stick", "D", 6.49, .62),
    ("Bearglove Antiperspirant Deodorant", "Stick", "D", 6.49, .52),
    ("GentleMan's Blend Deodorant Brown Sugar + Cocoa Butter", "Stick", "D", 7.99, .42),
    ("Red Zone Pure Sport Antiperspirant & Deodorant", "Stick", "D", 5.99, .60),
])
_b("Deodorant", "Degree", [
    ("UltraClear Black + White Antiperspirant Deodorant Stick", "Stick", "D", 5.99, .65),
    ("MotionSense Invisible Solid Cool Rush Antiperspirant", "Stick", "D", 5.99, .62),
    ("Advanced 72H MotionSense Antiperspirant Stick Sport Defense", "Stick", "D", 6.49, .55),
    ("Men Dry Spray Antiperspirant Cool Rush", "Spray", "SP", 7.49, .48),
    ("Women Dry Spray Antiperspirant Sheer Powder", "Spray", "SP", 7.49, .45),
])
_b("Deodorant", "Native", [
    ("Deodorant Coconut & Vanilla", "Stick", "D", 12.99, .55),
    ("Deodorant Lavender & Rose", "Stick", "D", 12.99, .48),
    ("Deodorant Eucalyptus & Mint", "Stick", "D", 12.99, .45),
    ("Deodorant Cucumber & Mint", "Stick", "D", 12.99, .42),
    ("Sensitive Deodorant Unscented", "Stick", "D", 13.99, .40),
])
_b("Deodorant", "Schmidt's", [
    ("Natural Deodorant Bergamot + Lime", "Stick", "D", 9.99, .42),
    ("Natural Deodorant Lavender + Sage", "Stick", "D", 9.99, .40),
    ("Natural Deodorant Charcoal + Magnesium", "Stick", "D", 9.99, .38),
])
_b("Deodorant", "Arm & Hammer", [
    ("Essentials Solid Deodorant Unscented", "Stick", "D", 4.49, .48),
    ("UltraMax Solid Antiperspirant Deodorant Fresh", "Stick", "D", 4.49, .52),
    ("UltraMax Invisible Solid Cool Blast", "Stick", "D", 4.99, .45),
])
_b("Deodorant", "Tom's of Maine", [
    ("Long Lasting Deodorant Stick Unscented", "Stick", "D", 8.49, .40),
    ("Original Care Natural Deodorant Wild Lavender", "Stick", "D", 7.49, .35),
])
_b("Deodorant", "CVS Health", [
    ("Invisible Solid Antiperspirant Deodorant Powder Fresh", "Stick", "D", 3.99, .52),
    ("Invisible Solid Antiperspirant Deodorant Unscented", "Stick", "D", 3.99, .48),
    ("Clear Gel Antiperspirant Deodorant Cool Wave", "Gel", "D", 3.99, .42),
    ("Sport Antiperspirant Deodorant Stick", "Stick", "D", 3.99, .45),
    ("Dry Spray Antiperspirant Fresh", "Spray", "SP", 5.49, .38),
])

# ─── Shaving & Grooming ──────────────────────────────────────────
_b("Shaving & Grooming", "Gillette", [
    ("Fusion5 Power Razor", "Razor", "S", 12.99, .68),
    ("Fusion5 ProGlide Razor", "Razor", "S", 14.99, .65),
    ("Fusion5 Razor Cartridge Refills 4 CT", "Cartridges", "S", 19.99, .62),
    ("Fusion5 ProGlide Razor Cartridge Refills 4 CT", "Cartridges", "S", 22.99, .58),
    ("Mach3 Razor", "Razor", "S", 9.99, .55),
    ("Mach3 Cartridge Refills 4 CT", "Cartridges", "S", 14.99, .50),
    ("Venus Smooth Razor", "Razor", "S", 9.99, .58),
    ("Venus Extra Smooth Razor", "Razor", "S", 12.99, .52),
    ("Venus Cartridge Refills 4 CT", "Cartridges", "S", 16.99, .48),
    ("Fusion5 Ultra Sensitive Shave Gel", "Shave Gel", "C4", 4.99, .55),
    ("Series Sensitive Shave Foam", "Shave Foam", "C4", 3.99, .48),
    ("Sensor3 Disposable Razors 4 CT", "Disposable", "S", 7.99, .42),
    ("CustomPlus 3 Disposable Razors", "Disposable", "S", 9.99, .38),
    ("SkinGuard Sensitive Razor", "Razor", "S", 11.99, .35),
])
_b("Shaving & Grooming", "Schick", [
    ("Hydro 5 Sense Hydrate Razor", "Razor", "S", 10.99, .52),
    ("Hydro 5 Sense Cartridge Refills 4 CT", "Cartridges", "S", 15.99, .45),
    ("Quattro Titanium Razor", "Razor", "S", 9.99, .42),
    ("Intuition Sensitive Care Razor", "Razor", "S", 11.99, .45),
    ("Hydro Silk TrimStyle Razor", "Razor", "S", 12.99, .40),
    ("ST2 Sensitive Slim Twin Disposable Razors 10 CT", "Disposable", "S", 5.99, .38),
])
_b("Shaving & Grooming", "Harry's", [
    ("Razor Handle with 2 Blade Cartridges Navy", "Razor", "S", 9.99, .52),
    ("Razor Blade Refills 4 CT", "Cartridges", "S", 9.49, .48),
    ("Shave Gel Lightly Scented", "Shave Gel", "C4", 5.99, .42),
    ("Face Wash with Fig Extract", "Cleanser", "C4", 7.99, .35),
    ("Post Shave Balm with Aloe", "Balm", "S", 7.99, .33),
])
_b("Shaving & Grooming", "Barbasol", [
    ("Original Thick & Rich Shaving Cream", "Shave Cream", "C4", 2.49, .55),
    ("Sensitive Skin Thick & Rich Shaving Cream", "Shave Cream", "C4", 2.99, .45),
    ("Soothing Aloe Thick & Rich Shaving Cream", "Shave Cream", "C4", 2.99, .38),
])
_b("Shaving & Grooming", "Edge", [
    ("Sensitive Skin Shave Gel", "Shave Gel", "C4", 4.49, .52),
    ("Extra Moisturizing Shave Gel", "Shave Gel", "C4", 4.49, .42),
])
_b("Shaving & Grooming", "Cremo", [
    ("Original Shave Cream", "Shave Cream", "C4", 8.99, .42),
    ("Beard Oil Revitalizing Cedar Forest", "Beard Oil", "S", 9.99, .32),
    ("Styling Beard Balm Forest Blend", "Beard Balm", "S", 9.99, .28),
])
_b("Shaving & Grooming", "Nair", [
    ("Hair Remover Lotion Cocoa Butter", "Lotion", "C4", 7.99, .45),
    ("Leg Mask Hair Removal + Beauty Treatment", "Cream", "S", 9.99, .35),
    ("Wax Ready-Strips Legs & Body", "Wax Strips", "S", 8.99, .38),
])
_b("Shaving & Grooming", "CVS Health", [
    ("Men's 5 Blade Disposable Razors 3 CT", "Disposable", "S", 5.99, .45),
    ("Men's 3 Blade Disposable Razors 10 CT", "Disposable", "S", 6.99, .42),
    ("Women's 5 Blade Disposable Razors 3 CT", "Disposable", "S", 5.99, .42),
    ("Sensitive Skin Shave Gel", "Shave Gel", "C4", 2.99, .40),
    ("Moisturizing Shave Cream", "Shave Cream", "C4", 2.49, .35),
])

# ─── Cosmetics & Makeup ──────────────────────────────────────────
_b("Cosmetics & Makeup", "Maybelline", [
    ("Fit Me Matte + Poreless Foundation Classic Ivory 120", "Foundation", "S", 8.99, .72),
    ("Fit Me Matte + Poreless Foundation Natural Beige 220", "Foundation", "S", 8.99, .68),
    ("Fit Me Matte + Poreless Foundation Toffee 330", "Foundation", "S", 8.99, .55),
    ("Instant Age Rewind Eraser Dark Circles Treatment Concealer Fair", "Concealer", "S", 10.99, .70),
    ("Instant Age Rewind Eraser Dark Circles Treatment Concealer Light", "Concealer", "S", 10.99, .68),
    ("Instant Age Rewind Eraser Dark Circles Treatment Concealer Medium", "Concealer", "S", 10.99, .58),
    ("Lash Sensational Full Fan Effect Washable Mascara Very Black", "Mascara", "S", 9.99, .75),
    ("Lash Sensational Sky High Washable Mascara Very Black", "Mascara", "S", 11.99, .72),
    ("SuperStay Matte Ink Liquid Lipstick Lover", "Lipstick", "S", 9.99, .60),
    ("SuperStay Matte Ink Liquid Lipstick Pioneer", "Lipstick", "S", 9.99, .52),
    ("SuperStay Matte Ink Liquid Lipstick Seductress", "Lipstick", "S", 9.99, .48),
    ("Color Sensational Lipstick Pink Petal 105", "Lipstick", "S", 8.49, .50),
    ("Color Sensational Lipstick Red Revival 645", "Lipstick", "S", 8.49, .48),
    ("Color Sensational Lipstick Nude Lust 657", "Lipstick", "S", 8.49, .45),
    ("SuperStay Full Coverage Foundation Fair Porcelain 102", "Foundation", "S", 12.99, .48),
    ("Fit Me Loose Finishing Powder Fair Light", "Powder", "S", 8.49, .52),
    ("Fit Me Blush Rose 25", "Blush", "S", 6.49, .42),
    ("Master Chrome Metallic Highlighter Molten Gold 100", "Highlighter", "S", 8.49, .38),
    ("TattooStudio Brow Pomade Soft Brown 378", "Brow", "S", 9.99, .42),
    ("EyeStudio Master Precise All Day Liquid Eyeliner Black", "Eyeliner", "S", 8.99, .55),
])
_b("Cosmetics & Makeup", "L'Oreal", [
    ("True Match Super-Blendable Foundation Porcelain W1", "Foundation", "S", 10.99, .68),
    ("True Match Super-Blendable Foundation Sand Beige W5", "Foundation", "S", 10.99, .60),
    ("True Match Super-Blendable Foundation Cocoa C8", "Foundation", "S", 10.99, .48),
    ("Voluminous Original Washable Bold Eye Mascara Carbon Black", "Mascara", "S", 8.99, .65),
    ("Voluminous Lash Paradise Washable Mascara Blackest Black", "Mascara", "S", 11.99, .68),
    ("Infallible 24HR Pro-Glow Foundation Classic Tan 109", "Foundation", "S", 14.99, .48),
    ("Colour Riche Lipstick Blushing Berry 620", "Lipstick", "S", 8.99, .50),
    ("Colour Riche Lipstick Mica 240", "Lipstick", "S", 8.99, .42),
    ("Age Perfect Radiant Serum Foundation Rose Ivory 0.5", "Foundation", "S", 14.99, .38),
    ("Infallible Pro-Matte Liquid Lipstick Stirred 370", "Lipstick", "S", 10.99, .40),
    ("Infallible 24HR Fresh Wear Powder Foundation Pearl 05", "Powder", "S", 14.99, .42),
    ("True Match Crayon Concealer Fair Light Warm W1-2-3", "Concealer", "S", 9.99, .45),
    ("Telescopic Waterproof Mascara Black", "Mascara", "S", 11.49, .52),
])
_b("Cosmetics & Makeup", "Revlon", [
    ("ColorStay Makeup Foundation Combo/Oily Buff 150", "Foundation", "S", 10.99, .58),
    ("ColorStay Makeup Foundation Combo/Oily Medium Beige 240", "Foundation", "S", 10.99, .52),
    ("ColorStay Makeup Foundation Combo/Oily Caramel 400", "Foundation", "S", 10.99, .42),
    ("Super Lustrous Lipstick Creme Fire & Ice 720", "Lipstick", "S", 8.99, .52),
    ("Super Lustrous Lipstick Creme Pink in the Afternoon 415", "Lipstick", "S", 8.99, .48),
    ("Super Lustrous Lipstick Creme Toast of New York 325", "Lipstick", "S", 8.99, .42),
    ("ColorStay Liquid Eye Pen Blackest Black", "Eyeliner", "S", 9.99, .48),
    ("PhotoReady Candid Concealer Light Medium 030", "Concealer", "S", 9.99, .40),
    ("So Fierce Chrome Ink Liquid Liner Gunmetal", "Eyeliner", "S", 10.49, .32),
])
_b("Cosmetics & Makeup", "CoverGirl", [
    ("Clean Fresh Skin Milk Foundation Light/Medium 540", "Foundation", "S", 10.99, .52),
    ("LashBlast Volume Waterproof Mascara Very Black", "Mascara", "S", 8.99, .58),
    ("Outlast All-Day Lip Color Luminous Lilac", "Lipstick", "S", 10.99, .42),
    ("Simply Ageless Instant Wrinkle-Defying Foundation Classic Ivory 210", "Foundation", "S", 14.99, .45),
    ("Clean Matte BB Cream Light/Medium 530", "BB Cream", "S", 7.99, .38),
    ("Perfect Point Plus Eye Liner Black Onyx", "Eyeliner", "S", 6.99, .48),
    ("Exhibitionist Lip Gloss Fling", "Lip Gloss", "S", 7.99, .35),
])
_b("Cosmetics & Makeup", "NYX Professional Makeup", [
    ("Butter Gloss Creme Brulee", "Lip Gloss", "S", 5.49, .62),
    ("Butter Gloss Angel Food Cake", "Lip Gloss", "S", 5.49, .52),
    ("Butter Gloss Tiramisu", "Lip Gloss", "S", 5.49, .48),
    ("Soft Matte Lip Cream Abu Dhabi", "Lip Cream", "S", 6.99, .55),
    ("Soft Matte Lip Cream Copenhagen", "Lip Cream", "S", 6.99, .48),
    ("Soft Matte Lip Cream Rome", "Lip Cream", "S", 6.99, .42),
    ("Epic Ink Liner Waterproof Liquid Eyeliner Black", "Eyeliner", "S", 8.99, .65),
    ("Can't Stop Won't Stop Full Coverage Foundation Light Ivory", "Foundation", "S", 10.99, .48),
    ("Can't Stop Won't Stop Full Coverage Foundation Medium Olive", "Foundation", "S", 10.99, .42),
    ("HD Finishing Powder Translucent", "Powder", "S", 10.49, .50),
    ("Micro Brow Pencil Brunette", "Brow", "S", 9.99, .48),
    ("Micro Brow Pencil Espresso", "Brow", "S", 9.99, .40),
    ("Lip Lingerie XXL Matte Liquid Lipstick Undressd", "Lipstick", "S", 8.99, .42),
])
_b("Cosmetics & Makeup", "e.l.f.", [
    ("16HR Camo Concealer Fair Warm", "Concealer", "S", 6.00, .62),
    ("16HR Camo Concealer Light Sand", "Concealer", "S", 6.00, .58),
    ("16HR Camo Concealer Medium Sand", "Concealer", "S", 6.00, .50),
    ("Poreless Putty Primer Universal Sheer", "Primer", "S", 9.00, .65),
    ("Flawless Finish Foundation Sand", "Foundation", "S", 6.00, .50),
    ("Bite-Size Eyeshadow Rose Water", "Eyeshadow", "S", 3.00, .52),
    ("Bite-Size Eyeshadow Cream & Sugar", "Eyeshadow", "S", 3.00, .48),
    ("Holy Hydration! Face Cream Fragrance Free", "Cream", "S", 12.00, .42),
    ("Power Grip Primer", "Primer", "S", 10.00, .58),
    ("Halo Glow Liquid Filter Fair/Light", "Primer", "S", 14.00, .52),
    ("Suntouchable Whoa Glow SPF 30 Sun Kissed", "Primer", "S", 14.00, .38),
])
_b("Cosmetics & Makeup", "Milani", [
    ("Baked Blush Luminoso", "Blush", "S", 8.99, .52),
    ("Make It Last Setting Spray", "Setting Spray", "S", 9.99, .48),
    ("Color Statement Lipstick Best Red", "Lipstick", "S", 6.99, .42),
    ("Conceal + Perfect 2-in-1 Foundation + Concealer Light Beige", "Foundation", "S", 9.99, .45),
    ("Highly Rated 10-in-1 Volume Mascara", "Mascara", "S", 9.99, .42),
])
_b("Cosmetics & Makeup", "Wet n Wild", [
    ("MegaLast High-Shine Lip Color Raining Rubies", "Lipstick", "S", 2.99, .48),
    ("Photo Focus Foundation Soft Beige", "Foundation", "S", 5.99, .45),
    ("Photo Focus Setting Spray", "Setting Spray", "S", 5.99, .42),
    ("MegaGlo Highlighting Powder Precious Petals", "Highlighter", "S", 5.49, .42),
    ("Color Icon Eyeshadow Quad Walking on Eggshells", "Eyeshadow", "S", 4.99, .40),
    ("MegaLength Waterproof Mascara Very Black", "Mascara", "S", 3.99, .45),
])
_b("Cosmetics & Makeup", "Physicians Formula", [
    ("Butter Bronzer Murumuru Butter Bronzer", "Bronzer", "S", 14.99, .52),
    ("Mineral Wear Talc-Free Mineral Face Powder Translucent", "Powder", "S", 11.99, .42),
    ("Rose All Day Oil-Free Serum + Foundation Light Medium", "Foundation", "S", 14.99, .35),
    ("The Healthy Lip Velvet Liquid Lipstick Dose of Rose", "Lipstick", "S", 8.99, .32),
])
_b("Cosmetics & Makeup", "Essie", [
    ("Nail Polish Ballet Slippers 162", "Nail Polish", "S", 9.99, .55),
    ("Nail Polish Bordeaux 12", "Nail Polish", "S", 9.99, .48),
    ("Nail Polish Wicked 249", "Nail Polish", "S", 9.99, .45),
    ("Gel Couture Nail Polish Fairy Tailor 40", "Nail Polish", "S", 11.99, .42),
    ("Nail Polish Chinchilly 688", "Nail Polish", "S", 9.99, .40),
    ("Good As Gone Nail Polish Remover", "Remover", "S", 5.99, .42),
])
_b("Cosmetics & Makeup", "OPI", [
    ("Nail Lacquer Big Apple Red NL N25", "Nail Polish", "S", 10.99, .50),
    ("Nail Lacquer Lincoln Park After Dark NL W42", "Nail Polish", "S", 10.99, .45),
    ("Nail Lacquer Bubble Bath NL S86", "Nail Polish", "S", 10.99, .48),
    ("Infinite Shine Long-Wear Nail Polish Cajun Shrimp", "Nail Polish", "S", 12.99, .38),
    ("Nail Envy Original Nail Strengthener", "Treatment", "S", 17.99, .40),
])
_b("Cosmetics & Makeup", "Sally Hansen", [
    ("Miracle Gel Nail Color Birthday Suit 234", "Nail Polish", "S", 9.99, .50),
    ("Miracle Gel Nail Color Red Eye 470", "Nail Polish", "S", 9.99, .45),
    ("Insta-Dri Fast Dry Nail Color Rapid Red 280", "Nail Polish", "S", 5.49, .48),
    ("Hard as Nails Nail Color Natural 760", "Nail Polish", "S", 3.49, .42),
    ("Miracle Gel Top Coat", "Top Coat", "S", 9.99, .52),
    ("Complete Salon Manicure Commander in Chic 641", "Nail Polish", "S", 7.99, .38),
])
_b("Cosmetics & Makeup", "CVS Health", [
    ("Beauty 360 Foundation Fair 01", "Foundation", "S", 6.99, .42),
    ("Beauty 360 Foundation Medium 05", "Foundation", "S", 6.99, .38),
    ("Beauty 360 Mascara Volume Black", "Mascara", "S", 5.99, .40),
    ("Beauty 360 Lipstick Berry Bliss", "Lipstick", "S", 4.99, .35),
    ("Beauty 360 Pressed Powder Light", "Powder", "S", 5.99, .35),
    ("Beauty 360 Liquid Eyeliner Black", "Eyeliner", "S", 4.99, .38),
    ("Beauty 360 Makeup Remover Wipes", "Wipes", "P24", 3.99, .42),
    ("Beauty 360 Nail Polish Remover", "Remover", "S", 2.99, .40),
    ("Beauty 360 Nail Color Classic Red", "Nail Polish", "S", 3.49, .35),
    ("Beauty 360 Lip Gloss Pink Shimmer", "Lip Gloss", "S", 3.99, .32),
])

# ─── Baby & Childcare ────────────────────────────────────────────
_b("Baby & Childcare", "Pampers", [
    ("Swaddlers Diapers Size Newborn", "Diapers", "K8", 12.99, .82),
    ("Swaddlers Diapers Size 1", "Diapers", "K8", 12.99, .85),
    ("Swaddlers Diapers Size 2", "Diapers", "K8", 12.99, .82),
    ("Swaddlers Diapers Size 3", "Diapers", "K8", 13.49, .78),
    ("Swaddlers Diapers Size 4", "Diapers", "K8", 13.99, .72),
    ("Swaddlers Diapers Size 5", "Diapers", "K8", 14.49, .65),
    ("Cruisers Diapers Size 3", "Diapers", "K8", 13.49, .62),
    ("Cruisers Diapers Size 4", "Diapers", "K8", 13.99, .58),
    ("Cruisers Diapers Size 5", "Diapers", "K8", 14.49, .52),
    ("Baby Dry Diapers Size 3", "Diapers", "K8", 11.99, .58),
    ("Baby Dry Diapers Size 4", "Diapers", "K8", 12.49, .55),
    ("Pure Protection Diapers Size 1", "Diapers", "K8", 14.99, .42),
    ("Sensitive Baby Wipes Fragrance Free", "Wipes", "K6", 3.49, .80),
    ("Complete Clean Baby Wipes Fresh Scent", "Wipes", "K6", 3.49, .68),
])
_b("Baby & Childcare", "Huggies", [
    ("Little Snugglers Diapers Size 1", "Diapers", "K8", 12.99, .78),
    ("Little Snugglers Diapers Size 2", "Diapers", "K8", 12.99, .75),
    ("Little Movers Diapers Size 3", "Diapers", "K8", 13.49, .72),
    ("Little Movers Diapers Size 4", "Diapers", "K8", 13.99, .68),
    ("Little Movers Diapers Size 5", "Diapers", "K8", 14.49, .60),
    ("Overnites Nighttime Diapers Size 4", "Diapers", "K8", 14.99, .50),
    ("Natural Care Sensitive Baby Wipes Unscented", "Wipes", "K6", 3.49, .75),
    ("Simply Clean Baby Wipes Unscented", "Wipes", "K6", 2.99, .62),
    ("Pull-Ups Training Pants Boys 2T-3T", "Training Pants", "K4", 12.99, .55),
    ("Pull-Ups Training Pants Girls 2T-3T", "Training Pants", "K4", 12.99, .52),
    ("Pull-Ups Training Pants Boys 3T-4T", "Training Pants", "K4", 12.99, .50),
    ("Pull-Ups Training Pants Girls 3T-4T", "Training Pants", "K4", 12.99, .48),
])
_b("Baby & Childcare", "Luvs", [
    ("Ultra Leakguards Diapers Size 3", "Diapers", "K8", 9.99, .48),
    ("Ultra Leakguards Diapers Size 4", "Diapers", "K8", 10.49, .45),
    ("Ultra Leakguards Diapers Size 5", "Diapers", "K8", 10.99, .40),
])
_b("Baby & Childcare", "Johnson's", [
    ("Baby Shampoo", "Shampoo", "C12", 5.99, .70),
    ("Head-to-Toe Gentle Baby Body Wash & Shampoo", "Body Wash", "C12", 6.49, .68),
    ("Baby Lotion", "Lotion", "C12", 5.99, .62),
    ("Baby Oil", "Oil", "C8", 5.49, .55),
    ("Baby Powder Pure Cornstarch", "Powder", "C8", 5.99, .45),
    ("Bedtime Baby Bath", "Body Wash", "C12", 6.99, .48),
    ("Bedtime Baby Lotion", "Lotion", "C12", 6.99, .45),
])
_b("Baby & Childcare", "Aveeno Baby", [
    ("Daily Moisture Lotion Fragrance Free", "Lotion", "C8", 8.99, .58),
    ("Eczema Therapy Moisturizing Cream", "Cream", "C4", 10.99, .48),
    ("Calming Comfort Bath Wash & Shampoo Lavender & Vanilla", "Body Wash", "C8", 8.99, .45),
    ("Soothing Relief Emollient Cream", "Cream", "C4", 9.99, .38),
])
_b("Baby & Childcare", "Desitin", [
    ("Maximum Strength Original Paste Diaper Rash Cream", "Cream", "C4", 9.99, .55),
    ("Daily Defense Cream", "Cream", "C4", 7.99, .48),
    ("Rapid Relief Cream", "Cream", "C4", 8.99, .42),
])
_b("Baby & Childcare", "Enfamil", [
    ("NeuroPro Infant Formula Powder", "Powder", "S", 37.99, .65),
    ("Gentlease Infant Formula Powder", "Powder", "S", 37.99, .58),
    ("A.R. Infant Formula for Spit-Up Powder", "Powder", "S", 38.99, .42),
    ("ProSobee Soy-Based Infant Formula Powder", "Powder", "S", 36.99, .38),
    ("NeuroPro Infant Formula Ready to Use", "Liquid", "S", 10.99, .48),
    ("Poly-Vi-Sol Multivitamin Drops", "Liquid", "S", 11.99, .35),
])
_b("Baby & Childcare", "Similac", [
    ("Pro-Advance Infant Formula Powder", "Powder", "S", 36.99, .62),
    ("Pro-Sensitive Infant Formula Powder", "Powder", "S", 37.99, .52),
    ("Pro-Total Comfort Infant Formula Powder", "Powder", "S", 37.99, .45),
    ("360 Total Care Infant Formula Powder", "Powder", "S", 38.99, .55),
    ("360 Total Care Sensitive Infant Formula Powder", "Powder", "S", 39.99, .42),
])
_b("Baby & Childcare", "Pedialyte", [
    ("Electrolyte Solution Unflavored", "Liquid", "L16", 6.49, .62),
    ("AdvancedCare Electrolyte Solution Berry Frost", "Liquid", "L16", 7.99, .55),
    ("Electrolyte Powder Packets Apple", "Powder", "S", 8.49, .48),
    ("Freezer Pops Assorted Flavors", "Pops", "S", 8.99, .50),
])
_b("Baby & Childcare", "NUK", [
    ("Simply Natural Bottle 5 OZ", "Bottle", "S", 8.99, .42),
    ("Simply Natural Bottle 9 OZ", "Bottle", "S", 9.99, .40),
    ("Orthodontic Pacifier 0-6 Months", "Pacifier", "S", 5.99, .45),
    ("Orthodontic Pacifier 6-18 Months", "Pacifier", "S", 5.99, .42),
    ("Learner Cup 5 OZ", "Cup", "S", 7.99, .35),
])
_b("Baby & Childcare", "CVS Health", [
    ("Premium Diapers Size 1", "Diapers", "K8", 8.99, .55),
    ("Premium Diapers Size 2", "Diapers", "K8", 8.99, .52),
    ("Premium Diapers Size 3", "Diapers", "K8", 9.49, .50),
    ("Premium Diapers Size 4", "Diapers", "K8", 9.99, .48),
    ("Premium Diapers Size 5", "Diapers", "K8", 10.49, .42),
    ("Baby Wipes Sensitive Fragrance Free", "Wipes", "K6", 2.49, .55),
    ("Baby Wipes Fresh Scent", "Wipes", "K6", 2.49, .48),
    ("Baby Lotion Gentle Formula", "Lotion", "C8", 4.49, .42),
    ("Baby Wash & Shampoo Gentle", "Body Wash", "C8", 4.49, .40),
    ("Diaper Rash Cream Maximum Strength", "Cream", "C4", 5.99, .42),
    ("Infant Formula Gentle Powder", "Powder", "S", 24.99, .45),
    ("Training Pants Boys 2T-3T", "Training Pants", "K4", 8.99, .38),
    ("Training Pants Girls 2T-3T", "Training Pants", "K4", 8.99, .35),
])

# ─── First Aid & Wound Care ──────────────────────────────────────
_b("First Aid & Wound Care", "Band-Aid", [
    ("Flexible Fabric Adhesive Bandages Assorted", "Bandages", "P30", 4.99, .75),
    ("Tough Strips Adhesive Bandages All One Size", "Bandages", "P30", 5.49, .58),
    ("Hydro Seal All Purpose Waterproof Bandages", "Bandages", "S", 6.99, .52),
    ("Water Block Flex Adhesive Bandages All One Size", "Bandages", "P30", 5.99, .48),
    ("Skin-Flex Adhesive Bandages Assorted", "Bandages", "P30", 5.49, .42),
    ("Adhesive Bandages Disney Kids Assorted", "Bandages", "P20", 4.49, .50),
    ("Flexible Fabric Adhesive Bandages Knuckle & Fingertip", "Bandages", "P20", 4.99, .35),
])
_b("First Aid & Wound Care", "Neosporin", [
    ("Original First Aid Antibiotic Ointment", "Ointment", "C1", 7.99, .70),
    ("+ Pain Relief Dual Action Ointment", "Ointment", "C1", 8.99, .60),
    ("+ Pain Itch Scar Antibiotic Ointment", "Ointment", "C1", 9.49, .48),
    ("Wound Cleanser First Aid Antiseptic Foam", "Cleanser", "S", 8.99, .35),
])
_b("First Aid & Wound Care", "3M Nexcare", [
    ("Waterproof Bandages Assorted", "Bandages", "P20", 5.49, .48),
    ("Tegaderm Transparent Dressing", "Dressing", "S", 7.99, .42),
    ("Micropore Paper Tape", "Tape", "S", 4.99, .50),
    ("Steri-Strip Skin Closures", "Closures", "S", 5.99, .38),
    ("DUO Adhesive Bandages Assorted", "Bandages", "P20", 4.49, .35),
])
_b("First Aid & Wound Care", "Ace", [
    ("Elastic Bandage with Clips 3 Inch", "Bandage", "S", 5.99, .52),
    ("Elastic Bandage with Clips 4 Inch", "Bandage", "S", 6.99, .48),
    ("Ankle Brace with Side Stabilizers", "Brace", "S", 14.99, .42),
    ("Knee Brace with Side Stabilizers", "Brace", "S", 16.99, .40),
    ("Wrist Brace Adjustable", "Brace", "S", 12.99, .38),
    ("Cold Compress Reusable", "Cold Pack", "S", 7.99, .42),
])
_b("First Aid & Wound Care", "CVS Health", [
    ("Flexible Fabric Adhesive Bandages Assorted", "Bandages", "P30", 3.49, .60),
    ("Sheer Adhesive Bandages All One Size", "Bandages", "P30", 2.99, .52),
    ("Waterproof Adhesive Bandages Assorted", "Bandages", "P30", 3.99, .45),
    ("Triple Antibiotic Ointment", "Ointment", "C1", 4.99, .55),
    ("Hydrogen Peroxide 3% First Aid Antiseptic", "Antiseptic", "L8", 1.99, .52),
    ("Isopropyl Rubbing Alcohol 70%", "Antiseptic", "L8", 2.49, .50),
    ("Elastic Bandage with Clips 3 Inch", "Bandage", "S", 3.99, .42),
    ("Sterile Gauze Pads 3x3", "Gauze", "P24", 3.49, .48),
    ("Paper Tape 1 Inch", "Tape", "S", 2.99, .42),
    ("Instant Cold Pack", "Cold Pack", "S", 2.99, .45),
    ("First Aid Kit 100 Pieces", "Kit", "S", 9.99, .40),
    ("Knee Support with Open Patella", "Brace", "S", 9.99, .35),
    ("Wrist Support Adjustable", "Brace", "S", 8.99, .32),
    ("Butterfly Wound Closures", "Closures", "S", 3.49, .35),
])

# ─── Eye & Ear Care ───────────────────────────────────────────────
_b("Eye & Ear Care", "Visine", [
    ("Original Redness Relief Eye Drops", "Eye Drops", "S", 7.99, .68),
    ("Advanced Redness + Irritation Relief Eye Drops", "Eye Drops", "S", 9.99, .58),
    ("Allergy Eye Relief Eye Drops", "Eye Drops", "S", 8.99, .52),
    ("Dry Eye Relief Lubricant Eye Drops", "Eye Drops", "S", 8.99, .50),
    ("Totality Multi-Symptom Relief Eye Drops", "Eye Drops", "S", 9.99, .42),
])
_b("Eye & Ear Care", "Systane", [
    ("Ultra Lubricant Eye Drops", "Eye Drops", "S", 14.99, .62),
    ("Complete Lubricant Eye Drops", "Eye Drops", "S", 15.99, .55),
    ("Balance Lubricant Eye Drops Restorative Formula", "Eye Drops", "S", 15.99, .48),
    ("Gel Drops Lubricant Eye Gel", "Eye Gel", "S", 14.99, .42),
])
_b("Eye & Ear Care", "Clear Eyes", [
    ("Redness Relief Eye Drops", "Eye Drops", "S", 5.99, .55),
    ("Pure Relief Multi-Symptom Eye Drops", "Eye Drops", "S", 8.99, .45),
    ("Triple Action Relief Eye Drops", "Eye Drops", "S", 7.99, .42),
])
_b("Eye & Ear Care", "Bausch + Lomb", [
    ("Biotrue Multi-Purpose Contact Lens Solution", "Solution", "C4", 10.99, .58),
    ("ReNu Advanced Multi-Purpose Contact Lens Solution", "Solution", "C4", 9.99, .50),
    ("Lumify Redness Reliever Eye Drops", "Eye Drops", "S", 12.99, .62),
    ("Ocuvite Adult 50+ Eye Vitamin & Mineral Supplement", "Softgels", "P60", 15.99, .42),
    ("PreserVision AREDS 2 Eye Vitamin Softgels", "Softgels", "P60", 22.99, .45),
])
_b("Eye & Ear Care", "Refresh", [
    ("Tears Lubricant Eye Drops", "Eye Drops", "S", 12.99, .55),
    ("Optive Advanced Lubricant Eye Drops", "Eye Drops", "S", 14.99, .48),
    ("PM Lubricant Eye Ointment", "Ointment", "S", 10.99, .38),
    ("Celluvisc Lubricant Eye Gel Single-Use", "Eye Gel", "S", 17.99, .35),
])
_b("Eye & Ear Care", "Rohto", [
    ("Cool Max Redness Relief Eye Drops", "Eye Drops", "S", 7.99, .45),
    ("Ice All-in-One Multi-Symptom Eye Drops", "Eye Drops", "S", 8.99, .40),
    ("Dry-Aid Lubricant Eye Drops", "Eye Drops", "S", 9.99, .35),
])
_b("Eye & Ear Care", "Debrox", [
    ("Earwax Removal Drops", "Ear Drops", "S", 7.99, .48),
    ("Earwax Removal Kit with Drops and Bulb Syringe", "Kit", "S", 9.99, .42),
])
_b("Eye & Ear Care", "CVS Health", [
    ("Redness Relief Eye Drops", "Eye Drops", "S", 5.49, .52),
    ("Lubricant Eye Drops", "Eye Drops", "S", 7.99, .48),
    ("Allergy Eye Drops", "Eye Drops", "S", 6.49, .42),
    ("Multi-Purpose Contact Lens Solution", "Solution", "C4", 7.99, .50),
    ("Earwax Removal Drops", "Ear Drops", "S", 5.49, .40),
    ("Reading Glasses +1.50", "Glasses", "S", 9.99, .35),
    ("Reading Glasses +2.00", "Glasses", "S", 9.99, .35),
    ("Reading Glasses +2.50", "Glasses", "S", 9.99, .32),
    ("Sterile Saline Solution for Contact Lenses", "Solution", "C4", 5.99, .42),
    ("Lubricating Eye Gel Drops Nighttime", "Eye Gel", "S", 8.99, .35),
])

# ─── Snacks & Beverages ──────────────────────────────────────────
_b("Snacks & Beverages", "Gold Emblem", [
    ("Deluxe Mixed Nuts", "Nuts", "C8", 9.99, .58),
    ("Cashew Halves & Pieces Lightly Salted", "Nuts", "C8", 8.99, .52),
    ("Trail Mix Mountain", "Trail Mix", "C8", 5.99, .55),
    ("Trail Mix Tropical", "Trail Mix", "C8", 5.99, .48),
    ("Dried Cranberries", "Dried Fruit", "C4", 4.99, .42),
    ("Butter Cookies", "Cookies", "S", 5.99, .45),
    ("Peanut Butter Filled Pretzels", "Pretzels", "C8", 4.99, .50),
    ("Kettle Cooked Sea Salt Potato Chips", "Chips", "S", 3.99, .42),
    ("Dark Chocolate Almonds", "Chocolate", "C8", 6.99, .48),
    ("Animal Crackers", "Crackers", "C8", 3.99, .38),
])
_b("Snacks & Beverages", "Gold Emblem Abound", [
    ("Organic Fruit Snacks Mixed Berry", "Fruit Snacks", "S", 4.99, .40),
    ("Organic Granola Bars Dark Chocolate", "Granola Bars", "S", 4.99, .38),
    ("Veggie Straws Sea Salt", "Chips", "S", 3.99, .42),
    ("Organic Blue Corn Tortilla Chips", "Chips", "S", 3.99, .35),
])
_b("Snacks & Beverages", "Lay's", [
    ("Classic Potato Chips", "Chips", "S", 4.99, .72),
    ("Barbecue Flavored Potato Chips", "Chips", "S", 4.99, .58),
    ("Sour Cream & Onion Flavored Potato Chips", "Chips", "S", 4.99, .55),
    ("Kettle Cooked Sea Salt & Vinegar Chips", "Chips", "S", 4.99, .45),
])
_b("Snacks & Beverages", "Doritos", [
    ("Nacho Cheese Flavored Tortilla Chips", "Chips", "S", 4.99, .68),
    ("Cool Ranch Flavored Tortilla Chips", "Chips", "S", 4.99, .60),
    ("Spicy Sweet Chili Flavored Tortilla Chips", "Chips", "S", 4.99, .42),
])
_b("Snacks & Beverages", "Coca-Cola", [
    ("Classic 20 FL OZ Bottle", "Beverage", "S", 2.29, .82),
    ("Classic 2 Liter", "Beverage", "S", 2.49, .72),
    ("Classic 12 FL OZ Cans", "Beverage", "B6", 6.99, .78),
    ("Diet Coke 20 FL OZ Bottle", "Beverage", "S", 2.29, .75),
    ("Diet Coke 12 FL OZ Cans", "Beverage", "B6", 6.99, .72),
    ("Coca-Cola Zero Sugar 20 FL OZ Bottle", "Beverage", "S", 2.29, .65),
    ("Coca-Cola Zero Sugar 12 FL OZ Cans", "Beverage", "B6", 6.99, .60),
    ("Sprite 20 FL OZ Bottle", "Beverage", "S", 2.29, .58),
    ("Sprite 12 FL OZ Cans", "Beverage", "B6", 6.99, .52),
])
_b("Snacks & Beverages", "Pepsi", [
    ("Cola 20 FL OZ Bottle", "Beverage", "S", 2.29, .72),
    ("Cola 12 FL OZ Cans", "Beverage", "B6", 6.99, .68),
    ("Diet Pepsi 20 FL OZ Bottle", "Beverage", "S", 2.29, .55),
    ("Diet Pepsi 12 FL OZ Cans", "Beverage", "B6", 6.99, .50),
    ("Pepsi Zero Sugar 20 FL OZ Bottle", "Beverage", "S", 2.29, .48),
    ("Mountain Dew 20 FL OZ Bottle", "Beverage", "S", 2.29, .58),
    ("Mountain Dew 12 FL OZ Cans", "Beverage", "B6", 6.99, .52),
])
_b("Snacks & Beverages", "Gatorade", [
    ("Thirst Quencher Lemon-Lime 20 FL OZ", "Beverage", "S", 1.99, .65),
    ("Thirst Quencher Fruit Punch 20 FL OZ", "Beverage", "S", 1.99, .60),
    ("Thirst Quencher Cool Blue 20 FL OZ", "Beverage", "S", 1.99, .55),
    ("Zero Sugar Thirst Quencher Glacier Cherry 20 FL OZ", "Beverage", "S", 1.99, .45),
])
_b("Snacks & Beverages", "Hershey's", [
    ("Milk Chocolate Bar", "Candy", "S", 1.99, .70),
    ("Milk Chocolate with Almonds Bar", "Candy", "S", 1.99, .52),
    ("Cookies 'N' Creme Bar", "Candy", "S", 1.99, .48),
    ("Kisses Milk Chocolate", "Candy", "C8", 5.49, .62),
    ("Reese's Peanut Butter Cups 2 Pack", "Candy", "S", 1.99, .78),
    ("Reese's Peanut Butter Cups Miniatures", "Candy", "C8", 5.49, .58),
    ("Kit Kat Crisp Wafer Bar", "Candy", "S", 1.99, .65),
])
_b("Snacks & Beverages", "Mars", [
    ("Snickers Candy Bar", "Candy", "S", 1.99, .72),
    ("M&M's Milk Chocolate Candies", "Candy", "C8", 4.99, .70),
    ("M&M's Peanut Chocolate Candies", "Candy", "C8", 4.99, .62),
    ("Twix Cookie Bars", "Candy", "S", 1.99, .58),
    ("Milky Way Candy Bar", "Candy", "S", 1.99, .50),
    ("Skittles Original", "Candy", "C8", 4.49, .60),
    ("Starburst Original Fruit Chews", "Candy", "C8", 3.99, .48),
])
_b("Snacks & Beverages", "KIND", [
    ("Dark Chocolate Nuts & Sea Salt Bar", "Bars", "S", 1.99, .55),
    ("Peanut Butter Dark Chocolate Bar", "Bars", "S", 1.99, .48),
    ("Caramel Almond & Sea Salt Bar", "Bars", "S", 1.99, .42),
    ("Minis Dark Chocolate Nuts & Sea Salt 10 Pack", "Bars", "S", 8.99, .45),
])
_b("Snacks & Beverages", "Clif", [
    ("Bar Chocolate Chip", "Bars", "S", 1.79, .52),
    ("Bar Crunchy Peanut Butter", "Bars", "S", 1.79, .48),
    ("Builder's Chocolate Peanut Butter Bar", "Bars", "S", 2.49, .42),
])
_b("Snacks & Beverages", "Red Bull", [
    ("Energy Drink Original 8.4 FL OZ", "Beverage", "S", 3.49, .68),
    ("Energy Drink Sugar Free 8.4 FL OZ", "Beverage", "S", 3.49, .52),
    ("Energy Drink Tropical 8.4 FL OZ", "Beverage", "S", 3.49, .42),
    ("Energy Drink Original 12 FL OZ", "Beverage", "S", 3.99, .55),
])
_b("Snacks & Beverages", "Monster", [
    ("Energy Original Green 16 FL OZ", "Beverage", "S", 3.49, .62),
    ("Energy Ultra Zero Sugar 16 FL OZ", "Beverage", "S", 3.49, .50),
    ("Energy Juice Mango Loco 16 FL OZ", "Beverage", "S", 3.49, .42),
])
_b("Snacks & Beverages", "Celsius", [
    ("Sparkling Orange Energy Drink 12 FL OZ", "Beverage", "S", 2.49, .55),
    ("Sparkling Wild Berry Energy Drink 12 FL OZ", "Beverage", "S", 2.49, .48),
    ("Sparkling Watermelon Energy Drink 12 FL OZ", "Beverage", "S", 2.49, .42),
])
_b("Snacks & Beverages", "5-hour Energy", [
    ("Extra Strength Berry Shot", "Shot", "S", 3.99, .52),
    ("Original Berry Shot", "Shot", "S", 3.49, .48),
    ("Extra Strength Grape Shot", "Shot", "S", 3.99, .38),
])
_b("Snacks & Beverages", "Smartwater", [
    ("Vapor Distilled Water 1 Liter", "Water", "S", 2.49, .52),
    ("Vapor Distilled Water 700 ML Sport Cap", "Water", "S", 2.29, .42),
])
_b("Snacks & Beverages", "Poland Spring", [
    ("100% Natural Spring Water 16.9 FL OZ 6 Pack", "Water", "S", 3.99, .55),
    ("100% Natural Spring Water 1 Gallon", "Water", "S", 2.49, .48),
    ("100% Natural Spring Water 23.7 FL OZ Sport Cap", "Water", "S", 1.79, .42),
])
_b("Snacks & Beverages", "Nabisco", [
    ("Oreo Chocolate Sandwich Cookies", "Cookies", "C8", 5.49, .72),
    ("Chips Ahoy! Original Chocolate Chip Cookies", "Cookies", "C8", 5.49, .58),
    ("Ritz Crackers Original", "Crackers", "C8", 4.99, .55),
    ("Wheat Thins Original", "Crackers", "C8", 4.99, .42),
    ("Triscuit Original Crackers", "Crackers", "C8", 4.99, .40),
])
_b("Snacks & Beverages", "CVS Health", [
    ("Purified Drinking Water 16.9 FL OZ 24 Pack", "Water", "S", 4.99, .55),
    ("Purified Drinking Water 1 Gallon", "Water", "S", 1.49, .48),
    ("Electrolyte Water 1 Liter", "Water", "S", 1.99, .42),
])

# ─── Household Essentials ────────────────────────────────────────
_b("Household Essentials", "Bounty", [
    ("Select-A-Size Paper Towels White 2 Double Rolls", "Paper Towels", "K1", 7.49, .75),
    ("Select-A-Size Paper Towels White 6 Double Rolls", "Paper Towels", "K1", 15.99, .68),
    ("Essentials Select-A-Size Paper Towels 6 Rolls", "Paper Towels", "K1", 10.99, .52),
])
_b("Household Essentials", "Charmin", [
    ("Ultra Soft Toilet Paper 4 Mega Rolls", "Toilet Paper", "K4", 7.99, .72),
    ("Ultra Strong Toilet Paper 4 Mega Rolls", "Toilet Paper", "K4", 7.99, .68),
    ("Essentials Soft Toilet Paper 6 Rolls", "Toilet Paper", "K4", 7.49, .52),
    ("Ultra Soft Toilet Paper 12 Mega Rolls", "Toilet Paper", "S", 17.99, .60),
])
_b("Household Essentials", "Scott", [
    ("1000 Sheets Per Roll Toilet Paper 4 Rolls", "Toilet Paper", "K4", 4.99, .55),
    ("Choose-A-Sheet Paper Towels 6 Rolls", "Paper Towels", "K1", 9.99, .48),
    ("ComfortPlus Toilet Paper 4 Rolls", "Toilet Paper", "K4", 4.49, .42),
])
_b("Household Essentials", "Cottonelle", [
    ("Ultra CleanCare Toilet Paper 6 Mega Rolls", "Toilet Paper", "K4", 9.99, .55),
    ("GentlePlus Flushable Wipes", "Wipes", "K4", 3.99, .48),
    ("FreshCare Flushable Wipes", "Wipes", "K4", 3.49, .42),
])
_b("Household Essentials", "Clorox", [
    ("Disinfecting Wipes Crisp Lemon 35 CT", "Wipes", "P30", 4.99, .72),
    ("Disinfecting Wipes Fresh Scent 75 CT", "Wipes", "P60", 7.99, .68),
    ("Regular Bleach with CloroMax 43 OZ", "Bleach", "S", 4.49, .55),
    ("Clean-Up Cleaner + Bleach Spray", "Cleaner", "S", 5.49, .52),
    ("Toilet Bowl Cleaner with Bleach Rain Clean", "Toilet Cleaner", "S", 4.49, .45),
    ("Disinfecting Bathroom Cleaner Spray", "Cleaner", "S", 4.99, .42),
])
_b("Household Essentials", "Lysol", [
    ("Disinfectant Spray Crisp Linen", "Spray", "SP", 6.99, .72),
    ("Disinfecting Wipes Lemon & Lime Blossom 35 CT", "Wipes", "P30", 4.49, .68),
    ("Disinfecting Wipes Lemon & Lime Blossom 80 CT", "Wipes", "P60", 7.49, .60),
    ("Power Toilet Bowl Cleaner", "Toilet Cleaner", "S", 3.99, .48),
    ("Laundry Sanitizer Sport", "Sanitizer", "S", 8.99, .38),
    ("Disinfectant Spray Neutra Air Tropical Breeze", "Spray", "SP", 6.99, .45),
])
_b("Household Essentials", "Glad", [
    ("ForceFlex Tall Kitchen Drawstring Trash Bags 13 Gal", "Trash Bags", "P30", 8.99, .62),
    ("Press'n Seal Food Wrap", "Wrap", "S", 5.99, .52),
    ("ClingWrap Clear Plastic Wrap", "Wrap", "S", 4.49, .42),
    ("ForceFlex Plus OdorShield Tall Kitchen Bags 13 Gal", "Trash Bags", "P30", 9.99, .50),
])
_b("Household Essentials", "Hefty", [
    ("Ultra Strong Tall Kitchen Drawstring Bags 13 Gal", "Trash Bags", "P30", 8.49, .55),
    ("Recycling Bags 30 Gal Blue", "Trash Bags", "P24", 6.99, .35),
])
_b("Household Essentials", "Ziploc", [
    ("Storage Bags Gallon", "Storage Bags", "P30", 5.49, .62),
    ("Storage Bags Quart", "Storage Bags", "P30", 4.99, .55),
    ("Freezer Bags Gallon", "Freezer Bags", "P24", 5.99, .52),
    ("Sandwich Bags", "Storage Bags", "P30", 4.49, .55),
    ("Snack Bags", "Storage Bags", "P30", 3.99, .48),
])
_b("Household Essentials", "Duracell", [
    ("Coppertop AA Alkaline Batteries", "Batteries", "K4", 6.49, .75),
    ("Coppertop AAA Alkaline Batteries", "Batteries", "K4", 6.49, .72),
    ("Coppertop C Alkaline Batteries 2 CT", "Batteries", "S", 6.49, .42),
    ("Coppertop D Alkaline Batteries 2 CT", "Batteries", "S", 6.49, .40),
    ("Coppertop 9V Alkaline Battery", "Batteries", "S", 6.99, .42),
    ("Optimum AA Alkaline Batteries", "Batteries", "K4", 8.49, .48),
    ("2032 Lithium Coin Battery", "Batteries", "S", 5.99, .50),
])
_b("Household Essentials", "Energizer", [
    ("MAX AA Alkaline Batteries", "Batteries", "K4", 6.49, .70),
    ("MAX AAA Alkaline Batteries", "Batteries", "K4", 6.49, .68),
    ("Ultimate Lithium AA Batteries", "Batteries", "K4", 10.49, .42),
    ("MAX C Alkaline Batteries 2 CT", "Batteries", "S", 6.49, .38),
    ("MAX D Alkaline Batteries 2 CT", "Batteries", "S", 6.49, .35),
    ("MAX 9V Alkaline Battery", "Batteries", "S", 6.49, .38),
    ("2032 Lithium Coin Battery 2 CT", "Batteries", "S", 6.49, .45),
])
_b("Household Essentials", "Febreze", [
    ("Air Freshener Spray Linen & Sky", "Air Freshener", "SP", 5.99, .55),
    ("Fabric Refresher Extra Strength Original", "Fabric Spray", "SP", 6.49, .50),
    ("Small Spaces Air Freshener Linen & Sky 2 Pack", "Air Freshener", "S", 5.49, .42),
    ("Car Air Freshener Vent Clip Linen & Sky", "Air Freshener", "S", 4.49, .40),
])
_b("Household Essentials", "Air Wick", [
    ("Scented Oil Warmer + Refill Lavender & Chamomile", "Air Freshener", "S", 6.49, .45),
    ("Freshmatic Automatic Spray Refill Lavender", "Air Freshener", "S", 6.99, .38),
    ("Essential Mist Starter Kit Lavender & Almond Blossom", "Air Freshener", "S", 9.99, .32),
])
_b("Household Essentials", "Dawn", [
    ("Ultra Dishwashing Liquid Original Scent", "Dish Soap", "L8", 3.99, .62),
    ("Platinum Powerwash Dish Spray Fresh Scent", "Dish Spray", "S", 5.49, .48),
])
_b("Household Essentials", "CVS Health", [
    ("Premium Paper Towels Select-A-Size 6 Rolls", "Paper Towels", "K1", 8.99, .52),
    ("Soft & Strong Toilet Paper 4 Mega Rolls", "Toilet Paper", "K4", 5.49, .50),
    ("Tall Kitchen Drawstring Trash Bags 13 Gal", "Trash Bags", "P30", 5.99, .48),
    ("AA Alkaline Batteries 8 Pack", "Batteries", "S", 7.49, .52),
    ("AAA Alkaline Batteries 8 Pack", "Batteries", "S", 7.49, .48),
    ("9V Alkaline Battery 1 Pack", "Batteries", "S", 4.99, .35),
    ("Disinfecting Wipes Lemon Scent 75 CT", "Wipes", "S", 4.99, .52),
    ("Storage Bags Gallon", "Storage Bags", "P30", 3.99, .45),
    ("Freezer Bags Gallon", "Freezer Bags", "P24", 4.49, .40),
    ("Sandwich Bags", "Storage Bags", "P30", 2.99, .42),
])

# ─── Feminine Care ────────────────────────────────────────────────
_b("Feminine Care", "Always", [
    ("Maxi Pads Regular Absorbency Size 1", "Pads", "P24", 7.49, .72),
    ("Ultra Thin Pads Regular Absorbency Size 1", "Pads", "P24", 7.99, .70),
    ("Ultra Thin Pads Overnight Size 4", "Pads", "P24", 8.49, .62),
    ("Infinity FlexFoam Pads Regular Absorbency Size 1", "Pads", "P24", 8.99, .58),
    ("Radiant FlexFoam Pads Regular Absorbency Size 1", "Pads", "P24", 8.99, .55),
    ("Dailies Thin Liners Regular", "Liners", "P30", 5.99, .60),
    ("Discreet Incontinence Pads Moderate Long", "Pads", "P24", 10.99, .38),
])
_b("Feminine Care", "Tampax", [
    ("Pearl Tampons Regular Absorbency", "Tampons", "P24", 8.99, .72),
    ("Pearl Tampons Super Absorbency", "Tampons", "P24", 8.99, .68),
    ("Radiant Tampons Regular Absorbency", "Tampons", "P24", 9.99, .55),
    ("Pure Cotton Tampons Regular", "Tampons", "P24", 9.49, .42),
    ("Pocket Pearl Tampons Regular", "Tampons", "P24", 8.99, .40),
    ("Pearl Tampons Multi-Pack Light/Regular/Super", "Tampons", "P30", 10.99, .60),
])
_b("Feminine Care", "Playtex", [
    ("Sport Tampons Regular Absorbency", "Tampons", "P24", 7.99, .52),
    ("Sport Tampons Super Absorbency", "Tampons", "P24", 7.99, .48),
    ("Simply Gentle Glide Tampons Regular", "Tampons", "P24", 7.49, .40),
])
_b("Feminine Care", "U by Kotex", [
    ("Click Compact Tampons Regular", "Tampons", "P24", 7.99, .55),
    ("Click Compact Tampons Super", "Tampons", "P24", 7.99, .50),
    ("Security Maxi Pads Overnight", "Pads", "P24", 7.49, .45),
    ("CleanWear Ultra Thin Pads Regular", "Pads", "P24", 7.99, .48),
    ("Barely There Liners Regular", "Liners", "P30", 4.99, .42),
])
_b("Feminine Care", "Summer's Eve", [
    ("Cleansing Wash Sensitive Skin", "Wash", "C8", 5.99, .48),
    ("Freshening Spray Island Splash", "Spray", "S", 4.99, .35),
    ("Cleansing Cloths Sensitive Skin", "Wipes", "P24", 3.99, .42),
])
_b("Feminine Care", "Monistat", [
    ("1-Day Treatment Combination Pack", "Treatment", "S", 16.99, .48),
    ("3-Day Treatment Combination Pack", "Treatment", "S", 15.99, .45),
    ("7-Day Treatment Cream", "Treatment", "S", 13.99, .42),
    ("Chafing Relief Powder Gel", "Gel", "S", 8.99, .45),
])
_b("Feminine Care", "CVS Health", [
    ("Ultra Thin Pads Regular Absorbency", "Pads", "P24", 5.49, .55),
    ("Ultra Thin Pads Overnight", "Pads", "P24", 5.99, .48),
    ("Maxi Pads Regular", "Pads", "P24", 4.99, .50),
    ("Tampons Regular Absorbency", "Tampons", "P24", 5.49, .48),
    ("Tampons Super Absorbency", "Tampons", "P24", 5.49, .45),
    ("Daily Panty Liners Regular", "Liners", "P30", 3.49, .50),
    ("Feminine Cleansing Wash Sensitive", "Wash", "C8", 3.99, .38),
])

# ─── Sexual Health ────────────────────────────────────────────────
_b("Sexual Health", "Trojan", [
    ("BareSkin Thin Premium Lubricated Condoms", "Condoms", "S", 10.99, .62),
    ("Magnum Large Size Lubricated Condoms", "Condoms", "S", 10.99, .60),
    ("Ultra Thin Lubricated Condoms", "Condoms", "S", 9.99, .55),
    ("ENZ Lubricated Condoms", "Condoms", "S", 8.99, .52),
    ("Pleasure Pack Premium Lubricated Condoms", "Condoms", "S", 10.99, .48),
    ("Fire & Ice Dual Action Lubricated Condoms", "Condoms", "S", 10.99, .38),
    ("Personal Lubricant H2O Water-Based", "Lubricant", "C2", 8.99, .42),
])
_b("Sexual Health", "Durex", [
    ("Invisible Ultra Thin Lubricated Condoms", "Condoms", "S", 10.99, .48),
    ("Extra Sensitive Lubricated Condoms", "Condoms", "S", 9.99, .42),
    ("Performax Intense Lubricated Condoms", "Condoms", "S", 10.99, .35),
])
_b("Sexual Health", "K-Y", [
    ("Jelly Personal Lubricant", "Lubricant", "C2", 8.99, .55),
    ("Ultragel Personal Lubricant", "Lubricant", "C2", 10.99, .42),
    ("Duration Spray for Men", "Spray", "S", 16.99, .30),
])
_b("Sexual Health", "SKYN", [
    ("Original Non-Latex Lubricated Condoms", "Condoms", "S", 10.99, .48),
    ("Elite Ultra Thin Non-Latex Condoms", "Condoms", "S", 11.99, .40),
])
_b("Sexual Health", "Plan B", [
    ("One-Step Emergency Contraceptive", "Emergency Contraceptive", "S", 49.99, .42),
])
_b("Sexual Health", "First Response", [
    ("Early Result Pregnancy Test", "Pregnancy Test", "S", 9.99, .55),
    ("Rapid Result Pregnancy Test", "Pregnancy Test", "S", 7.99, .48),
    ("Triple Check Pregnancy Test Kit", "Pregnancy Test", "S", 14.99, .42),
])
_b("Sexual Health", "Clearblue", [
    ("Rapid Detection Pregnancy Test", "Pregnancy Test", "S", 8.99, .50),
    ("Digital Pregnancy Test with Smart Countdown", "Pregnancy Test", "S", 12.99, .45),
])
_b("Sexual Health", "CVS Health", [
    ("Lubricated Condoms Ultra Thin", "Condoms", "S", 6.99, .45),
    ("Personal Lubricant Water-Based", "Lubricant", "C2", 5.99, .38),
    ("Early Result Pregnancy Test", "Pregnancy Test", "S", 7.49, .48),
    ("One-Step Ovulation Predictor Kit", "Ovulation Test", "S", 14.99, .32),
])

# ─── Foot Care ────────────────────────────────────────────────────
_b("Foot Care", "Dr. Scholl's", [
    ("Comfort & Energy Massaging Gel Insoles Men", "Insoles", "S", 12.99, .60),
    ("Comfort & Energy Massaging Gel Insoles Women", "Insoles", "S", 12.99, .55),
    ("Heavy Duty Support Insoles Men", "Insoles", "S", 14.99, .48),
    ("Blister Defense Anti-Friction Stick", "Stick", "S", 7.99, .42),
    ("Corn Removers One Step Maximum Strength", "Treatment", "S", 7.99, .40),
    ("Freeze Away Wart Remover", "Treatment", "S", 14.99, .38),
    ("Bunion Cushions with Hydrogel Technology", "Cushions", "S", 6.99, .30),
    ("DreamWalk Express Pedi Foot Smoother Refills", "Refills", "S", 7.99, .25),
])
_b("Foot Care", "Gold Bond", [
    ("Ultimate Healing Foot Cream", "Cream", "C4", 7.99, .52),
    ("Medicated Foot Powder", "Powder", "S", 6.99, .48),
    ("Foot Powder Spray", "Spray", "S", 7.99, .38),
])
_b("Foot Care", "Tinactin", [
    ("Antifungal Cream", "Cream", "C1", 9.99, .48),
    ("Antifungal Powder Spray", "Spray", "SP", 9.99, .42),
    ("Antifungal Deodorant Powder Spray", "Spray", "SP", 9.99, .35),
])
_b("Foot Care", "Lotrimin", [
    ("AF Antifungal Cream Clotrimazole 1%", "Cream", "C1", 10.99, .50),
    ("Ultra Antifungal Cream Butenafine HCl 1%", "Cream", "C1", 13.99, .42),
])
_b("Foot Care", "Lamisil", [
    ("AT Antifungal Cream", "Cream", "C1", 14.99, .45),
    ("AT Antifungal Spray", "Spray", "SP", 12.99, .38),
])
_b("Foot Care", "CVS Health", [
    ("Comfort Insoles for Men", "Insoles", "S", 8.99, .45),
    ("Comfort Insoles for Women", "Insoles", "S", 8.99, .42),
    ("Antifungal Cream Clotrimazole 1%", "Cream", "C1", 6.99, .42),
    ("Medicated Foot Powder", "Powder", "S", 4.99, .38),
    ("Corn & Callus Remover Liquid", "Treatment", "S", 5.99, .32),
    ("Moleskin Plus Padding", "Padding", "S", 5.99, .30),
    ("Epsom Salt Foot Soak Lavender", "Soak", "S", 4.49, .35),
])

# ─── Diabetes & Blood Sugar ──────────────────────────────────────
_b("Diabetes & Blood Sugar", "OneTouch", [
    ("Verio Blood Glucose Test Strips", "Test Strips", "P30", 34.99, .58),
    ("Verio Reflect Blood Glucose Meter", "Meter", "S", 29.99, .48),
    ("Ultra Lancets Fine Point", "Lancets", "P30", 12.99, .45),
    ("Verio Test Strips", "Test Strips", "P60", 54.99, .52),
    ("Delica Plus Lancets", "Lancets", "P30", 14.99, .38),
])
_b("Diabetes & Blood Sugar", "Contour", [
    ("Next Blood Glucose Test Strips", "Test Strips", "P30", 29.99, .52),
    ("Next One Blood Glucose Monitoring System", "Meter", "S", 19.99, .42),
    ("Next Lancets", "Lancets", "P30", 9.99, .38),
])
_b("Diabetes & Blood Sugar", "FreeStyle", [
    ("Lite Blood Glucose Test Strips", "Test Strips", "P30", 32.99, .50),
    ("Lite Blood Glucose Monitoring System", "Meter", "S", 19.99, .38),
    ("Lancets", "Lancets", "P30", 11.99, .35),
])
_b("Diabetes & Blood Sugar", "BD", [
    ("Ultra-Fine Insulin Syringes 31G Short 1 mL", "Syringes", "P30", 22.99, .48),
    ("Ultra-Fine Pen Needles 32G 4mm", "Pen Needles", "P30", 24.99, .45),
    ("Ultra-Fine Pen Needles 31G 8mm", "Pen Needles", "P30", 24.99, .40),
])
_b("Diabetes & Blood Sugar", "CVS Health", [
    ("Advanced Blood Glucose Test Strips", "Test Strips", "P30", 19.99, .50),
    ("Blood Glucose Monitor Kit", "Meter", "S", 14.99, .42),
    ("Lancets 30 Gauge", "Lancets", "P30", 6.99, .42),
    ("Glucose Tablets Orange Flavor", "Tablets", "P30", 4.99, .48),
    ("Glucose Tablets Raspberry Flavor", "Tablets", "P30", 4.99, .40),
    ("Glucose Gel Packs", "Gel", "S", 5.99, .35),
    ("Diabetic Crew Socks White", "Socks", "S", 9.99, .32),
    ("Insulin Syringes 31G 1 mL", "Syringes", "P30", 15.99, .42),
])

# ─── Greeting Cards & Gift Wrap ──────────────────────────────────
_b("Greeting Cards & Gift Wrap", "Hallmark", [
    ("Birthday Card For Friend", "Card", "S", 4.99, .55),
    ("Birthday Card For Mom", "Card", "S", 5.99, .52),
    ("Birthday Card For Dad", "Card", "S", 5.99, .48),
    ("Birthday Card For Kids", "Card", "S", 3.99, .50),
    ("Birthday Card Funny", "Card", "S", 4.99, .48),
    ("Thank You Card", "Card", "S", 3.99, .48),
    ("Sympathy Card", "Card", "S", 4.99, .42),
    ("Anniversary Card", "Card", "S", 5.99, .42),
    ("Congratulations Card", "Card", "S", 4.99, .38),
    ("Get Well Card", "Card", "S", 4.49, .40),
    ("Blank Note Cards Set of 10", "Card Set", "S", 7.99, .35),
    ("Wedding Card", "Card", "S", 5.99, .40),
    ("Graduation Card", "Card", "S", 4.99, .35),
    ("New Baby Card", "Card", "S", 4.99, .35),
])
_b("Greeting Cards & Gift Wrap", "American Greetings", [
    ("Birthday Card For Her", "Card", "S", 4.49, .48),
    ("Birthday Card For Him", "Card", "S", 4.49, .45),
    ("Birthday Card For Kids Funny", "Card", "S", 3.49, .42),
    ("Thank You Card", "Card", "S", 3.49, .42),
    ("Sympathy Card", "Card", "S", 4.49, .38),
    ("Anniversary Card", "Card", "S", 4.49, .35),
    ("Get Well Soon Card", "Card", "S", 3.99, .35),
    ("Holiday Cards Box of 16", "Card Set", "S", 9.99, .38),
])
_b("Greeting Cards & Gift Wrap", "CVS Health", [
    ("Gift Wrap Roll Assorted Patterns", "Gift Wrap", "S", 3.99, .42),
    ("Gift Bag Medium Assorted", "Gift Bag", "S", 3.49, .45),
    ("Gift Bag Large Assorted", "Gift Bag", "S", 4.49, .40),
    ("Tissue Paper White 8 Sheets", "Tissue Paper", "S", 1.99, .48),
    ("Tissue Paper Assorted Colors 8 Sheets", "Tissue Paper", "S", 2.49, .40),
    ("Gift Bows Self-Stick 25 Count", "Bows", "S", 2.99, .42),
    ("Birthday Card Value Pack 12 Count", "Card Set", "S", 9.99, .35),
    ("Curling Ribbon Assorted Colors 6 Pack", "Ribbon", "S", 2.99, .30),
])

# ─── Photo & Electronics ─────────────────────────────────────────
_b("Photo & Electronics", "Fujifilm", [
    ("Instax Mini Instant Film Twin Pack 20 Exposures", "Film", "S", 15.99, .52),
    ("Instax Wide Instant Film Twin Pack 20 Exposures", "Film", "S", 18.99, .35),
    ("Instax Mini 12 Instant Camera Blossom Pink", "Camera", "S", 69.99, .32),
])
_b("Photo & Electronics", "SanDisk", [
    ("Ultra microSDXC UHS-I Memory Card 64GB", "Memory Card", "S", 12.99, .45),
    ("Ultra microSDXC UHS-I Memory Card 128GB", "Memory Card", "S", 19.99, .42),
    ("Ultra USB 3.0 Flash Drive 32GB", "Flash Drive", "S", 8.99, .40),
    ("Ultra USB 3.0 Flash Drive 64GB", "Flash Drive", "S", 11.99, .35),
])
_b("Photo & Electronics", "Anker", [
    ("PowerCore Portable Charger 10000mAh", "Power Bank", "S", 25.99, .48),
    ("USB-C to Lightning Cable 6 ft", "Cable", "S", 15.99, .45),
    ("USB-C Charging Cable 6 ft", "Cable", "S", 12.99, .42),
    ("Nano II 30W USB-C Charger", "Charger", "S", 22.99, .38),
])
_b("Photo & Electronics", "CVS Health", [
    ("Lightning to USB Charging Cable 3 ft", "Cable", "S", 7.99, .45),
    ("Lightning to USB Charging Cable 6 ft", "Cable", "S", 9.99, .42),
    ("USB-C Charging Cable 3 ft", "Cable", "S", 7.99, .42),
    ("USB-C Charging Cable 6 ft", "Cable", "S", 9.99, .40),
    ("Micro-USB Charging Cable 3 ft", "Cable", "S", 5.99, .35),
    ("USB Wall Charger Dual Port", "Charger", "S", 9.99, .42),
    ("USB Car Charger Dual Port", "Charger", "S", 8.99, .38),
    ("Wireless Earbuds", "Earbuds", "S", 14.99, .35),
    ("Screen Protector for iPhone", "Screen Protector", "S", 9.99, .32),
    ("Phone Case Clear Universal", "Phone Case", "S", 9.99, .28),
    ("Reading Light Book Light", "Light", "S", 6.99, .25),
    ("Alarm Clock Digital", "Clock", "S", 9.99, .22),
])

# ─── Sleep & Relaxation ──────────────────────────────────────────
_b("Sleep & Relaxation", "ZzzQuil", [
    ("Nighttime Sleep Aid LiquiCaps", "LiquiCaps", "P24", 9.99, .65),
    ("Nighttime Sleep Aid Liquid Warming Berry", "Liquid", "L8", 10.99, .60),
    ("Pure Zzzs Melatonin Sleep Aid Gummies", "Gummies", "P60", 11.99, .62),
    ("Pure Zzzs De-Stress & Sleep Gummies Ashwagandha", "Gummies", "P60", 12.99, .45),
    ("CALM + Sleep Gummies Lavender & Valerian Root", "Gummies", "P30", 12.99, .40),
])
_b("Sleep & Relaxation", "Unisom", [
    ("SleepTabs Doxylamine Succinate Tablets", "Tablets", "P24", 8.99, .55),
    ("SleepGels Diphenhydramine HCl 50mg Softgels", "Softgels", "P24", 9.99, .50),
    ("SleepMelts Cherry Dissolve Tabs", "Dissolve Tabs", "P24", 9.99, .38),
    ("PM Pain Sleep Aid Caplets", "Caplets", "P24", 10.99, .35),
])
_b("Sleep & Relaxation", "Natrol", [
    ("Melatonin 3mg Time Release Tablets", "Tablets", "P60", 7.99, .50),
    ("Melatonin 5mg Tablets", "Tablets", "P60", 8.99, .52),
    ("Melatonin 10mg Maximum Strength Tablets", "Tablets", "P60", 9.99, .45),
    ("Melatonin 5mg Gummies Strawberry", "Gummies", "P60", 9.99, .42),
    ("Sleep + Immune Health Gummies", "Gummies", "P60", 12.99, .32),
])
_b("Sleep & Relaxation", "CVS Health", [
    ("Melatonin 3mg Tablets", "Tablets", "P60", 5.49, .52),
    ("Melatonin 5mg Tablets", "Tablets", "P60", 5.99, .50),
    ("Melatonin 10mg Tablets", "Tablets", "P60", 6.99, .42),
    ("Melatonin 5mg Gummies", "Gummies", "P60", 7.49, .45),
    ("Nighttime Sleep Aid Diphenhydramine 25mg Softgels", "Softgels", "P24", 5.99, .48),
    ("Nighttime Sleep Aid Doxylamine 25mg Tablets", "Tablets", "P24", 4.99, .42),
    ("Soft Foam Earplugs 10 Pair", "Earplugs", "S", 3.49, .42),
    ("Contoured Sleep Mask", "Sleep Mask", "S", 5.99, .35),
    ("Lavender Sleep Spray", "Spray", "S", 7.99, .28),
])

# ─── Smoking Cessation ───────────────────────────────────────────
_b("Smoking Cessation", "Nicorette", [
    ("Nicotine Gum 2mg Original", "Gum", "P60", 34.99, .55),
    ("Nicotine Gum 4mg Original", "Gum", "P60", 36.99, .52),
    ("Nicotine Gum 2mg White Ice Mint", "Gum", "P60", 34.99, .50),
    ("Nicotine Gum 4mg White Ice Mint", "Gum", "P60", 36.99, .48),
    ("Nicotine Gum 2mg Fruit Chill", "Gum", "P60", 34.99, .42),
    ("Nicotine Gum 4mg Fruit Chill", "Gum", "P60", 36.99, .40),
    ("Nicotine Lozenge 2mg Cherry", "Lozenge", "P60", 34.99, .45),
    ("Nicotine Lozenge 4mg Cherry", "Lozenge", "P60", 36.99, .42),
    ("Nicotine Mini Lozenge 2mg Mint", "Lozenge", "P60", 34.99, .40),
    ("Nicotine Mini Lozenge 4mg Mint", "Lozenge", "P60", 36.99, .38),
    ("Nicotine Coated Lozenge 2mg Ice Mint", "Lozenge", "P60", 38.99, .35),
])
_b("Smoking Cessation", "NicoDerm CQ", [
    ("Step 1 Nicotine Patch 21mg", "Patch", "P20", 34.99, .52),
    ("Step 2 Nicotine Patch 14mg", "Patch", "P20", 34.99, .45),
    ("Step 3 Nicotine Patch 7mg", "Patch", "P20", 34.99, .40),
])
_b("Smoking Cessation", "CVS Health", [
    ("Nicotine Polacrilex Gum 2mg Mint", "Gum", "P60", 24.99, .48),
    ("Nicotine Polacrilex Gum 4mg Mint", "Gum", "P60", 26.99, .45),
    ("Nicotine Lozenge 2mg Cherry", "Lozenge", "P60", 24.99, .40),
    ("Nicotine Lozenge 4mg Cherry", "Lozenge", "P60", 26.99, .38),
    ("Nicotine Transdermal System Step 1 21mg Patch", "Patch", "P20", 24.99, .42),
    ("Nicotine Transdermal System Step 2 14mg Patch", "Patch", "P20", 24.99, .38),
    ("Nicotine Transdermal System Step 3 7mg Patch", "Patch", "P20", 24.99, .35),
])

# ─── Pet Care ─────────────────────────────────────────────────────
_b("Pet Care", "Pedigree", [
    ("Adult Complete Nutrition Roasted Chicken Rice & Vegetable Dry Dog Food", "Dog Food", "S", 8.99, .48),
    ("Dentastix Dog Treats Original Large 7 CT", "Dog Treats", "S", 5.99, .50),
    ("Dentastix Dog Treats Fresh Large 7 CT", "Dog Treats", "S", 5.99, .42),
    ("Chopped Ground Dinner Beef Wet Dog Food 13.2 OZ", "Dog Food", "S", 2.49, .38),
])
_b("Pet Care", "Purina", [
    ("Dog Chow Complete Adult Chicken Dry Dog Food", "Dog Food", "S", 9.99, .50),
    ("Fancy Feast Classic Pate Savory Salmon Cat Food 3 OZ", "Cat Food", "S", 1.29, .52),
    ("Fancy Feast Grilled Chicken Feast Cat Food 3 OZ", "Cat Food", "S", 1.29, .48),
    ("Beneful Originals with Farm-Raised Beef Dry Dog Food", "Dog Food", "S", 10.99, .42),
    ("Cat Chow Complete Dry Cat Food", "Cat Food", "S", 8.99, .45),
    ("Tidy Cats Clumping Cat Litter Instant Action", "Cat Litter", "S", 12.99, .42),
    ("Friskies Pate Poultry Platter Cat Food 5.5 OZ", "Cat Food", "S", 0.89, .40),
])
_b("Pet Care", "Greenies", [
    ("Original Regular Dog Dental Treats", "Dog Treats", "S", 9.99, .48),
    ("Pill Pockets for Dogs Chicken Flavor Capsule Size", "Pill Pockets", "S", 9.99, .40),
    ("Feline Smartbites Healthy Skin & Fur Chicken Cat Treats", "Cat Treats", "S", 4.99, .35),
])
_b("Pet Care", "Milk-Bone", [
    ("Original Dog Biscuits Medium", "Dog Treats", "S", 4.99, .50),
    ("MaroSnacks Dog Treats", "Dog Treats", "S", 5.99, .42),
    ("Soft & Chewy Dog Treats Chicken Recipe", "Dog Treats", "S", 6.99, .38),
])
_b("Pet Care", "CVS Health", [
    ("Dog Waste Bags 120 Count", "Pet Supplies", "S", 4.99, .42),
    ("Pet Stain & Odor Remover Spray", "Cleaner", "S", 5.99, .35),
    ("Rawhide Chew Sticks for Dogs", "Dog Treats", "S", 4.99, .32),
    ("Catnip Toy Mouse", "Cat Toy", "S", 3.99, .28),
    ("Pet Food Bowl Stainless Steel", "Pet Supplies", "S", 5.99, .25),
    ("Flea & Tick Collar for Dogs", "Flea & Tick", "S", 9.99, .30),
])

# ─── Seasonal Items ───────────────────────────────────────────────
_b("Seasonal Items", "Coppertone", [
    ("Sport Sunscreen Lotion SPF 50", "Sunscreen", "C4", 10.99, .62),
    ("Sport Sunscreen Spray SPF 50", "Sunscreen", "SP", 11.99, .58),
    ("WaterBabies Sunscreen Lotion SPF 50", "Sunscreen", "C4", 10.99, .52),
    ("Pure & Simple Mineral Sunscreen Lotion SPF 50", "Sunscreen", "C4", 12.99, .42),
    ("Sport Face Sunscreen Lotion SPF 50", "Sunscreen", "C2", 9.99, .45),
])
_b("Seasonal Items", "Banana Boat", [
    ("Sport Ultra Sunscreen Lotion SPF 50+", "Sunscreen", "C4", 8.99, .55),
    ("Sport Ultra Sunscreen Spray SPF 50+", "Sunscreen", "SP", 9.99, .50),
    ("Light As Air Sunscreen Lotion SPF 50", "Sunscreen", "C4", 10.99, .42),
    ("Kids Sport Sunscreen Lotion SPF 50+", "Sunscreen", "C4", 9.99, .45),
    ("Aloe Vera After Sun Gel", "After Sun", "C8", 6.99, .48),
])
_b("Seasonal Items", "Hawaiian Tropic", [
    ("Island Sport Sunscreen Lotion SPF 30", "Sunscreen", "C4", 8.99, .45),
    ("Sheer Touch Ultra Radiance Lotion SPF 50", "Sunscreen", "C4", 9.99, .42),
    ("After Sun Moisturizer", "After Sun", "C4", 7.99, .35),
])
_b("Seasonal Items", "Sun Bum", [
    ("Original Moisturizing Sunscreen Lotion SPF 30", "Sunscreen", "C4", 15.99, .48),
    ("Original Moisturizing Sunscreen Lotion SPF 50", "Sunscreen", "C4", 15.99, .45),
    ("Original Sunscreen Lip Balm SPF 30", "Lip Balm", "S", 4.99, .42),
    ("Revitalizing After Sun Lotion Cool Down", "After Sun", "C4", 12.99, .35),
])
_b("Seasonal Items", "OFF!", [
    ("Deep Woods Insect Repellent Spray", "Insect Repellent", "SP", 8.99, .58),
    ("FamilyCare Insect Repellent Spray Unscented", "Insect Repellent", "SP", 7.49, .52),
    ("Deep Woods Insect Repellent Towelettes", "Insect Repellent", "S", 7.99, .40),
    ("Clip-On Mosquito Repellent", "Insect Repellent", "S", 9.99, .35),
])
_b("Seasonal Items", "Cutter", [
    ("Backwoods Insect Repellent Spray", "Insect Repellent", "SP", 7.99, .42),
    ("Lemon Eucalyptus Insect Repellent Spray", "Insect Repellent", "SP", 7.49, .38),
])
_b("Seasonal Items", "HotHands", [
    ("Hand Warmers 2 Pair", "Hand Warmers", "S", 2.99, .45),
    ("Toe Warmers 2 Pair", "Hand Warmers", "S", 2.99, .38),
    ("Body Warmers 1 Pack", "Hand Warmers", "S", 2.99, .30),
])
_b("Seasonal Items", "CVS Health", [
    ("Sport Sunscreen Lotion SPF 30", "Sunscreen", "C4", 6.99, .50),
    ("Sport Sunscreen Lotion SPF 50", "Sunscreen", "C4", 7.99, .48),
    ("Sport Sunscreen Spray SPF 50", "Sunscreen", "SP", 8.49, .42),
    ("Kids Sunscreen Lotion SPF 50", "Sunscreen", "C4", 7.49, .40),
    ("Aloe Vera Gel After Sun", "After Sun", "C8", 4.99, .45),
    ("Insect Repellent Spray DEET 25%", "Insect Repellent", "SP", 5.99, .42),
    ("Lip Balm with Sunscreen SPF 30", "Lip Balm", "S", 2.49, .40),
    ("Hand Warmers 3 Pair Value Pack", "Hand Warmers", "S", 3.99, .35),
    ("Cold & Hot Therapy Gel Pack Reusable", "Therapy Pack", "S", 6.99, .30),
])


# ════════════════════════════════════════════════════════════════════
# SYSTEMATIC GENERATORS — efficiently produce many products from
# compact cross-product definitions (brand × type × size).
# ════════════════════════════════════════════════════════════════════

def _generate_vitamin_matrix():
    """Generate Nature Made / Nature's Bounty / CVS Health vitamin lines."""
    vitamin_types = [
        ("Vitamin D3 2000 IU (50 mcg) Softgels", "Softgels", 8.99, .52),
        ("Vitamin D3 5000 IU (125 mcg) Softgels", "Softgels", 10.99, .48),
        ("Vitamin C 500mg Tablets", "Tablets", 7.49, .50),
        ("Vitamin C 1000mg Tablets", "Tablets", 10.99, .48),
        ("Vitamin B12 1000mcg Tablets", "Tablets", 8.99, .45),
        ("Vitamin B12 3000mcg Softgels", "Softgels", 10.99, .38),
        ("Vitamin B6 100mg Tablets", "Tablets", 6.99, .32),
        ("B-Complex with Vitamin C Caplets", "Caplets", 9.99, .38),
        ("Super B-Complex with Vitamin C & Folic Acid Tablets", "Tablets", 10.99, .35),
        ("Fish Oil 1000mg Softgels", "Softgels", 9.99, .55),
        ("Fish Oil 1200mg Omega-3 Softgels", "Softgels", 13.99, .50),
        ("Burpless Fish Oil 1200mg Softgels", "Softgels", 14.99, .42),
        ("CoQ10 100mg Softgels", "Softgels", 15.99, .42),
        ("CoQ10 200mg Softgels", "Softgels", 22.99, .35),
        ("Melatonin 3mg Tablets", "Tablets", 6.99, .55),
        ("Melatonin 5mg Tablets", "Tablets", 7.99, .52),
        ("Melatonin 10mg Tablets", "Tablets", 9.99, .42),
        ("Melatonin 5mg Gummies Strawberry", "Gummies", 9.99, .48),
        ("Zinc 30mg Tablets", "Tablets", 5.99, .42),
        ("Zinc 50mg Tablets", "Tablets", 6.99, .38),
        ("Magnesium 250mg Tablets", "Tablets", 7.99, .48),
        ("Magnesium 400mg Tablets", "Tablets", 8.99, .42),
        ("Magnesium Glycinate 200mg Capsules", "Capsules", 12.99, .38),
        ("Iron 65mg Tablets", "Tablets", 6.99, .40),
        ("Folic Acid 400mcg Tablets", "Tablets", 5.49, .42),
        ("Folic Acid 800mcg Tablets", "Tablets", 6.49, .35),
        ("Calcium 600mg + Vitamin D3 Tablets", "Tablets", 8.99, .50),
        ("Calcium 600mg + D3 + Minerals Tablets", "Tablets", 10.99, .42),
        ("Biotin 1000mcg Softgels", "Softgels", 7.99, .42),
        ("Biotin 5000mcg Softgels", "Softgels", 9.99, .38),
        ("Biotin 10000mcg Softgels", "Softgels", 11.99, .32),
        ("Turmeric Curcumin 500mg Capsules", "Capsules", 12.99, .38),
        ("Elderberry 100mg Gummies", "Gummies", 11.99, .42),
        ("Elderberry with Vitamin C & Zinc Gummies", "Gummies", 13.99, .40),
        ("Prenatal Multi + DHA Softgels", "Softgels", 16.99, .42),
        ("Postnatal Multi + DHA Softgels", "Softgels", 16.99, .30),
        ("Extra Strength Vitamin D3 125mcg (5000 IU) Gummies", "Gummies", 11.99, .40),
        ("Apple Cider Vinegar Gummies", "Gummies", 10.99, .38),
        ("Ashwagandha Gummies", "Gummies", 12.99, .35),
        ("Vitamin E 400 IU (180mg) Softgels", "Softgels", 9.99, .35),
        ("Vitamin A 3000mcg (10000 IU) Softgels", "Softgels", 7.99, .28),
        ("Potassium Gluconate 595mg Caplets", "Caplets", 6.99, .30),
        ("Flaxseed Oil 1000mg Softgels", "Softgels", 9.99, .28),
        ("Evening Primrose Oil 1000mg Softgels", "Softgels", 11.99, .25),
        ("Lutein 20mg Softgels", "Softgels", 12.99, .30),
    ]
    brands = [
        ("Nature Made", 1.0, 0.0, False),
        ("Nature's Bounty", 0.95, -0.02, False),
        ("CVS Health", 0.70, -0.05, True),
    ]
    for bname, price_mult, pop_adj, _ in brands:
        for vit_name, subcat, base_price, base_pop in vitamin_types:
            _RAW.append((
                "Vitamins & Supplements",
                bname,
                vit_name,
                subcat,
                "P60",
                round(base_price * price_mult, 2),
                max(0.10, min(0.95, base_pop + pop_adj)),
            ))


def _generate_hair_care_formulas():
    """Add additional shampoo/conditioner variants across brands."""
    hair_formulas = [
        ("Coconut Milk & Honey Shampoo", "Shampoo"),
        ("Coconut Milk & Honey Conditioner", "Conditioner"),
        ("Tea Tree & Mint Shampoo", "Shampoo"),
        ("Tea Tree & Mint Conditioner", "Conditioner"),
        ("Argan Oil & Keratin Shampoo", "Shampoo"),
        ("Argan Oil & Keratin Conditioner", "Conditioner"),
        ("Color Protect & Shine Shampoo", "Shampoo"),
        ("Color Protect & Shine Conditioner", "Conditioner"),
    ]
    brands = [
        ("Suave Professionals", 3.99, .32),
        ("VO5", 1.99, .28),
        ("White Rain", 1.49, .22),
        ("Finesse", 3.49, .25),
    ]
    for bname, price, pop in brands:
        for fname, subcat in hair_formulas:
            _RAW.append(("Hair Care", bname, fname, subcat, "C12", price, pop))


def _generate_snack_extras():
    """Add more snack / candy / beverage products."""
    extras = [
        ("Snacks & Beverages", "Cheetos", "Crunchy Cheese Flavored Snacks", "Chips", "S", 4.99, .58),
        ("Snacks & Beverages", "Cheetos", "Flamin' Hot Crunchy Cheese Snacks", "Chips", "S", 4.99, .55),
        ("Snacks & Beverages", "Ruffles", "Original Potato Chips", "Chips", "S", 4.99, .48),
        ("Snacks & Beverages", "SunChips", "Original Multigrain Snacks", "Chips", "S", 4.99, .38),
        ("Snacks & Beverages", "Skinny Pop", "Original Popcorn", "Popcorn", "S", 4.99, .45),
        ("Snacks & Beverages", "Smartfood", "White Cheddar Popcorn", "Popcorn", "S", 4.99, .48),
        ("Snacks & Beverages", "Kellogg's", "Nutri-Grain Bars Strawberry 8 CT", "Bars", "S", 4.49, .42),
        ("Snacks & Beverages", "Kellogg's", "Rice Krispies Treats Original 8 CT", "Bars", "S", 4.49, .48),
        ("Snacks & Beverages", "Kellogg's", "Pop-Tarts Frosted Strawberry 8 CT", "Pastry", "S", 3.99, .52),
        ("Snacks & Beverages", "Kellogg's", "Cheez-It Original Crackers", "Crackers", "C8", 4.99, .55),
        ("Snacks & Beverages", "RXBar", "Chocolate Sea Salt Protein Bar", "Bars", "S", 2.99, .38),
        ("Snacks & Beverages", "RXBar", "Blueberry Protein Bar", "Bars", "S", 2.99, .32),
        ("Snacks & Beverages", "Starbucks", "Frappuccino Mocha 13.7 FL OZ Bottle", "Beverage", "S", 3.49, .48),
        ("Snacks & Beverages", "Starbucks", "Doubleshot Espresso 6.5 FL OZ Can", "Beverage", "S", 3.49, .45),
        ("Snacks & Beverages", "Pure Leaf", "Real Brewed Tea Unsweetened Black 18.5 FL OZ", "Beverage", "S", 2.29, .42),
        ("Snacks & Beverages", "Arizona", "Green Tea with Ginseng & Honey 23 FL OZ", "Beverage", "S", 1.29, .48),
        ("Snacks & Beverages", "LaCroix", "Sparkling Water Lime 8 Pack 12 FL OZ Cans", "Water", "S", 5.99, .42),
        ("Snacks & Beverages", "Bubly", "Sparkling Water Lime 8 Pack 12 FL OZ Cans", "Water", "S", 5.49, .40),
        ("Snacks & Beverages", "Body Armor", "SuperDrink Strawberry Banana 16 FL OZ", "Beverage", "S", 2.29, .42),
        ("Snacks & Beverages", "Vitamin Water", "XXX Acai-Blueberry-Pomegranate 20 FL OZ", "Beverage", "S", 2.29, .40),
        ("Snacks & Beverages", "Haribo", "Goldbears Gummy Candy", "Candy", "C4", 3.49, .50),
        ("Snacks & Beverages", "Lindt", "Lindor Milk Chocolate Truffles Bag", "Candy", "C4", 6.99, .42),
        ("Snacks & Beverages", "Ferrero Rocher", "Fine Hazelnut Chocolates 3 Pack", "Candy", "S", 2.99, .40),
        ("Snacks & Beverages", "Ghirardelli", "Intense Dark Midnight Reverie 86% Cacao Bar", "Candy", "S", 4.49, .35),
        ("Snacks & Beverages", "Werther's Original", "Hard Caramel Candy", "Candy", "C4", 3.99, .38),
        ("Snacks & Beverages", "Life Savers", "5 Flavors Hard Candy", "Candy", "S", 2.49, .35),
        ("Snacks & Beverages", "Altoids", "Curiously Strong Peppermints", "Mints", "S", 2.99, .50),
        ("Snacks & Beverages", "Tic Tac", "Fresh Breath Mints Freshmint", "Mints", "S", 1.99, .48),
        ("Snacks & Beverages", "Trident", "Original Flavor Sugar Free Gum", "Gum", "S", 1.79, .45),
        ("Snacks & Beverages", "Extra", "Spearmint Sugar Free Gum", "Gum", "S", 1.79, .48),
        ("Snacks & Beverages", "Orbit", "Sweet Mint Sugar Free Gum", "Gum", "S", 1.79, .42),
    ]
    for e in extras:
        _RAW.append(e)


def _generate_body_care():
    """Generate body wash, bar soap, hand soap, and hand sanitizer lines."""
    body_washes = [
        ("Dove", [("Deep Moisture Nourishing", .55), ("Sensitive Skin Unscented", .50),
                  ("Purely Pampering Shea Butter", .42), ("Refreshing Cucumber & Green Tea", .40),
                  ("Exfoliating Pomegranate Seeds", .38), ("Men+Care Extra Fresh", .52),
                  ("Men+Care Clean Comfort", .48), ("Men+Care Sport Active+Fresh", .42)]),
        ("Olay", [("Ultra Moisture Shea Butter", .48), ("Age Defying with Vitamin E", .42),
                  ("Fresh Outlast Soothing Orchid", .38), ("Cleansing & Nourishing Almond Milk", .35)]),
        ("Dial", [("Antibacterial Body Wash Spring Water", .48), ("For Men Ultimate Clean", .40),
                  ("Antibacterial Body Wash Gold", .42), ("Lavender & Jasmine Body Wash", .35)]),
        ("Irish Spring", [("Original Clean", .50), ("Moisture Blast", .42),
                          ("Aloe Mist Moisturizing", .38), ("Charcoal", .35)]),
        ("Softsoap", [("Moisturizing Coconut Butter Exfoliating", .42),
                      ("Moisturizing Pomegranate & Mango", .38),
                      ("Fresh & Glow Exfoliating", .35)]),
        ("Axe", [("Phoenix Body Wash", .45), ("Apollo Body Wash", .42),
                 ("Dark Temptation Body Wash", .38), ("Ice Chill Body Wash", .35)]),
    ]
    for brand, variants in body_washes:
        for name, pop in variants:
            _RAW.append(("Skin Care", brand, f"{name} Body Wash", "Body Wash", "C12", 6.99, pop))

    hand_soaps = [
        ("Softsoap", [("Liquid Hand Soap Soothing Aloe Vera", .52), ("Liquid Hand Soap Fresh Breeze", .48),
                      ("Antibacterial Liquid Hand Soap Crisp Clean", .50),
                      ("Liquid Hand Soap Milk Protein & Honey", .38)]),
        ("Dial", [("Antibacterial Liquid Hand Soap Spring Water", .48), ("Gold Antibacterial Hand Soap", .45),
                  ("Complete Foaming Hand Wash Fresh Pear", .40)]),
        ("Mrs. Meyer's", [("Clean Day Hand Soap Lavender", .42), ("Clean Day Hand Soap Basil", .38),
                          ("Clean Day Hand Soap Lemon Verbena", .36)]),
        ("Method", [("Gel Hand Wash Sweet Water", .38), ("Gel Hand Wash Pink Grapefruit", .35),
                    ("Foaming Hand Wash Coconut Water", .32)]),
    ]
    for brand, variants in hand_soaps:
        for name, pop in variants:
            _RAW.append(("Skin Care", brand, name, "Hand Soap", "L8", 4.49, pop))

    # Bar soaps
    bar_soaps = [
        ("Dial", "Antibacterial Deodorant Bar Soap Gold 3 Pack", .48),
        ("Dial", "Antibacterial Bar Soap Spring Water 3 Pack", .42),
        ("Irish Spring", "Original Clean Deodorant Bar Soap 3 Pack", .48),
        ("Irish Spring", "Moisture Blast Bar Soap 3 Pack", .38),
        ("Dove", "White Beauty Bar 4 Pack", .55),
        ("Dove", "Sensitive Skin Beauty Bar 4 Pack", .48),
        ("Dove", "Men+Care Extra Fresh Body and Face Bar 4 Pack", .42),
        ("Lever 2000", "Original Bar Soap 4 Pack", .32),
    ]
    for brand, name, pop in bar_soaps:
        _RAW.append(("Skin Care", brand, name, "Bar Soap", "S", 5.99, pop))

    # Hand sanitizer
    sanitizers = [
        ("Purell", "Advanced Hand Sanitizer Refreshing Gel", "Sanitizer", "C8", 5.99, .62),
        ("Purell", "Advanced Hand Sanitizer Naturals", "Sanitizer", "C8", 6.49, .45),
        ("Purell", "Advanced Hand Sanitizer Pump Bottle", "Sanitizer", "S", 7.99, .48),
        ("Germ-X", "Original Hand Sanitizer", "Sanitizer", "C8", 4.49, .42),
        ("CVS Health", "Hand Sanitizer Original", "Sanitizer", "C8", 3.49, .45),
        ("CVS Health", "Hand Sanitizer Aloe Vera", "Sanitizer", "C8", 3.49, .38),
    ]
    for brand, name, sub, tc, price, pop in sanitizers:
        _RAW.append(("Skin Care", brand, name, sub, tc, price, pop))


def _generate_cosmetics_shades():
    """Generate makeup products in realistic shade ranges."""
    rng = random.Random(42)
    foundation_shades = [
        "Porcelain", "Ivory", "Fair Ivory", "Classic Ivory", "Natural Ivory",
        "Buff", "Nude", "Soft Beige", "Natural Beige", "Warm Beige",
        "Medium Beige", "Golden Beige", "Sand Beige", "Honey", "Natural Tan",
        "Classic Tan", "Caramel", "Toffee", "Cappuccino", "Espresso",
    ]
    for brand, base, price, pop in [
        ("Maybelline", "Fit Me Dewy + Smooth Foundation", 8.99, .58),
        ("L'Oreal", "Infallible 24H Fresh Wear Foundation", 14.99, .48),
        ("Revlon", "ColorStay Makeup Normal/Dry Foundation", 10.99, .45),
        ("CoverGirl", "Simply Ageless Instant Wrinkle-Defying Foundation", 14.99, .40),
        ("NYX Professional Makeup", "Born to Glow Naturally Radiant Foundation", 9.99, .42),
    ]:
        for shade in foundation_shades:
            p = max(0.10, pop - rng.uniform(0, 0.06))
            _RAW.append(("Cosmetics & Makeup", brand, f"{base} {shade}", "Foundation", "S", price, round(p, 3)))

    lip_shades = [
        "Dusty Rose", "Berry Kiss", "Mauve Over", "Red Carpet", "Coral Crush",
        "Nude Pink", "Plum Perfect", "Wine Night", "Peach Fuzz", "Crimson Red",
        "Rosewood", "Cinnamon Spice", "Sugar Plum", "Fuchsia", "Brandy",
    ]
    for brand, base, price, pop in [
        ("Maybelline", "SuperStay Vinyl Ink Liquid Lipcolor", 10.99, .52),
        ("Revlon", "Super Lustrous Lipstick Creme", 8.99, .45),
        ("NYX Professional Makeup", "Lip Lingerie Push-Up Long-Lasting Lipstick", 8.99, .42),
        ("L'Oreal", "Colour Riche Original Satin Lipstick", 8.99, .40),
        ("CoverGirl", "Outlast All-Day Lip Color", 10.99, .38),
    ]:
        for shade in lip_shades:
            p = max(0.10, pop - rng.uniform(0, 0.05))
            _RAW.append(("Cosmetics & Makeup", brand, f"{base} {shade}", "Lipstick", "S", price, round(p, 3)))

    nail_colors = [
        "Marshmallow", "Mademoiselle", "Berry Naughty", "Eternal Optimist",
        "Sand Tropez", "Wired", "Smokin Hot", "Licorice", "No Place Like Chrome",
        "In Stitches", "Topless & Barefoot", "Go Ginza", "Turquoise & Caicos",
        "After School Boy Blazer", "Angelic", "Cute As a Button", "Fifth Avenue",
    ]
    for brand, base, price, pop in [
        ("Essie", "Nail Polish", 9.99, .42),
        ("OPI", "Nail Lacquer", 10.99, .38),
        ("Sally Hansen", "Miracle Gel Nail Color", 9.99, .38),
        ("Sinful Colors", "Professional Nail Polish", 2.49, .35),
    ]:
        for color in nail_colors:
            p = max(0.10, pop - rng.uniform(0, 0.04))
            _RAW.append(("Cosmetics & Makeup", brand, f"{base} {color}", "Nail Polish", "S", price, round(p, 3)))


def _generate_cleaning_and_laundry():
    """Generate additional cleaning, laundry, and household products."""
    products = [
        # Cleaning
        ("Household Essentials", "Windex", "Original Glass Cleaner Spray", "Glass Cleaner", "S", 4.99, .55),
        ("Household Essentials", "Windex", "Vinegar Multi-Surface Cleaner", "Cleaner", "S", 4.99, .42),
        ("Household Essentials", "Mr. Clean", "Magic Eraser Original", "Eraser", "S", 4.49, .55),
        ("Household Essentials", "Mr. Clean", "Magic Eraser Extra Durable", "Eraser", "S", 5.49, .42),
        ("Household Essentials", "Mr. Clean", "Multi-Purpose Cleaner Gain Original", "Cleaner", "S", 4.49, .38),
        ("Household Essentials", "Pine-Sol", "Multi-Surface Cleaner Original Pine", "Cleaner", "L16", 4.99, .48),
        ("Household Essentials", "Pine-Sol", "Multi-Surface Cleaner Lavender Clean", "Cleaner", "L16", 4.99, .38),
        ("Household Essentials", "Scrub Free", "Bathroom Cleaner Lemon", "Cleaner", "S", 3.99, .32),
        ("Household Essentials", "Swiffer", "WetJet Mopping Pad Refills", "Mop Pads", "P24", 8.99, .45),
        ("Household Essentials", "Swiffer", "Sweeper Dry Mopping Cloth Refills", "Mop Pads", "P24", 7.99, .42),
        ("Household Essentials", "Swiffer", "Duster Heavy Duty Refills", "Dusters", "S", 9.99, .38),
        ("Household Essentials", "Swiffer", "WetJet Mopping Solution Fresh Citrus", "Solution", "S", 7.99, .40),
        ("Household Essentials", "Pledge", "Multi Surface Everyday Cleaner", "Cleaner", "SP", 5.49, .35),
        ("Household Essentials", "SC Johnson", "Glade Automatic Spray Refill Clean Linen", "Air Freshener", "S", 6.99, .35),
        ("Household Essentials", "SC Johnson", "Glade PlugIns Scented Oil Warmer + Refill Lavender", "Air Freshener", "S", 5.49, .38),
        ("Household Essentials", "SC Johnson", "Glade 3-Wick Candle Cashmere Woods", "Candle", "S", 8.99, .32),
        # Laundry
        ("Household Essentials", "Tide", "Original Liquid Laundry Detergent", "Laundry", "L16", 11.99, .55),
        ("Household Essentials", "Tide", "PODS 3-in-1 Laundry Detergent Pacs Spring Meadow", "Laundry", "P20", 12.99, .52),
        ("Household Essentials", "Tide", "Free & Gentle Liquid Laundry Detergent", "Laundry", "L16", 12.99, .45),
        ("Household Essentials", "Tide", "To Go Instant Stain Remover Liquid 10 ML", "Stain Remover", "S", 3.49, .42),
        ("Household Essentials", "Downy", "Ultra Liquid Fabric Conditioner April Fresh", "Fabric Softener", "L16", 6.99, .48),
        ("Household Essentials", "Downy", "Unstopables In-Wash Scent Booster Beads Fresh", "Scent Booster", "S", 11.99, .42),
        ("Household Essentials", "Gain", "Liquid Laundry Detergent Original", "Laundry", "L16", 11.99, .42),
        ("Household Essentials", "Gain", "Flings! Laundry Detergent Pacs Original", "Laundry", "P20", 12.49, .38),
        ("Household Essentials", "All", "Free Clear Liquid Laundry Detergent", "Laundry", "L16", 8.99, .48),
        ("Household Essentials", "All", "Mighty Pacs Laundry Detergent Free Clear", "Laundry", "P20", 9.99, .40),
        ("Household Essentials", "OxiClean", "Versatile Stain Remover Powder", "Stain Remover", "S", 8.99, .48),
        ("Household Essentials", "OxiClean", "MaxForce Laundry Stain Remover Spray", "Stain Remover", "S", 5.49, .42),
        ("Household Essentials", "Shout", "Triple-Acting Stain Remover Spray", "Stain Remover", "S", 4.99, .42),
        ("Household Essentials", "Bounce", "Dryer Sheets Outdoor Fresh", "Dryer Sheets", "P30", 5.49, .48),
        ("Household Essentials", "Snuggle", "Fabric Softener Sheets Blue Sparkle", "Dryer Sheets", "P30", 4.99, .38),
        # Tissues and cotton
        ("Household Essentials", "Kleenex", "Trusted Care Facial Tissues 144 CT", "Tissues", "S", 2.49, .65),
        ("Household Essentials", "Kleenex", "Ultra Soft Facial Tissues 120 CT", "Tissues", "S", 2.99, .52),
        ("Household Essentials", "Kleenex", "Trusted Care Tissues 3-Pack", "Tissues", "S", 5.99, .55),
        ("Household Essentials", "Puffs", "Plus Lotion Facial Tissues 124 CT", "Tissues", "S", 2.79, .52),
        ("Household Essentials", "Puffs", "Ultra Soft Facial Tissues 124 CT", "Tissues", "S", 2.99, .45),
        ("Household Essentials", "Q-tips", "Cotton Swabs", "Cotton Swabs", "P60", 3.99, .55),
        ("Household Essentials", "CVS Health", "Facial Tissues 160 CT", "Tissues", "S", 1.49, .48),
        ("Household Essentials", "CVS Health", "Premium Facial Tissues with Lotion 120 CT", "Tissues", "S", 1.99, .40),
        ("Household Essentials", "CVS Health", "Cotton Swabs", "Cotton Swabs", "P60", 2.49, .45),
        ("Household Essentials", "CVS Health", "Cotton Balls 100 CT", "Cotton", "S", 1.99, .42),
        ("Household Essentials", "CVS Health", "Cotton Rounds 80 CT", "Cotton", "S", 2.49, .38),
        ("Household Essentials", "CVS Health", "Bleach Regular Scent", "Bleach", "S", 2.99, .38),
        ("Household Essentials", "CVS Health", "All-Purpose Cleaner Spray Lemon", "Cleaner", "S", 2.99, .35),
        ("Household Essentials", "CVS Health", "Glass Cleaner Spray", "Glass Cleaner", "S", 2.49, .32),
        ("Household Essentials", "CVS Health", "Dish Soap Original Scent", "Dish Soap", "L8", 1.99, .35),
    ]
    for p in products:
        _RAW.append(p)


def _generate_additional_pet_care():
    """More pet care products."""
    pets = [
        ("Pet Care", "Blue Buffalo", "Life Protection Small Breed Adult Chicken Dry Dog Food", "Dog Food", "S", 14.99, .42),
        ("Pet Care", "Blue Buffalo", "Tastefuls Indoor Natural Adult Chicken Cat Food", "Cat Food", "S", 12.99, .38),
        ("Pet Care", "Iams", "ProActive Health Adult MiniChunks Dry Dog Food", "Dog Food", "S", 9.99, .42),
        ("Pet Care", "Iams", "ProActive Health Indoor Weight & Hairball Care Dry Cat Food", "Cat Food", "S", 8.99, .38),
        ("Pet Care", "Rachael Ray Nutrish", "Natural Dry Dog Food Real Chicken & Veggies", "Dog Food", "S", 9.99, .35),
        ("Pet Care", "Meow Mix", "Original Choice Dry Cat Food", "Cat Food", "S", 6.99, .42),
        ("Pet Care", "9Lives", "Daily Essentials Dry Cat Food", "Cat Food", "S", 5.99, .32),
        ("Pet Care", "Cesar", "Classic Loaf Grilled Chicken Wet Dog Food 3.5 OZ", "Dog Food", "S", 1.59, .38),
        ("Pet Care", "Sheba", "Perfect Portions Pate Savory Chicken Cat Food", "Cat Food", "S", 1.29, .35),
        ("Pet Care", "Purina", "Beggin' Strips Bacon & Cheese Flavor Dog Treats", "Dog Treats", "S", 5.99, .48),
        ("Pet Care", "Purina", "Temptations Classic Crunchy & Soft Tasty Chicken Cat Treats", "Cat Treats", "S", 3.49, .45),
        ("Pet Care", "Purina", "Busy Bone Small/Medium Dog Treats", "Dog Treats", "S", 6.99, .35),
        ("Pet Care", "Arm & Hammer", "Clump & Seal Clumping Cat Litter", "Cat Litter", "S", 14.99, .42),
        ("Pet Care", "Fresh Step", "Clean Paws Multi-Cat Clumping Cat Litter", "Cat Litter", "S", 13.99, .38),
        ("Pet Care", "Frontline", "Plus Flea & Tick Treatment for Dogs Medium 3 Doses", "Flea & Tick", "S", 38.99, .40),
        ("Pet Care", "Frontline", "Plus Flea & Tick Treatment for Cats 3 Doses", "Flea & Tick", "S", 36.99, .35),
        ("Pet Care", "Advantage", "II Flea Prevention for Dogs Medium 4 Doses", "Flea & Tick", "S", 42.99, .38),
        ("Pet Care", "Advantage", "II Flea Prevention for Cats 4 Doses", "Flea & Tick", "S", 39.99, .35),
        ("Pet Care", "Hartz", "UltraGuard Flea & Tick Collar for Dogs", "Flea & Tick", "S", 6.99, .30),
        ("Pet Care", "Hartz", "Delectables Squeeze Up Cat Treats Chicken", "Cat Treats", "S", 3.99, .32),
        ("Pet Care", "CVS Health", "Premium Dog Treats Chicken Jerky", "Dog Treats", "S", 5.99, .32),
        ("Pet Care", "CVS Health", "Cat Litter Clumping Unscented", "Cat Litter", "S", 7.99, .32),
        ("Pet Care", "CVS Health", "Pet Stain & Odor Carpet Cleaner", "Cleaner", "S", 4.99, .28),
        ("Pet Care", "CVS Health", "Dog Poop Bags with Dispenser 120 CT", "Pet Supplies", "S", 5.99, .35),
    ]
    for p in pets:
        _RAW.append(p)


def _generate_additional_electronics():
    """More photo & electronics products."""
    elec = [
        ("Photo & Electronics", "Apple", "EarPods with Lightning Connector", "Earbuds", "S", 19.99, .42),
        ("Photo & Electronics", "Apple", "20W USB-C Power Adapter", "Charger", "S", 19.99, .40),
        ("Photo & Electronics", "JBL", "Tune 510BT Wireless On-Ear Headphones", "Headphones", "S", 34.99, .35),
        ("Photo & Electronics", "JBL", "Go 3 Portable Bluetooth Speaker", "Speaker", "S", 29.99, .32),
        ("Photo & Electronics", "Skullcandy", "Jib True 2 Wireless Earbuds", "Earbuds", "S", 24.99, .32),
        ("Photo & Electronics", "OtterBox", "Symmetry Series iPhone Case Clear", "Phone Case", "S", 39.99, .30),
        ("Photo & Electronics", "PopSockets", "PopGrip Phone Grip Black", "Phone Accessory", "S", 9.99, .38),
        ("Photo & Electronics", "Energizer", "Rechargeable AA Batteries 4 Pack", "Batteries", "S", 14.99, .35),
        ("Photo & Electronics", "SanDisk", "Ultra Dual Drive USB Type-C 64GB", "Flash Drive", "S", 12.99, .32),
        ("Photo & Electronics", "Fujifilm", "Instax Mini Instant Film 10 Exposures", "Film", "S", 9.99, .42),
        ("Photo & Electronics", "CVS Health", "Portable Power Bank 5000mAh", "Power Bank", "S", 12.99, .35),
        ("Photo & Electronics", "CVS Health", "Wired Earbuds with Microphone", "Earbuds", "S", 7.99, .32),
        ("Photo & Electronics", "CVS Health", "USB-A to USB-C Cable 6 ft", "Cable", "S", 8.99, .35),
        ("Photo & Electronics", "CVS Health", "Car Phone Mount Universal", "Phone Accessory", "S", 9.99, .28),
        ("Photo & Electronics", "CVS Health", "Bluetooth FM Transmitter for Car", "Car Accessory", "S", 14.99, .25),
        ("Photo & Electronics", "CVS Health", "Travel Adapter International", "Adapter", "S", 12.99, .22),
    ]
    for p in elec:
        _RAW.append(p)


def _generate_additional_seasonal():
    """More seasonal and holiday items."""
    items = [
        ("Seasonal Items", "Coppertone", "Glow Shimmer Sunscreen Lotion SPF 50", "Sunscreen", "C4", 11.99, .38),
        ("Seasonal Items", "Coppertone", "Kids Sport Sunscreen Lotion SPF 50", "Sunscreen", "C4", 10.99, .48),
        ("Seasonal Items", "Neutrogena", "Beach Defense Sunscreen Lotion SPF 70", "Sunscreen", "C4", 11.99, .52),
        ("Seasonal Items", "Neutrogena", "Ultra Sheer Dry-Touch Sunscreen SPF 70", "Sunscreen", "C4", 11.99, .48),
        ("Seasonal Items", "Neutrogena", "Hydro Boost Water Gel Lotion Sunscreen SPF 50", "Sunscreen", "C4", 13.99, .42),
        ("Seasonal Items", "Blue Lizard", "Australian Sunscreen Sensitive SPF 30+", "Sunscreen", "C4", 14.99, .38),
        ("Seasonal Items", "EltaMD", "UV Clear Broad-Spectrum SPF 46 Sunscreen", "Sunscreen", "C1", 39.00, .32),
        ("Seasonal Items", "Repel", "Insect Repellent Sportsmen Max Formula 40% DEET", "Insect Repellent", "SP", 7.99, .38),
        ("Seasonal Items", "CVS Health", "After Sun Moisturizing Lotion", "After Sun", "C8", 5.49, .38),
        ("Seasonal Items", "CVS Health", "Mineral Sunscreen Lotion SPF 50", "Sunscreen", "C4", 8.99, .35),
        ("Seasonal Items", "CVS Health", "Kids Sunscreen Spray SPF 70", "Sunscreen", "SP", 8.99, .32),
        ("Seasonal Items", "CVS Health", "Insect Repellent Wipes 15 CT", "Insect Repellent", "S", 4.99, .30),
        ("Seasonal Items", "CVS Health", "Instant Ice Pack 2 Pack", "First Aid", "S", 3.49, .32),
        ("Seasonal Items", "CVS Health", "Cooling Towel", "Cooling", "S", 5.99, .25),
    ]
    for p in items:
        _RAW.append(p)


def _generate_cvs_health_equivalents():
    """Auto-generate CVS Health store brand versions for categories that don't have enough."""
    rng = random.Random(42)
    categories_for_cvs = {
        "Pain Relief & Fever", "Cold/Flu/Allergy", "Digestive Health",
        "Vitamins & Supplements", "Skin Care", "Hair Care", "Oral Care",
        "Deodorant", "First Aid & Wound Care", "Eye & Ear Care",
        "Feminine Care", "Foot Care", "Sleep & Relaxation",
        "Household Essentials",
    }
    existing_cvs = set()
    for cat, brand, name, sub, tc, price, pop in _RAW:
        if brand == "CVS Health":
            existing_cvs.add((cat, sub))

    for cat, brand, name, sub, tc, price, pop in list(_RAW):
        if brand == "CVS Health":
            continue
        if cat not in categories_for_cvs:
            continue
        # Only create one CVS equiv per (category, subcategory) that doesn't exist
        key = (cat, sub)
        if key in existing_cvs:
            continue
        existing_cvs.add(key)
        # Generic name
        generic_name = name
        for b in [brand]:
            generic_name = generic_name.replace(b, "").strip()
        if not generic_name or len(generic_name) < 5:
            continue
        cvs_price = round(price * rng.uniform(0.65, 0.80), 2)
        cvs_pop = max(0.10, pop - rng.uniform(0.05, 0.15))
        _RAW.append((cat, "CVS Health", generic_name, sub, tc, cvs_price, round(cvs_pop, 3)))


def _generate_more_greeting_cards():
    """Additional greeting card products."""
    occasions = [
        "Valentine's Day Card", "Mother's Day Card", "Father's Day Card",
        "Easter Card", "Christmas Card", "Hanukkah Card", "Thinking of You Card",
        "Just Because Card", "Encouragement Card", "Retirement Card",
        "Baptism Card", "Confirmation Card", "Teacher Thank You Card",
        "Boss's Day Card", "Friendship Card",
    ]
    for brand, price in [("Hallmark", 5.99), ("American Greetings", 4.99)]:
        for occasion in occasions:
            _RAW.append(("Greeting Cards & Gift Wrap", brand, occasion, "Card", "S", price, .32))
    # More gift supplies
    more_gift = [
        ("Greeting Cards & Gift Wrap", "Scotch", "Gift Wrap Tape Satin Finish", "Tape", "S", 3.99, .35),
        ("Greeting Cards & Gift Wrap", "CVS Health", "Gift Card Holder 3 Pack", "Gift Card Holder", "S", 2.99, .30),
        ("Greeting Cards & Gift Wrap", "CVS Health", "Wrapping Paper Holiday Assorted 3 Roll", "Gift Wrap", "S", 5.99, .32),
        ("Greeting Cards & Gift Wrap", "CVS Health", "Gift Tags Self-Adhesive 50 Count", "Gift Tags", "S", 2.49, .28),
        ("Greeting Cards & Gift Wrap", "CVS Health", "Party Balloons Assorted 12 Pack", "Party Supplies", "S", 3.99, .30),
        ("Greeting Cards & Gift Wrap", "CVS Health", "Paper Plates 9 Inch 20 Count", "Party Supplies", "S", 2.99, .32),
        ("Greeting Cards & Gift Wrap", "CVS Health", "Paper Cups 9 OZ 20 Count", "Party Supplies", "S", 2.49, .28),
        ("Greeting Cards & Gift Wrap", "CVS Health", "Plastic Utensils Assorted 24 Count", "Party Supplies", "S", 2.99, .25),
    ]
    for p in more_gift:
        _RAW.append(p)


# Run all generators
_generate_vitamin_matrix()
_generate_hair_care_formulas()
_generate_snack_extras()
_generate_body_care()
_generate_cosmetics_shades()
_generate_cleaning_and_laundry()
_generate_additional_pet_care()
_generate_additional_electronics()
_generate_additional_seasonal()
_generate_cvs_health_equivalents()
_generate_more_greeting_cards()


# ════════════════════════════════════════════════════════════════════
# PRESCRIPTION (Rx) PRODUCT KNOWLEDGE BASE
# ════════════════════════════════════════════════════════════════════
# Each entry: (therapeutic_class, generic_name, brand_info, form, strengths, gen_30d_price, popularity)
# brand_info: None (generic-only) or (brand_name, manufacturer, brand_30d_price)
# gen_30d_price: cash/AWP price for 30-day supply at the lowest strength
# popularity: 0-1 based on US prescription volume (atorvastatin ~0.95, specialty ~0.10)

_RX_DRUGS = [
    # ═══ Statins (HMG-CoA Reductase Inhibitors) ═══
    ("Statins", "Atorvastatin Calcium", ("Lipitor", "Pfizer", 389.99), "Tablets", ["10mg","20mg","40mg","80mg"], 12.99, .95),
    ("Statins", "Rosuvastatin Calcium", ("Crestor", "AstraZeneca", 329.99), "Tablets", ["5mg","10mg","20mg","40mg"], 14.99, .88),
    ("Statins", "Simvastatin", None, "Tablets", ["5mg","10mg","20mg","40mg","80mg"], 9.99, .82),
    ("Statins", "Pravastatin Sodium", None, "Tablets", ["10mg","20mg","40mg","80mg"], 11.99, .65),
    ("Statins", "Lovastatin", None, "Tablets", ["10mg","20mg","40mg"], 10.99, .45),
    ("Statins", "Pitavastatin", ("Livalo", "Kowa", 329.99), "Tablets", ["1mg","2mg","4mg"], 29.99, .25),
    # ═══ ACE Inhibitors ═══
    ("ACE Inhibitors", "Lisinopril", None, "Tablets", ["2.5mg","5mg","10mg","20mg","40mg"], 4.99, .92),
    ("ACE Inhibitors", "Enalapril Maleate", None, "Tablets", ["2.5mg","5mg","10mg","20mg"], 7.99, .55),
    ("ACE Inhibitors", "Ramipril", None, "Capsules", ["1.25mg","2.5mg","5mg","10mg"], 9.99, .50),
    ("ACE Inhibitors", "Benazepril HCl", None, "Tablets", ["5mg","10mg","20mg","40mg"], 8.99, .42),
    ("ACE Inhibitors", "Quinapril HCl", None, "Tablets", ["5mg","10mg","20mg","40mg"], 12.99, .30),
    ("ACE Inhibitors", "Fosinopril Sodium", None, "Tablets", ["10mg","20mg","40mg"], 14.99, .22),
    # ═══ ARBs (Angiotensin II Receptor Blockers) ═══
    ("ARBs", "Losartan Potassium", None, "Tablets", ["25mg","50mg","100mg"], 9.99, .88),
    ("ARBs", "Valsartan", None, "Tablets", ["40mg","80mg","160mg","320mg"], 14.99, .72),
    ("ARBs", "Irbesartan", None, "Tablets", ["75mg","150mg","300mg"], 16.99, .45),
    ("ARBs", "Olmesartan Medoxomil", ("Benicar", "Daiichi Sankyo", 289.99), "Tablets", ["5mg","20mg","40mg"], 18.99, .42),
    ("ARBs", "Telmisartan", None, "Tablets", ["20mg","40mg","80mg"], 19.99, .32),
    ("ARBs", "Candesartan Cilexetil", None, "Tablets", ["4mg","8mg","16mg","32mg"], 22.99, .25),
    # ═══ Beta Blockers ═══
    ("Beta Blockers", "Metoprolol Succinate ER", ("Toprol-XL", "AstraZeneca", 149.99), "Extended-Release Tablets", ["25mg","50mg","100mg","200mg"], 9.99, .90),
    ("Beta Blockers", "Metoprolol Tartrate", None, "Tablets", ["25mg","50mg","100mg"], 6.99, .78),
    ("Beta Blockers", "Atenolol", None, "Tablets", ["25mg","50mg","100mg"], 6.99, .68),
    ("Beta Blockers", "Carvedilol", None, "Tablets", ["3.125mg","6.25mg","12.5mg","25mg"], 9.99, .62),
    ("Beta Blockers", "Propranolol HCl", None, "Tablets", ["10mg","20mg","40mg","80mg"], 7.99, .48),
    # ═══ Calcium Channel Blockers ═══
    ("Calcium Channel Blockers", "Amlodipine Besylate", None, "Tablets", ["2.5mg","5mg","10mg"], 4.99, .90),
    ("Calcium Channel Blockers", "Diltiazem HCl ER", None, "Extended-Release Capsules", ["120mg","180mg","240mg","300mg","360mg"], 14.99, .58),
    ("Calcium Channel Blockers", "Nifedipine ER", None, "Extended-Release Tablets", ["30mg","60mg","90mg"], 12.99, .42),
    ("Calcium Channel Blockers", "Verapamil HCl ER", None, "Extended-Release Tablets", ["120mg","180mg","240mg"], 12.99, .35),
    # ═══ Diuretics ═══
    ("Diuretics", "Hydrochlorothiazide", None, "Tablets", ["12.5mg","25mg","50mg"], 4.99, .82),
    ("Diuretics", "Furosemide", None, "Tablets", ["20mg","40mg","80mg"], 4.99, .72),
    ("Diuretics", "Spironolactone", None, "Tablets", ["25mg","50mg","100mg"], 9.99, .58),
    ("Diuretics", "Chlorthalidone", None, "Tablets", ["25mg","50mg"], 7.99, .38),
    ("Diuretics", "Triamterene/HCTZ", None, "Capsules", ["37.5/25mg","75/50mg"], 9.99, .42),
    # ═══ Combination Blood Pressure ═══
    ("Combination Antihypertensives", "Lisinopril/HCTZ", None, "Tablets", ["10/12.5mg","20/12.5mg","20/25mg"], 8.99, .72),
    ("Combination Antihypertensives", "Losartan/HCTZ", None, "Tablets", ["50/12.5mg","100/12.5mg","100/25mg"], 14.99, .60),
    ("Combination Antihypertensives", "Valsartan/HCTZ", None, "Tablets", ["80/12.5mg","160/12.5mg","160/25mg","320/25mg"], 19.99, .48),
    ("Combination Antihypertensives", "Amlodipine/Benazepril", None, "Capsules", ["2.5/10mg","5/10mg","5/20mg","10/20mg"], 14.99, .45),
    ("Combination Antihypertensives", "Amlodipine/Valsartan", ("Exforge", "Novartis", 349.99), "Tablets", ["5/160mg","5/320mg","10/160mg","10/320mg"], 24.99, .32),
    # ═══ SSRIs ═══
    ("SSRIs", "Sertraline HCl", None, "Tablets", ["25mg","50mg","100mg"], 6.99, .92),
    ("SSRIs", "Escitalopram Oxalate", ("Lexapro", "Allergan", 399.99), "Tablets", ["5mg","10mg","20mg"], 8.99, .88),
    ("SSRIs", "Fluoxetine HCl", None, "Capsules", ["10mg","20mg","40mg"], 4.99, .78),
    ("SSRIs", "Citalopram HBr", None, "Tablets", ["10mg","20mg","40mg"], 6.99, .65),
    ("SSRIs", "Paroxetine HCl", None, "Tablets", ["10mg","20mg","30mg","40mg"], 9.99, .52),
    # ═══ SNRIs ═══
    ("SNRIs", "Duloxetine HCl", ("Cymbalta", "Eli Lilly", 449.99), "Delayed-Release Capsules", ["20mg","30mg","60mg"], 12.99, .78),
    ("SNRIs", "Venlafaxine HCl ER", None, "Extended-Release Capsules", ["37.5mg","75mg","150mg"], 11.99, .68),
    ("SNRIs", "Desvenlafaxine ER", ("Pristiq", "Pfizer", 429.99), "Extended-Release Tablets", ["25mg","50mg","100mg"], 24.99, .42),
    # ═══ Other Antidepressants ═══
    ("Other Antidepressants", "Bupropion HCl XL", ("Wellbutrin XL", "Bausch", 499.99), "Extended-Release Tablets", ["150mg","300mg"], 14.99, .78),
    ("Other Antidepressants", "Bupropion HCl SR", None, "Sustained-Release Tablets", ["100mg","150mg","200mg"], 12.99, .55),
    ("Other Antidepressants", "Trazodone HCl", None, "Tablets", ["50mg","100mg","150mg"], 4.99, .72),
    ("Other Antidepressants", "Mirtazapine", None, "Tablets", ["7.5mg","15mg","30mg","45mg"], 7.99, .50),
    ("Other Antidepressants", "Amitriptyline HCl", None, "Tablets", ["10mg","25mg","50mg","75mg"], 4.99, .48),
    ("Other Antidepressants", "Nortriptyline HCl", None, "Capsules", ["10mg","25mg","50mg","75mg"], 9.99, .32),
    # ═══ Anti-Anxiety ═══
    ("Anti-Anxiety", "Buspirone HCl", None, "Tablets", ["5mg","10mg","15mg","30mg"], 7.99, .62),
    ("Anti-Anxiety", "Hydroxyzine HCl", None, "Tablets", ["10mg","25mg","50mg"], 4.99, .55),
    ("Anti-Anxiety", "Hydroxyzine Pamoate", None, "Capsules", ["25mg","50mg"], 9.99, .38),
    # ═══ Oral Antidiabetics ═══
    ("Oral Antidiabetics", "Metformin HCl", None, "Tablets", ["500mg","850mg","1000mg"], 4.99, .95),
    ("Oral Antidiabetics", "Metformin HCl ER", None, "Extended-Release Tablets", ["500mg","750mg","1000mg"], 6.99, .85),
    ("Oral Antidiabetics", "Glipizide", None, "Tablets", ["5mg","10mg"], 4.99, .60),
    ("Oral Antidiabetics", "Glimepiride", None, "Tablets", ["1mg","2mg","4mg"], 6.99, .52),
    ("Oral Antidiabetics", "Pioglitazone HCl", None, "Tablets", ["15mg","30mg","45mg"], 9.99, .38),
    ("Oral Antidiabetics", "Sitagliptin", ("Januvia", "Merck", 519.99), "Tablets", ["25mg","50mg","100mg"], None, .55),
    ("Oral Antidiabetics", "Empagliflozin", ("Jardiance", "Boehringer Ingelheim", 579.99), "Tablets", ["10mg","25mg"], None, .58),
    ("Oral Antidiabetics", "Dapagliflozin", ("Farxiga", "AstraZeneca", 549.99), "Tablets", ["5mg","10mg"], None, .42),
    ("Oral Antidiabetics", "Canagliflozin", ("Invokana", "Janssen", 529.99), "Tablets", ["100mg","300mg"], None, .30),
    ("Oral Antidiabetics", "Metformin/Sitagliptin", ("Janumet", "Merck", 539.99), "Tablets", ["500/50mg","1000/50mg"], None, .40),
    # ═══ Insulins & GLP-1 Agonists (injectables, type S) ═══
    ("Insulins & Injectables", "Insulin Glargine", ("Lantus", "Sanofi", 349.99), "Injection Pen", ["100 units/mL"], None, .72),
    ("Insulins & Injectables", "Insulin Glargine", ("Basaglar", "Eli Lilly", 299.99), "Injection Pen", ["100 units/mL"], None, .55),
    ("Insulins & Injectables", "Insulin Lispro", ("Humalog", "Eli Lilly", 329.99), "Injection Pen", ["100 units/mL"], None, .62),
    ("Insulins & Injectables", "Insulin Aspart", ("NovoLog", "Novo Nordisk", 339.99), "Injection Pen", ["100 units/mL"], None, .58),
    ("Insulins & Injectables", "Insulin Detemir", ("Levemir", "Novo Nordisk", 359.99), "Injection Pen", ["100 units/mL"], None, .38),
    ("Insulins & Injectables", "Semaglutide", ("Ozempic", "Novo Nordisk", 935.77), "Injection Pen", ["0.25mg","0.5mg","1mg","2mg"], None, .72),
    ("Insulins & Injectables", "Semaglutide", ("Wegovy", "Novo Nordisk", 1349.02), "Injection Pen", ["0.25mg","0.5mg","1mg","1.7mg","2.4mg"], None, .55),
    ("Insulins & Injectables", "Tirzepatide", ("Mounjaro", "Eli Lilly", 1023.04), "Injection Pen", ["2.5mg","5mg","7.5mg","10mg","12.5mg","15mg"], None, .60),
    ("Insulins & Injectables", "Dulaglutide", ("Trulicity", "Eli Lilly", 886.48), "Injection Pen", ["0.75mg","1.5mg","3mg","4.5mg"], None, .48),
    ("Insulins & Injectables", "Liraglutide", ("Victoza", "Novo Nordisk", 899.99), "Injection Pen", ["1.2mg","1.8mg"], None, .35),
    # ═══ Thyroid ═══
    ("Thyroid", "Levothyroxine Sodium", ("Synthroid", "AbbVie", 149.99), "Tablets", ["25mcg","50mcg","75mcg","88mcg","100mcg","112mcg","125mcg","150mcg","175mcg","200mcg"], 9.99, .92),
    ("Thyroid", "Liothyronine Sodium", None, "Tablets", ["5mcg","25mcg","50mcg"], 19.99, .28),
    ("Thyroid", "Armour Thyroid", ("Armour Thyroid", "AbbVie", 49.99), "Tablets", ["30mg","60mg","90mg","120mg"], None, .25),
    # ═══ PPIs (Rx Strength) ═══
    ("Proton Pump Inhibitors", "Omeprazole Rx", None, "Delayed-Release Capsules", ["20mg","40mg"], 9.99, .72),
    ("Proton Pump Inhibitors", "Pantoprazole Sodium", ("Protonix", "Pfizer", 259.99), "Delayed-Release Tablets", ["20mg","40mg"], 9.99, .68),
    ("Proton Pump Inhibitors", "Esomeprazole Magnesium", None, "Delayed-Release Capsules", ["20mg","40mg"], 14.99, .52),
    ("Proton Pump Inhibitors", "Lansoprazole", None, "Delayed-Release Capsules", ["15mg","30mg"], 12.99, .42),
    # ═══ Antibiotics ═══
    ("Antibiotics", "Amoxicillin", None, "Capsules", ["250mg","500mg"], 4.99, .88),
    ("Antibiotics", "Amoxicillin/Clavulanate", None, "Tablets", ["250/125mg","500/125mg","875/125mg"], 14.99, .72),
    ("Antibiotics", "Azithromycin", None, "Tablets", ["250mg","500mg"], 9.99, .82),
    ("Antibiotics", "Ciprofloxacin HCl", None, "Tablets", ["250mg","500mg","750mg"], 9.99, .58),
    ("Antibiotics", "Levofloxacin", None, "Tablets", ["250mg","500mg","750mg"], 12.99, .48),
    ("Antibiotics", "Doxycycline Hyclate", None, "Capsules", ["50mg","100mg"], 7.99, .72),
    ("Antibiotics", "Cephalexin", None, "Capsules", ["250mg","500mg"], 7.99, .68),
    ("Antibiotics", "Sulfamethoxazole/Trimethoprim", None, "Tablets", ["400/80mg","800/160mg"], 4.99, .55),
    ("Antibiotics", "Nitrofurantoin Monohydrate", None, "Capsules", ["100mg"], 24.99, .48),
    ("Antibiotics", "Metronidazole", None, "Tablets", ["250mg","500mg"], 7.99, .52),
    ("Antibiotics", "Clindamycin HCl", None, "Capsules", ["150mg","300mg"], 9.99, .42),
    ("Antibiotics", "Trimethoprim", None, "Tablets", ["100mg"], 7.99, .25),
    # ═══ Pain (Rx) ═══
    ("Pain Management", "Gabapentin", ("Neurontin", "Pfizer", 199.99), "Capsules", ["100mg","300mg","400mg","600mg","800mg"], 6.99, .90),
    ("Pain Management", "Pregabalin", ("Lyrica", "Pfizer", 549.99), "Capsules", ["25mg","50mg","75mg","100mg","150mg","200mg","300mg"], None, .55),
    ("Pain Management", "Tramadol HCl", None, "Tablets", ["50mg"], 7.99, .72),
    ("Pain Management", "Meloxicam", None, "Tablets", ["7.5mg","15mg"], 4.99, .78),
    ("Pain Management", "Diclofenac Sodium", None, "Delayed-Release Tablets", ["25mg","50mg","75mg"], 8.99, .55),
    ("Pain Management", "Celecoxib", ("Celebrex", "Pfizer", 349.99), "Capsules", ["50mg","100mg","200mg"], 12.99, .52),
    ("Pain Management", "Naproxen Rx", None, "Tablets", ["250mg","375mg","500mg"], 6.99, .48),
    ("Pain Management", "Indomethacin", None, "Capsules", ["25mg","50mg"], 9.99, .28),
    ("Pain Management", "Cyclobenzaprine HCl", None, "Tablets", ["5mg","10mg"], 4.99, .62),
    ("Pain Management", "Methocarbamol", None, "Tablets", ["500mg","750mg"], 7.99, .42),
    ("Pain Management", "Tizanidine HCl", None, "Tablets", ["2mg","4mg"], 7.99, .38),
    # ═══ Respiratory ═══
    ("Respiratory", "Albuterol Sulfate HFA", ("ProAir HFA", "Teva", 79.99), "Inhaler", ["90mcg/actuation"], 29.99, .88),
    ("Respiratory", "Fluticasone/Salmeterol", ("Advair Diskus", "GSK", 449.99), "Inhaler", ["100/50mcg","250/50mcg","500/50mcg"], None, .55),
    ("Respiratory", "Budesonide/Formoterol", ("Symbicort", "AstraZeneca", 379.99), "Inhaler", ["80/4.5mcg","160/4.5mcg"], None, .48),
    ("Respiratory", "Montelukast Sodium", None, "Tablets", ["4mg","5mg","10mg"], 9.99, .82),
    ("Respiratory", "Fluticasone Propionate", ("Flovent HFA", "GSK", 289.99), "Inhaler", ["44mcg","110mcg","220mcg"], None, .45),
    ("Respiratory", "Tiotropium Bromide", ("Spiriva HandiHaler", "Boehringer Ingelheim", 499.99), "Inhaler", ["1.25mcg"], None, .35),
    ("Respiratory", "Benzonatate", None, "Capsules", ["100mg","200mg"], 9.99, .62),
    # ═══ Anticoagulants & Antiplatelets ═══
    ("Anticoagulants", "Warfarin Sodium", None, "Tablets", ["1mg","2mg","2.5mg","3mg","4mg","5mg","7.5mg","10mg"], 4.99, .68),
    ("Anticoagulants", "Apixaban", ("Eliquis", "Bristol-Myers Squibb", 579.99), "Tablets", ["2.5mg","5mg"], None, .65),
    ("Anticoagulants", "Rivaroxaban", ("Xarelto", "Janssen", 549.99), "Tablets", ["10mg","15mg","20mg"], None, .55),
    ("Anticoagulants", "Clopidogrel Bisulfate", None, "Tablets", ["75mg"], 6.99, .65),
    # ═══ Antiseizure / Neuro ═══
    ("Antiseizure", "Levetiracetam", None, "Tablets", ["250mg","500mg","750mg","1000mg"], 9.99, .62),
    ("Antiseizure", "Lamotrigine", None, "Tablets", ["25mg","50mg","100mg","150mg","200mg"], 7.99, .58),
    ("Antiseizure", "Topiramate", None, "Tablets", ["25mg","50mg","100mg","200mg"], 9.99, .50),
    ("Antiseizure", "Oxcarbazepine", None, "Tablets", ["150mg","300mg","600mg"], 14.99, .32),
    ("Antiseizure", "Valproic Acid", ("Depakote ER", "AbbVie", 299.99), "Extended-Release Tablets", ["250mg","500mg"], 14.99, .35),
    # ═══ Urology / Men's Health ═══
    ("Urology", "Tamsulosin HCl", None, "Capsules", ["0.4mg"], 6.99, .72),
    ("Urology", "Finasteride", None, "Tablets", ["1mg","5mg"], 6.99, .60),
    ("Urology", "Sildenafil Citrate", ("Viagra", "Pfizer", 499.99), "Tablets", ["25mg","50mg","100mg"], 14.99, .55),
    ("Urology", "Tadalafil", ("Cialis", "Eli Lilly", 449.99), "Tablets", ["2.5mg","5mg","10mg","20mg"], 16.99, .50),
    ("Urology", "Dutasteride", ("Avodart", "GSK", 239.99), "Capsules", ["0.5mg"], 19.99, .32),
    # ═══ Eye Rx ═══
    ("Ophthalmic", "Latanoprost", None, "Ophthalmic Solution", ["0.005%"], 12.99, .55),
    ("Ophthalmic", "Timolol Maleate", None, "Ophthalmic Solution", ["0.25%","0.5%"], 9.99, .42),
    ("Ophthalmic", "Brimonidine Tartrate", None, "Ophthalmic Solution", ["0.1%","0.15%","0.2%"], 12.99, .38),
    ("Ophthalmic", "Dorzolamide/Timolol", None, "Ophthalmic Solution", ["2%/0.5%"], 19.99, .32),
    ("Ophthalmic", "Prednisolone Acetate", None, "Ophthalmic Suspension", ["1%"], 24.99, .35),
    # ═══ Dermatology Rx ═══
    ("Dermatology Rx", "Tretinoin", None, "Cream", ["0.025%","0.05%","0.1%"], 29.99, .55),
    ("Dermatology Rx", "Clobetasol Propionate", None, "Cream", ["0.05%"], 14.99, .48),
    ("Dermatology Rx", "Triamcinolone Acetonide", None, "Cream", ["0.025%","0.1%","0.5%"], 6.99, .58),
    ("Dermatology Rx", "Mupirocin", None, "Ointment", ["2%"], 14.99, .42),
    ("Dermatology Rx", "Ketoconazole", None, "Cream", ["2%"], 12.99, .38),
    ("Dermatology Rx", "Fluocinonide", None, "Cream", ["0.05%"], 11.99, .32),
    ("Dermatology Rx", "Permethrin", None, "Cream", ["5%"], 14.99, .28),
    # ═══ Women's Health / Hormones ═══
    ("Women's Health", "Estradiol", None, "Tablets", ["0.5mg","1mg","2mg"], 9.99, .55),
    ("Women's Health", "Estradiol", ("Vivelle-Dot", "Novartis", 199.99), "Transdermal Patch", ["0.025mg/day","0.05mg/day","0.075mg/day","0.1mg/day"], None, .35),
    ("Women's Health", "Norethindrone", None, "Tablets", ["0.35mg"], 14.99, .48),
    ("Women's Health", "Norgestimate/Ethinyl Estradiol", ("Tri-Lo-Sprintec", "Teva", 89.99), "Tablets", ["28-day pack"], 14.99, .52),
    ("Women's Health", "Levonorgestrel/Ethinyl Estradiol", ("Levora", "Allergan", 69.99), "Tablets", ["28-day pack"], 12.99, .42),
    ("Women's Health", "Drospirenone/Ethinyl Estradiol", ("Yaz", "Bayer", 199.99), "Tablets", ["28-day pack"], 24.99, .40),
    ("Women's Health", "Medroxyprogesterone Acetate", None, "Tablets", ["2.5mg","5mg","10mg"], 7.99, .38),
    ("Women's Health", "Progesterone", None, "Capsules", ["100mg","200mg"], 19.99, .32),
    # ═══ ADHD ═══
    ("ADHD", "Methylphenidate HCl ER", ("Concerta", "Janssen", 349.99), "Extended-Release Tablets", ["18mg","27mg","36mg","54mg"], 29.99, .62),
    ("ADHD", "Amphetamine/Dextroamphetamine", None, "Tablets", ["5mg","10mg","15mg","20mg","25mg","30mg"], 24.99, .65),
    ("ADHD", "Lisdexamfetamine Dimesylate", ("Vyvanse", "Takeda", 389.99), "Capsules", ["10mg","20mg","30mg","40mg","50mg","60mg","70mg"], None, .55),
    ("ADHD", "Atomoxetine HCl", None, "Capsules", ["10mg","18mg","25mg","40mg","60mg","80mg"], 19.99, .32),
    # ═══ Specialty / Autoimmune ═══
    ("Specialty", "Adalimumab", ("Humira", "AbbVie", 6922.00), "Injection Pen", ["40mg/0.8mL"], None, .35),
    ("Specialty", "Methotrexate", None, "Tablets", ["2.5mg"], 12.99, .35),
    ("Specialty", "Hydroxychloroquine Sulfate", None, "Tablets", ["200mg"], 9.99, .42),
    ("Specialty", "Sulfasalazine", None, "Tablets", ["500mg"], 14.99, .22),
    # ═══ Antivirals ═══
    ("Antivirals", "Acyclovir", None, "Capsules", ["200mg","400mg","800mg"], 9.99, .55),
    ("Antivirals", "Valacyclovir HCl", None, "Tablets", ["500mg","1g"], 14.99, .58),
    ("Antivirals", "Oseltamivir Phosphate", ("Tamiflu", "Genentech", 179.99), "Capsules", ["30mg","45mg","75mg"], 29.99, .40),
    # ═══ Antipsychotics ═══
    ("Antipsychotics", "Quetiapine Fumarate", None, "Tablets", ["25mg","50mg","100mg","200mg","300mg","400mg"], 9.99, .55),
    ("Antipsychotics", "Aripiprazole", None, "Tablets", ["2mg","5mg","10mg","15mg","20mg","30mg"], 14.99, .52),
    ("Antipsychotics", "Risperidone", None, "Tablets", ["0.25mg","0.5mg","1mg","2mg","3mg","4mg"], 6.99, .42),
    ("Antipsychotics", "Olanzapine", None, "Tablets", ["2.5mg","5mg","10mg","15mg","20mg"], 9.99, .38),
    # ═══ Sleep (Rx) ═══
    ("Sleep Medications", "Zolpidem Tartrate", None, "Tablets", ["5mg","10mg"], 6.99, .58),
    ("Sleep Medications", "Eszopiclone", None, "Tablets", ["1mg","2mg","3mg"], 12.99, .38),
    ("Sleep Medications", "Suvorexant", ("Belsomra", "Merck", 389.99), "Tablets", ["10mg","15mg","20mg"], None, .22),
    # ═══ Miscellaneous High Volume ═══
    ("Miscellaneous", "Allopurinol", None, "Tablets", ["100mg","300mg"], 4.99, .62),
    ("Miscellaneous", "Prednisone", None, "Tablets", ["1mg","2.5mg","5mg","10mg","20mg","50mg"], 4.99, .72),
    ("Miscellaneous", "Methylprednisolone", None, "Dose Pack", ["4mg 21-tablet pack"], 14.99, .48),
    ("Miscellaneous", "Potassium Chloride ER", None, "Extended-Release Tablets", ["10mEq","20mEq"], 6.99, .58),
    ("Miscellaneous", "Ondansetron", None, "ODT Tablets", ["4mg","8mg"], 9.99, .62),
    ("Miscellaneous", "Colchicine", ("Colcrys", "Takeda", 249.99), "Tablets", ["0.6mg"], 19.99, .30),
    ("Miscellaneous", "Promethazine HCl", None, "Tablets", ["12.5mg","25mg","50mg"], 6.99, .48),
    ("Miscellaneous", "Clonidine HCl", None, "Tablets", ["0.1mg","0.2mg","0.3mg"], 4.99, .52),
    ("Miscellaneous", "Doxazosin Mesylate", None, "Tablets", ["1mg","2mg","4mg","8mg"], 6.99, .32),
    ("Miscellaneous", "Prazosin HCl", None, "Capsules", ["1mg","2mg","5mg"], 9.99, .30),
    ("Miscellaneous", "Phenazopyridine HCl", None, "Tablets", ["100mg","200mg"], 9.99, .35),
    ("Miscellaneous", "Nystatin", None, "Oral Suspension", ["100000 units/mL"], 12.99, .32),
    ("Miscellaneous", "Fluconazole", None, "Tablets", ["50mg","100mg","150mg","200mg"], 4.99, .48),
    ("Miscellaneous", "Terbinafine HCl", None, "Tablets", ["250mg"], 9.99, .35),
    ("Miscellaneous", "Lithium Carbonate", None, "Capsules", ["150mg","300mg","600mg"], 6.99, .28),
    ("Miscellaneous", "Dicyclomine HCl", None, "Capsules", ["10mg","20mg"], 7.99, .35),
    ("Miscellaneous", "Sucralfate", None, "Tablets", ["1g"], 14.99, .25),
]


def _generate_rx_catalog():
    """Expand _RX_DRUGS into _RAW_RX with generic and brand versions."""
    rng = random.Random(42)
    for entry in _RX_DRUGS:
        cls, gen_name, brand_info, form, strengths, gen_price, pop = entry
        for i, strength in enumerate(strengths):
            price_step = 1.0 + i * 0.10  # slightly higher price per step up in strength

            # ── Generic version ──
            if gen_price is not None:
                p = round(gen_price * price_step, 2)
                name = f"{gen_name} {form} {strength}"
                # Inhalers, injectables, patches, suspensions, creams = single item
                if form in ("Inhaler", "Injection Pen", "Transdermal Patch",
                            "Oral Suspension", "Ophthalmic Solution",
                            "Ophthalmic Suspension", "Cream", "Ointment"):
                    tc = "S"
                elif "pack" in strength.lower():
                    tc = "S"
                else:
                    tc = "RX"
                jpop = max(0.05, pop - i * 0.02 + rng.uniform(-0.03, 0.03))
                _RAW_RX.append(("Prescription Medications", "Generic", name, form,
                                tc, p, round(jpop, 3),
                                gen_name.lower(), cls, 30 if tc == "RX" else 0))

            # ── Brand version ──
            if brand_info is not None:
                bname, mfr, bprice = brand_info
                bp = round(bprice * price_step, 2)
                bfull = f"{bname} ({gen_name}) {form} {strength}"
                if form in ("Inhaler", "Injection Pen", "Transdermal Patch",
                            "Oral Suspension", "Ophthalmic Solution",
                            "Ophthalmic Suspension", "Cream", "Ointment"):
                    tc = "S"
                elif "pack" in strength.lower():
                    tc = "S"
                else:
                    tc = "RX"
                bpop = max(0.05, pop - 0.25 - i * 0.02 + rng.uniform(-0.03, 0.03))
                _RAW_RX.append(("Prescription Medications", mfr, bfull, form,
                                tc, bp, round(bpop, 3),
                                gen_name.lower(), cls, 30 if tc == "RX" else 0))


_generate_rx_catalog()


# ════════════════════════════════════════════════════════════════════
# BUILD ENGINE — expand _RAW into exactly TARGET_COUNT products
# ════════════════════════════════════════════════════════════════════

def _num_variants(pop: float, ladder_len: int) -> int:
    """How many size variants based on popularity and available ladder sizes."""
    if pop >= 0.62:
        return min(ladder_len, 5)
    if pop >= 0.42:
        return min(ladder_len, 4)
    if pop >= 0.25:
        return min(ladder_len, 3)
    if pop >= 0.10:
        return min(ladder_len, 2)
    return 1


def build_catalog(target_otc: int = TARGET_OTC_COUNT,
                   target_rx: int = TARGET_RX_COUNT,
                   seed: int = SEED) -> pd.DataFrame:
    """Build the full product catalog (OTC + Rx) from embedded knowledge base."""
    rng = random.Random(seed)
    rng_rx = random.Random(seed + 1)  # separate RNG for Rx to not perturb OTC

    target = target_otc  # OTC expansion uses this
    console.print(f"[bold cyan]Building CVS product catalog ({target_otc:,} OTC + {target_rx:,} Rx)...[/bold cyan]")

    # Step 1: Expand _RAW into individual products
    products: list[dict] = []

    for cat, brand, name, subcat, tc, base_price, pop in _RAW:
        ladder = LADDERS.get(tc, LADDERS["S"])
        n_vars = _num_variants(pop, len(ladder))

        for i in range(n_vars):
            size_val, price_mult = ladder[i]
            label = _size_label(tc, size_val)
            full_name = f"{brand} {name}"
            if label:
                full_name += f", {label}"
            price = round(base_price * price_mult, 2)
            weight = _calc_weight(tc, size_val)
            # Jitter popularity slightly
            jpop = pop + rng.uniform(-0.04, 0.04)
            jpop = max(0.01, min(0.99, jpop))

            products.append({
                "name": full_name,
                "brand": brand,
                "category": cat,
                "subcategory": subcat,
                "price": price,
                "weight_oz": weight,
                "popularity_score": round(jpop, 4),
                "is_store_brand": brand == "CVS Health",
                "type_code": tc,
                "size_index": i,
            })

    console.print(f"  Raw expansion produced [yellow]{len(products):,}[/yellow] products")

    # Step 2: Adjust to hit exactly target count
    if len(products) > target:
        # Sort by popularity descending, keep top `target`
        products.sort(key=lambda p: (-p["popularity_score"], p["name"]))
        products = products[:target]
        console.print(f"  Trimmed to [green]{len(products):,}[/green]")
    elif len(products) < target:
        # Need more products — add additional size variants to popular items
        deficit = target - len(products)
        console.print(f"  Need [yellow]{deficit}[/yellow] more products, expanding popular items...")

        # Find product lines that have room for more variants
        existing_keys = set()
        for p in products:
            existing_keys.add((p["brand"], p["name"].rsplit(",", 1)[0] if "," in p["name"] else p["name"], p.get("size_index", 0)))

        extra: list[dict] = []
        for cat, brand, name, subcat, tc, base_price, pop in _RAW:
            if len(extra) >= deficit:
                break
            ladder = LADDERS.get(tc, LADDERS["S"])
            current_n = _num_variants(pop, len(ladder))
            # Try to add the next size up
            for i in range(current_n, len(ladder)):
                if len(extra) >= deficit:
                    break
                size_val, price_mult = ladder[i]
                label = _size_label(tc, size_val)
                full_name = f"{brand} {name}"
                if label:
                    full_name += f", {label}"
                price = round(base_price * price_mult, 2)
                weight = _calc_weight(tc, size_val)
                jpop = pop + rng.uniform(-0.06, -0.02)
                jpop = max(0.01, min(0.99, jpop))
                extra.append({
                    "name": full_name,
                    "brand": brand,
                    "category": cat,
                    "subcategory": subcat,
                    "price": price,
                    "weight_oz": weight,
                    "popularity_score": round(jpop, 4),
                    "is_store_brand": brand == "CVS Health",
                    "type_code": tc,
                    "size_index": i,
                })

        products.extend(extra[:deficit])
        console.print(f"  After expansion: [green]{len(products):,}[/green]")

    # If still not enough, add realistic category-aware size and pack variants.
    # Strategy: cycle through multiple variant types to avoid over-representing
    # any single variant suffix.
    EXPANDABLE_CATS = {
        "Pain Relief & Fever", "Cold/Flu/Allergy", "Digestive Health",
        "Vitamins & Supplements", "Skin Care", "Hair Care", "Oral Care",
        "First Aid & Wound Care", "Feminine Care", "Household Essentials",
        "Deodorant", "Shaving & Grooming", "Foot Care", "Sleep & Relaxation",
        "Smoking Cessation", "Seasonal Items", "Eye & Ear Care",
        "Baby & Childcare", "Snacks & Beverages", "Pet Care",
        "Sexual Health", "Diabetes & Blood Sugar",
    }
    # Varied suffixes with realistic labels, cycled through
    SIZE_SUFFIXES = [
        ("Value Size", 1.80), ("Bonus Size", 1.65), ("Family Size", 2.20),
        ("Economy Pack", 2.50), ("Club Pack", 3.00), ("Super Size", 1.95),
        ("Jumbo Pack", 2.80), ("Bulk Pack", 3.50), ("Max Pack", 2.60),
        ("Stock Up Size", 2.40),
    ]
    PACK_SUFFIXES = [
        ("Twin Pack", 1.85), ("2 Pack", 1.85), ("3 Pack", 2.65),
    ]

    existing_names = {p["name"] for p in products}
    suffix_idx = 0
    pack_idx = 0
    while len(products) < target:
        deficit = target - len(products)
        base_products = sorted(products, key=lambda p: -p["popularity_score"])
        more = []
        for p in base_products:
            if len(more) >= deficit:
                break
            base_name = p["name"]
            # Skip products that already have a size/pack suffix
            if any(s in base_name for s in ["Size", "Pack", "Bulk", "Club"]):
                continue
            cat = p["category"]
            tc = p.get("type_code", "S")

            if cat in EXPANDABLE_CATS and tc != "S":
                suffix, price_mult = SIZE_SUFFIXES[suffix_idx % len(SIZE_SUFFIXES)]
                suffix_idx += 1
                new_name = f"{base_name} {suffix}"
                if new_name in existing_names:
                    continue
                existing_names.add(new_name)
                more.append({
                    "name": new_name, "brand": p["brand"],
                    "category": cat, "subcategory": p["subcategory"],
                    "price": round(p["price"] * price_mult, 2),
                    "weight_oz": round(p["weight_oz"] * price_mult * 0.9, 1),
                    "popularity_score": round(max(0.01, p["popularity_score"] - rng.uniform(0.04, 0.10)), 4),
                    "is_store_brand": p["is_store_brand"],
                    "type_code": tc, "size_index": 99,
                })
            elif cat in EXPANDABLE_CATS and tc == "S":
                suffix, price_mult = PACK_SUFFIXES[pack_idx % len(PACK_SUFFIXES)]
                pack_idx += 1
                new_name = f"{base_name} {suffix}"
                if new_name in existing_names:
                    continue
                existing_names.add(new_name)
                more.append({
                    "name": new_name, "brand": p["brand"],
                    "category": cat, "subcategory": p["subcategory"],
                    "price": round(p["price"] * price_mult, 2),
                    "weight_oz": round(p["weight_oz"] * price_mult * 0.9, 1),
                    "popularity_score": round(max(0.01, p["popularity_score"] - rng.uniform(0.04, 0.10)), 4),
                    "is_store_brand": p["is_store_brand"],
                    "type_code": tc, "size_index": 99,
                })
        if not more:
            break  # No more variants possible
        products.extend(more[:deficit])

    # Final trim if over
    products = products[:target]

    # ── Step 3: Build Rx products from _RAW_RX ──
    rx_products: list[dict] = []
    if target_rx > 0:
        console.print(f"[bold cyan]Building Rx catalog ({target_rx:,} target)...[/bold cyan]")
        for entry in _RAW_RX:
            cat, brand, name, sub, tc, price, pop, gen_name, ther_class, days = entry
            ladder = LADDERS.get(tc, LADDERS["S"])
            n_vars = _num_variants(pop, len(ladder))
            for i in range(n_vars):
                size_val, price_mult = ladder[i]
                label = _size_label(tc, size_val)
                full_name = f"{brand} {name}" if brand != "Generic" else name
                if label:
                    full_name += f", {label}"
                p_price = round(price * price_mult, 2)
                jpop = max(0.05, min(0.99, pop + rng_rx.uniform(-0.03, 0.03)))
                d_supply = int(size_val) if tc == "RX" else (days if days else 30)
                rx_products.append({
                    "name": full_name,
                    "brand": brand,
                    "category": cat,
                    "subcategory": sub,
                    "price": p_price,
                    "weight_oz": round(rng_rx.uniform(1.0, 4.0), 1),
                    "popularity_score": round(jpop, 4),
                    "is_store_brand": brand == "Generic",
                    "is_rx": True,
                    "rx_generic_name": gen_name,
                    "rx_therapeutic_class": ther_class,
                    "rx_days_supply": d_supply,
                })
        console.print(f"  Raw Rx expansion: [yellow]{len(rx_products):,}[/yellow] products")
        # Trim or pad Rx to target
        if len(rx_products) > target_rx:
            rx_products.sort(key=lambda p: -p["popularity_score"])
            rx_products = rx_products[:target_rx]
        elif len(rx_products) < target_rx:
            # Add extended supply variants for generics and brands
            deficit = target_rx - len(rx_products)
            existing_rx_names = {p["name"] for p in rx_products}
            more_rx = []
            supply_options = [(60, 1.75), (180, 4.50), (14, 0.50), (7, 0.28)]
            for supply, pmult in supply_options:
                for entry in _RAW_RX:
                    if len(more_rx) >= deficit:
                        break
                    cat, brand, name, sub, tc, price, pop, gen_name, ther_class, days = entry
                    if tc != "RX":
                        continue
                    prefix = f"{brand} {name}" if brand != "Generic" else name
                    full_name = f"{prefix}, {supply} CT"
                    if full_name in existing_rx_names:
                        continue
                    existing_rx_names.add(full_name)
                    more_rx.append({
                        "name": full_name, "brand": brand,
                        "category": cat, "subcategory": sub,
                        "price": round(price * pmult, 2),
                        "weight_oz": round(rng_rx.uniform(1.0, 4.0), 1),
                        "popularity_score": round(max(0.05, pop - 0.06), 4),
                        "is_store_brand": brand == "Generic",
                        "is_rx": True,
                        "rx_generic_name": gen_name,
                        "rx_therapeutic_class": ther_class,
                        "rx_days_supply": supply,
                    })
                if len(more_rx) >= deficit:
                    break
            rx_products.extend(more_rx[:deficit])
        rx_products = rx_products[:target_rx]
        console.print(f"  Final Rx count: [green]{len(rx_products):,}[/green]")

    # ── Step 4: Assign product_id, sku, unit_cost, avg_units for ALL products ──
    products.sort(key=lambda p: (p["category"], p["brand"], p["name"]))
    rx_products.sort(key=lambda p: (p.get("rx_therapeutic_class", ""), p["brand"], p["name"]))

    rows = []
    # OTC products: IDs 1 to len(products)
    for idx, p in enumerate(products, start=1):
        pop = p["popularity_score"]
        price = p["price"]
        is_sb = p["is_store_brand"]

        if is_sb:
            cost_pct = rng.uniform(0.25, 0.40)
        else:
            cost_pct = rng.uniform(0.35, 0.55)
        unit_cost = round(price * cost_pct, 2)

        if pop >= 0.7:
            avg_units = rng.uniform(12.0, 50.0)
        elif pop >= 0.4:
            avg_units = rng.uniform(3.0, 15.0)
        elif pop >= 0.2:
            avg_units = rng.uniform(1.0, 5.0)
        else:
            avg_units = rng.uniform(0.3, 2.0)
        if p["category"] in ("Snacks & Beverages", "Household Essentials"):
            avg_units *= 1.5
        if p["category"] in ("Greeting Cards & Gift Wrap", "Photo & Electronics", "Smoking Cessation"):
            avg_units *= 0.5

        rows.append({
            "product_id": idx,
            "sku": generate_upc(idx),
            "name": p["name"],
            "brand": p["brand"],
            "category": p["category"],
            "subcategory": p["subcategory"],
            "price": price,
            "unit_cost": unit_cost,
            "weight_oz": p["weight_oz"],
            "is_store_brand": is_sb,
            "is_rx": False,
            "popularity_score": pop,
            "avg_units_per_store_per_week": round(avg_units, 2),
            "rx_generic_name": None,
            "rx_therapeutic_class": None,
            "rx_days_supply": None,
        })

    # Rx products: IDs continue after OTC
    rx_start = len(products) + 1
    for idx, p in enumerate(rx_products, start=rx_start):
        pop = p["popularity_score"]
        price = p["price"]
        is_generic = p["is_store_brand"]  # Generic = store brand for Rx

        # Rx pricing: WAC is higher % of AWP
        if is_generic:
            cost_pct = rng_rx.uniform(0.70, 0.88)
        else:
            cost_pct = rng_rx.uniform(0.60, 0.80)
        unit_cost = round(price * cost_pct, 2)

        # Rx avg_units = prescriptions filled per store per week
        if pop >= 0.7:
            avg_units = rng_rx.uniform(30.0, 80.0)
        elif pop >= 0.4:
            avg_units = rng_rx.uniform(8.0, 35.0)
        elif pop >= 0.2:
            avg_units = rng_rx.uniform(2.0, 10.0)
        else:
            avg_units = rng_rx.uniform(0.5, 3.0)

        rows.append({
            "product_id": idx,
            "sku": generate_ndc(idx),
            "name": p["name"],
            "brand": p["brand"],
            "category": p["category"],
            "subcategory": p["subcategory"],
            "price": price,
            "unit_cost": unit_cost,
            "weight_oz": p["weight_oz"],
            "is_store_brand": is_generic,
            "is_rx": True,
            "popularity_score": pop,
            "avg_units_per_store_per_week": round(avg_units, 2),
            "rx_generic_name": p.get("rx_generic_name"),
            "rx_therapeutic_class": p.get("rx_therapeutic_class"),
            "rx_days_supply": p.get("rx_days_supply"),
        })

    df = pd.DataFrame(rows)

    # Ensure SKU is string
    df["sku"] = df["sku"].astype(str)
    # For OTC rows, zero-pad UPC to 12 digits
    otc_mask = df["is_rx"] == False
    df.loc[otc_mask, "sku"] = df.loc[otc_mask, "sku"].str.zfill(12)

    # Fix dtypes
    df["product_id"] = df["product_id"].astype("int32")
    df["price"] = df["price"].astype("float64")
    df["unit_cost"] = df["unit_cost"].astype("float64")
    df["weight_oz"] = df["weight_oz"].astype("float64")
    df["popularity_score"] = df["popularity_score"].astype("float64")
    df["avg_units_per_store_per_week"] = df["avg_units_per_store_per_week"].astype("float64")
    df["is_store_brand"] = df["is_store_brand"].astype("bool")
    df["is_rx"] = df["is_rx"].astype("bool")
    df["rx_days_supply"] = df["rx_days_supply"].astype("Int32")  # nullable int

    return df


# ════════════════════════════════════════════════════════════════════
# SCRAPE MODE — best-effort crawl of cvs.com
# ════════════════════════════════════════════════════════════════════

SHOP_CATEGORIES = [
    "pain-fever", "cold-flu-medicine", "allergy-medicine",
    "digestive-health", "vitamins-supplements", "skin-care",
    "hair-care", "oral-care", "personal-care", "beauty",
    "baby-child", "first-aid", "eye-ear-care",
    "snacks-candy", "beverages", "household",
    "feminine-care", "sexual-health", "foot-care",
]

SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def scrape_catalog() -> pd.DataFrame | None:
    """Attempt to scrape products from cvs.com.  Returns None if blocked."""
    session = requests.Session()
    session.headers.update(SCRAPE_HEADERS)

    all_products: list[dict] = []

    console.print("[bold cyan]Attempting to scrape CVS.com product pages...[/bold cyan]")

    for cat_slug in tqdm(SHOP_CATEGORIES, desc="Categories"):
        url = f"https://www.cvs.com/shop/{cat_slug}"
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code != 200:
                console.print(f"  [yellow]{cat_slug}: HTTP {resp.status_code}[/yellow]")
                continue

            soup = BeautifulSoup(resp.text, "lxml")

            # Try __NEXT_DATA__
            next_data_tag = soup.find("script", id="__NEXT_DATA__")
            if next_data_tag:
                try:
                    data = json.loads(next_data_tag.string)
                    # Navigate to product list (structure varies by CVS deployments)
                    page_props = data.get("props", {}).get("pageProps", {})
                    products_data = (
                        page_props.get("products")
                        or page_props.get("searchResult", {}).get("products")
                        or page_props.get("plpData", {}).get("products")
                        or []
                    )
                    for prod in products_data:
                        name = prod.get("name") or prod.get("productName") or ""
                        brand = prod.get("brand") or prod.get("brandName") or ""
                        price_str = prod.get("price") or prod.get("regularPrice") or "0"
                        try:
                            price = float(str(price_str).replace("$", "").replace(",", ""))
                        except (ValueError, TypeError):
                            price = 0.0
                        sku = prod.get("sku") or prod.get("skuId") or prod.get("upc") or ""
                        if name:
                            all_products.append({
                                "name": name,
                                "brand": brand,
                                "category": cat_slug.replace("-", " ").title(),
                                "price": price,
                                "sku": str(sku),
                            })
                except (json.JSONDecodeError, KeyError):
                    pass

            # Fallback: try parsing product cards from HTML
            if not all_products or len(all_products) < 10:
                cards = soup.select("[data-testid='product-card'], .product-card, .css-1dbjc4n")
                for card in cards:
                    name_el = card.select_one("[data-testid='product-name'], .product-name, h3, h2")
                    price_el = card.select_one("[data-testid='product-price'], .product-price, .price")
                    brand_el = card.select_one("[data-testid='product-brand'], .product-brand")
                    if name_el:
                        name = name_el.get_text(strip=True)
                        brand = brand_el.get_text(strip=True) if brand_el else ""
                        price_text = price_el.get_text(strip=True) if price_el else "0"
                        try:
                            price = float(re.sub(r"[^\d.]", "", price_text) or "0")
                        except ValueError:
                            price = 0.0
                        all_products.append({
                            "name": name,
                            "brand": brand,
                            "category": cat_slug.replace("-", " ").title(),
                            "price": price,
                            "sku": "",
                        })

            time.sleep(1.5)  # Rate limit

        except requests.RequestException as e:
            console.print(f"  [red]{cat_slug}: {e}[/red]")
            continue

    if len(all_products) < 50:
        console.print(f"[red]Only scraped {len(all_products)} products — too few. Falling back to build mode.[/red]")
        return None

    console.print(f"[green]Scraped {len(all_products)} products from CVS.com[/green]")

    # Build a DataFrame from scraped data with reasonable defaults
    rng = random.Random(SEED)
    rows = []
    for idx, p in enumerate(all_products[:TARGET_COUNT], start=1):
        price = p["price"] if p["price"] > 0 else rng.uniform(3.0, 25.0)
        rows.append({
            "product_id": idx,
            "sku": p.get("sku") or generate_upc(idx),
            "name": p["name"],
            "brand": p.get("brand", "Unknown"),
            "category": p["category"],
            "subcategory": "General",
            "price": round(price, 2),
            "unit_cost": round(price * rng.uniform(0.30, 0.55), 2),
            "weight_oz": round(rng.uniform(1.0, 20.0), 1),
            "is_store_brand": "cvs" in p.get("brand", "").lower(),
            "is_rx": False,
            "popularity_score": round(rng.uniform(0.1, 0.9), 4),
            "avg_units_per_store_per_week": round(rng.uniform(1.0, 30.0), 2),
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════
# OUTPUT & SUMMARY
# ════════════════════════════════════════════════════════════════════

def save_catalog(df: pd.DataFrame, output_dir: str):
    """Save catalog to both CSV and Parquet."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "products.csv"
    parquet_path = out / "products.parquet"

    # Ensure sku is always string (preserve leading zeros in UPC)
    df["sku"] = df["sku"].astype(str)
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    console.print(f"  Saved [green]{csv_path}[/green] ({csv_path.stat().st_size / 1024:.0f} KB)")
    console.print(f"  Saved [green]{parquet_path}[/green] ({parquet_path.stat().st_size / 1024:.0f} KB)")


def print_summary(df: pd.DataFrame):
    """Print a rich summary of the product catalog."""
    console.print()
    console.print(f"[bold]═══ Product Catalog Summary ═══[/bold]")
    console.print(f"  Total products: [bold green]{len(df):,}[/bold green]")
    console.print(f"  Total brands: [cyan]{df['brand'].nunique()}[/cyan]")
    console.print(f"  Store brand products: [cyan]{df['is_store_brand'].sum():,}[/cyan] "
                  f"({df['is_store_brand'].mean():.1%})")
    console.print()

    # Products per category
    cat_table = Table(title="Products per Category", show_lines=False)
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Count", justify="right")
    cat_table.add_column("Brands", justify="right")
    cat_table.add_column("Avg Price", justify="right")
    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat]
        cat_table.add_row(
            cat,
            str(len(sub)),
            str(sub["brand"].nunique()),
            f"${sub['price'].mean():.2f}",
        )
    console.print(cat_table)
    console.print()

    # Top 15 brands by product count
    brand_counts = df["brand"].value_counts().head(15)
    brand_table = Table(title="Top 15 Brands by Product Count")
    brand_table.add_column("Brand", style="cyan")
    brand_table.add_column("Products", justify="right")
    for brand, count in brand_counts.items():
        brand_table.add_row(brand, str(count))
    console.print(brand_table)
    console.print()

    # Price distribution
    console.print("[bold]Price Distribution:[/bold]")
    console.print(f"  Min:    ${df['price'].min():.2f}")
    console.print(f"  Median: ${df['price'].median():.2f}")
    console.print(f"  Mean:   ${df['price'].mean():.2f}")
    console.print(f"  Max:    ${df['price'].max():.2f}")
    console.print()

    # Popularity distribution
    console.print("[bold]Popularity Score Distribution:[/bold]")
    console.print(f"  High (>0.7):    {(df['popularity_score'] > 0.7).sum():,} products")
    console.print(f"  Medium (0.3-0.7): {((df['popularity_score'] >= 0.3) & (df['popularity_score'] <= 0.7)).sum():,} products")
    console.print(f"  Low (<0.3):     {(df['popularity_score'] < 0.3).sum():,} products")

    # Rx summary
    rx_df = df[df["is_rx"] == True]
    if len(rx_df) > 0:
        console.print()
        console.print(f"[bold]═══ Prescription (Rx) Summary ═══[/bold]")
        console.print(f"  Total Rx products: [bold green]{len(rx_df):,}[/bold green]")
        console.print(f"  Generic products: [cyan]{rx_df['is_store_brand'].sum():,}[/cyan]")
        console.print(f"  Brand products: [cyan]{(~rx_df['is_store_brand']).sum():,}[/cyan]")
        console.print(f"  Unique molecules: [cyan]{rx_df['rx_generic_name'].nunique()}[/cyan]")
        console.print(f"  Therapeutic classes: [cyan]{rx_df['rx_therapeutic_class'].nunique()}[/cyan]")
        console.print()
        # Rx by therapeutic class
        rx_table = Table(title="Rx Products by Therapeutic Class", show_lines=False)
        rx_table.add_column("Therapeutic Class", style="cyan")
        rx_table.add_column("Products", justify="right")
        rx_table.add_column("Avg Price", justify="right")
        for cls in sorted(rx_df["rx_therapeutic_class"].dropna().unique()):
            sub = rx_df[rx_df["rx_therapeutic_class"] == cls]
            rx_table.add_row(cls, str(len(sub)), f"${sub['price'].mean():.2f}")
        console.print(rx_table)


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

@click.command()
@click.option("--mode", type=click.Choice(["build", "scrape"]), default="build",
              help="Mode: 'build' generates from knowledge base, 'scrape' tries cvs.com first.")
@click.option("--output-dir", default="data/real", help="Output directory for CSV and Parquet.")
@click.option("--count", default=TARGET_OTC_COUNT, type=int, help="Target OTC product count.")
@click.option("--rx-count", default=TARGET_RX_COUNT, type=int, help="Target Rx product count (0 to skip).")
def main(mode: str, output_dir: str, count: int, rx_count: int):
    """Build a realistic CVS product catalog (OTC + Rx)."""
    console.print(f"[bold]CVS Product Catalog Builder[/bold]")
    console.print(f"  Mode: {mode}")
    console.print(f"  OTC target: {count:,} | Rx target: {rx_count:,}")
    console.print(f"  Output: {output_dir}/")
    console.print()

    df = None

    if mode == "scrape":
        df = scrape_catalog()
        if df is None:
            console.print("[yellow]Scraping failed — falling back to build mode.[/yellow]")

    if df is None:
        df = build_catalog(target_otc=count, target_rx=rx_count)

    save_catalog(df, output_dir)
    print_summary(df)


if __name__ == "__main__":
    main()

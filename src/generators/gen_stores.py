"""Generate synthetic CVS store locations (fallback when scraping fails).

Creates ~9,000 realistic store locations distributed across US states
using the same CVS store density weights as the customer generator.

Output: data/real/stores.csv AND data/real/stores.parquet
"""

from pathlib import Path

import click
import numpy as np
import pandas as pd
from faker import Faker
from rich.console import Console

console = Console()

# Same state weights as gen_customers.py (CVS store density)
STATES = [
    "CA", "FL", "TX", "NY", "OH", "PA", "IL", "MA", "NJ", "GA",
    "NC", "VA", "MI", "AZ", "MD", "IN", "TN", "MO", "CT", "SC",
    "MN", "WI", "CO", "AL", "KY", "LA", "OR", "OK", "NV", "IA",
    "RI", "MS", "AR", "UT", "KS", "NE", "NM", "NH", "WV", "ME",
    "HI", "ID", "DE", "MT", "VT", "SD", "ND", "WY", "AK", "DC",
]
_STATE_WEIGHTS = [
    120, 90, 80, 50, 50, 45, 40, 38, 35, 30,
    28, 27, 25, 22, 20, 18, 17, 15, 14, 13,
    12, 11, 10, 10, 9, 9, 8, 8, 7, 7,
    6, 6, 5, 5, 5, 4, 4, 4, 3, 3,
    3, 3, 3, 2, 2, 2, 1, 1, 1, 1,
]
_sw_total = sum(_STATE_WEIGHTS)
STATE_PROBS = [w / _sw_total for w in _STATE_WEIGHTS]

# Approximate lat/lng centers per state for realistic coordinates
STATE_CENTERS = {
    "CA": (36.78, -119.42), "FL": (27.66, -81.52), "TX": (31.97, -99.90),
    "NY": (42.17, -74.95), "OH": (40.42, -82.91), "PA": (41.20, -77.19),
    "IL": (40.63, -89.40), "MA": (42.41, -71.38), "NJ": (40.06, -74.41),
    "GA": (32.16, -82.90), "NC": (35.76, -79.02), "VA": (37.43, -78.66),
    "MI": (44.31, -85.60), "AZ": (34.05, -111.09), "MD": (39.05, -76.64),
    "IN": (40.27, -86.13), "TN": (35.52, -86.58), "MO": (38.46, -92.29),
    "CT": (41.60, -72.76), "SC": (33.84, -81.16), "MN": (46.73, -94.69),
    "WI": (43.78, -88.79), "CO": (39.55, -105.78), "AL": (32.32, -86.90),
    "KY": (37.84, -84.27), "LA": (30.98, -91.96), "OR": (43.80, -120.55),
    "OK": (35.47, -97.52), "NV": (38.80, -116.42), "IA": (41.88, -93.10),
    "RI": (41.58, -71.48), "MS": (32.35, -89.40), "AR": (35.20, -91.83),
    "UT": (39.32, -111.09), "KS": (39.01, -98.48), "NE": (41.13, -98.27),
    "NM": (34.97, -105.03), "NH": (43.19, -71.57), "WV": (38.60, -80.95),
    "ME": (45.25, -69.45), "HI": (19.90, -155.58), "ID": (44.07, -114.74),
    "DE": (38.91, -75.53), "MT": (46.88, -110.36), "VT": (44.56, -72.58),
    "SD": (43.97, -99.90), "ND": (47.55, -101.00), "WY": (43.08, -107.29),
    "AK": (64.20, -152.49), "DC": (38.91, -77.04),
}

STORE_TYPES = ["pharmacy", "pharmacy", "pharmacy", "pharmacy",
               "minuteclinic", "target_cvs"]


@click.command()
@click.option("--count", default=9000, help="Number of stores to generate.")
@click.option("--output-dir", default="data/real", help="Output directory.")
@click.option("--seed", default=42, help="Random seed.")
def main(count: int, output_dir: str, seed: int) -> None:
    """Generate synthetic CVS store locations."""
    console.print(f"[bold]Generating {count:,} synthetic store locations[/bold]")
    console.print("  (fallback: CVS scraper was blocked)")

    rng = np.random.default_rng(seed)
    fake = Faker("en_US")
    Faker.seed(seed)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Distribute stores across states by weight
    state_indices = rng.choice(len(STATES), size=count, p=STATE_PROBS)
    states = [STATES[i] for i in state_indices]

    store_ids = list(range(1, count + 1))
    names = [f"CVS Pharmacy #{sid}" for sid in store_ids]
    addresses = [fake.street_address() for _ in range(count)]
    cities = [fake.city() for _ in range(count)]
    zip_codes = []
    for st in states:
        try:
            zip_codes.append(fake.zipcode_in_state(state_abbr=st))
        except Exception:
            zip_codes.append(fake.zipcode())

    # Generate lat/lng near state centers
    lats = []
    lngs = []
    for st in states:
        center_lat, center_lng = STATE_CENTERS.get(st, (39.8, -98.6))
        lats.append(round(center_lat + rng.normal(0, 0.8), 6))
        lngs.append(round(center_lng + rng.normal(0, 0.8), 6))

    phones = [fake.numerify("(###) ###-####") for _ in range(count)]
    store_types = [STORE_TYPES[i % len(STORE_TYPES)] for i in range(count)]

    # Hours
    hours_mf = ["8:00 AM - 10:00 PM"] * count
    hours_sat = ["9:00 AM - 9:00 PM"] * count
    hours_sun = ["10:00 AM - 6:00 PM"] * count

    df = pd.DataFrame({
        "store_id": store_ids,
        "name": names,
        "address": addresses,
        "city": cities,
        "state": states,
        "zip_code": zip_codes,
        "lat": lats,
        "lng": lngs,
        "phone": phones,
        "store_type": store_types,
        "hours_mon_fri": hours_mf,
        "hours_sat": hours_sat,
        "hours_sun": hours_sun,
    })

    csv_path = Path(output_dir) / "stores.csv"
    pq_path = Path(output_dir) / "stores.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False, compression="snappy")

    console.print(f"  Saved {csv_path} ({csv_path.stat().st_size // 1024} KB)")
    console.print(f"  Saved {pq_path} ({pq_path.stat().st_size // 1024} KB)")
    console.print(f"[bold green]✓ Generated {count:,} synthetic stores[/bold green]")


if __name__ == "__main__":
    main()

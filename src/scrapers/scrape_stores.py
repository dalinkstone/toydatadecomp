"""Scrape CVS store locations from cvs.com store locator.

Strategy (tried in order):
1. PRIMARY: Crawl CVS's Next.js store locator pages via JSON data routes.
   Enumerate states -> cities -> stores from the city page slcp-props JSON.
2. FALLBACK: Crawl the HTML pages directly, parsing __NEXT_DATA__ for the same data.
3. LAST RESORT: Hit the store-detail endpoint for known store IDs to fill gaps.

CVS has ~9,000 stores. Target: at minimum 8,000 unique stores.

Output: data/real/stores.parquet AND data/real/stores.csv
"""

import html
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import click
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from rich.console import Console

console = Console()

BASE_URL = "https://www.cvs.com"
LOCATOR_PATH = "/store-locator/cvs-pharmacy-locations"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.cvs.com/store-locator/landing",
}

ALL_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR",
}


class CVSStoreScraper:
    """Scrapes CVS store locations from cvs.com."""

    def __init__(
        self,
        output_dir: str = "data/real",
        delay: float = 0.5,
        max_retries: int = 3,
    ):
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.build_id: str | None = None
        self.stores: dict[str, dict] = {}  # store_id -> store data
        self._partial_path = self.output_dir / "stores_partial.json"

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _fetch(self, url: str, expect_json: bool = False) -> requests.Response | None:
        """Fetch URL with retry logic, exponential backoff, and rate limiting."""
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.delay)
                resp = self.session.get(url, timeout=30)

                if resp.status_code == 200:
                    return resp

                if resp.status_code == 404 and expect_json and attempt == 0:
                    # buildId may have rotated after a CVS deployment
                    console.print(
                        "[yellow]Got 404 on JSON data route — refreshing buildId...[/yellow]"
                    )
                    self._refresh_build_id()
                    continue

                if resp.status_code in (403, 429):
                    wait = (2 ** attempt) * 2
                    console.print(
                        f"[yellow]HTTP {resp.status_code} — backing off {wait}s...[/yellow]"
                    )
                    time.sleep(wait)
                    continue

                # Other error
                console.print(f"[yellow]HTTP {resp.status_code} for {url}[/yellow]")

            except requests.RequestException as e:
                wait = (2 ** attempt) * 1
                console.print(
                    f"[yellow]Request error ({e.__class__.__name__}), retry in {wait}s...[/yellow]"
                )
                time.sleep(wait)

        return None

    # ------------------------------------------------------------------
    # Build-ID management (Next.js)
    # ------------------------------------------------------------------

    def _get_build_id(self) -> str | None:
        """Extract the Next.js buildId from any store-locator HTML page."""
        if self.build_id:
            return self.build_id

        resp = self._fetch(f"{BASE_URL}{LOCATOR_PATH}")
        if not resp:
            return None

        # Try __NEXT_DATA__ script tag first
        m = re.search(
            r'<script\s+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
            resp.text,
            re.DOTALL,
        )
        if m:
            try:
                nd = json.loads(m.group(1))
                self.build_id = nd.get("buildId")
            except json.JSONDecodeError:
                pass

        # Fallback: regex in the raw HTML
        if not self.build_id:
            m = re.search(r'"buildId"\s*:\s*"([a-f0-9]{20,})"', resp.text)
            if m:
                self.build_id = m.group(1)

        if self.build_id:
            console.print(f"[green]buildId: {self.build_id[:16]}...[/green]")
        else:
            console.print("[red]Could not extract buildId[/red]")

        return self.build_id

    def _refresh_build_id(self) -> None:
        old = self.build_id
        self.build_id = None
        self._get_build_id()
        if self.build_id and self.build_id != old:
            console.print(f"[green]buildId refreshed to {self.build_id[:16]}...[/green]")

    def _json_url(self, path: str, query: str = "") -> str:
        """Construct a Next.js JSON data-route URL."""
        base = f"{BASE_URL}/retail-locator/_next/data/{self.build_id}{path}.json"
        return f"{base}?{query}" if query else base

    # ------------------------------------------------------------------
    # State enumeration
    # ------------------------------------------------------------------

    def get_states(self) -> list[str]:
        """Return state slug names (e.g. 'New-York') from the country page."""
        bid = self._get_build_id()
        if not bid:
            return []

        html_content = None

        # --- attempt 1: JSON data route ---
        url = self._json_url(f"{LOCATOR_PATH}")
        resp = self._fetch(url, expect_json=True)
        if resp:
            try:
                data = resp.json()
                html_content = data.get("pageProps", {}).get("countryPage", "")
            except (json.JSONDecodeError, KeyError):
                pass

        # --- attempt 2: __NEXT_DATA__ from HTML page ---
        if not html_content:
            resp = self._fetch(f"{BASE_URL}{LOCATOR_PATH}")
            if resp:
                m = re.search(
                    r'<script\s+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                    resp.text,
                    re.DOTALL,
                )
                if m:
                    try:
                        nd = json.loads(m.group(1))
                        html_content = (
                            nd.get("props", {})
                            .get("pageProps", {})
                            .get("countryPage", "")
                        )
                    except json.JSONDecodeError:
                        pass

        if not html_content:
            return []

        return self._extract_state_slugs(html_content)

    @staticmethod
    def _extract_state_slugs(html_body: str) -> list[str]:
        """Pull unique state slug strings out of the country-page HTML."""
        pattern = re.compile(
            rf"{re.escape(LOCATOR_PATH)}/([A-Z][A-Za-z-]+?)(?:/|\"|\s|&|<|$)"
        )
        seen: set[str] = set()
        slugs: list[str] = []
        for m in pattern.finditer(html_body):
            slug = m.group(1)
            if slug not in seen:
                seen.add(slug)
                slugs.append(slug)
        return slugs

    # ------------------------------------------------------------------
    # City enumeration
    # ------------------------------------------------------------------

    def get_cities(self, state_slug: str) -> list[str]:
        """Return city slug names for *state_slug*."""
        html_content = None

        # --- attempt 1: JSON data route ---
        url = self._json_url(
            f"{LOCATOR_PATH}/{state_slug}",
            f"state={state_slug}",
        )
        resp = self._fetch(url, expect_json=True)
        if resp:
            try:
                data = resp.json()
                html_content = data.get("pageProps", {}).get("statePage", "")
            except (json.JSONDecodeError, KeyError):
                pass

        # --- attempt 2: HTML page ---
        if not html_content:
            resp = self._fetch(f"{BASE_URL}{LOCATOR_PATH}/{state_slug}")
            if resp:
                m = re.search(
                    r'<script\s+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                    resp.text,
                    re.DOTALL,
                )
                if m:
                    try:
                        nd = json.loads(m.group(1))
                        html_content = (
                            nd.get("props", {})
                            .get("pageProps", {})
                            .get("statePage", "")
                        )
                    except json.JSONDecodeError:
                        pass

        if not html_content:
            return []

        return self._extract_city_slugs(html_content, state_slug)

    @staticmethod
    def _extract_city_slugs(html_body: str, state_slug: str) -> list[str]:
        pattern = re.compile(
            rf"{re.escape(LOCATOR_PATH)}/{re.escape(state_slug)}"
            r"/([A-Za-z0-9][A-Za-z0-9 .'-]+?)(?:\"|&|<|\s|$)"
        )
        seen: set[str] = set()
        slugs: list[str] = []
        for m in pattern.finditer(html_body):
            slug = m.group(1).rstrip("/")
            if slug and slug not in seen:
                seen.add(slug)
                slugs.append(slug)
        return slugs

    # ------------------------------------------------------------------
    # Store extraction for a single city
    # ------------------------------------------------------------------

    def get_stores_for_city(
        self, state_slug: str, city_slug: str
    ) -> list[dict]:
        """Fetch and parse all stores for a single city."""
        html_content = None

        # --- attempt 1: JSON data route ---
        url = self._json_url(
            f"{LOCATOR_PATH}/{state_slug}/{city_slug}",
            f"state={state_slug}&city={city_slug}",
        )
        resp = self._fetch(url, expect_json=True)
        if resp:
            try:
                data = resp.json()
                html_content = data.get("pageProps", {}).get("cityPageSsr", "")
            except (json.JSONDecodeError, KeyError):
                pass

        # --- attempt 2: HTML page ---
        if not html_content:
            resp = self._fetch(
                f"{BASE_URL}{LOCATOR_PATH}/{state_slug}/{city_slug}"
            )
            if resp:
                m = re.search(
                    r'<script\s+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                    resp.text,
                    re.DOTALL,
                )
                if m:
                    try:
                        nd = json.loads(m.group(1))
                        html_content = (
                            nd.get("props", {})
                            .get("pageProps", {})
                            .get("cityPageSsr", "")
                        )
                    except json.JSONDecodeError:
                        pass

        if not html_content:
            return []

        return self._parse_city_stores(html_content)

    def _parse_city_stores(self, html_body: str) -> list[dict]:
        """Extract store records from the city page HTML (slcp-props JSON)."""
        stores: list[dict] = []

        # --- method 1: BeautifulSoup on custom element attribute ---
        soup = BeautifulSoup(html_body, "lxml")
        el = soup.find("cvs-store-locator-city-page")
        if el:
            raw_props = el.get("slcp-props")
            if raw_props:
                try:
                    props = json.loads(raw_props)
                    for raw in props.get("storeResult", []):
                        parsed = self._parse_store(raw)
                        if parsed:
                            stores.append(parsed)
                    if stores:
                        return stores
                except json.JSONDecodeError:
                    pass

        # --- method 2: regex for slcp-props attribute value ---
        # The attribute value is a JSON blob inside an HTML attribute.
        # BeautifulSoup should handle entity decoding, but as a fallback
        # we try a manual extraction.
        m = re.search(r'slcp-props="(.*?)"', html_body, re.DOTALL)
        if not m:
            m = re.search(r"slcp-props='(.*?)'", html_body, re.DOTALL)
        if m:
            try:
                raw_json = html.unescape(m.group(1))
                props = json.loads(raw_json)
                for raw in props.get("storeResult", []):
                    parsed = self._parse_store(raw)
                    if parsed:
                        stores.append(parsed)
                if stores:
                    return stores
            except (json.JSONDecodeError, ValueError):
                pass

        # --- method 3: JSON-LD structured data ---
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                ld = json.loads(script.string or "")
                items = ld if isinstance(ld, list) else [ld]
                for item in items:
                    parsed = self._parse_jsonld_store(item)
                    if parsed:
                        stores.append(parsed)
            except json.JSONDecodeError:
                continue

        return stores

    # ------------------------------------------------------------------
    # Individual store record parsing
    # ------------------------------------------------------------------

    def _parse_store(self, raw: dict) -> dict | None:
        """Convert a raw slcp-props store object into our output schema."""
        try:
            store_id = str(raw.get("storeNumber", "")).strip()
            if not store_id:
                return None

            # ---- address ----
            addr_obj = raw.get("address", {})
            address = (addr_obj.get("firstLine") or "").strip()
            second_line = (addr_obj.get("secondLine") or "").strip()

            city, state, zip_code = "", "", ""
            if second_line:
                parts = [p.strip() for p in second_line.split(",")]
                if len(parts) >= 3:
                    city = parts[0]
                    state = parts[1].strip().upper()
                    zip_code = parts[2].strip()[:5]
                elif len(parts) == 2:
                    city = parts[0]
                    rest = parts[1].strip().split()
                    state = rest[0].upper() if rest else ""
                    zip_code = rest[1][:5] if len(rest) > 1 else ""

            # Supplement from weeklyAdData.address if fields are missing
            wa = raw.get("weeklyAdData", {}).get("address", {})
            if not city and wa.get("city"):
                city = wa["city"]
            if not state and wa.get("state"):
                state = wa["state"].upper()
            if not zip_code and wa.get("zip"):
                zip_code = str(wa["zip"])[:5]
            if not address and wa.get("street"):
                address = wa["street"].strip()

            # Title-case the city (data often comes in lower/mixed case)
            city = city.strip().title()
            # Title-case addresses that are ALL CAPS
            if address and address == address.upper():
                address = address.title()

            # ---- coordinates ----
            pin = raw.get("mapPin", {})
            lat = float(pin.get("lat", 0))
            lng = float(pin.get("lng", 0))
            if lat == 0.0 and lng == 0.0:
                return None
            if not address:
                return None

            # ---- phone ----
            phone_raw = (raw.get("phone") or "").strip()
            phone = self._format_phone(phone_raw)

            # ---- store type ----
            identifiers = [str(x).lower() for x in raw.get("identifiers", [])]
            indicators = [str(x).lower() for x in raw.get("indicators", [])]
            all_flags = " ".join(identifiers + indicators)

            if "minuteclinic" in all_flags:
                store_type = "minuteclinic"
                name = "CVS MinuteClinic"
            elif "target" in all_flags:
                store_type = "target_cvs"
                name = "CVS Pharmacy (Target)"
            else:
                store_type = "pharmacy"
                name = "CVS Pharmacy"

            # ---- hours ----
            hours_mon_fri, hours_sat, hours_sun = self._extract_hours(raw)

            return {
                "store_id": store_id,
                "name": name,
                "address": address,
                "city": city,
                "state": state,
                "zip_code": zip_code,
                "latitude": lat,
                "longitude": lng,
                "phone": phone,
                "store_type": store_type,
                "hours_mon_fri": hours_mon_fri,
                "hours_sat": hours_sat,
                "hours_sun": hours_sun,
            }
        except Exception as exc:
            # Never crash on a single store
            console.print(f"[dim yellow]Parse error: {exc}[/dim yellow]")
            return None

    @staticmethod
    def _format_phone(raw: str) -> str:
        digits = re.sub(r"\D", "", raw)
        if len(digits) == 11 and digits[0] == "1":
            digits = digits[1:]
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        return raw

    @staticmethod
    def _extract_hours(raw: dict) -> tuple[str, str, str]:
        """Return (hours_mon_fri, hours_sat, hours_sun) strings."""
        # Try structured hours from weeklyAdData
        departments = (
            raw.get("weeklyAdData", {})
            .get("hours", {})
            .get("departments", [])
        )
        for dept in departments:
            if dept.get("name", "").lower() in ("retail", "store"):
                by_day: dict[str, str] = {}
                for entry in dept.get("regHours", []):
                    day = entry.get("weekday", "").upper()
                    start = entry.get("startTime", "")
                    end = entry.get("endTime", "")
                    if start and end:
                        by_day[day] = f"{start} - {end}"
                if by_day:
                    mon_fri = by_day.get("MON", "")
                    sat = by_day.get("SAT", "")
                    sun = by_day.get("SUN", "")
                    return mon_fri, sat, sun

        # Fallback: storeListHours
        slh = raw.get("storeListHours", {})
        store_photo = slh.get("Store & Photo:", [])
        if store_photo:
            joined = " ".join(str(s).strip() for s in store_photo if s).strip()
            return joined, joined, joined

        return "", "", ""

    @staticmethod
    def _parse_jsonld_store(item: dict) -> dict | None:
        """Parse a JSON-LD (schema.org) store object — last-resort fallback."""
        t = item.get("@type", "")
        if t not in ("Pharmacy", "Store", "LocalBusiness", "DrugStore", "MedicalBusiness"):
            return None

        addr = item.get("address", {})
        geo = item.get("geo", {})
        lat = float(geo.get("latitude", 0))
        lng = float(geo.get("longitude", 0))
        if lat == 0.0 and lng == 0.0:
            return None

        return {
            "store_id": item.get("branchCode", item.get("identifier", "")),
            "name": item.get("name", "CVS Pharmacy"),
            "address": addr.get("streetAddress", ""),
            "city": addr.get("addressLocality", ""),
            "state": addr.get("addressRegion", ""),
            "zip_code": str(addr.get("postalCode", ""))[:5],
            "latitude": lat,
            "longitude": lng,
            "phone": item.get("telephone", ""),
            "store_type": "pharmacy",
            "hours_mon_fri": "",
            "hours_sat": "",
            "hours_sun": "",
        }

    # ------------------------------------------------------------------
    # Intermediate save / resume
    # ------------------------------------------------------------------

    def _save_intermediate(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self._partial_path, "w") as fh:
            json.dump(list(self.stores.values()), fh)

    def _load_intermediate(self) -> int:
        """Load partial results from a previous run; return count loaded."""
        if not self._partial_path.exists():
            return 0
        try:
            with open(self._partial_path) as fh:
                data = json.load(fh)
            for store in data:
                sid = store.get("store_id")
                if sid:
                    self.stores[sid] = store
            console.print(
                f"[green]Resumed: loaded {len(self.stores)} stores from partial save[/green]"
            )
            return len(self.stores)
        except (json.JSONDecodeError, KeyError):
            return 0

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------

    def scrape_all(self, dry_run: bool = False) -> None:
        """Top-level loop: states -> cities -> stores."""
        console.print("[bold]Step 1/3: Enumerating states...[/bold]")
        states = self.get_states()

        if not states:
            console.print(
                "[bold red]Could not retrieve state list from CVS.[/bold red]\n"
                "[yellow]CVS may be blocking requests. See CONFIGURE.md for "
                "alternative data sources (ScrapeHero ~$89, AggData ~$49).[/yellow]"
            )
            return

        console.print(f"[green]Found {len(states)} states/territories[/green]")

        if dry_run:
            states = states[:2]
            console.print(f"[yellow]--dry-run: limiting to {len(states)} states[/yellow]")

        self._load_intermediate()
        prev_count = len(self.stores)

        console.print("[bold]Step 2/3: Crawling cities per state...[/bold]")
        all_city_tasks: list[tuple[str, str]] = []

        for state_slug in tqdm(states, desc="Listing cities", unit="state"):
            cities = self.get_cities(state_slug)
            if not cities:
                console.print(f"[yellow]  {state_slug}: 0 cities[/yellow]")
                continue
            if dry_run:
                cities = cities[:3]
            for city in cities:
                all_city_tasks.append((state_slug, city))

        console.print(
            f"[green]Total city pages to fetch: {len(all_city_tasks)}[/green]"
        )

        console.print("[bold]Step 3/3: Fetching store data per city...[/bold]")
        pbar = tqdm(all_city_tasks, desc="Cities", unit="city")

        for state_slug, city_slug in pbar:
            pbar.set_postfix(
                state=state_slug,
                city=city_slug[:15],
                stores=len(self.stores),
            )

            try:
                city_stores = self.get_stores_for_city(state_slug, city_slug)
            except Exception as exc:
                console.print(
                    f"[yellow]Error fetching {state_slug}/{city_slug}: {exc}[/yellow]"
                )
                continue

            new_this_city = 0
            for store in city_stores:
                sid = store["store_id"]
                if sid not in self.stores:
                    self.stores[sid] = store
                    new_this_city += 1

            # Save every ~500 new stores
            if (len(self.stores) - prev_count) // 500 > (
                (len(self.stores) - new_this_city) - prev_count
            ) // 500:
                self._save_intermediate()

        pbar.close()
        console.print(
            f"\n[bold green]Scraping complete: {len(self.stores)} unique stores[/bold green]"
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_results(self) -> pd.DataFrame | None:
        """Write stores.csv, stores.parquet, and print summary."""
        if not self.stores:
            console.print("[red]No stores to save.[/red]")
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(list(self.stores.values()))
        df = df.sort_values(["state", "city", "store_id"]).reset_index(drop=True)

        # Enforce types
        df["store_id"] = df["store_id"].astype(str)
        df["zip_code"] = df["zip_code"].astype(str)
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        # Fill NaN hours with empty string (some stores lack hours data)
        for col in ("hours_mon_fri", "hours_sat", "hours_sun"):
            df[col] = df[col].fillna("")

        csv_path = self.output_dir / "stores.csv"
        parquet_path = self.output_dir / "stores.parquet"

        df.to_csv(csv_path, index=False)
        console.print(f"[green]Saved {len(df)} stores -> {csv_path}[/green]")

        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        console.print(f"[green]Saved {len(df)} stores -> {parquet_path}[/green]")

        # Clean up partial file
        if self._partial_path.exists():
            self._partial_path.unlink()

        self._print_summary(df)
        return df

    @staticmethod
    def _print_summary(df: pd.DataFrame) -> None:
        console.print("\n[bold underline]Summary[/bold underline]")
        console.print(f"  Total stores:       {len(df):,}")
        console.print(f"  Unique states:      {df['state'].nunique()}")
        console.print(f"  Unique cities:      {df['city'].nunique()}")
        console.print(f"  Unique zip codes:   {df['zip_code'].nunique()}")

        state_counts = df.groupby("state").size().sort_values(ascending=False)
        console.print("\n[bold]Top 15 states by store count:[/bold]")
        for st, cnt in state_counts.head(15).items():
            console.print(f"    {st}: {cnt:,}")

        found = set(df["state"].unique())
        missing = ALL_US_STATES - found
        if missing:
            console.print(
                f"\n[yellow]States with 0 stores: {', '.join(sorted(missing))}[/yellow]"
            )
        else:
            console.print("\n[green]All 50 states + DC represented.[/green]")

        type_counts = df.groupby("store_type").size()
        console.print("\n[bold]Store types:[/bold]")
        for stype, cnt in type_counts.items():
            console.print(f"    {stype}: {cnt:,}")


# ======================================================================
# CLI
# ======================================================================


@click.command()
@click.option(
    "--output-dir",
    default="data/real",
    show_default=True,
    help="Directory for stores.csv and stores.parquet.",
)
@click.option(
    "--delay",
    default=0.5,
    type=float,
    show_default=True,
    help="Seconds between HTTP requests (politeness).",
)
@click.option(
    "--max-retries",
    default=3,
    type=int,
    show_default=True,
    help="Retry attempts per failed request.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Quick test: fetch only 2 states x 3 cities.",
)
def main(output_dir: str, delay: float, max_retries: int, dry_run: bool) -> None:
    """Scrape all CVS Pharmacy store locations in the United States."""
    console.print("[bold]CVS Store Location Scraper[/bold]")
    console.print(f"  Output dir : {output_dir}")
    console.print(f"  Delay      : {delay}s")
    console.print(f"  Retries    : {max_retries}")
    if dry_run:
        console.print("  [yellow]MODE: dry-run (2 states, 3 cities each)[/yellow]")
    console.print()

    scraper = CVSStoreScraper(
        output_dir=output_dir,
        delay=delay,
        max_retries=max_retries,
    )
    scraper.scrape_all(dry_run=dry_run)
    scraper.save_results()


if __name__ == "__main__":
    main()

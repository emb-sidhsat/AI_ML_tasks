"""
Phase 1 — Data Ingestion: Recalls
Ground-truth recall records used as labels during validation.
"""
import re
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[2]))
import config

warnings.filterwarnings("ignore", message=".*Unverified HTTPS request.*")


def _normalise(s: str) -> str:
    """Recalls API is stricter about make/model spelling than the complaints API."""
    return re.sub(r"[-\s]+", " ", s.strip().upper())


def fetch_recalls_for_vehicle(make: str, model: str) -> list[dict]:
    """Fetch recalls for a make/model across all target years."""
    url  = f"{config.NHTSA_BASE}/recalls/recallsByVehicle"
    rows = []
    failures = 0
    for year in config.YEARS_TO_TARGET:
        try:
            resp = requests.get(
                url,
                params={"make": _normalise(make), "model": _normalise(model), "modelYear": year},
                timeout=15,
                verify=config.NHTSA_API_VERIFY_SSL,
            )
            if resp.status_code != 200:
                failures += 1
                print(f"  WARNING recall request failed for {make} {model} {year}: "
                      f"HTTP {resp.status_code} — {resp.text[:160]}")
                continue
            results = resp.json().get("results", [])
            for r in results:
                r["make_pulled"]  = make
                r["model_pulled"] = model
                r["year_pulled"]  = year
            rows.extend(results)
        except requests.RequestException as exc:
            failures += 1
            print(f"  WARNING recall request failed for {make} {model} {year}: "
                  f"{type(exc).__name__}: {exc}")
        time.sleep(config.API_DELAY)
    if failures == len(config.YEARS_TO_TARGET) and failures:
        raise RuntimeError(f"Recall API failed for every target year for {make} {model}")
    return rows


def load_or_fetch_recalls(vehicles: list[tuple[str, str]]) -> pd.DataFrame:
    """Load from cache if available; otherwise fetch from NHTSA API and cache."""
    cache = config.DATA_RAW / "recalls_raw.csv"
    if cache.exists() and not config.FORCE_REFETCH:
        try:
            cached = pd.read_csv(cache)
        except pd.errors.EmptyDataError:
            cached = None
        if cached is not None:
            print(f"[cache] recalls_raw.csv → {len(cached):,} rows")
            return cached
        print("[cache] recalls_raw.csv is empty or malformed; refetching")

    all_rows: list[dict] = []
    for make, model in tqdm(vehicles, desc="Fetching recalls"):
        try:
            rows = fetch_recalls_for_vehicle(make, model)
        except RuntimeError as exc:
            print(f"  WARNING skipping {make} {model}: {exc}")
            continue
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    if not df.empty and "make_pulled" in df.columns:
        df["vehicle_key"] = (
            df["make_pulled"].str.upper() + "_" +
            df["model_pulled"].str.upper() + "_" +
            df["year_pulled"].astype(str)
        )
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"\n[saved] {len(df):,} recalls → {cache}")
    return df

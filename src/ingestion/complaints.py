"""
Phase 1 — Data Ingestion: Complaints
Fetches NHTSA consumer complaint narratives with disk caching.
"""
import re
import time
import requests
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
import config


def fetch_makes(year: int, issue_type: str = "c") -> list[str]:
    """All vehicle makes that have complaint data for a given model year."""
    url  = f"{config.NHTSA_BASE}/products/vehicle/makes"
    resp = requests.get(url, params={"modelYear": year, "issueType": issue_type}, timeout=15)
    resp.raise_for_status()
    return [m["make"] for m in resp.json().get("results", [])]


def fetch_models_for_make_year(make: str, year: int) -> list[str]:
    """All models with complaint data for a given make and model year."""
    url  = f"{config.NHTSA_BASE}/products/vehicle/models"
    resp = requests.get(url, params={"make": make, "modelYear": year, "issueType": "c"}, timeout=15)
    resp.raise_for_status()
    time.sleep(config.API_DELAY)
    return [m["model"] for m in resp.json().get("results", [])]


def _complaint_count(make: str, model: str, year: int) -> int:
    url  = f"{config.NHTSA_BASE}/complaints/complaintsByVehicle"
    resp = requests.get(url, params={"make": make, "model": model, "modelYear": year}, timeout=15)
    resp.raise_for_status()
    time.sleep(config.API_DELAY)
    return len(resp.json().get("results", []))


def build_vehicle_catalogue() -> list[tuple[str, str]]:
    """
    Returns (make, model) pairs that:
      1. Have complaint data in EVERY target year (consistent coverage).
      2. Exceed MIN_COMPLAINTS in the probe year (sufficient volume).
    Capped at MAX_VEHICLES to keep the full pipeline tractable.
    """
    print(f"Discovering makes for {config.PROBE_YEAR}...")
    makes = fetch_makes(config.PROBE_YEAR)
    print(f"  Found {len(makes)} makes")

    # Find models present across all target years
    consistent_pairs: list[tuple[str, str]] = []
    for make in makes:
        model_sets = []
        for year in config.YEARS_TO_TARGET:
            models = set(fetch_models_for_make_year(make, year))
            model_sets.append(models)
        consistent = set.intersection(*model_sets) if model_sets else set()
        for model in consistent:
            consistent_pairs.append((make, model))

    print(f"  {len(consistent_pairs)} (make, model) pairs with consistent cross-year coverage")

    # Volume filter: keep only pairs with enough complaints in probe year
    rows = []
    for make, model in consistent_pairs:
        count = _complaint_count(make, model, config.PROBE_YEAR)
        if count >= config.MIN_COMPLAINTS:
            rows.append({"make": make, "model": model, "complaints": count})

    df = pd.DataFrame(rows).sort_values("complaints", ascending=False)
    top = df.head(config.MAX_VEHICLES)
    catalogue = list(zip(top["make"], top["model"]))
    print(f"  Final catalogue: {len(catalogue)} vehicles (≥{config.MIN_COMPLAINTS} complaints, top {config.MAX_VEHICLES})")
    return catalogue


def fetch_complaints_for_vehicle(make: str, model: str, year: int) -> list[dict]:
    url    = f"{config.NHTSA_BASE}/complaints/complaintsByVehicle"
    resp   = requests.get(url, params={"make": make, "model": model, "modelYear": year}, timeout=15)
    resp.raise_for_status()
    time.sleep(config.API_DELAY)
    results = resp.json().get("results", [])
    for r in results:
        r["make_pulled"]  = make
        r["model_pulled"] = model
        r["year_pulled"]  = year
    return results


def load_or_fetch_complaints(vehicles: list[tuple[str, str]]) -> pd.DataFrame:
    """Load from cache if available; otherwise fetch from NHTSA API and cache."""
    cache = config.DATA_RAW / "complaints_raw.csv"
    if cache.exists() and not config.FORCE_REFETCH:
        print(f"[cache] complaints_raw.csv → {len(pd.read_csv(cache)):,} rows")
        return pd.read_csv(cache)

    all_rows: list[dict] = []
    for make, model in vehicles:
        for year in config.YEARS_TO_TARGET:
            rows = fetch_complaints_for_vehicle(make, model, year)
            all_rows.extend(rows)
            print(f"  {make:<15} {model:<20} {year}  →  {len(rows):>4} complaints")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["vehicle_key"] = (
            df["make_pulled"].str.upper() + "_" +
            df["model_pulled"].str.upper() + "_" +
            df["year_pulled"].astype(str)
        )
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"\n[saved] {len(df):,} complaints → {cache}")
    return df

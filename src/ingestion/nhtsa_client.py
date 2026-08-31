"""
NHTSA API client — fetches makes, models, complaints, and recalls.
All functions are cache-aware; set FORCE_REFETCH=True in config to bypass.
"""

import re
import time
import json
import os
import requests
import pandas as pd
from tqdm import tqdm
from typing import List, Dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict, delay: float = 0.4, timeout: int = 60):
    """Single GET with polite delay. Returns parsed JSON results list or []."""
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        time.sleep(delay)
        return r.json().get("results", [])
    except Exception as e:
        print(f"  API error [{url}] params={params}: {e}")
        return []


def _normalise_model(model: str) -> str:
    """Collapse hyphens/spaces for fuzzy recall lookup."""
    return re.sub(r"[-\s]+", " ", model.strip().upper())


# ---------------------------------------------------------------------------
# Makes
# ---------------------------------------------------------------------------

def fetch_makes_for_year(year: int, delay: float = 0.4) -> List[str]:
    results = _get(
        "https://api.nhtsa.gov/products/vehicle/makes",
        {"modelYear": year, "issueType": "c"},
        delay=delay,
    )
    return [m.get("make", "") for m in results if m.get("make")]


def fetch_all_makes(years: List[int], cache_path: str, delay: float = 0.4, force: bool = False) -> Dict[int, List[str]]:
    if not force and os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:
        with open(cache_path) as f:
            data = json.load(f)
        makes_by_year = {int(k): v for k, v in data.items()}
        print(f"  Cache hit -> {cache_path}  ({sum(len(v) for v in makes_by_year.values())} makes)")
        return makes_by_year

    makes_by_year = {}
    for year in years:
        makes = fetch_makes_for_year(year, delay)
        makes_by_year[year] = makes
        print(f"  {year}: {len(makes)} makes")
    with open(cache_path, "w") as f:
        json.dump(makes_by_year, f)
    return makes_by_year


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def fetch_models_for_make_year(make: str, year: int, delay: float = 0.4) -> List[str]:
    results = _get(
        "https://api.nhtsa.gov/products/vehicle/models",
        {"modelYear": year, "issueType": "c", "make": make},
        delay=delay,
    )
    return [m.get("model", "") for m in results if m.get("model")]


def fetch_all_models(makes: List[str], years: List[int], cache_path: str,
                     delay: float = 0.4, force: bool = False) -> Dict[str, Dict[int, List[str]]]:
    if not force and os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:
        with open(cache_path) as f:
            data = json.load(f)
        registry = {make: {int(y): models for y, models in yd.items()} for make, yd in data.items()}
        print(f"  Cache hit -> {cache_path}  ({len(registry)} makes)")
        return registry

    registry = {}
    for make in tqdm(makes, desc="Fetching models"):
        registry[make] = {}
        for year in years:
            registry[make][year] = fetch_models_for_make_year(make, year, delay)
    with open(cache_path, "w") as f:
        json.dump(registry, f)
    return registry


# ---------------------------------------------------------------------------
# Complaints
# ---------------------------------------------------------------------------

def fetch_complaints(make: str, model: str, year: int, delay: float = 0.4) -> List[dict]:
    results = _get(
        "https://api.nhtsa.gov/complaints/complaintsByVehicle",
        {"make": make, "model": model, "modelYear": year},
        delay=delay,
    )
    for rec in results:
        rec["make_pulled"] = make
        rec["model_pulled"] = model
        rec["year_pulled"] = year
    return results


def fetch_all_complaints(vehicles: List[dict], cache_path: str,
                         delay: float = 0.4, force: bool = False) -> pd.DataFrame:
    if not force and os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:
        df = pd.read_csv(cache_path)
        print(f"  Cache hit -> {cache_path}  ({len(df):,} complaints)")
        return df

    all_records = []
    for vehicle in vehicles:
        for year in vehicle["years"]:
            records = fetch_complaints(vehicle["make"], vehicle["model"], year, delay)
            all_records.extend(records)
            print(f"  {vehicle['make']} {vehicle['model']} {year}: {len(records)} complaints")

    df = pd.DataFrame(all_records)
    df.to_csv(cache_path, index=False)
    print(f"  Saved {len(df):,} complaints -> {cache_path}")
    return df


# ---------------------------------------------------------------------------
# Recalls
# ---------------------------------------------------------------------------

def fetch_recalls(make: str, model: str, year: int, delay: float = 0.4) -> List[dict]:
    """
    Attempts lookup with original model name, then normalised name.
    Returns first successful (non-empty) result.
    """
    for model_attempt in sorted({model, _normalise_model(model)}):
        results = _get(
            "https://api.nhtsa.gov/recalls/recallsByVehicle",
            {"make": make, "model": model_attempt, "modelYear": year},
            delay=delay,
        )
        if results:
            for rec in results:
                rec["make_pulled"] = make
                rec["model_pulled"] = model
                rec["year_pulled"] = year
            return results
    return []


def fetch_all_recalls(vehicles: List[dict], cache_path: str,
                      delay: float = 0.4, force: bool = False) -> pd.DataFrame:
    if not force and os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:
        df = pd.read_csv(cache_path)
        print(f"  Cache hit -> {cache_path}  ({len(df):,} recalls)")
        return df

    all_records = []
    for vehicle in vehicles:
        for year in vehicle["years"]:
            records = fetch_recalls(vehicle["make"], vehicle["model"], year, delay)
            all_records.extend(records)

    df = pd.DataFrame(all_records) if all_records else pd.DataFrame()
    df.to_csv(cache_path, index=False)
    print(f"  Saved {len(df):,} recalls -> {cache_path}")
    return df


# ---------------------------------------------------------------------------
# Coverage audit
# ---------------------------------------------------------------------------

def audit_recall_coverage(vehicles: List[dict], df_recalls: pd.DataFrame,
                           cache_path: str) -> dict:
    """
    Tracks which vehicle_keys returned zero recall records from the API.
    These should be EXCLUDED from evaluation (not labeled as non-recalled).
    """
    attempted = set()
    for vehicle in vehicles:
        for year in vehicle["years"]:
            vk = f"{vehicle['make'].upper()}_{vehicle['model'].upper()}_{year}"
            attempted.add(vk)

    with_recalls = set()
    if len(df_recalls) > 0:
        for _, row in df_recalls.iterrows():
            vk = f"{str(row['make_pulled']).upper()}_{str(row['model_pulled']).upper()}_{row['year_pulled']}"
            with_recalls.add(vk)

    no_data = attempted - with_recalls
    coverage = {
        "vehicles_attempted": sorted(attempted),
        "vehicles_with_recalls": sorted(with_recalls),
        "vehicles_no_data": sorted(no_data),
    }
    with open(cache_path, "w") as f:
        json.dump(coverage, f, indent=2)

    print(f"  Attempted  : {len(attempted)}")
    print(f"  With recalls: {len(with_recalls)}")
    print(f"  No API data : {len(no_data)}  (excluded from evaluation)")
    return coverage

"""
Phase 1 — Data Ingestion: Recalls
Ground-truth recall records used as labels during validation.
"""
import re
import time
import requests
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
import config


def _normalise(s: str) -> str:
    """Recalls API is stricter about make/model spelling than the complaints API."""
    return re.sub(r"[-\s]+", " ", s.strip().upper())


def fetch_recalls_for_vehicle(make: str, model: str) -> list[dict]:
    """Fetch recalls for a make/model across all target years."""
    url  = f"{config.NHTSA_BASE}/products/vehicle/recallsByVehicle"
    rows = []
    for year in config.YEARS_TO_TARGET:
        try:
            resp = requests.get(
                url,
                params={"make": _normalise(make), "model": _normalise(model), "modelYear": year},
                timeout=15,
            )
            if resp.status_code != 200:
                continue
            results = resp.json().get("results", [])
            for r in results:
                r["make_pulled"]  = make
                r["model_pulled"] = model
                r["year_pulled"]  = year
            rows.extend(results)
        except requests.RequestException:
            pass
        time.sleep(config.API_DELAY)
    return rows


def load_or_fetch_recalls(vehicles: list[tuple[str, str]]) -> pd.DataFrame:
    """Load from cache if available; otherwise fetch from NHTSA API and cache."""
    cache = config.DATA_RAW / "recalls_raw.csv"
    if cache.exists() and not config.FORCE_REFETCH:
        print(f"[cache] recalls_raw.csv → {len(pd.read_csv(cache)):,} rows")
        return pd.read_csv(cache)

    all_rows: list[dict] = []
    for make, model in vehicles:
        rows = fetch_recalls_for_vehicle(make, model)
        all_rows.extend(rows)
        print(f"  {make:<15} {model:<20}  →  {len(rows):>3} recalls")

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

"""
Phase 1 — Data Ingestion: Recalls
Ground-truth recall records used as labels during validation.
"""
import re
import logging
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
logger = logging.getLogger(__name__)


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
                logger.warning("Recall request failed for %s %s %s: HTTP %s - %s", make, model, year, resp.status_code, resp.text[:160])
                continue
            results = resp.json().get("results", [])
            for r in results:
                r["make_pulled"]  = make
                r["model_pulled"] = model
                r["year_pulled"]  = year
            rows.extend(results)
        except requests.RequestException as exc:
            failures += 1
            logger.warning("Recall request failed for %s %s %s: %s: %s", make, model, year, type(exc).__name__, exc)
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
            logger.info("Cache hit: %s (%s rows)", cache, f"{len(cached):,}")
            return cached
        logger.warning("Cache %s is empty or malformed; refetching", cache)

    all_rows: list[dict] = []
    for make, model in tqdm(vehicles, desc="Fetching recalls"):
        try:
            rows = fetch_recalls_for_vehicle(make, model)
        except RuntimeError as exc:
            logger.warning("Skipping %s %s: %s", make, model, exc)
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
    logger.info("Saved %s recalls to %s", f"{len(df):,}", cache)
    return df

"""
Central configuration for the NHTSA Early Warning System.
Edit this file to change years, catalogue size, scoring thresholds, or caching.
"""
from pathlib import Path

ROOT = Path(__file__).parent

# ── NHTSA API ──────────────────────────────────────────────────────────────
NHTSA_BASE = "https://api.nhtsa.gov"
API_DELAY  = 0.5   # seconds between requests (be polite to the public API)

# ── Vehicle Catalogue ──────────────────────────────────────────────────────
YEARS_TO_TARGET = [2016, 2017, 2018, 2019, 2020]
PROBE_YEAR      = 2020   # year used to measure complaint volume for filtering
MIN_COMPLAINTS  = 50     # minimum complaints in probe year to include a vehicle
MAX_VEHICLES    = 30     # cap on catalogue size (keeps full pipeline tractable)

# ── Scoring Thresholds ─────────────────────────────────────────────────────
HIGH_SCORE_THRESHOLD = 45    # composite_score_v1 cutoff for SBERT encoding
ZS_SCORE_THRESHOLD   = 55    # composite_score_v1 cutoff for zero-shot classification
ZS_MAX_COMPLAINTS    = 800   # runtime cap on zero-shot (BART-MNLI is slow on CPU)

COMPOSITE_CRITICAL = 60      # complaints above this are "critical" in aggregation

# ── Caching ────────────────────────────────────────────────────────────────
FORCE_REFETCH = False   # True → always call API; False → use cached CSVs if present

# ── Data Paths ─────────────────────────────────────────────────────────────
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_OUTPUTS   = ROOT / "data" / "outputs"

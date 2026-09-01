"""
Central configuration for the NHTSA Early Warning System.
Edit this file to change years, catalogue size, scoring thresholds, or caching.

Environment variable overrides (useful in Docker / CI):
  FORCE_REFETCH=true      — always re-call the NHTSA API regardless of cached CSVs
  NHTSA_BASE=<url>        — override API base (e.g. point at a mock server in tests)
  API_DELAY=<seconds>     — override per-request polite delay (set 0 in tests)
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent

# ── NHTSA API ──────────────────────────────────────────────────────────────
# Both overridable via env var: point NHTSA_BASE at a mock server in tests,
# set API_DELAY=0 in CI to skip polite delays.
NHTSA_BASE = os.getenv("NHTSA_BASE", "https://api.nhtsa.gov")
API_DELAY  = float(os.getenv("API_DELAY", "0.5"))

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
# Read from env var so Docker / CI can override without rebuilding the image
FORCE_REFETCH = os.getenv("FORCE_REFETCH", "false").lower() == "true"

# ── Data Lake Paths ────────────────────────────────────────────────────────
# Bronze preserves API extracts; Silver contains cleaned and scored records;
# Gold contains vehicle-level products and validation artifacts.
DATA_BRONZE = ROOT / "data" / "bronze"
DATA_SILVER = ROOT / "data" / "silver"
DATA_GOLD   = ROOT / "data" / "gold"

# Compatibility aliases for modules outside the pipeline scripts.
DATA_RAW       = DATA_BRONZE
DATA_PROCESSED = DATA_SILVER
DATA_OUTPUTS   = DATA_GOLD

"""
Phase 3 — Criticality Scoring, Layer 1: Rule-Based
Keyword matching + structured signal extraction.

This is the interpretable baseline. Always implement this first.
A rule-based system tells you what to beat with fancier methods.

NLP concepts covered:
  - Tiered keyword matching
  - Feature engineering from structured metadata
  - TF-IDF (statistical text representation)
"""
import ast
import re
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))


# ── Tiered safety vocabulary (curated from NHTSA recall terminology) ───────
# Baseline = 5.  Most complaints score 5–30.  Genuinely dangerous ones → 60+.
# Caps per tier prevent 10 weak words from outscoring 1 strong word.

TIER1_DANGER: list[str] = [
    "fire", "explode", "explosion", "rollaway", "roll away",
    "loss of control", "unintended acceleration", "sudden acceleration",
    "cannot stop", "no brakes", "brake fail", "brake failure",
    "airbag deploy", "airbag explode", "death", "fatality", "died",
    "serious injury", "hospitalized",
]

TIER2_SAFETY: list[str] = [
    "crash", "accident", "collision", "injury", "injured",
    "unsafe", "dangerous", "hazard", "stall", "stalled",
    "sudden", "unexpected", "without warning", "spontaneous",
    "airbag", "air bag",
]

TIER3_WEAK: list[str] = [
    "failed", "failure", "malfunction", "broken", "defect",
    "recall", "not working", "problem", "issue",
]

# ── Component risk tiers ────────────────────────────────────────────────────
# Score reflects danger of failure, not frequency.
COMPONENT_RISK_MAP: dict[str, int] = {
    "SERVICE BRAKES": 60, "STEERING": 60, "AIR BAGS": 60,
    "FUEL SYSTEM": 55,    "TIRES": 50,
    "ENGINE": 40,         "SUSPENSION": 40, "ELECTRICAL SYSTEM": 35,
    "TRANSMISSION": 35,   "POWER TRAIN": 35,
    "VISIBILITY": 20,     "EXTERIOR LIGHTING": 20, "STRUCTURE": 20,
    "INTERIOR LINING": 5, "SEATS": 5, "AUDIO SYSTEM": 5,
}


def keyword_score(text: str) -> dict:
    """
    Score one complaint text using tiered keyword matching.
    Returns score (0–100), keywords found, and tier summary.
    """
    if not isinstance(text, str) or not text.strip():
        return {"score": 5.0, "keywords_found": [], "tier_summary": "no text"}

    t = text.lower()
    score = 5.0
    found: list[str] = []

    # Tier 1: each hit +20, cap 3 hits
    t1 = [w for w in TIER1_DANGER if w in t]
    score += min(len(t1), 3) * 20
    found += [f"[T1]{w}" for w in t1[:3]]

    # Tier 2: each hit +10, cap 3 hits
    t2 = [w for w in TIER2_SAFETY if w in t]
    score += min(len(t2), 3) * 10
    found += [f"[T2]{w}" for w in t2[:3]]

    # Tier 3: each hit +3, cap 4 hits
    t3 = [w for w in TIER3_WEAK if w in t]
    score += min(len(t3), 4) * 3
    found += [f"[T3]{w}" for w in t3[:4]]

    return {
        "score":          round(min(100.0, score), 1),
        "keywords_found": found,
        "tier_summary":   f"T1:{len(t1)} T2:{len(t2)} T3:{len(t3)}",
    }


def parse_components(raw) -> list[str]:
    """
    Parse the NHTSA 'components' field which can be a plain string,
    a comma/AND-separated string, or a Python-list-as-string.
    """
    if pd.isna(raw) or raw == "":
        return []
    if isinstance(raw, list):
        return [c.upper().strip() for c in raw]
    raw_str = str(raw).strip()
    try:
        parsed = ast.literal_eval(raw_str)
        if isinstance(parsed, list):
            return [c.upper().strip() for c in parsed]
    except Exception:
        pass
    parts = re.split(r"\bAND\b|,", raw_str, flags=re.IGNORECASE)
    return [p.upper().strip() for p in parts if p.strip()]


def component_risk_score(raw) -> int:
    """Return the HIGHEST risk score across all components in the field."""
    scores = [COMPONENT_RISK_MAP.get(c, 0) for c in parse_components(raw)]
    return max(scores) if scores else 0


def extract_structured_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary safety flags and component/date signals to the DataFrame.

    New columns:
      has_crash, has_fire, has_injury, has_fatality  — from NHTSA flags
      component_parsed, component_risk               — from 'components' column
      days_to_file, filed_quickly                    — from incident/complaint dates
    """
    df = df.copy()

    # Safety flags from NHTSA structured fields
    df["has_crash"]    = df.get("crash",   pd.Series(False, index=df.index)).fillna(False).astype(bool).astype(int)
    df["has_fire"]     = df.get("fire",    pd.Series(False, index=df.index)).fillna(False).astype(bool).astype(int)
    df["has_injury"]   = df.get("injured", pd.Series(0,     index=df.index)).fillna(0).astype(int).clip(0, 1)
    df["has_fatality"] = df.get("deaths",  pd.Series(0,     index=df.index)).fillna(0).astype(int).clip(0, 1)

    # Component risk
    if "components" in df.columns:
        df["component_parsed"] = df["components"].apply(parse_components)
        df["component_risk"]   = df["components"].apply(component_risk_score)
    else:
        df["component_risk"] = 0

    # Date lag: complaints filed quickly after an incident = acute / alarming event
    for col in ("dateOfIncident", "dateComplaint"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", unit="ms")

    if "dateOfIncident" in df.columns and "dateComplaint" in df.columns:
        df["days_to_file"]  = (df["dateComplaint"] - df["dateOfIncident"]).dt.days
        df["filed_quickly"] = (df["days_to_file"].fillna(999) < 30).astype(int)
    else:
        df["days_to_file"]  = None
        df["filed_quickly"] = 0

    return df


def apply_keyword_scoring(df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    """Apply keyword_score() to every row; adds keyword_score and keywords_found columns."""
    col = text_col if text_col in df.columns else "summary"
    print(f"Keyword scoring on '{col}' for {len(df):,} complaints...")
    df = df.copy()
    results             = df[col].fillna("").apply(keyword_score)
    df["keyword_score"] = results.apply(lambda x: x["score"])
    df["keywords_found"]= results.apply(lambda x: str(x["keywords_found"]))
    return df


def build_composite_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic composite score (v1) — used later as training labels for Layer 2.
    Range 0–100.
    """
    df = df.copy()
    s  = df["keyword_score"].fillna(0) * 0.40
    s += df.get("has_crash",    pd.Series(0, index=df.index)).fillna(0) * 20
    s += df.get("has_fire",     pd.Series(0, index=df.index)).fillna(0) * 20
    s += df.get("has_injury",   pd.Series(0, index=df.index)).fillna(0) * 15
    s += df.get("has_fatality", pd.Series(0, index=df.index)).fillna(0) * 25
    s += df.get("component_risk", pd.Series(0, index=df.index)).fillna(0) * 0.20
    s += df.get("filed_quickly",  pd.Series(0, index=df.index)).fillna(0) * 5
    df["composite_score_v1"] = s.clip(0, 100)
    return df

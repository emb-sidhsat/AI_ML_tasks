"""
Structured signal extraction from NHTSA API fields.
These are the most reliable signals — ground truth from filed data.
"""

import ast
import pandas as pd
from typing import Dict

# ---------------------------------------------------------------------------
# Component risk hierarchy
# ---------------------------------------------------------------------------

CRITICAL_COMPONENTS: Dict[str, int] = {
    "AIR BAGS": 95,
    "STEERING": 90,
    "SERVICE BRAKES": 90,
    "FORWARD COLLISION AVOIDANCE": 90,
    "FUEL SYSTEM": 85,
    "VEHICLE SPEED CONTROL": 85,
    "ENGINE AND ENGINE COOLING": 80,
    "ENGINE": 80,
    "ELECTRICAL SYSTEM": 75,
    "POWER TRAIN": 75,
    "TRACTION CONTROL SYSTEM": 70,
    "LANE DEPARTURE": 70,
    "TIRES": 70,
    "SUSPENSION": 65,
    "VISIBILITY": 65,
    "LATCHES": 60,
    "EXTERIOR LIGHTING": 50,
    "UNKNOWN OR OTHER": 30,
    "INTERIOR FEATURES": 20,
}


def _parse_components(raw) -> list:
    if not isinstance(raw, str) or not raw.strip():
        return []
    stripped = raw.strip()
    if stripped.startswith("["):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                return [str(c).upper().strip() for c in parsed if c]
        except (ValueError, SyntaxError):
            pass
    return [stripped.upper()]


def _component_risk(raw) -> int:
    components = _parse_components(raw)
    if not components:
        return 30
    scores = []
    for comp in components:
        matched = False
        for key, score in CRITICAL_COMPONENTS.items():
            if key in comp:
                scores.append(score)
                matched = True
                break
        if not matched:
            scores.append(30)
    return max(scores)


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def extract_structured_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts binary flags and component risk from NHTSA structured fields.
    Column names validated against actual NHTSA API: numberOfInjuries, numberOfDeaths, components.
    """
    df = df.copy()
    df["has_crash"] = df["crash"].fillna(False).astype(bool).astype(int)
    df["has_fire"] = df["fire"].fillna(False).astype(bool).astype(int)
    df["has_injury"] = df["numberOfInjuries"].fillna(0).astype(float).gt(0).astype(int)
    df["has_fatality"] = df["numberOfDeaths"].fillna(0).astype(float).gt(0).astype(int)

    df["date_incident"] = pd.to_datetime(df["dateOfIncident"], errors="coerce")
    df["date_filed"] = pd.to_datetime(df["dateComplaintFiled"], errors="coerce")
    df["days_to_file"] = (df["date_filed"] - df["date_incident"]).dt.days
    df.loc[df["days_to_file"] < 0, "days_to_file"] = None
    df["filed_quickly"] = df["days_to_file"].fillna(999).lt(30).astype(int)

    df["component_parsed"] = df["components"].apply(_parse_components)
    df["component_risk"] = df["components"].apply(_component_risk)

    return df


# ---------------------------------------------------------------------------
# Score-B: Incident severity (0-25)
# ---------------------------------------------------------------------------

def compute_score_b(row: pd.Series) -> float:
    """
    Structured incident severity (0-25).
    Includes component risk bonus (0-5 pts) — normalised from 0-95 scale.
    component_risk feeds in here since it was previously computed but never used in scoring.
    """
    score = 0.0
    score += min(row["has_fatality"] * 20.0, 25.0)
    if row["has_fatality"] == 0:
        deaths = float(row.get("numberOfDeaths", 0) or 0)
        score += min(deaths * 6.25, 12.5)
    score += min(row["has_injury"] * 2.5, 12.5)
    score += row["has_crash"] * 10.0
    score += row["has_fire"] * 8.0
    score += row.get("filed_quickly", 0) * 2.0

    # Component risk bonus (0-5 pts)
    comp_risk = float(row.get("component_risk", 30) or 30)
    score += min((comp_risk / 95) * 5.0, 5.0)

    return round(min(score, 25.0), 2)

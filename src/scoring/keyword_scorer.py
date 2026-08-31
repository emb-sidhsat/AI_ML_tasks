"""
Tiered keyword criticality scoring.
Runs on RAW summary text (pre-lemmatization) to preserve inflected danger words.
Hard cap at 25 prevents keyword inflation dominating the composite score.
"""

from typing import List, Dict

# ---------------------------------------------------------------------------
# Keyword tiers
# ---------------------------------------------------------------------------

TIER1_DANGER: List[str] = [
    "fire", "explode", "explosion", "rollaway", "roll away",
    "loss of control", "unintended acceleration", "sudden acceleration",
    "cannot stop", "no brakes", "brake fail", "brake failure",
    "airbag deploy", "airbag explode", "death", "fatality", "died",
    "serious injury", "hospitalized",
]

TIER2_SAFETY: List[str] = [
    "crash", "accident", "collision", "injury", "injured",
    "unsafe", "dangerous", "hazard", "stall", "stalled",
    "sudden", "unexpected", "without warning", "spontaneous",
    "airbag", "air bag",
]

TIER3_WEAK: List[str] = [
    "failed", "failure", "malfunction", "broken", "defect",
    "recall", "not working", "problem", "issue",
]


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def keyword_score(text: str,
                  baseline: float = 5.0,
                  t1_pts: float = 20.0,
                  t2_pts: float = 10.0,
                  t3_pts: float = 3.0,
                  t1_max: int = 3,
                  t2_max: int = 3,
                  t3_max: int = 4,
                  hard_cap: float = 25.0) -> Dict:
    """
    Returns dict with:
      score         : 0-25 float (hard cap)
      keywords_found: list with tier prefix [T1]/[T2]/[T3]
      tier_summary  : "T1:x T2:y T3:z"

    NOTE: Run on raw `summary` field, not text_clean.
    Lemmatization converts "died"->"die", "hospitalized"->"hospitalize",
    breaking matches against inflected tier keywords.
    """
    if not isinstance(text, str) or not text.strip():
        return {"score": baseline, "keywords_found": [], "tier_summary": "no text"}

    t = text.lower()
    score = baseline
    found = []

    t1_words = [w for w in TIER1_DANGER if w in t][:t1_max]
    score += len(t1_words) * t1_pts
    found += [f"[T1]{w}" for w in t1_words]

    t2_words = [w for w in TIER2_SAFETY if w in t][:t2_max]
    score += len(t2_words) * t2_pts
    found += [f"[T2]{w}" for w in t2_words]

    t3_words = [w for w in TIER3_WEAK if w in t][:t3_max]
    score += len(t3_words) * t3_pts
    found += [f"[T3]{w}" for w in t3_words]

    return {
        "score": round(min(hard_cap, score), 1),
        "keywords_found": found,
        "tier_summary": f"T1:{len(t1_words)} T2:{len(t2_words)} T3:{len(t3_words)}",
    }


def apply_keyword_scoring(df, raw_text_col: str = "summary"):
    """Apply keyword scoring to a DataFrame. Returns df with new columns."""
    results = df[raw_text_col].fillna("").apply(keyword_score)
    df = df.copy()
    df["keyword_score"] = results.apply(lambda x: x["score"])
    df["keywords_found"] = results.apply(lambda x: x["keywords_found"])
    df["tier_summary"] = results.apply(lambda x: x["tier_summary"])
    return df

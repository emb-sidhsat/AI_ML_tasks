"""Tests for layer1_rule_based — the deterministic scoring baseline."""
import pandas as pd
import pytest

from src.scoring.layer1_rule_based import (
    build_composite_v1,
    component_risk_score,
    extract_structured_signals,
    keyword_score,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# keyword_score — layer1 version (cap 100, baseline 5)
# Unlike the standalone keyword_scorer.py (cap 25), this one is uncapped at
# the tier level so genuine multi-keyword danger can reach 100.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,lo,hi", [
    ("vehicle caught fire and driver died in crash", 45, 100),  # T1 + T2
    ("crash accident collision injury airbag",        25, 100),  # T2 heavy
    ("problem issue malfunction broken defect",        5,  20),  # T3 only
    ("great car smooth ride",                          5,   6),  # no keywords
    ("",                                               5,   5),  # empty → baseline
])
def test_keyword_score_ranges(text, lo, hi):
    result = keyword_score(text)
    assert lo <= result["score"] <= hi, f"score={result['score']} not in [{lo},{hi}] for: {text!r}"


def test_layer1_cap_is_100_not_25():
    """Layer1 keyword_score must reach 100 on an extreme text (cap differs from standalone scorer)."""
    extreme = "fire explode death fatality brake failure crash accident injury airbag problem"
    assert keyword_score(extreme)["score"] == 100.0


def test_tier_prefixes_in_keywords_found():
    result = keyword_score("fire crash broken")
    labels = [k[:4] for k in result["keywords_found"]]
    assert "[T1" in " ".join(labels)
    assert "[T2" in " ".join(labels)
    assert "[T3" in " ".join(labels)


# ---------------------------------------------------------------------------
# component_risk_score — tier mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("component,expected_min", [
    ("SERVICE BRAKES",   60),
    ("STEERING",         60),
    ("AIR BAGS",         60),
    ("FUEL SYSTEM",      55),
    ("TIRES",            50),
    ("AUDIO SYSTEM",      0),
    ("INTERIOR LINING",   0),
    ("",                  0),
])
def test_component_risk_tiers(component, expected_min):
    assert component_risk_score(component) >= expected_min


def test_highest_risk_wins_for_comma_separated():
    assert component_risk_score("SERVICE BRAKES, AUDIO SYSTEM") >= 60


def test_and_separated_components_parsed():
    assert component_risk_score("STEERING AND AUDIO SYSTEM") >= 60


# ---------------------------------------------------------------------------
# extract_structured_signals — binary flag extraction
# ---------------------------------------------------------------------------

def test_crash_flag_set(crash_complaint_df):
    result = extract_structured_signals(crash_complaint_df)
    assert result["has_crash"].iloc[0] == 1
    assert result["has_fire"].iloc[0] == 0


def test_injury_count_clipped_to_binary():
    """injured=3 must produce has_injury=1, not 3."""
    df = pd.DataFrame([{"crash": False, "fire": False, "injured": 3, "deaths": 0}])
    result = extract_structured_signals(df)
    assert result["has_injury"].iloc[0] == 1


def test_missing_columns_default_to_zero():
    """Must not raise when optional NHTSA fields are absent from the DataFrame."""
    df = pd.DataFrame([{"summary": "brake noise only"}])
    result = extract_structured_signals(df)
    assert result["has_crash"].iloc[0] == 0
    assert result["component_risk"].iloc[0] == 0


def test_filed_quickly_flag_set_under_30_days(crash_complaint_df):
    result = extract_structured_signals(crash_complaint_df)
    # gap is 13 days → filed_quickly=1
    assert result["filed_quickly"].iloc[0] == 1


# ---------------------------------------------------------------------------
# build_composite_v1 — formula correctness
# ---------------------------------------------------------------------------

def _scored_row(**overrides) -> pd.DataFrame:
    base = {
        "keyword_score": 0.0, "has_crash": 0, "has_fire": 0,
        "has_injury": 0, "has_fatality": 0, "component_risk": 0, "filed_quickly": 0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_composite_v1_fatality_contributes_25_pts():
    result = build_composite_v1(_scored_row(has_fatality=1))
    assert result["composite_score_v1"].iloc[0] == pytest.approx(25.0)


def test_composite_v1_all_signals_capped_at_100():
    df = _scored_row(
        keyword_score=100.0, has_crash=1, has_fire=1,
        has_injury=1, has_fatality=1, component_risk=95, filed_quickly=1,
    )
    assert build_composite_v1(df)["composite_score_v1"].iloc[0] == 100.0


def test_composite_v1_no_signals_is_zero():
    assert build_composite_v1(_scored_row())["composite_score_v1"].iloc[0] == 0.0


def test_composite_v1_crash_contributes_more_than_injury():
    crash  = build_composite_v1(_scored_row(has_crash=1))["composite_score_v1"].iloc[0]
    injury = build_composite_v1(_scored_row(has_injury=1))["composite_score_v1"].iloc[0]
    assert crash > injury   # crash=20 pts, injury=15 pts

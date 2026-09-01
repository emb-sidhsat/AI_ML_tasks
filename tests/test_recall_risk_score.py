"""Tests for the vehicle-level recall risk score formula."""
import numpy as np
import pandas as pd
import pytest

from src.aggregation.vehicle_risk import recall_risk_score

pytestmark = pytest.mark.unit


def _row(**overrides) -> pd.Series:
    base = {
        "mean_composite_score": 0.0,
        "crash_rate": 0.0,
        "injury_rate": 0.0,
        "n_complaints": 0,
        "safety_category_rate": 0.0,
        "sbert_cluster_rate": 0.0,
        "sbert_mean_cluster_depth": 0.0,
    }
    base.update(overrides)
    return pd.Series(base)


# ---------------------------------------------------------------------------
# Individual component contributions (formula: score += component * weight)
# ---------------------------------------------------------------------------

def test_all_zeros_gives_zero(minimal_vehicle_row):
    assert recall_risk_score(minimal_vehicle_row) == 0.0


def test_composite_score_capped_at_40():
    """mean_composite_score * 0.40 must not exceed 40 even at score=200."""
    assert recall_risk_score(_row(mean_composite_score=200)) == 40.0


def test_crash_rate_full_contributes_20():
    assert recall_risk_score(_row(crash_rate=1.0)) == pytest.approx(20.0)


def test_injury_rate_full_contributes_10():
    assert recall_risk_score(_row(injury_rate=1.0)) == pytest.approx(10.0)


@pytest.mark.parametrize("field,value,expected", [
    ("safety_category_rate",     1.0, 15.0),
    ("sbert_cluster_rate",       1.0,  5.0),
])
def test_single_signal_exact_contribution(field, value, expected):
    assert recall_risk_score(_row(**{field: value})) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Ordering and relationships
# ---------------------------------------------------------------------------

def test_crash_outweighs_injury_at_equal_rate():
    """crash_rate weight (20) > injury_rate weight (10)."""
    assert recall_risk_score(_row(crash_rate=0.5)) > recall_risk_score(_row(injury_rate=0.5))


def test_volume_signal_is_log_scaled():
    """Doubling n_complaints should give diminishing marginal returns."""
    s100  = recall_risk_score(_row(n_complaints=100))
    s200  = recall_risk_score(_row(n_complaints=200))
    s400  = recall_risk_score(_row(n_complaints=400))
    gap_1 = s200 - s100
    gap_2 = s400 - s200
    assert gap_2 < gap_1, "Expected diminishing returns from log-scale volume signal"


# ---------------------------------------------------------------------------
# Boundary and cap behaviour
# ---------------------------------------------------------------------------

def test_score_hard_capped_at_100():
    row = _row(
        mean_composite_score=100, crash_rate=1.0, injury_rate=1.0,
        n_complaints=10_000, safety_category_rate=1.0,
        sbert_cluster_rate=1.0, sbert_mean_cluster_depth=200,
    )
    assert recall_risk_score(row) == 100.0


def test_missing_optional_fields_default_to_zero():
    """recall_risk_score must not raise when SBERT/ZS columns are absent."""
    row = pd.Series({"mean_composite_score": 50.0, "crash_rate": 0.1})
    score = recall_risk_score(row)
    assert 0 <= score <= 100

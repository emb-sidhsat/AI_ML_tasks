"""Unit tests for structured signal extraction and Score-B."""
import pandas as pd
import pytest
from src.scoring.structured_signals import compute_score_b


def _make_row(**kwargs):
    defaults = {
        "has_crash": 0, "has_fire": 0, "has_injury": 0, "has_fatality": 0,
        "numberOfDeaths": 0, "filed_quickly": 0, "component_risk": 30,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def test_fatality_maxes_score_b():
    row = _make_row(has_fatality=1, component_risk=95)
    assert compute_score_b(row) == 25.0


def test_crash_and_injury_accumulate():
    row = _make_row(has_crash=1, has_injury=1)
    score = compute_score_b(row)
    assert score > 12  # crash=10 + injury=2.5 + component_bonus


def test_component_risk_bonus():
    """High component_risk should give more points than low."""
    low = compute_score_b(_make_row(component_risk=30))
    high = compute_score_b(_make_row(component_risk=95))
    assert high > low


def test_no_signals_is_low():
    row = _make_row()
    score = compute_score_b(row)
    assert score < 5

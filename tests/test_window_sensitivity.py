"""
Tests for window sensitivity analysis.
Validates that different window configurations produce expected label counts.
"""

import pandas as pd
import pytest
from src.aggregation.recall_labeler import temporal_recall_label


def _build_scenario():
    """
    Build a controlled scenario with 5 vehicles and known recall timing:
      V1: recall 30d after complaint  → in all windows
      V2: recall 200d after complaint → in 365d but not 180d
      V3: recall 60d before complaint → in 90d-before window, not 0
      V4: recall 400d after complaint → outside all windows
      V5: no recall data              → always 0
    """
    earliest = pd.Series({
        "MAKE_A_2018": pd.Timestamp("2018-06-01"),
        "MAKE_B_2018": pd.Timestamp("2018-06-01"),
        "MAKE_C_2018": pd.Timestamp("2018-06-01"),
        "MAKE_D_2018": pd.Timestamp("2018-06-01"),
        "MAKE_E_2018": pd.Timestamp("2018-06-01"),
    })
    recalls = pd.DataFrame([
        {"vehicle_key": "MAKE_A_2018",
         "recall_date": pd.Timestamp("2018-07-01")},   # +30d
        {"vehicle_key": "MAKE_B_2018",
         "recall_date": pd.Timestamp("2018-12-18")},   # +200d
        {"vehicle_key": "MAKE_C_2018",
         "recall_date": pd.Timestamp("2018-04-01")},   # -61d
        {"vehicle_key": "MAKE_D_2018",
         "recall_date": pd.Timestamp("2019-08-06")},   # +431d
        # MAKE_E has no recall record
    ])
    return earliest, recalls


def test_strict_window_180d():
    e, r = _build_scenario()
    results = {vk: temporal_recall_label(vk, e, r, days_after=180, days_before=0)
               for vk in e.index}
    assert results["MAKE_A_2018"] == 1  # +30d ✓
    assert results["MAKE_B_2018"] == 0  # +200d outside 180d
    assert results["MAKE_C_2018"] == 0  # -61d, no before buffer
    assert results["MAKE_D_2018"] == 0  # +431d outside
    assert results["MAKE_E_2018"] == 0  # no recall


def test_extended_window_365d():
    e, r = _build_scenario()
    results = {vk: temporal_recall_label(vk, e, r, days_after=365, days_before=0)
               for vk in e.index}
    assert results["MAKE_A_2018"] == 1
    assert results["MAKE_B_2018"] == 1  # +200d now within 365d ✓
    assert results["MAKE_C_2018"] == 0
    assert results["MAKE_D_2018"] == 0
    assert results["MAKE_E_2018"] == 0


def test_bidirectional_window():
    e, r = _build_scenario()
    results = {vk: temporal_recall_label(vk, e, r, days_after=365, days_before=90)
               for vk in e.index}
    assert results["MAKE_A_2018"] == 1
    assert results["MAKE_B_2018"] == 1
    assert results["MAKE_C_2018"] == 1  # -61d now within 90d buffer ✓
    assert results["MAKE_D_2018"] == 0
    assert results["MAKE_E_2018"] == 0

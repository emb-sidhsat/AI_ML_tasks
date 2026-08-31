"""
Tests for temporal recall labeling — bidirectional window version.
Validates causality constraint and admin-lag handling.

Key scenarios covered:
  - Recall after complaint (core hypothesis)
  - Recall before complaint strict (days_before=0 → label=0)
  - Recall within admin lag buffer (days_before=90 → label=1)
  - Recall outside any window (label=0)
  - Unknown vehicle (label=0)
  - NaT complaint date (label=0)
  - Field name resolution (auto-detect fallback)
"""

import pandas as pd
import pytest
from src.aggregation.recall_labeler import (
    temporal_recall_label,
    build_earliest_complaint_map,
    resolve_recall_date_field,
)


def _recall_df(vehicle_key: str, recall_date: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "vehicle_key": vehicle_key,
        "recall_date": pd.to_datetime(recall_date),
    }])


def _earliest(vehicle_key: str, complaint_date: str) -> pd.Series:
    return pd.Series(
        {vehicle_key: pd.to_datetime(complaint_date)},
        name="earliest_complaint_date"
    )


# ── Core causality ────────────────────────────────────────────────────────

def test_recall_after_complaint_within_window():
    """Recall 90 days after first complaint → label=1."""
    e = _earliest("TOYOTA_CAMRY_2018", "2018-01-01")
    r = _recall_df("TOYOTA_CAMRY_2018", "2018-04-01")   # +90d
    assert temporal_recall_label("TOYOTA_CAMRY_2018", e, r,
                                  days_after=365, days_before=0) == 1


def test_recall_before_complaint_strict_zero():
    """Recall 5 months before complaint with days_before=0 → label=0."""
    e = _earliest("HONDA_CRV_2016", "2016-06-01")
    r = _recall_df("HONDA_CRV_2016", "2016-01-01")   # -151d
    assert temporal_recall_label("HONDA_CRV_2016", e, r,
                                  days_after=365, days_before=0) == 0


def test_recall_after_window_boundary():
    """Recall exactly 1 day past days_after → label=0."""
    e = _earliest("FORD_F150_2017", "2017-01-01")
    r = _recall_df("FORD_F150_2017", "2018-01-02")   # 366d after
    assert temporal_recall_label("FORD_F150_2017", e, r,
                                  days_after=365, days_before=0) == 0


def test_recall_at_window_boundary():
    """Recall exactly at days_after boundary → label=1."""
    e = _earliest("FORD_F150_2017", "2017-01-01")
    r = _recall_df("FORD_F150_2017", "2018-01-01")   # exactly 365d after
    assert temporal_recall_label("FORD_F150_2017", e, r,
                                  days_after=365, days_before=0) == 1


# ── Bidirectional window (admin lag) ─────────────────────────────────────

def test_recall_within_admin_lag_buffer():
    """
    Recall 61 days before complaint with days_before=90 → label=1.
    Models NHTSA ReportReceivedDate admin lag for voluntary recalls.
    Production observation: mean lead_time = -72 days.
    """
    e = _earliest("RAM_1500_2018", "2018-06-01")
    r = _recall_df("RAM_1500_2018", "2018-04-01")   # -61d before complaint
    assert temporal_recall_label("RAM_1500_2018", e, r,
                                  days_after=365, days_before=90) == 1


def test_recall_outside_admin_lag_buffer():
    """Recall 151 days before complaint with days_before=90 → label=0."""
    e = _earliest("BMW_X5_2019", "2019-06-01")
    r = _recall_df("BMW_X5_2019", "2019-01-01")   # -151d before complaint
    assert temporal_recall_label("BMW_X5_2019", e, r,
                                  days_after=365, days_before=90) == 0


# ── Edge cases ────────────────────────────────────────────────────────────

def test_unknown_vehicle_is_zero():
    e = _earliest("HONDA_CRV_2016", "2016-06-01")
    r = _recall_df("HONDA_CRV_2016", "2016-08-01")
    assert temporal_recall_label("UNKNOWN_VEHICLE_2018", e, r,
                                  days_after=365, days_before=90) == 0


def test_nat_earliest_complaint_is_zero():
    e = pd.Series({"FORD_F150_2019": pd.NaT}, name="earliest_complaint_date")
    r = _recall_df("FORD_F150_2019", "2019-06-01")
    assert temporal_recall_label("FORD_F150_2019", e, r,
                                  days_after=365, days_before=90) == 0


def test_no_recalls_for_vehicle_is_zero():
    e = _earliest("TOYOTA_CAMRY_2018", "2018-01-01")
    r = _recall_df("OTHER_VEHICLE_2018", "2018-06-01")   # different vehicle
    assert temporal_recall_label("TOYOTA_CAMRY_2018", e, r,
                                  days_after=365, days_before=90) == 0


# ── Field name resolution ─────────────────────────────────────────────────

def test_resolve_correct_field():
    df = pd.DataFrame({"ReportReceivedDate": ["2020-01-01"]})
    assert resolve_recall_date_field(df, "ReportReceivedDate") == "ReportReceivedDate"


def test_resolve_fallback_to_date_column():
    df = pd.DataFrame({"recallInitiationDate": ["2020-01-01"]})
    result = resolve_recall_date_field(df, "ReportReceivedDate")
    assert "date" in result.lower()


def test_resolve_raises_when_no_date_column():
    df = pd.DataFrame({"make": ["TOYOTA"], "model": ["CAMRY"]})
    with pytest.raises(KeyError):
        resolve_recall_date_field(df, "ReportReceivedDate")


# ── build_earliest_complaint_map ──────────────────────────────────────────

def test_earliest_takes_minimum_date():
    df = pd.DataFrame([
        {"make_pulled": "TOYOTA", "model_pulled": "CAMRY",
         "year_pulled": 2018, "dateComplaintFiled": "2018-03-01"},
        {"make_pulled": "TOYOTA", "model_pulled": "CAMRY",
         "year_pulled": 2018, "dateComplaintFiled": "2018-01-15"},
    ])
    result = build_earliest_complaint_map(df)
    assert result["TOYOTA_CAMRY_2018"] == pd.Timestamp("2018-01-15")


def test_vehicle_key_is_uppercased():
    df = pd.DataFrame([
        {"make_pulled": "honda", "model_pulled": "cr-v",
         "year_pulled": 2019, "dateComplaintFiled": "2019-05-01"},
    ])
    result = build_earliest_complaint_map(df)
    assert "HONDA_CR-V_2019" in result.index

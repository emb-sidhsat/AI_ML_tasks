"""Shared fixtures for the NHTSA Early Warning System test suite."""
import pandas as pd
import pytest


@pytest.fixture
def crash_complaint_df():
    """Single row with a crash + brakes component — highest-risk profile."""
    return pd.DataFrame([{
        "crash": True, "fire": False, "injured": 1, "deaths": 0,
        "components": "SERVICE BRAKES",
        "dateOfIncident": 1546300800000,   # 2019-01-01 UTC in ms
        "dateComplaint":  1547424000000,   # 2019-01-14 UTC in ms
        "summary": "The brakes failed completely causing a crash with injury.",
    }])


@pytest.fixture
def cosmetic_complaint_df():
    """Single row with no safety signals — lowest-risk profile."""
    return pd.DataFrame([{
        "crash": False, "fire": False, "injured": 0, "deaths": 0,
        "components": "AUDIO SYSTEM",
        "dateOfIncident": 1551398400000,   # 2019-03-01 UTC in ms
        "dateComplaint":  1551657600000,   # 2019-03-04 UTC in ms  (3 days → filed quickly)
        "summary": "The radio cuts out occasionally.",
    }])


@pytest.fixture
def minimal_vehicle_row():
    """Baseline vehicle row with all signals at zero for recall_risk_score tests."""
    return pd.Series({
        "mean_composite_score": 0.0,
        "crash_rate": 0.0,
        "injury_rate": 0.0,
        "n_complaints": 0,
        "safety_category_rate": 0.0,
        "sbert_cluster_rate": 0.0,
        "sbert_mean_cluster_depth": 0.0,
    })

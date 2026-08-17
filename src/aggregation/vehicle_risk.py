"""
Phase 4 — Vehicle Aggregation
Converts complaint-level composite scores into a vehicle-level recall risk score.

NLP concepts covered:
  - Aggregation of NLP-derived signals
  - Temporal feature engineering (complaint acceleration)
  - Recall risk scoring formula
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
import config
from src.scoring.layer3_semantic import SAFETY_CRITICAL_CATS


def _temporal_acceleration(group: pd.DataFrame) -> float:
    """
    Linear slope of monthly complaint counts.
    Positive = complaints trending up = early warning signal.
    """
    if "complaint_date" not in group.columns:
        return 0.0
    monthly = group.set_index("complaint_date").resample("ME").size()
    if len(monthly) < 3:
        return 0.0
    x = np.arange(len(monthly))
    return float(np.polyfit(x, monthly.values, 1)[0])


def compute_vehicle_features(group: pd.DataFrame) -> dict:
    """
    Aggregate complaint-level signals to vehicle-level features.
    Incorporates SBERT cluster depth and zero-shot category signals.
    """
    vehicle_key  = group["vehicle_key"].iloc[0] if "vehicle_key" in group.columns else "UNKNOWN"
    n_complaints = len(group)

    scores          = group["composite_score"].fillna(0)
    mean_composite  = scores.mean()
    max_composite   = scores.max()
    high_crit_count = int((scores > config.COMPOSITE_CRITICAL).sum())

    crash_rate  = group.get("has_crash",  pd.Series([0] * n_complaints, index=group.index)).fillna(0).mean()
    injury_rate = group.get("has_injury", pd.Series([0] * n_complaints, index=group.index)).fillna(0).mean()
    fatal_rate  = group.get("has_fatality", pd.Series([0] * n_complaints, index=group.index)).fillna(0).mean()

    # Component concentration: how focused are complaints on a single component?
    comp_concentration = 0.0
    if "component_parsed" in group.columns:
        exploded = group["component_parsed"].explode().dropna()
        if not exploded.empty:
            comp_concentration = float(exploded.value_counts(normalize=True).iloc[0])

    # SBERT cluster signals
    cluster_rate       = float(group["sbert_is_clustered"].mean()) if "sbert_is_clustered" in group else 0.0
    mean_cluster_depth = float(group["sbert_cluster_size"].mean())  if "sbert_cluster_size"  in group else 0.0

    # Zero-shot category signals
    zs_cats = group["zs_category"] if "zs_category" in group.columns else pd.Series(["LOW_SCORE"] * n_complaints)
    safety_cat_rate = float(zs_cats.isin(SAFETY_CRITICAL_CATS).mean())
    cat_counts      = zs_cats.value_counts()
    dominant_cat    = str(cat_counts.index[0]) if len(cat_counts) > 0 else "UNKNOWN"
    distinct_cats   = int(zs_cats[zs_cats != "UNCLASSIFIED"].nunique())

    return {
        "vehicle_key":              vehicle_key,
        "n_complaints":             n_complaints,
        "mean_composite_score":     round(mean_composite, 2),
        "max_composite_score":      round(max_composite, 2),
        "high_criticality_count":   high_crit_count,
        "crash_rate":               round(crash_rate, 3),
        "injury_rate":              round(injury_rate, 3),
        "fatality_rate":            round(fatal_rate, 3),
        "component_concentration":  round(comp_concentration, 3),
        "temporal_acceleration":    round(_temporal_acceleration(group), 4),
        "sbert_cluster_rate":       round(cluster_rate, 3),
        "sbert_mean_cluster_depth": round(mean_cluster_depth, 1),
        "safety_category_rate":     round(safety_cat_rate, 3),
        "dominant_defect_category": dominant_cat,
        "distinct_defect_categories": distinct_cats,
    }


def recall_risk_score(row: pd.Series) -> float:
    """
    Compute a 0–100 recall risk score for one vehicle.
    All components are independently interpretable.

    Components:
      mean_composite_score   (0–40)  — average NLP criticality of complaints
      crash_rate             (0–20)  — fraction of complaints with crash flag
      injury_rate            (0–10)  — fraction with injury flag
      n_complaints log scale (0–10)  — volume signal (diminishing returns)
      safety_category_rate   (0–15)  — fraction in safety-critical ZS categories
      sbert_cluster_rate     (0–5)   — same defect reported repeatedly
      sbert_mean_cluster_depth (0–5) — how large the repeated-defect clusters are
    """
    score = 0.0
    score += min(row.get("mean_composite_score", 0) * 0.40, 40)
    score += row.get("crash_rate",  0) * 20
    score += row.get("injury_rate", 0) * 10
    score += min(np.log1p(row.get("n_complaints", 0)) / np.log1p(200) * 10, 10)
    score += row.get("safety_category_rate",     0) * 15
    score += row.get("sbert_cluster_rate",        0) * 5
    score += min(row.get("sbert_mean_cluster_depth", 0) / 50 * 5, 5)
    return round(min(score, 100.0), 2)


def aggregate_to_vehicles(df: pd.DataFrame, df_recalls: pd.DataFrame) -> pd.DataFrame:
    """
    Group complaints by vehicle_key, compute features and recall risk,
    then attach ground-truth actually_recalled label.

    Returns a DataFrame sorted by recall_risk_score descending.
    """
    # Parse complaint date if not already done
    if "complaint_date" not in df.columns and "dateComplaint" in df.columns:
        df = df.copy()
        df["complaint_date"] = pd.to_datetime(df["dateComplaint"], errors="coerce", unit="ms")

    print(f"Aggregating {len(df):,} complaints across {df['vehicle_key'].nunique()} vehicles...")
    vehicle_rows = (
        df.groupby("vehicle_key", group_keys=False)
        .apply(compute_vehicle_features)
    )
    df_vehicles = pd.DataFrame(list(vehicle_rows))
    df_vehicles["recall_risk_score"] = df_vehicles.apply(recall_risk_score, axis=1)

    # Ground-truth label from recalls dataset
    if not df_recalls.empty and "vehicle_key" in df_recalls.columns:
        recalled = set(df_recalls["vehicle_key"].dropna().str.upper())
        df_vehicles["actually_recalled"] = df_vehicles["vehicle_key"].isin(recalled).astype(int)
    else:
        df_vehicles["actually_recalled"] = 0

    # Risk tier for readability
    df_vehicles["risk_tier"] = pd.cut(
        df_vehicles["recall_risk_score"],
        bins=[0, 30, 50, 70, 100],
        labels=["Low", "Medium", "High", "Critical"],
        include_lowest=True,
    )

    result = df_vehicles.sort_values("recall_risk_score", ascending=False).reset_index(drop=True)
    print(f"Done. Recall rate in dataset: {result['actually_recalled'].mean():.1%}")
    return result

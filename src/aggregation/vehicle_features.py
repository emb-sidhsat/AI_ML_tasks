"""
Vehicle-level feature aggregation and recall risk scoring.
Aggregates complaint-level signals to vehicle (make_model_year) level.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from src.classification.zero_shot_classifier import SAFETY_CRITICAL_CATS


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_vehicle_features(group: pd.DataFrame) -> dict:
    """Per-vehicle feature dict from aggregated complaint group."""
    n = len(group)
    vehicle_key = group["vehicle_key"].iloc[0]

    mean_composite = group["composite_score"].mean()
    max_composite = group["composite_score"].max()
    high_crit_count = (group["composite_score"] > 60).sum()
    crash_rate = group.get("has_crash", pd.Series([0] * n)).mean()
    injury_rate = group.get("has_injury", pd.Series([0] * n)).mean()

    # Temporal acceleration (complaint volume trend)
    acceleration = 0.0
    if "complaint_month" in group.columns and group["complaint_date"].notna().sum() > 2:
        monthly = group.groupby("complaint_month").size()
        if len(monthly) >= 3:
            thirds = max(len(monthly) // 3, 1)
            acceleration = (
                monthly.iloc[-thirds:].mean() - monthly.iloc[:thirds].mean()
            ) / (monthly.iloc[:thirds].mean() + 1)

    # Monthly rate
    if group["complaint_date"].notna().sum() > 0:
        date_range = max((group["complaint_date"].max() - group["complaint_date"].min()).days / 30, 1)
        monthly_rate = n / date_range
    else:
        monthly_rate = float(n)

    # SBERT cluster features
    cluster_rate = group["sbert_is_clustered"].mean() if "sbert_is_clustered" in group else 0.0
    mean_cluster_depth = group["sbert_cluster_size"].mean() if "sbert_cluster_size" in group else 0.0

    # Zero-shot features
    zs_cats = group["zs_category"] if "zs_category" in group else pd.Series(["LOW_SCORE"] * n)
    safety_category_rate = zs_cats.isin(SAFETY_CRITICAL_CATS).mean()
    cat_counts = zs_cats.value_counts()
    dominant_category = cat_counts.index[0] if len(cat_counts) > 0 else "UNKNOWN"
    distinct_categories = zs_cats[zs_cats != "LOW_SCORE"].nunique()
    high_conf_zs = int((group["zs_confidence"] > 0.6).sum()) if "zs_confidence" in group.columns else 0

    return {
        "vehicle_key": vehicle_key,
        "n_complaints": n,
        "mean_composite_score": round(mean_composite, 2),
        "max_composite_score": round(max_composite, 2),
        "high_criticality_count": int(high_crit_count),
        "crash_rate": round(crash_rate, 3),
        "injury_rate": round(injury_rate, 3),
        "temporal_acceleration": round(acceleration, 3),
        "monthly_complaint_rate": round(monthly_rate, 2),
        "sbert_cluster_rate": round(cluster_rate, 3),
        "sbert_mean_cluster_depth": round(mean_cluster_depth, 1),
        "safety_category_rate": round(safety_category_rate, 3),
        "dominant_defect_category": dominant_category,
        "distinct_defect_categories": int(distinct_categories),
        "high_confidence_zs_count": high_conf_zs,
    }


def aggregate_vehicle_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.groupby("vehicle_key", group_keys=False).apply(compute_vehicle_features)
    return pd.DataFrame(list(features))


# ---------------------------------------------------------------------------
# Temporal anomaly detection
# ---------------------------------------------------------------------------

def detect_temporal_anomaly(group: pd.DataFrame,
                             contamination: float = 0.15,
                             min_complaints: int = 8,
                             min_months: int = 4) -> dict:
    group = group.copy()
    group["complaint_date"] = pd.to_datetime(group.get("dateComplaintFiled", pd.NaT), errors="coerce")
    group = group.dropna(subset=["complaint_date"])

    if len(group) < min_complaints:
        return {"temporal_anomaly": 0, "anomaly_score": 0.0}

    group["month"] = group["complaint_date"].dt.to_period("M")
    monthly = group.groupby("month").agg(
        volume=("composite_score", "count"),
        severity=("composite_score", "mean"),
        crash_vol=("has_crash", "sum"),
    ).reset_index()

    if len(monthly) < min_months:
        return {"temporal_anomaly": 0, "anomaly_score": 0.0}

    features = monthly[["volume", "severity", "crash_vol"]].fillna(0).values
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=50)
    iso.fit(features)
    preds = iso.predict(features)
    scores = iso.decision_function(features)

    return {
        "temporal_anomaly": int(preds[-1] == -1),
        "anomaly_score": round(float(-scores[-1]), 4),
    }


def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    results = df.groupby("vehicle_key", group_keys=False).apply(detect_temporal_anomaly)
    return pd.DataFrame(list(results), index=results.index).reset_index().rename(
        columns={"index": "vehicle_key"}
    )


# ---------------------------------------------------------------------------
# Hand-tuned vehicle risk score
# ---------------------------------------------------------------------------

def vehicle_recall_risk_v2(row: pd.Series) -> float:
    """
    Composite recall risk (0-100) with hand-tuned weights.
    Replace with GBM scorer (below) once temporal labeling is fixed.
    """
    score = min(row["mean_composite_score"] * 0.40, 40)
    score += row["crash_rate"] * 20
    score += row["injury_rate"] * 10
    score += min(np.log1p(row["n_complaints"]) / np.log1p(200) * 10, 10)
    if "safety_category_rate" in row:     score += row["safety_category_rate"] * 15
    if "sbert_cluster_rate" in row:       score += row["sbert_cluster_rate"] * 3
    if "sbert_mean_cluster_depth" in row: score += min(row["sbert_mean_cluster_depth"] / 50 * 2, 5)
    return round(min(score, 100), 2)


# ---------------------------------------------------------------------------
# GBM learned vehicle scorer
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "mean_composite_score", "max_composite_score", "high_criticality_count",
    "crash_rate", "injury_rate", "temporal_acceleration", "monthly_complaint_rate",
    "sbert_cluster_rate", "sbert_mean_cluster_depth",
    "safety_category_rate", "distinct_defect_categories", "high_confidence_zs_count",
    "n_complaints",
]

OPTIONAL_COLS = ["temporal_anomaly", "anomaly_score", "cross_vehicle_flag"]


def train_gbm_scorer(df_vehicles: pd.DataFrame,
                     n_splits: int = 5) -> tuple:
    """
    Trains GBM on vehicle-level features to predict actually_recalled.
    Requires temporal labeling (Cell 7.3) to be fixed first.

    Returns (gbm_model, feature_importances_series, cv_roc_auc_mean).
    """
    feature_cols = FEATURE_COLS + [c for c in OPTIONAL_COLS if c in df_vehicles.columns]
    available = [c for c in feature_cols if c in df_vehicles.columns]

    eval_mask = (
        df_vehicles.get("recall_data_available", pd.Series(1, index=df_vehicles.index)) == 1
    ) & df_vehicles["actually_recalled"].notna()

    df_eval = df_vehicles[eval_mask].copy()
    X = df_eval[available].fillna(0)
    y = df_eval["actually_recalled"].astype(int)

    print(f"GBM: {len(df_eval)} vehicles  |  recall rate: {y.mean():.1%}")

    if y.nunique() < 2:
        print("SKIPPED: only one class present. Fix temporal recall labeling first.")
        return None, None, 0.0

    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=3, random_state=42,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(gbm, X, y, cv=cv, scoring="roc_auc")
    print(f"GBM 5-fold ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    baseline_auc = roc_auc_score(y, df_eval["recall_risk_v2"])
    print(f"Hand-tuned baseline: {baseline_auc:.3f}")
    print(f"GBM improvement:    {cv_scores.mean() - baseline_auc:+.3f}")

    gbm.fit(X, y)
    importances = pd.Series(gbm.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 feature importances:")
    for feat, imp in importances.head(10).items():
        print(f"  {feat:<35} {imp:.3f}  {'#' * int(imp * 60)}")

    return gbm, importances, cv_scores.mean()

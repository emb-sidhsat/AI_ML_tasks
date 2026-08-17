"""
Phase 5 — Validation
Proves the system works against historical recall reality.

Three checks:
  1. Spearman rank correlation — does higher score predict recalled vehicles?
  2. ROC-AUC               — binary classification performance vs baseline
  3. Timeline case study    — visual proof that complaints precede recalls
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, RocCurveDisplay

sys.path.insert(0, str(Path(__file__).parents[2]))
import config


def rank_correlation(df_vehicles: pd.DataFrame) -> tuple[float, float]:
    """
    Spearman ρ between recall_risk_score and actually_recalled.
    Tests whether the score monotonically predicts recall outcomes.
    """
    corr, pvalue = stats.spearmanr(
        df_vehicles["recall_risk_score"],
        df_vehicles["actually_recalled"],
    )
    print(f"Spearman ρ = {corr:.3f}  (p = {pvalue:.4f})")
    if pvalue < 0.05:
        print("  ✓ Statistically significant — higher score correlates with actual recalls")
    else:
        print("  ✗ Not significant — more data or signal tuning needed")
    return corr, pvalue


def roc_auc(df_vehicles: pd.DataFrame, output_dir: Path | None = None) -> float:
    """
    ROC-AUC for the binary recalled / not-recalled prediction task.
    Saves a curve plot to output_dir if provided.
    """
    y_true  = df_vehicles["actually_recalled"]
    y_score = df_vehicles["recall_risk_score"]

    if y_true.nunique() < 2:
        print("ROC-AUC: only one class present — cannot compute (need both recalled and not-recalled vehicles)")
        return float("nan")

    auc = roc_auc_score(y_true, y_score)
    print(f"ROC-AUC = {auc:.3f}  (0.5 = random, 1.0 = perfect)")

    if output_dir:
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name="recall_risk_score")
        ax.set_title("ROC Curve — Recall Prediction")
        fig.tight_layout()
        path = output_dir / "roc_curve.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved → {path}")

    return auc


def score_distribution_plot(df_vehicles: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Score distribution by recall outcome + risk tier breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    recalled     = df_vehicles[df_vehicles["actually_recalled"] == 1]["recall_risk_score"]
    not_recalled = df_vehicles[df_vehicles["actually_recalled"] == 0]["recall_risk_score"]

    axes[0].hist(not_recalled, bins=30, alpha=0.6, color="#4A80C0", label="Not Recalled")
    axes[0].hist(recalled,     bins=30, alpha=0.7, color="#C04040", label="Recalled")
    axes[0].set_xlabel("Recall Risk Score")
    axes[0].set_ylabel("Vehicle Count")
    axes[0].set_title("Risk Score Distribution by Recall Outcome")
    axes[0].legend()

    tier_counts = df_vehicles.groupby(["risk_tier", "actually_recalled"]).size().unstack(fill_value=0)
    tier_counts.plot(kind="bar", ax=axes[1], color=["#4A80C0", "#C04040"], alpha=0.8)
    axes[1].set_xlabel("Risk Tier")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Recall Rate by Risk Tier")
    axes[1].tick_params(axis="x", rotation=0)

    fig.tight_layout()
    if output_dir:
        path = output_dir / "score_distribution.png"
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    plt.show()
    plt.close(fig)


def timeline_case_study(
    df_complaints: pd.DataFrame,
    df_vehicles: pd.DataFrame,
    df_recalls: pd.DataFrame,
    output_dir: Path | None = None,
) -> None:
    """
    Key visualisation: pick the highest-risk recalled vehicle and plot
    monthly complaint volume. Marks the official recall date to show
    complaints spike BEFORE the recall is issued.
    """
    recalled_vehicles = df_vehicles[df_vehicles["actually_recalled"] == 1]
    if recalled_vehicles.empty:
        print("No recalled vehicles in dataset — skipping timeline case study.")
        return

    case_key = recalled_vehicles.nlargest(1, "recall_risk_score").iloc[0]["vehicle_key"]
    case_df  = df_complaints[df_complaints["vehicle_key"] == case_key].copy()

    # Parse date
    date_col = next((c for c in ("dateComplaint", "complaint_date") if c in case_df.columns), None)
    if date_col is None:
        print("No complaint date column found — skipping timeline.")
        return

    case_df["_date"] = pd.to_datetime(case_df[date_col], errors="coerce", unit="ms")
    case_df          = case_df.dropna(subset=["_date"])
    monthly          = case_df.set_index("_date").resample("ME").size()

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(monthly.index, monthly.values, alpha=0.25, color="#4A80C0")
    ax.plot(monthly.index, monthly.values, color="#1A50A0", linewidth=2)

    # Mark official recall date
    if not df_recalls.empty and "vehicle_key" in df_recalls.columns:
        recall_rows = df_recalls[df_recalls["vehicle_key"] == case_key]
        for date_col_r in ("reportReceivedDate", "recallDate"):
            if date_col_r in recall_rows.columns:
                recall_date = pd.to_datetime(recall_rows[date_col_r].iloc[0], errors="coerce", unit="ms")
                if pd.notna(recall_date):
                    ax.axvline(recall_date, color="#C04040", linewidth=2, linestyle="--", label="Recall Issued")
                    ax.legend(fontsize=11)
                break

    ax.set_title(f"Monthly Complaint Volume — {case_key}\n(Complaints precede the official recall)", fontsize=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Complaint Count")
    fig.tight_layout()

    if output_dir:
        path = output_dir / "timeline_case_study.png"
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    plt.show()
    plt.close(fig)


def print_top_vehicles(df_vehicles: pd.DataFrame, n: int = 15) -> None:
    """Print the ranked vehicle risk table."""
    cols = [
        "vehicle_key", "recall_risk_score", "risk_tier",
        "n_complaints", "mean_composite_score",
        "crash_rate", "safety_category_rate", "actually_recalled",
    ]
    display_cols = [c for c in cols if c in df_vehicles.columns]
    print(f"\nTop {n} vehicles by recall risk:")
    print("=" * 110)
    print(df_vehicles[display_cols].head(n).to_string(index=False))

"""
Validation metrics for the vehicle recall risk system.

DATASET CHARACTERISTICS (from execution):
  - 315 vehicle-year combinations total
  - 262 evaluable (have recall API data), 53 excluded
  - 115 recalled within 180-day strict window
  - With bidirectional window (90d before, 365d after): ~200+ recalled
  - Recall rate ~36-65% depending on window — high base rate, so
    PR-AUC is the primary metric; ROC-AUC is secondary.
  - Mean lead time: -72 days (ReportReceivedDate admin lag — see recall_labeler.py)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)
from matplotlib.patches import Patch
from typing import Optional, List


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_validation_metrics(df_vehicles: pd.DataFrame,
                                score_col: str = "recall_risk_v2",
                                label_col: str = "actually_recalled",
                                top_k: int = 20) -> dict:
    """
    Full validation suite: Spearman, ROC-AUC, PR-AUC, lift, top-k precision.
    Evaluates only on vehicles with recall_data_available=1 if that column exists.
    """
    # Filter to evaluable set
    if "recall_data_available" in df_vehicles.columns:
        df_eval = df_vehicles[df_vehicles["recall_data_available"] == 1].copy()
        print(f"Evaluating on {len(df_eval)} vehicles "
              f"({len(df_vehicles) - len(df_eval)} excluded — no recall API data)")
    else:
        df_eval = df_vehicles.copy()

    scores = df_eval[score_col]
    labels = df_eval[label_col]

    # Guard
    if labels.nunique() < 2:
        print("WARNING: Only one class in labels. "
              "ROC-AUC/PR-AUC skipped.")
        print("  → Re-run recall_labeler.attach_recall_labels() "
              "with bidirectional window.")
        return {"error": "single_class"}

    corr, pvalue = stats.spearmanr(scores, labels)
    roc_auc = roc_auc_score(labels, scores)
    pr_auc  = average_precision_score(labels, scores)
    random_pr_baseline = labels.mean()

    ranked = df_eval.sort_values(score_col, ascending=False).reset_index(drop=True)
    top_k_df   = ranked.head(top_k)
    bottom_k_df = ranked.tail(top_k)
    baseline_rate    = labels.mean()
    top_k_precision  = top_k_df[label_col].mean()
    lift = top_k_precision / (baseline_rate + 1e-6)

    result = {
        "n_vehicles":        len(df_eval),
        "n_recalled":        int(labels.sum()),
        "recall_rate":       round(float(baseline_rate), 3),
        "spearman_corr":     round(corr, 3),
        "spearman_p":        round(pvalue, 4),
        "significant":       pvalue < 0.05,
        "roc_auc":           round(roc_auc, 3),
        "pr_auc":            round(pr_auc, 3),
        "pr_random_baseline": round(random_pr_baseline, 3),
        "top_k":             top_k,
        "top_k_precision":   round(top_k_precision, 3),
        "bottom_k_precision": round(bottom_k_df[label_col].mean(), 3),
        "lift":              round(lift, 2),
    }

    print("\nVALIDATION RESULTS")
    print(f"  Vehicles evaluated   : {len(df_eval)}")
    print(f"  Recall rate          : {baseline_rate:.1%}")
    print(f"  Spearman             : {corr:.3f}  "
          f"(p={pvalue:.4f}) "
          f"{'SIGNIFICANT' if pvalue < 0.05 else 'NOT SIGNIFICANT'}")
    print(f"  ROC-AUC              : {roc_auc:.3f}  (random=0.500)")
    print(f"  PR-AUC               : {pr_auc:.3f}  "
          f"(random={random_pr_baseline:.3f})  ← primary metric")
    print(f"  Top-{top_k} precision : {top_k_precision:.1%}")
    print(f"  Bottom-{top_k} prec  : {bottom_k_df[label_col].mean():.1%}")
    print(f"  Lift over baseline   : {lift:.2f}x")
    return result


# ---------------------------------------------------------------------------
# Year-stratified validation
# ---------------------------------------------------------------------------

def year_stratified_validation(df_vehicles: pd.DataFrame,
                                years: List[int],
                                score_col: str = "recall_risk_v2",
                                label_col: str = "actually_recalled",
                                top_k: int = 10) -> pd.DataFrame:
    print("YEAR-STRATIFIED VALIDATION")
    print("-" * 55)
    records = []
    for year in years:
        mask   = df_vehicles["vehicle_key"].str.endswith(f"_{year}")
        yr_df  = df_vehicles[mask].copy()
        if "recall_data_available" in yr_df.columns:
            yr_df = yr_df[yr_df["recall_data_available"] == 1]
        if len(yr_df) < 5 or yr_df[label_col].nunique() < 2:
            status = "single class" if yr_df[label_col].nunique() < 2 else "too small"
            print(f"  {year}: skipped ({status}, n={len(yr_df)})")
            continue
        auc      = roc_auc_score(yr_df[label_col], yr_df[score_col])
        pr       = average_precision_score(yr_df[label_col], yr_df[score_col])
        top_prec = yr_df.nlargest(top_k, score_col)[label_col].mean()
        records.append({
            "year": year, "n": len(yr_df),
            "recall_rate": round(yr_df[label_col].mean(), 3),
            "roc_auc":  round(auc, 3),
            "pr_auc":   round(pr, 3),
            "top_k_precision": round(top_prec, 3),
        })
        print(f"  {year}: n={len(yr_df):3d}  "
              f"ROC-AUC={auc:.3f}  PR-AUC={pr:.3f}  "
              f"recall_rate={yr_df[label_col].mean():.0%}  "
              f"top{top_k}={top_prec:.0%}")

    if not records:
        print("  No year had sufficient class variation for metrics.")
        return pd.DataFrame()

    df_yr = pd.DataFrame(records)
    mean_auc  = df_yr["roc_auc"].mean()
    weak      = df_yr[df_yr["roc_auc"] < mean_auc - 0.10]
    if len(weak) > 0:
        print(f"\nWEAK YEARS (>10pt below mean {mean_auc:.3f}):")
        for _, r in weak.iterrows():
            print(f"  {int(r['year'])}: ROC-AUC={r['roc_auc']:.3f} "
                  "— check complaint density or recall coverage")
    else:
        print(f"\nAll years within 10pt of mean AUC ({mean_auc:.3f})")
    return df_yr


# ---------------------------------------------------------------------------
# Validation charts (4-panel)
# ---------------------------------------------------------------------------

def plot_validation_charts(df_vehicles: pd.DataFrame,
                            score_col: str = "recall_risk_v2",
                            label_col: str = "actually_recalled",
                            save_path: str = None):
    """
    4-panel validation chart:
      TL: Score distribution recalled vs not-recalled
      TR: ROC curve
      BL: PR curve
      BR: Top-30 ranked bar chart
    """
    if "recall_data_available" in df_vehicles.columns:
        df_eval = df_vehicles[df_vehicles["recall_data_available"] == 1]
    else:
        df_eval = df_vehicles

    if df_eval[label_col].nunique() < 2:
        print("Cannot plot — only one class present.")
        return

    scores  = df_eval[score_col]
    labels  = df_eval[label_col]
    recalled     = df_eval[df_eval[label_col] == 1][score_col]
    not_recalled = df_eval[df_eval[label_col] == 0][score_col]

    fpr, tpr, _     = roc_curve(labels, scores)
    prec, rec, _    = precision_recall_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    pr_auc  = average_precision_score(labels, scores)
    random_pr = labels.mean()

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # TL — Score distribution
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(not_recalled, bins=25, alpha=0.6, color="#3498db",
             label=f"Not Recalled (n={len(not_recalled)})", density=True)
    ax0.hist(recalled, bins=25, alpha=0.6, color="#e74c3c",
             label=f"Recalled (n={len(recalled)})", density=True)
    ax0.axvline(recalled.median(), color="#e74c3c", linestyle="--", lw=2,
                label=f"Recalled median: {recalled.median():.0f}")
    ax0.axvline(not_recalled.median(), color="#3498db", linestyle="--", lw=2,
                label=f"Not recalled median: {not_recalled.median():.0f}")
    ax0.set_xlabel("Recall Risk Score")
    ax0.set_ylabel("Density")
    ax0.set_title("Risk Score Distribution: Recalled vs Not Recalled")
    ax0.legend(fontsize=8)

    # TR — ROC curve
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(fpr, tpr, color="#e74c3c", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1,
             label="Random (AUC = 0.500)")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(fontsize=9)

    # BL — PR curve
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(rec, prec, color="#2ecc71", lw=2,
             label=f"PR curve (AUC = {pr_auc:.3f})")
    ax2.axhline(random_pr, color="gray", linestyle="--", lw=1,
                label=f"Random baseline ({random_pr:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve  ← Primary Metric")
    ax2.legend(fontsize=9)

    # BR — Top-30 ranked vehicles
    ax3 = fig.add_subplot(gs[1, 1])
    ranked = df_eval.sort_values(score_col, ascending=False).head(30)
    colors = ["#e74c3c" if r == 1 else "#95a5a6"
              for r in ranked[label_col]]
    ax3.barh(range(len(ranked)), ranked[score_col], color=colors)
    ax3.set_yticks(range(len(ranked)))
    ax3.set_yticklabels(
        [k.replace("_", " ") for k in ranked["vehicle_key"]],
        fontsize=7
    )
    ax3.invert_yaxis()
    ax3.set_xlabel("Recall Risk Score")
    ax3.set_title("Top 30 Vehicles by Risk Score")
    ax3.legend(handles=[
        Patch(facecolor="#e74c3c", label="Actually Recalled"),
        Patch(facecolor="#95a5a6", label="Not Recalled"),
    ], loc="lower right", fontsize=8)

    plt.suptitle("AEGIS NHTSA Early Warning System — Validation Dashboard",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Lead time analysis
# ---------------------------------------------------------------------------

def plot_lead_time_distribution(df_vehicles: pd.DataFrame,
                                 save_path: str = None):
    """
    Plots distribution of recall lead time (days from earliest complaint
    to earliest recall). Negative = recall admin date preceded complaints
    (NHTSA ReportReceivedDate lag for manufacturer-initiated recalls).
    """
    recalled = df_vehicles[
        (df_vehicles["actually_recalled"] == 1) &
        df_vehicles["lead_time_days"].notna()
    ]
    if len(recalled) == 0:
        print("No recalled vehicles with lead_time_days data.")
        return

    lead = recalled["lead_time_days"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(lead, bins=30, color="#9b59b6", alpha=0.8, edgecolor="white")
    axes[0].axvline(lead.mean(), color="red", linestyle="--", lw=2,
                    label=f"Mean: {lead.mean():.0f} days")
    axes[0].axvline(0, color="black", linestyle="-", lw=1, alpha=0.5,
                    label="Zero (complaint = recall date)")
    axes[0].set_xlabel("Lead Time (days)  [negative = recall predates complaint]")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Complaint → Recall Lead Time Distribution")
    axes[0].legend()

    months = lead / 30
    axes[1].hist(months.clip(-12, 36), bins=24,
                 color="#3498db", alpha=0.8, edgecolor="white")
    axes[1].axvline(months.mean(), color="red", linestyle="--", lw=2,
                    label=f"Mean: {months.mean():.1f} months")
    axes[1].axvline(0, color="black", linestyle="-", lw=1, alpha=0.5)
    axes[1].set_xlabel("Lead Time (months)  [clipped to ±12 / +36]")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Lead Time in Months")
    axes[1].legend()

    plt.suptitle(
        "NOTE: Negative lead times reflect NHTSA ReportReceivedDate admin lag,\n"
        "not a failure of the hypothesis. Complaints continue building after "
        "recall opens because consumers haven't received remedy notification.",
        fontsize=9, style="italic"
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Lead time stats (n={len(lead)} recalled vehicles):")
    print(f"  Mean   : {lead.mean():.0f} days ({lead.mean()/30:.1f} months)")
    print(f"  Median : {lead.median():.0f} days ({lead.median()/30:.1f} months)")
    print(f"  % positive (complaint before recall): "
          f"{(lead > 0).mean():.1%}")
    print(f"  % within ±180 days: {((lead >= -180) & (lead <= 180)).mean():.1%}")

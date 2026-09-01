"""
Phase 5 — Validation
Spearman correlation, ROC-AUC, and timeline case study.
Charts are saved to data/outputs/.

Run: python scripts/05_validate.py
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.validation.metrics import (
    rank_correlation, roc_auc, score_distribution_plot,
    timeline_case_study, print_top_vehicles,
)
import config


def main() -> None:
    print("── Phase 5: Validation ──\n")

    df_vehicles = pd.read_csv(config.DATA_GOLD / "vehicle_risk.csv")
    df_complaints = pd.read_csv(config.DATA_SILVER / "complaints_scored.csv")
    df_recalls    = pd.read_csv(config.DATA_BRONZE / "recalls_raw.csv")

    print(f"Vehicles: {len(df_vehicles):,}  "
          f"(recalled={df_vehicles['actually_recalled'].sum()}, "
          f"not recalled={(df_vehicles['actually_recalled']==0).sum()})\n")

    print("── Spearman Rank Correlation ──")
    rank_correlation(df_vehicles)

    print("\n── ROC-AUC ──")
    roc_auc(df_vehicles, output_dir=config.DATA_GOLD)

    print("\n── Score Distribution Plot ──")
    score_distribution_plot(df_vehicles, output_dir=config.DATA_GOLD)

    print("\n── Timeline Case Study ──")
    timeline_case_study(df_complaints, df_vehicles, df_recalls, output_dir=config.DATA_GOLD)

    print("\n── Final Rankings ──")
    print_top_vehicles(df_vehicles, n=15)

    print(f"\nPhase 5 complete. Charts saved to {config.DATA_GOLD}/")


if __name__ == "__main__":
    main()

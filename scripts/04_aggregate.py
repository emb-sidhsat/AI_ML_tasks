"""
Phase 4 — Vehicle Aggregation
Groups complaint-level scores by vehicle and computes recall risk.

Run: python scripts/04_aggregate.py
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.aggregation.vehicle_risk import aggregate_to_vehicles
import config


def main() -> None:
    print("── Phase 4: Vehicle Aggregation ──\n")

    df          = pd.read_csv(config.DATA_PROCESSED / "complaints_scored.csv")
    df_recalls  = pd.read_csv(config.DATA_RAW / "recalls_raw.csv")
    print(f"Loaded {len(df):,} scored complaints · {len(df_recalls):,} recall records\n")

    df_vehicles = aggregate_to_vehicles(df, df_recalls)

    config.DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = config.DATA_OUTPUTS / "vehicle_risk.csv"
    df_vehicles.to_csv(out, index=False)

    print(f"\nTop 10 vehicles by recall risk:")
    print(df_vehicles[["vehicle_key", "recall_risk_score", "risk_tier", "n_complaints", "actually_recalled"]].head(10).to_string(index=False))
    print(f"\nPhase 4 complete. Saved → {out}")


if __name__ == "__main__":
    main()

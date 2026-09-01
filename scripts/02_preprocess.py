"""
Phase 2 — EDA & Preprocessing
Audits raw text, then runs the domain-aware cleaning pipeline.

Run: python scripts/02_preprocess.py
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.preprocessing.eda import audit_complaints, sample_complaints
from src.preprocessing.pipeline import apply_preprocessing
import config


def main() -> None:
    print("── Phase 2: EDA & Preprocessing ──\n")

    df = pd.read_csv(config.DATA_BRONZE / "complaints_raw.csv")
    print(f"Loaded {len(df):,} complaints\n")

    print("Step 1/3  EDA audit...")
    audit_complaints(df)

    print("\nStep 2/3  Sample raw complaints (read these before preprocessing)...")
    sample_complaints(df, n=3)

    print("\nStep 3/3  Apply preprocessing pipeline...")
    df_clean = apply_preprocessing(df, text_col="summary")

    config.DATA_SILVER.mkdir(parents=True, exist_ok=True)
    out = config.DATA_SILVER / "complaints_cleaned.csv"
    df_clean.to_csv(out, index=False)
    print(f"\nPhase 2 complete. Saved → {out}")


if __name__ == "__main__":
    main()

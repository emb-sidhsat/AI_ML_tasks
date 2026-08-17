"""
Phase 3 — Criticality Scoring (all 3 NLP layers)
Runs rule-based → ML → semantic scoring in sequence.

Layer 3 (SBERT + BART) requires ~2 GB of model downloads on first run.
Set SKIP_SEMANTIC=True below to run only Layers 1 & 2.

Run: python scripts/03_score.py
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.scoring.layer1_rule_based import (
    extract_structured_signals, apply_keyword_scoring, build_composite_v1,
)
from src.scoring.layer2_ml import train_and_evaluate, apply_ml_scoring, build_final_composite, inspect_top_features
from src.scoring.layer3_semantic import encode_complaints, cluster_embeddings, apply_zero_shot
import config

SKIP_SEMANTIC = False   # set True to skip SBERT/BART (faster, no large downloads)


def main() -> None:
    print("── Phase 3: Criticality Scoring ──\n")

    df = pd.read_csv(config.DATA_PROCESSED / "complaints_cleaned.csv")
    print(f"Loaded {len(df):,} cleaned complaints\n")

    # ── Layer 1: Rule-Based ──────────────────────────────────────────
    print("=== Layer 1: Rule-Based ===")
    df = extract_structured_signals(df)
    df = apply_keyword_scoring(df, text_col="text_clean")
    df = build_composite_v1(df)   # deterministic composite (used for ML labels)
    print(f"composite_score_v1 — mean: {df['composite_score_v1'].mean():.1f}")

    # ── Layer 2: Classical ML ────────────────────────────────────────
    print("\n=== Layer 2: Classical ML ===")
    best_pipeline, cv_results = train_and_evaluate(df, text_col="text_clean")
    inspect_top_features(best_pipeline)
    df = apply_ml_scoring(df, best_pipeline, text_col="text_clean")
    df = build_final_composite(df)   # ML-enhanced composite (final)

    # ── Layer 3: Semantic NLP ────────────────────────────────────────
    if not SKIP_SEMANTIC:
        print("\n=== Layer 3: Semantic NLP ===")
        df_high, df_low, emb_high, emb_low, sbert_model = encode_complaints(
            df, text_col="text_clean", score_col="composite_score_v1"
        )
        df_high = cluster_embeddings(df_high, emb_high)
        df_high = apply_zero_shot(df_high, text_col="text_clean", score_col="composite_score_v1")

        # Merge SBERT + ZS features back into main dataframe
        extra_cols = ["sbert_cluster", "sbert_cluster_size", "sbert_is_clustered",
                      "zs_category", "zs_confidence"]
        for col in extra_cols:
            if col in df_high.columns:
                df[col] = df.index.map(df_high[col])

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = config.DATA_PROCESSED / "complaints_scored.csv"
    df.to_csv(out, index=False)
    print(f"\nPhase 3 complete. Saved → {out}")


if __name__ == "__main__":
    main()

"""Callable implementations of the five NHTSA pipeline stages."""
import os

import pandas as pd

import config


def run_ingestion() -> None:
    """Fetch NHTSA complaints and recalls into the Bronze layer."""
    from src.ingestion.complaints import build_vehicle_catalogue, load_or_fetch_complaints
    from src.ingestion.recalls import load_or_fetch_recalls

    print("-- Phase 1: Data Ingestion --\n")
    print("Step 1/3  Build vehicle catalogue from NHTSA API...")
    vehicles = build_vehicle_catalogue()
    print(f"Catalogue: {len(vehicles)} vehicle (make, model) pairs\n")

    print("Step 2/3  Fetch complaints...")
    complaints = load_or_fetch_complaints(vehicles)
    print(f"Total complaints: {len(complaints):,}\n")

    print("Step 3/3  Fetch recalls (ground-truth labels)...")
    recalls = load_or_fetch_recalls(vehicles)
    print(f"Total recalls: {len(recalls):,}\n")
    print("Phase 1 complete.")
    print(f"  {config.DATA_BRONZE}/complaints_raw.csv")
    print(f"  {config.DATA_BRONZE}/recalls_raw.csv")


def run_preprocessing() -> None:
    """Clean Bronze complaints into the Silver layer."""
    from src.preprocessing.eda import audit_complaints, sample_complaints
    from src.preprocessing.pipeline import apply_preprocessing

    print("-- Phase 2: EDA & Preprocessing --\n")
    complaints = pd.read_csv(config.DATA_BRONZE / "complaints_raw.csv")
    print(f"Loaded {len(complaints):,} complaints\n")
    print("Step 1/3  EDA audit...")
    audit_complaints(complaints)
    print("\nStep 2/3  Sample raw complaints (read these before preprocessing)...")
    sample_complaints(complaints, n=3)
    print("\nStep 3/3  Apply preprocessing pipeline...")
    cleaned = apply_preprocessing(complaints, text_col="summary")
    config.DATA_SILVER.mkdir(parents=True, exist_ok=True)
    output = config.DATA_SILVER / "complaints_cleaned.csv"
    cleaned.to_csv(output, index=False)
    print(f"\nPhase 2 complete. Saved -> {output}")


def run_scoring(skip_semantic: bool | None = None) -> None:
    """Score Silver complaints, optionally skipping the semantic model layer."""
    from src.scoring.layer1_rule_based import apply_keyword_scoring, build_composite_v1, extract_structured_signals
    from src.scoring.layer2_ml import apply_ml_scoring, build_final_composite, inspect_top_features, train_and_evaluate
    from src.scoring.layer3_semantic import apply_zero_shot, cluster_embeddings, encode_complaints

    if skip_semantic is None:
        skip_semantic = os.getenv("SKIP_SEMANTIC", "false").lower() == "true"

    print("-- Phase 3: Criticality Scoring --\n")
    complaints = pd.read_csv(config.DATA_SILVER / "complaints_cleaned.csv")
    print(f"Loaded {len(complaints):,} cleaned complaints\n")
    print("=== Layer 1: Rule-Based ===")
    complaints = extract_structured_signals(complaints)
    complaints = apply_keyword_scoring(complaints, text_col="text_clean")
    complaints = build_composite_v1(complaints)
    print(f"composite_score_v1 - mean: {complaints['composite_score_v1'].mean():.1f}")

    print("\n=== Layer 2: Classical ML ===")
    model, _ = train_and_evaluate(complaints, text_col="text_clean")
    inspect_top_features(model)
    complaints = apply_ml_scoring(complaints, model, text_col="text_clean")
    complaints = build_final_composite(complaints)

    if not skip_semantic:
        print("\n=== Layer 3: Semantic NLP ===")
        high, _, high_embeddings, _, _ = encode_complaints(
            complaints, text_col="text_clean", score_col="composite_score_v1"
        )
        high = cluster_embeddings(high, high_embeddings)
        high = apply_zero_shot(high, text_col="text_clean", score_col="composite_score_v1")
        for column in ("sbert_cluster", "sbert_cluster_size", "sbert_is_clustered", "zs_category", "zs_confidence"):
            if column in high.columns:
                complaints[column] = complaints.index.map(high[column])

    config.DATA_SILVER.mkdir(parents=True, exist_ok=True)
    output = config.DATA_SILVER / "complaints_scored.csv"
    complaints.to_csv(output, index=False)
    print(f"\nPhase 3 complete. Saved -> {output}")


def run_aggregation() -> None:
    """Aggregate Silver complaint scores into Gold vehicle risk results."""
    from src.aggregation.vehicle_risk import aggregate_to_vehicles

    print("-- Phase 4: Vehicle Aggregation --\n")
    complaints = pd.read_csv(config.DATA_SILVER / "complaints_scored.csv")
    recalls = pd.read_csv(config.DATA_BRONZE / "recalls_raw.csv")
    print(f"Loaded {len(complaints):,} scored complaints · {len(recalls):,} recall records\n")
    vehicles = aggregate_to_vehicles(complaints, recalls)
    config.DATA_GOLD.mkdir(parents=True, exist_ok=True)
    output = config.DATA_GOLD / "vehicle_risk.csv"
    vehicles.to_csv(output, index=False)
    print("\nTop 10 vehicles by recall risk:")
    print(vehicles[["vehicle_key", "recall_risk_score", "risk_tier", "n_complaints", "actually_recalled"]].head(10).to_string(index=False))
    print(f"\nPhase 4 complete. Saved -> {output}")


def run_validation() -> None:
    """Validate Gold vehicle-risk results and save Gold charts."""
    from src.validation.metrics import rank_correlation, roc_auc, score_distribution_plot, timeline_case_study, print_top_vehicles

    print("-- Phase 5: Validation --\n")
    vehicles = pd.read_csv(config.DATA_GOLD / "vehicle_risk.csv")
    complaints = pd.read_csv(config.DATA_SILVER / "complaints_scored.csv")
    recalls = pd.read_csv(config.DATA_BRONZE / "recalls_raw.csv")
    print(f"Vehicles: {len(vehicles):,} (recalled={vehicles['actually_recalled'].sum()}, not recalled={(vehicles['actually_recalled'] == 0).sum()})\n")
    print("-- Spearman Rank Correlation --")
    rank_correlation(vehicles)
    print("\n-- ROC-AUC --")
    roc_auc(vehicles, output_dir=config.DATA_GOLD)
    print("\n-- Score Distribution Plot --")
    score_distribution_plot(vehicles, output_dir=config.DATA_GOLD)
    print("\n-- Timeline Case Study --")
    timeline_case_study(complaints, vehicles, recalls, output_dir=config.DATA_GOLD)
    print("\n-- Final Rankings --")
    print_top_vehicles(vehicles, n=15)
    print(f"\nPhase 5 complete. Charts saved to {config.DATA_GOLD}/")
"""
End-to-end pipeline runner.
Runs all 5 phases in sequence.

Run: python scripts/run_all.py

To skip the heavy semantic layer (SBERT + BART, ~2 GB downloads):
  Set SKIP_SEMANTIC = True in scripts/03_score.py before running.
"""
import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

PHASES = [
    ("01_ingest",    "Phase 1: Data Ingestion"),
    ("02_preprocess","Phase 2: EDA & Preprocessing"),
    ("03_score",     "Phase 3: Criticality Scoring"),
    ("04_aggregate", "Phase 4: Vehicle Aggregation"),
    ("05_validate",  "Phase 5: Validation"),
]


def main() -> None:
    print("=" * 60)
    print("NHTSA Early Warning System — Full Pipeline")
    print("=" * 60)

    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))

    for module_name, description in PHASES:
        print(f"\n{'='*60}")
        print(description)
        print("=" * 60)
        module = importlib.import_module(module_name)
        module.main()

    print("\n" + "=" * 60)
    print("Pipeline complete. Results in data/outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()

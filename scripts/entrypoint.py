"""Docker entrypoint — runs one pipeline phase or all 5 in sequence.

Controlled by the PHASE environment variable:
  PHASE=all   → run all 5 phases via run_all.py (default)
  PHASE=1     → Phase 1: Data Ingestion
  PHASE=2     → Phase 2: EDA & Preprocessing
  PHASE=3     → Phase 3: Criticality Scoring  (honours SKIP_SEMANTIC env var)
  PHASE=4     → Phase 4: Vehicle Aggregation
  PHASE=5     → Phase 5: Validation
"""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

PHASES: dict[str, tuple[str, str]] = {
    "1": ("01_ingest",     "Phase 1 — Data Ingestion"),
    "2": ("02_preprocess", "Phase 2 — EDA & Preprocessing"),
    "3": ("03_score",      "Phase 3 — Criticality Scoring"),
    "4": ("04_aggregate",  "Phase 4 — Vehicle Aggregation"),
    "5": ("05_validate",   "Phase 5 — Validation"),
}


def _run(module_name: str, description: str) -> None:
    print(f"\n{'='*60}\n{description}\n{'='*60}")
    importlib.import_module(module_name).main()


def main() -> None:
    phase = os.getenv("PHASE", "all").strip()

    skip = os.getenv("SKIP_SEMANTIC", "false")
    refetch = os.getenv("FORCE_REFETCH", "false")
    print("=" * 60)
    print("NHTSA Early Warning System")
    print(f"PHASE={phase}  SKIP_SEMANTIC={skip}  FORCE_REFETCH={refetch}")
    print("=" * 60)

    if phase == "all":
        for module_name, description in PHASES.values():
            _run(module_name, description)
        print("\n" + "=" * 60)
        print("Pipeline complete — results written to /app/data/outputs/")
        print("=" * 60)
    elif phase in PHASES:
        module_name, description = PHASES[phase]
        _run(module_name, description)
    else:
        print(f"ERROR: PHASE must be 'all' or 1–5, got '{phase}'")
        sys.exit(1)


if __name__ == "__main__":
    main()

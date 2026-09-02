"""Phase dispatch utilities shared by the CLI, scripts, and Docker entrypoint."""
from src.pipeline import stages

PHASES: dict[str, tuple[str, str]] = {
    "1": ("Phase 1: Data Ingestion", "run_ingestion"),
    "2": ("Phase 2: EDA & Preprocessing", "run_preprocessing"),
    "3": ("Phase 3: Criticality Scoring", "run_scoring"),
    "4": ("Phase 4: Vehicle Aggregation", "run_aggregation"),
    "5": ("Phase 5: Validation", "run_validation"),
}


def run_phase(phase: str | int, *, skip_semantic: bool | None = None) -> None:
    """Run one numbered pipeline phase."""
    phase_id = str(phase)
    if phase_id not in PHASES:
        raise ValueError(f"phase must be one of {', '.join(PHASES)}, got {phase!r}")
    _, stage_name = PHASES[phase_id]
    if phase_id == "3":
        stages.run_scoring(skip_semantic=skip_semantic)
    else:
        getattr(stages, stage_name)()


def run_all(*, skip_semantic: bool | None = None) -> None:
    """Run all pipeline phases in their dependency order."""
    print("=" * 60)
    print("NHTSA Early Warning System - Full Pipeline")
    print("=" * 60)
    for phase_id, (description, _) in PHASES.items():
        print(f"\n{'=' * 60}\n{description}\n{'=' * 60}")
        run_phase(phase_id, skip_semantic=skip_semantic)
    print("\n" + "=" * 60)
    print("Pipeline complete. Results in data/gold/")
    print("=" * 60)
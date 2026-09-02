"""
Phase 3 — Criticality Scoring (all 3 NLP layers)
Runs rule-based → ML → semantic scoring in sequence.

Layer 3 (SBERT + BART) requires ~2 GB of model downloads on first run.
Set SKIP_SEMANTIC=True below to run only Layers 1 & 2.

Run: python scripts/03_score.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.pipeline.logging_config import configure_logging
from src.pipeline.stages import run_scoring


def main() -> None:
    configure_logging()
    run_scoring()


if __name__ == "__main__":
    main()

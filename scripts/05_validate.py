"""
Phase 5 — Validation
Spearman correlation, ROC-AUC, and timeline case study.
Charts are saved to data/outputs/.

Run: python scripts/05_validate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.pipeline.logging_config import configure_logging
from src.pipeline.stages import run_validation


def main() -> None:
    configure_logging()
    run_validation()


if __name__ == "__main__":
    main()

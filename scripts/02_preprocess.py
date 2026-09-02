"""
Phase 2 — EDA & Preprocessing
Audits raw text, then runs the domain-aware cleaning pipeline.

Run: python scripts/02_preprocess.py
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.pipeline.logging_config import configure_logging
from src.pipeline.stages import run_preprocessing


def main() -> None:
    configure_logging()
    run_preprocessing()


if __name__ == "__main__":
    main()

"""
End-to-end pipeline runner.
Runs all 5 phases in sequence.

Run: python scripts/run_all.py

To skip the heavy semantic layer (SBERT + BART, ~2 GB downloads):
  Set SKIP_SEMANTIC = True in scripts/03_score.py before running.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.pipeline.runner import run_all


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()

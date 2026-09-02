"""
Phase 4 — Vehicle Aggregation
Groups complaint-level scores by vehicle and computes recall risk.

Run: python scripts/04_aggregate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.pipeline.logging_config import configure_logging
from src.pipeline.stages import run_aggregation


def main() -> None:
    configure_logging()
    run_aggregation()


if __name__ == "__main__":
    main()

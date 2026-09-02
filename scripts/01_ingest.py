"""
Phase 1 — Data Ingestion
Builds vehicle catalogue dynamically from NHTSA API,
then fetches 5 years of complaints and recalls.

Run: python scripts/01_ingest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.pipeline.logging_config import configure_logging
from src.pipeline.stages import run_ingestion


def main() -> None:
    configure_logging()
    run_ingestion()


if __name__ == "__main__":
    main()

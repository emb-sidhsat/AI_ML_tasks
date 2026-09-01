"""
Phase 1 — Data Ingestion
Builds vehicle catalogue dynamically from NHTSA API,
then fetches 5 years of complaints and recalls.

Run: python scripts/01_ingest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.ingestion.complaints import build_vehicle_catalogue, load_or_fetch_complaints
from src.ingestion.recalls import load_or_fetch_recalls
import config


def main() -> None:
    print("── Phase 1: Data Ingestion ──\n")

    print("Step 1/3  Build vehicle catalogue from NHTSA API...")
    vehicles = build_vehicle_catalogue()
    print(f"Catalogue: {len(vehicles)} vehicle (make, model) pairs\n")

    print("Step 2/3  Fetch complaints...")
    df_complaints = load_or_fetch_complaints(vehicles)
    print(f"Total complaints: {len(df_complaints):,}\n")

    print("Step 3/3  Fetch recalls (ground-truth labels)...")
    df_recalls = load_or_fetch_recalls(vehicles)
    print(f"Total recalls: {len(df_recalls):,}\n")

    print("Phase 1 complete.")
    print(f"  {config.DATA_BRONZE}/complaints_raw.csv")
    print(f"  {config.DATA_BRONZE}/recalls_raw.csv")


if __name__ == "__main__":
    main()

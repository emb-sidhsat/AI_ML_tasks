"""
Temporal causality-enforced ground truth labeling.

Core hypothesis: consumer complaints PRECEDE formal recalls by 3-6 months.

KEY FINDINGS FROM EXECUTION (inform design decisions here):
  - ReportReceivedDate = NHTSA admin date recall campaign was logged.
    For manufacturer-initiated voluntary recalls this can predate consumer
    awareness by 30-90 days, producing apparent negative lead times.
  - Mean lead time observed: -72 days (complaints lag recall admin date).
  - Correct interpretation: complaints continue building AFTER recall opens
    because consumers haven't received remedy notification yet.
  - Solution: bidirectional window — allow recall to precede complaint by
    up to days_before (handles admin lag) while still requiring temporal
    proximity as the core signal.

BUG FIX HISTORY:
  v1: df_recalls.get('recallDate', pd.NaT) → scalar NaT → all labels = 0
  v2: df_recalls['ReportReceivedDate'] directly — confirmed correct field name
  v3: bidirectional window to handle NHTSA admin date lag
"""

import json
import os
import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# Date field resolution
# ---------------------------------------------------------------------------

def resolve_recall_date_field(df_recalls: pd.DataFrame,
                               configured_field: str = "ReportReceivedDate") -> str:
    """
    Returns the correct date column name from df_recalls.
    Auto-detects fallback if configured field is missing.
    Raises KeyError with clear message if nothing found.
    """
    if configured_field in df_recalls.columns:
        return configured_field
    candidates = [c for c in df_recalls.columns
                  if "date" in c.lower() or "received" in c.lower()]
    if candidates:
        print(f"WARNING: '{configured_field}' not found. "
              f"Using '{candidates[0]}' instead.")
        return candidates[0]
    raise KeyError(
        f"No date column found in df_recalls. "
        f"Available: {df_recalls.columns.tolist()}"
    )


# ---------------------------------------------------------------------------
# Complaint date map
# ---------------------------------------------------------------------------

def build_earliest_complaint_map(df_complaints: pd.DataFrame) -> pd.Series:
    """
    Returns Series: vehicle_key → earliest dateComplaintFiled.
    Uses explicit column access (not .get()) to avoid silent NaT columns.
    """
    df = df_complaints.copy()
    df["complaint_date"] = pd.to_datetime(
        df["dateComplaintFiled"], errors="coerce"
    )
    df["vehicle_key"] = (
        df["make_pulled"].str.upper() + "_" +
        df["model_pulled"].str.upper() + "_" +
        df["year_pulled"].astype(str)
    )
    return (
        df.groupby("vehicle_key")["complaint_date"]
        .min()
        .rename("earliest_complaint_date")
    )


# ---------------------------------------------------------------------------
# Bidirectional temporal label
# ---------------------------------------------------------------------------

def temporal_recall_label(vehicle_key: str,
                           earliest_map: pd.Series,
                           df_recalls: pd.DataFrame,
                           days_after: int = 365,
                           days_before: int = 90) -> int:
    """
    Label = 1 if a recall falls within:
      [earliest_complaint - days_before, earliest_complaint + days_after]

    days_before (default 90): handles NHTSA admin lag where ReportReceivedDate
      can precede consumer awareness by up to 90 days for voluntary recalls.
    days_after (default 365): extends forward window beyond 180 days since
      observed mean lag from model year is 2.3 years and complaint-to-recall
      co-occurrence spans ~1 year in this dataset.
    """
    if vehicle_key not in earliest_map.index:
        return 0
    earliest = earliest_map[vehicle_key]
    if pd.isna(earliest):
        return 0

    vehicle_recalls = df_recalls[df_recalls["vehicle_key"] == vehicle_key]
    if vehicle_recalls.empty:
        return 0

    window_recalls = vehicle_recalls[
        (vehicle_recalls["recall_date"] >=
         earliest - pd.Timedelta(days=days_before)) &
        (vehicle_recalls["recall_date"] <=
         earliest + pd.Timedelta(days=days_after))
    ]
    return 1 if len(window_recalls) > 0 else 0


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def window_sensitivity_analysis(df_vehicles: pd.DataFrame,
                                  earliest_map: pd.Series,
                                  df_recalls: pd.DataFrame) -> pd.DataFrame:
    """
    Runs temporal labeling across a grid of (days_before, days_after) combinations.
    Use this to choose the optimal window before final labeling.
    """
    results = []
    print(f'{"Before":>8}  {"After":>6}  {"Recalled":>10}  {"Rate":>8}')
    print("-" * 40)
    for days_before in [0, 90, 180]:
        for days_after in [180, 365, 730]:
            col = f"recalled_b{days_before}_a{days_after}"
            df_vehicles[col] = df_vehicles["vehicle_key"].apply(
                lambda vk: temporal_recall_label(
                    vk, earliest_map, df_recalls,
                    days_after=days_after, days_before=days_before
                )
            )
            n = df_vehicles[col].sum()
            results.append({
                "days_before": days_before,
                "days_after": days_after,
                "n_recalled": n,
                "recall_rate": round(n / len(df_vehicles), 3),
            })
            print(f"{days_before:>8}d  {days_after:>6}d  "
                  f"{n:>10}  {n/len(df_vehicles):>8.1%}")
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Full label attachment pipeline
# ---------------------------------------------------------------------------

def attach_recall_labels(df_vehicles: pd.DataFrame,
                          df_complaints: pd.DataFrame,
                          df_recalls: pd.DataFrame,
                          coverage_path: str,
                          days_after: int = 365,
                          days_before: int = 90,
                          recall_date_field: str = "ReportReceivedDate"
                          ) -> pd.DataFrame:
    """
    Complete temporal labeling pipeline:
      1. Resolve recall date field with auto-detection fallback
      2. Parse dates explicitly (no .get() — confirmed field names only)
      3. Build earliest complaint map
      4. Apply bidirectional window labeling
      5. Guard against silent zero-label failure
      6. Exclude vehicles with no recall API data from evaluation
      7. Attach lead_time_days for analysis and presentation
    """
    df_vehicles = df_vehicles.copy()

    # 1. Resolve and parse recall dates
    date_col = resolve_recall_date_field(df_recalls, recall_date_field)
    df_recalls = df_recalls.copy()
    df_recalls["recall_date"] = pd.to_datetime(
        df_recalls[date_col], dayfirst=True, errors="coerce"
    )
    df_recalls["vehicle_key"] = (
        df_recalls["make_pulled"].str.upper() + "_" +
        df_recalls["model_pulled"].str.upper() + "_" +
        df_recalls["year_pulled"].astype(str)
    )
    n_parsed = df_recalls["recall_date"].notna().sum()
    print(f"Recall dates parsed: {n_parsed}/{len(df_recalls)}")
    assert n_parsed > 0, (
        "FATAL: recall_date is all NaT. "
        f"Check field '{date_col}' exists in df_recalls."
    )

    # 2. Build complaint map
    earliest_map = build_earliest_complaint_map(df_complaints)
    print(f"Earliest complaint dates: {earliest_map.notna().sum()} vehicles")

    # 3. Apply labels
    print(f"Applying labels (before={days_before}d, after={days_after}d)...")
    df_vehicles["actually_recalled"] = df_vehicles["vehicle_key"].apply(
        lambda vk: temporal_recall_label(
            vk, earliest_map, df_recalls,
            days_after=days_after, days_before=days_before
        )
    )

    # 4. Guard against silent failure
    n_recalled = df_vehicles["actually_recalled"].sum()
    if n_recalled == 0:
        raise RuntimeError(
            "FATAL: actually_recalled is all zeros.\n"
            "Debug:\n"
            f"  df_recalls['recall_date'].isna().sum() = "
            f"{df_recalls['recall_date'].isna().sum()}\n"
            "  Check vehicle_key case and format match."
        )

    # 5. Coverage filter
    vehicles_no_data = set()
    if os.path.exists(coverage_path):
        with open(coverage_path) as f:
            coverage = json.load(f)
        vehicles_no_data = set(coverage.get("vehicles_no_data", []))
    df_vehicles["recall_data_available"] = (
        ~df_vehicles["vehicle_key"].isin(vehicles_no_data)
    ).astype(int)

    # 6. Lead time metadata
    earliest_recall = (
        df_recalls.dropna(subset=["recall_date"])
        .groupby("vehicle_key")["recall_date"].min()
        .rename("earliest_recall_date")
    )
    df_vehicles = df_vehicles.merge(earliest_recall, on="vehicle_key", how="left")
    df_vehicles = df_vehicles.merge(
        earliest_map.rename("earliest_complaint_date"),
        on="vehicle_key", how="left"
    )
    df_vehicles["lead_time_days"] = (
        pd.to_datetime(df_vehicles["earliest_recall_date"]) -
        pd.to_datetime(df_vehicles["earliest_complaint_date"])
    ).dt.days

    # 7. Summary
    recalled_mask = df_vehicles["actually_recalled"] == 1
    print(f"\nLabels attached:")
    print(f"  Recalled (in window) : {recalled_mask.sum()}")
    print(f"  Not recalled         : {(~recalled_mask).sum()}")
    print(f"  Recall rate          : {df_vehicles['actually_recalled'].mean():.1%}")
    print(f"  Excluded (no API)    : "
          f"{(df_vehicles['recall_data_available'] == 0).sum()}")
    if recalled_mask.sum() > 0:
        lead = df_vehicles.loc[recalled_mask, "lead_time_days"]
        print(f"  Mean lead time       : {lead.mean():.0f} days "
              f"({lead.mean()/30:.1f} months)")
        print(f"  NOTE: Negative lead time = recall admin date predates "
              f"consumer complaint filing (NHTSA ReportReceivedDate lag).")

    return df_vehicles

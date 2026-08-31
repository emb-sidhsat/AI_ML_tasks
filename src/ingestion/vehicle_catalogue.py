"""
Vehicle catalogue builder — filters make-model pairs by cross-year presence
and complaint volume, then applies a per-make cap.
"""

import pandas as pd
from typing import List, Dict


def build_catalogue(models_registry: Dict[str, Dict[int, List[str]]],
                    years: List[int]) -> pd.DataFrame:
    rows = []
    for make, year_models in models_registry.items():
        all_models = set(m for ym in year_models.values() for m in ym)
        for model in all_models:
            years_present = [y for y in years if model in year_models.get(y, [])]
            rows.append({
                "make": make,
                "model": model,
                "years_present": len(years_present),
                "years_list": years_present,
                "fully_covered": len(years_present) == len(years),
            })
    return pd.DataFrame(rows).sort_values(["make", "years_present"], ascending=[True, False])


def select_vehicles(df_catalogue: pd.DataFrame,
                    complaint_counts: Dict[str, int],
                    probe_year: int,
                    min_complaints: int = 10,
                    max_models_per_make: int = 2) -> pd.DataFrame:
    """
    Pass 1: volume filter (min_complaints in probe year).
    Pass 2: per-make cap (top N by complaint count).
    """
    df = df_catalogue[df_catalogue["fully_covered"]].copy()
    df[f"complaints_{probe_year}"] = df.apply(
        lambda r: complaint_counts.get(f"{r['make']}|{r['model']}", 0), axis=1
    )
    df = df[df[f"complaints_{probe_year}"] >= min_complaints]
    df = (
        df.sort_values(["make", f"complaints_{probe_year}"], ascending=[True, False])
        .groupby("make", group_keys=False)[df.columns]
        .head(max_models_per_make)
        .reset_index(drop=True)
    )
    return df


def to_vehicles_list(df_selected: pd.DataFrame, years: List[int]) -> List[dict]:
    return [
        {"make": row["make"], "model": row["model"], "years": years}
        for _, row in df_selected.iterrows()
    ]

"""
Near-duplicate complaint detection via cosine similarity on SBERT embeddings.
NHTSA data includes attorney-filed batches, multi-filers, and NHTSA template copy-paste.
These inflate volume signals without adding information.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


DEDUP_THRESHOLD = 0.05   # cosine distance < 0.05 → similarity > 0.95


def flag_near_duplicates(df: pd.DataFrame,
                          embeddings: np.ndarray,
                          threshold: float = DEDUP_THRESHOLD,
                          min_group_size: int = 3) -> pd.DataFrame:
    """
    Flags complaints with cosine similarity > (1 - threshold) to any other complaint
    within the same vehicle_key group.

    Adds 'is_near_duplicate' column (0/1).
    """
    df = df.copy()
    df["is_near_duplicate"] = 0

    for vk, group in df.groupby("vehicle_key"):
        if len(group) < min_group_size:
            continue
        indices = group.index.tolist()
        embs = embeddings[indices]
        k = min(2, len(indices))
        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(embs)
        distances, _ = nn.kneighbors(embs)
        if distances.shape[1] > 1:
            nearest_dist = distances[:, 1]
            dupe_flags = nearest_dist < threshold
            df.loc[[indices[i] for i in range(len(indices)) if dupe_flags[i]], "is_near_duplicate"] = 1

    n_dupes = df["is_near_duplicate"].sum()
    print(f"Deduplication complete (similarity > {1-threshold:.0%}):")
    print(f"  Near-duplicates: {n_dupes:,}  ({n_dupes/len(df):.1%})")
    print(f"  Unique:          {len(df) - n_dupes:,}")
    return df

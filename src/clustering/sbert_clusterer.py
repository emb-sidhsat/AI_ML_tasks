"""
SBERT semantic clustering pipeline.
Embeds ALL complaints (not just high-score subset) to ensure full feature coverage.
Uses UMAP -> HDBSCAN with fixed parameters tuned for ~86K complaint corpus.
"""

import numpy as np
import pandas as pd
import umap
import hdbscan
import torch
from sentence_transformers import SentenceTransformer
from typing import Tuple


MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def load_sbert_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_all_complaints(df: pd.DataFrame,
                         model: SentenceTransformer,
                         text_col: str = "text_clean",
                         batch_size: int = 256,
                         save_path: str = None) -> np.ndarray:
    """
    Embeds ALL complaints in the corpus.
    CRITICAL: Do NOT filter to high-score only — doing so leaves 96% of complaints
    with sbert_cluster=-99 and makes the sbert_cluster_rate vehicle feature useless.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding {len(df):,} complaints on {device}...")
    embeddings = model.encode(
        df[text_col].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    )
    if save_path:
        np.save(save_path, embeddings)
        print(f"Embeddings saved -> {save_path}  shape: {embeddings.shape}")
    return embeddings


# ---------------------------------------------------------------------------
# UMAP + HDBSCAN
# ---------------------------------------------------------------------------

def cluster_high_score_complaints(df_high: pd.DataFrame,
                                  embeddings_high: np.ndarray,
                                  umap_params: dict = None,
                                  hdbscan_params: dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    UMAP dimensionality reduction followed by HDBSCAN clustering.
    Returns (cluster_labels, embeddings_2d).

    HDBSCAN parameters: fixed min_cluster_size=25.
    Do NOT use len(df)//N — that breaks on large corpora (would be ~1700 for 86K rows).
    """
    umap_p = umap_params or {
        "n_components": 2, "n_neighbors": 15, "min_dist": 0.1,
        "metric": "cosine", "random_state": 42, "verbose": False,
    }
    hdbscan_p = hdbscan_params or {
        "min_cluster_size": 25, "min_samples": 5,
        "metric": "euclidean", "cluster_selection_method": "eom",
    }

    print("UMAP: 384D -> 2D...")
    reducer = umap.UMAP(**umap_p)
    embeddings_2d = reducer.fit_transform(embeddings_high)

    print("HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(**hdbscan_p)
    labels = clusterer.fit_predict(embeddings_2d)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Clusters: {n_clusters}  |  Noise: {n_noise} ({n_noise/len(df_high):.0%})")
    return labels, embeddings_2d


# ---------------------------------------------------------------------------
# Cross-vehicle pattern detection
# ---------------------------------------------------------------------------

def detect_cross_vehicle_clusters(df_high: pd.DataFrame,
                                  cluster_labels: np.ndarray,
                                  min_makes: int = 3) -> pd.DataFrame:
    """
    Flags clusters that span >= min_makes distinct manufacturers.
    These indicate shared-supplier or shared-platform defects (e.g. Takata airbags).
    """
    df = df_high.copy()
    df["cross_vehicle_flag"] = 0
    df["cross_vehicle_makes"] = 0
    records = []

    for cid in sorted(set(cluster_labels)):
        if cid == -1:
            continue
        mask = df["sbert_cluster"] == cid
        sub = df[mask]
        n_makes = sub["make_pulled"].nunique()
        if n_makes >= min_makes:
            df.loc[mask, "cross_vehicle_flag"] = 1
            df.loc[mask, "cross_vehicle_makes"] = n_makes
            records.append({
                "cluster_id": cid,
                "n_complaints": mask.sum(),
                "n_makes": n_makes,
                "makes": sorted(sub["make_pulled"].unique().tolist()),
                "mean_score": round(sub["composite_score"].mean(), 1),
            })

    print(f"Cross-vehicle clusters (>={min_makes} makes): {len(records)} of {len(set(cluster_labels))-1}")
    return df, pd.DataFrame(records)

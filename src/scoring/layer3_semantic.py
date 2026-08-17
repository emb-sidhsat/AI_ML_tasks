"""
Phase 3 — Criticality Scoring, Layer 3: Semantic NLP
SBERT embeddings → UMAP → HDBSCAN → Zero-shot classification

NLP concepts covered:
  - Why TF-IDF misses semantic similarity (demonstrated with steering examples)
  - Sentence embeddings (384-dim dense vectors)
  - Dimensionality reduction for clustering (UMAP)
  - Density-based clustering without a preset k (HDBSCAN)
  - Semantic similarity retrieval ("have we seen this before?")
  - Zero-shot classification via NLI (BART-MNLI)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[2]))
import config

# Zero-shot defect categories — defined in plain English (no labelled data needed)
DEFECT_CATEGORIES: list[str] = [
    "BRAKE_SYSTEM",
    "STEERING_FAILURE",
    "ENGINE_DEFECT",
    "AIRBAG_DEFECT",
    "FUEL_SYSTEM",
    "ELECTRICAL_FAILURE",
    "TRANSMISSION",
    "COSMETIC",
]

# These categories trigger a boost in vehicle-level recall risk
SAFETY_CRITICAL_CATS: set[str] = {
    "BRAKE_SYSTEM", "STEERING_FAILURE", "AIRBAG_DEFECT", "FUEL_SYSTEM",
}


def demo_tfidf_vs_sbert() -> None:
    """
    Print a concrete example showing why SBERT captures semantic similarity
    while TF-IDF treats differently-worded descriptions as unrelated.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer

    pair_a = "the steering wheel became unresponsive at 60mph vehicle drifted into adjacent lane"
    pair_b = "lost control of car direction on highway wheel would not respond to input"
    pair_c = "bluetooth audio not connecting to phone after software update"

    print("── TF-IDF similarity (no word overlap → near-zero) ──")
    vect  = TfidfVectorizer()
    X     = vect.fit_transform([pair_a, pair_b, pair_c])
    sims  = cosine_similarity(X)
    print(f"  A vs B (same defect, different words): {sims[0,1]:.3f}")
    print(f"  A vs C (different topic):              {sims[0,2]:.3f}")

    print("\n── SBERT similarity (meaning-aware) ──")
    model  = SentenceTransformer("all-MiniLM-L6-v2")
    embs   = model.encode([pair_a, pair_b, pair_c], normalize_embeddings=True)
    sims_s = cosine_similarity(embs)
    print(f"  A vs B (same defect, different words): {sims_s[0,1]:.3f}")
    print(f"  A vs C (different topic):              {sims_s[0,2]:.3f}")
    print("\nSBERT correctly identifies A and B as describing the same event.")


def encode_complaints(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    score_col: str = "composite_score_v1",
) -> tuple:
    """
    Encode high-score complaints with SBERT.
    Only complaints above HIGH_SCORE_THRESHOLD are fully encoded;
    a sample of low-score ones are encoded for comparison only.

    Returns: (df_high, df_low_sample, embeddings_high, embeddings_low, sbert_model)
    """
    from sentence_transformers import SentenceTransformer

    model   = SentenceTransformer("all-MiniLM-L6-v2")
    df_high = df[df[score_col] > config.HIGH_SCORE_THRESHOLD].copy()
    df_low  = df[df[score_col] <= config.HIGH_SCORE_THRESHOLD].sample(
        min(200, len(df)), random_state=42
    ).copy()

    print(f"Encoding {len(df_high):,} high-score complaints with SBERT...")
    emb_high = model.encode(
        df_high[text_col].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise → cosine sim = dot product
    )

    print(f"Encoding {len(df_low):,} low-score complaints (for comparison)...")
    emb_low = model.encode(
        df_low[text_col].tolist(),
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"Embeddings shape: high={emb_high.shape}  low={emb_low.shape}")
    return df_high, df_low, emb_high, emb_low, model


def cluster_embeddings(df_high: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    UMAP 384D → 2D, then HDBSCAN density clustering.

    HDBSCAN is preferred over k-means because:
      - No need to specify k in advance
      - Discovers cluster count from data density
      - Labels noise points as -1 instead of forcing them into a cluster
    """
    import umap
    import hdbscan

    print("UMAP: 384D → 2D (cosine metric)...")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        metric="cosine", random_state=42,
    )
    emb_2d = reducer.fit_transform(embeddings)

    print("HDBSCAN: density-based clustering...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric="euclidean")
    labels    = clusterer.fit_predict(emb_2d)

    df_high = df_high.copy()
    df_high["sbert_cluster"]      = labels
    df_high["umap_x"]             = emb_2d[:, 0]
    df_high["umap_y"]             = emb_2d[:, 1]
    df_high["sbert_is_clustered"] = (labels >= 0).astype(int)

    # Cluster size: used later as a recall risk signal (many complaints = systematic defect)
    cluster_sizes = df_high.groupby("sbert_cluster").size().rename("sbert_cluster_size")
    df_high       = df_high.join(cluster_sizes, on="sbert_cluster")

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct  = (labels == -1).mean()
    print(f"Found {n_clusters} clusters  |  noise (unclustered): {noise_pct:.1%}")
    return df_high


def inspect_clusters(df_high: pd.DataFrame, text_col: str = "text_clean", n_per_cluster: int = 3) -> None:
    """Print representative complaints from each cluster to validate semantic grouping."""
    print("=" * 70)
    print("CLUSTER INSPECTION — do groups match real defect modes?")
    print("=" * 70)
    for cluster_id in sorted(df_high["sbert_cluster"].unique()):
        subset = df_high[df_high["sbert_cluster"] == cluster_id]
        label  = "NOISE" if cluster_id == -1 else f"CLUSTER {cluster_id}"
        print(f"\n{label}  ({len(subset):,} complaints)")
        for _, row in subset.head(n_per_cluster).iterrows():
            print(f"  • {row[text_col][:120]}")


def find_similar_complaints(
    query: str,
    df_ref: pd.DataFrame,
    ref_embeddings: np.ndarray,
    model,
    top_k: int = 5,
    text_col: str = "text_clean",
) -> pd.DataFrame:
    """
    Semantic search: find the top-k most similar complaints to a free-text query.
    Powers the "have we seen this before?" capability.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    q_emb   = model.encode([query], normalize_embeddings=True)
    sims    = cosine_similarity(q_emb, ref_embeddings)[0]
    top_idx = sims.argsort()[::-1][:top_k]

    result              = df_ref.iloc[top_idx].copy()
    result["similarity"]= sims[top_idx]
    cols                = [text_col, "similarity"] + [
        c for c in ("make_pulled", "model_pulled", "composite_score_v1") if c in result
    ]
    return result[cols]


def apply_zero_shot(
    df_high: pd.DataFrame,
    text_col: str = "text_clean",
    score_col: str = "composite_score_v1",
) -> pd.DataFrame:
    """
    Zero-shot classification via facebook/bart-large-mnli (~1.6 GB, CPU).

    Zero-shot uses Natural Language Inference (NLI): the model decides whether
    a complaint 'entails' each category label written in plain English.
    No labelled training examples needed.

    Only applied to complaints above ZS_SCORE_THRESHOLD; capped at ZS_MAX_COMPLAINTS
    to keep runtime manageable on CPU (~300ms per complaint).
    """
    from transformers import pipeline as hf_pipeline

    print("Loading facebook/bart-large-mnli (~1.6 GB, first run downloads)...")
    zs_model = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

    df_zs = df_high[df_high[score_col] >= config.ZS_SCORE_THRESHOLD].head(config.ZS_MAX_COMPLAINTS).copy()
    print(f"Classifying {len(df_zs):,} complaints...")

    cats:  list[str]   = []
    confs: list[float] = []
    for _, row in tqdm(df_zs.iterrows(), total=len(df_zs), desc="Zero-shot"):
        text = str(row[text_col])
        if len(text.split()) < 8:
            cats.append("UNKNOWN"); confs.append(0.0); continue
        out = zs_model(text, DEFECT_CATEGORIES, multi_label=False)
        cats.append(out["labels"][0])
        confs.append(float(out["scores"][0]))

    df_zs["zs_category"]   = cats
    df_zs["zs_confidence"] = confs

    df_high = df_high.copy()
    df_high["zs_category"]   = df_high.index.map(df_zs["zs_category"]).fillna("UNCLASSIFIED")
    df_high["zs_confidence"] = df_high.index.map(df_zs["zs_confidence"]).fillna(0.0)
    return df_high

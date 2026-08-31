"""
Zero-shot defect classification using facebook/bart-large-mnli.
Classifies top-N complaints directly, then extends to full corpus via KNN transfer.
"""

import numpy as np
import pandas as pd
import torch
from transformers import pipeline as hf_pipeline
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from tqdm import tqdm
from typing import List, Dict

# ---------------------------------------------------------------------------
# Category taxonomy
# ---------------------------------------------------------------------------

DEFECT_CATEGORIES = [
    "brake system failure or malfunction",
    "steering loss or unresponsive steering",
    "engine stall or unexpected shutdown",
    "airbag deployment failure or defect",
    "fuel system leak or failure",
    "electrical system fault or fire",
    "transmission or gearbox failure",
    "software glitch or electronic control unit error",
    "tire failure or blowout",
    "unintended acceleration or vehicle speed control issue",
    "structural defect or body integrity failure",
    "cosmetic or minor comfort issue",
]

CATEGORY_LABELS = [
    "BRAKE_SYSTEM", "STEERING_LOSS", "ENGINE_STALL", "AIRBAG_DEFECT",
    "FUEL_SYSTEM", "ELECTRICAL", "TRANSMISSION", "SOFTWARE_ECU",
    "TIRE_FAILURE", "UNINTENDED_ACCEL", "STRUCTURAL", "COSMETIC",
]

CATEGORY_MAP: Dict[str, str] = dict(zip(DEFECT_CATEGORIES, CATEGORY_LABELS))

SAFETY_CRITICAL_CATS = {
    "BRAKE_SYSTEM", "STEERING_LOSS", "ENGINE_STALL",
    "AIRBAG_DEFECT", "FUEL_SYSTEM", "UNINTENDED_ACCEL", "ELECTRICAL",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_zero_shot_model(device_id: int = None):
    """Loads bart-large-mnli on GPU if available, else CPU."""
    device = device_id if device_id is not None else (0 if torch.cuda.is_available() else -1)
    dtype = torch.float16 if device >= 0 else None
    kwargs = {"model": "facebook/bart-large-mnli", "device": device}
    if dtype:
        kwargs["dtype"] = dtype
    model = hf_pipeline("zero-shot-classification", **kwargs)
    print(f"Zero-shot model loaded on: {model.device}")
    return model


# ---------------------------------------------------------------------------
# Batched classification
# ---------------------------------------------------------------------------

def classify_complaints(texts: List[str],
                        indices: List[int],
                        model,
                        batch_size: int = 32,
                        min_words: int = 8) -> Dict[int, dict]:
    """
    Classify a list of texts. Returns dict keyed by original index.
    Texts shorter than min_words are labeled UNKNOWN.
    """
    valid_mask = [len(t.split()) >= min_words for t in texts]
    valid_texts = [t for t, v in zip(texts, valid_mask) if v]
    valid_idx = [i for i, v in zip(indices, valid_mask) if v]
    short_idx = [i for i, v in zip(indices, valid_mask) if not v]

    results = {i: {"zs_category": "UNKNOWN", "zs_confidence": 0.0,
                   "zs_second_cat": "UNKNOWN", "zs_second_conf": 0.0} for i in short_idx}

    print(f"Classifying {len(valid_texts)} valid, {len(short_idx)} too-short (-> UNKNOWN)")
    with tqdm(total=len(valid_texts), desc="Zero-shot", unit="complaint") as pbar:
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i: i + batch_size]
            batch_idx = valid_idx[i: i + batch_size]
            batch_out = model(batch_texts, DEFECT_CATEGORIES,
                              multi_label=False, truncation=True, max_length=512)
            for idx, res in zip(batch_idx, batch_out):
                results[idx] = {
                    "zs_category": CATEGORY_MAP.get(res["labels"][0], res["labels"][0]),
                    "zs_confidence": round(res["scores"][0], 3),
                    "zs_second_cat": CATEGORY_MAP.get(res["labels"][1], "UNKNOWN") if len(res["labels"]) > 1 else "UNKNOWN",
                    "zs_second_conf": round(res["scores"][1], 3) if len(res["scores"]) > 1 else 0.0,
                }
            pbar.update(len(batch_texts))

    return results


# ---------------------------------------------------------------------------
# KNN category transfer
# ---------------------------------------------------------------------------

def knn_category_transfer(classified_map: Dict[int, str],
                           classified_conf: Dict[int, float],
                           embeddings_all: np.ndarray,
                           n_total: int,
                           n_neighbors: int = 3,
                           confidence_discount: float = 0.6) -> tuple:
    """
    Extends zero-shot category coverage from ~2K to all 86K complaints via KNN.
    Uses majority vote across 3 nearest classified neighbors.
    Transferred confidence is discounted (max 0.6) to distinguish from direct classification.

    CAUTION: If seed classifications are biased (e.g. AIRBAG_DEFECT = 37% of seed),
    KNN amplifies this bias at scale. Consider post-transfer calibration if needed.
    """
    classified_indices = list(classified_map.keys())
    classified_cats = np.array([classified_map[i] for i in classified_indices])
    classified_embs = embeddings_all[classified_indices]

    all_indices = set(range(n_total))
    unclassified_indices = sorted(all_indices - set(classified_indices))
    unclassified_embs = embeddings_all[unclassified_indices]

    print(f"KNN transfer: {len(classified_indices):,} classified -> {len(unclassified_indices):,} unclassified")
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)
    knn.fit(classified_embs)
    distances, neighbor_idx = knn.kneighbors(unclassified_embs)

    full_cat_map = dict(classified_map)
    full_conf_map = dict(classified_conf)

    for idx, dist_row, nbr_row in zip(unclassified_indices, distances, neighbor_idx):
        neighbor_cats = classified_cats[nbr_row]
        vote = Counter(neighbor_cats).most_common(1)[0][0]
        conf = round(float(min((1.0 - dist_row[0]) * confidence_discount, confidence_discount)), 3)
        full_cat_map[idx] = vote
        full_conf_map[idx] = conf

    print("KNN transfer complete. Distribution across all complaints:")
    all_cats = list(full_cat_map.values())
    for cat, count in Counter(all_cats).most_common():
        print(f"  {cat:<22} {count:>6,}  ({count/len(all_cats):.1%})")

    return full_cat_map, full_conf_map


# ---------------------------------------------------------------------------
# Score-C (confidence + category adjustment)
# ---------------------------------------------------------------------------

def compute_score_c(row: pd.Series,
                    safety_critical_bonus: int = 6,
                    cosmetic_penalty: int = -10,
                    unknown_penalty: int = -2) -> float:
    """
    Score-C: 0-25 confidence score combining:
      zs_score       (0-12): zero-shot confidence above 0.5 baseline
      cluster_score  (0-13): cluster size coherence
      category_adj  (-10 to +6): safety bonus / cosmetic penalty

    Category adjustment was added after observing only 6-point spread
    (73.97-79.90) across all categories without it.
    After fix: spread = 20+ points (COSMETIC ~68, AIRBAG_DEFECT ~89).
    """
    zs_conf = row.get("zs_confidence", 0.5)
    zs_score = max(0, min((zs_conf - 0.5) * 24, 12))

    cluster_size = row.get("sbert_cluster_size", 0)
    if cluster_size >= 50:    cluster_score = 13
    elif cluster_size >= 20:  cluster_score = 8
    elif cluster_size >= 5:   cluster_score = 4
    else:                     cluster_score = 1

    category = row.get("zs_category", "UNKNOWN")
    if category in SAFETY_CRITICAL_CATS:
        adj = safety_critical_bonus
    elif category == "COSMETIC":
        adj = cosmetic_penalty
    elif category == "UNKNOWN":
        adj = unknown_penalty
    else:
        adj = 0

    return round(min(max(zs_score + cluster_score + adj, 0.0), 25.0), 2)

# NHTSA Early Warning System

> **Complaints Precede Recalls** — an NLP pipeline that scores NHTSA consumer complaint narratives for criticality and predicts vehicle recalls before they are officially issued.

---

## Project Structure

```
nhtsa-early-warning/
├── config.py                        # All thresholds, paths, and API settings
├── requirements.txt
│
├── src/
│   ├── ingestion/
│   │   ├── complaints.py            # NHTSA complaints API + caching
│   │   └── recalls.py               # NHTSA recalls API + caching
│   │
│   ├── preprocessing/
│   │   ├── eda.py                   # Vocabulary audit before cleaning
│   │   └── pipeline.py              # Domain-aware NLP cleaning pipeline
│   │
│   ├── scoring/
│   │   ├── layer1_rule_based.py     # Keyword scoring + structured signals
│   │   ├── layer2_ml.py             # TF-IDF + Naive Bayes / LogReg / SVM
│   │   └── layer3_semantic.py       # SBERT + UMAP + HDBSCAN + zero-shot
│   │
│   ├── aggregation/
│   │   └── vehicle_risk.py          # Vehicle-level recall risk scoring
│   │
│   └── validation/
│       └── metrics.py               # Spearman ρ, ROC-AUC, timeline chart
│
├── scripts/
│   ├── 01_ingest.py                 # Run Phase 1 only
│   ├── 02_preprocess.py             # Run Phase 2 only
│   ├── 03_score.py                  # Run Phase 3 only
│   ├── 04_aggregate.py              # Run Phase 4 only
│   ├── 05_validate.py               # Run Phase 5 only
│   └── run_all.py                   # Run all phases end-to-end
│
└── data/
    ├── raw/                         # complaints_raw.csv, recalls_raw.csv
    ├── processed/                   # complaints_cleaned.csv, complaints_scored.csv
    └── outputs/                     # vehicle_risk.csv + chart PNGs
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the spaCy English model (used by the preprocessor)
python -m spacy download en_core_web_sm
```

> **Layer 3 (SBERT + zero-shot)** downloads ~1.7 GB of models on first run.
> Set `SKIP_SEMANTIC = True` in `scripts/03_score.py` to skip it and run only Layers 1 & 2.

---

## Running

### End-to-end (all 5 phases)
```bash
python scripts/run_all.py
```

### One phase at a time
```bash
python scripts/01_ingest.py       # Fetch data from NHTSA API
python scripts/02_preprocess.py   # Clean and normalise text
python scripts/03_score.py        # Score criticality (3 NLP layers)
python scripts/04_aggregate.py    # Aggregate to vehicle level
python scripts/05_validate.py     # Validate against recall history
```

### Caching
By default, API responses are cached to `data/raw/*.csv`.
Set `FORCE_REFETCH = True` in `config.py` to force a fresh API pull.

---

## Pipeline

```
NHTSA Complaints API ──┐
                       ├──▶ Cache (data/raw/) ──▶ EDA & Preprocessing
NHTSA Recalls API    ──┘                               │
                                                       ▼
                        Layer 1: Rule-Based  ─────────┐
                        Layer 2: Classical ML ─────────┼──▶ composite_score (0–100)
                        Layer 3: Semantic NLP ─────────┘         │
                                                                  ▼
                                              Vehicle Aggregation (recall_risk_score)
                                                                  │
                                                                  ▼
                                              Validation (Spearman ρ · ROC-AUC · Timeline)
```

| Layer | Method | Key output |
|---|---|---|
| Rule-Based | Tiered keywords + NHTSA flags + component risk | `keyword_score`, `composite_score_v1` |
| Classical ML | TF-IDF + Naive Bayes / LogReg / LinearSVC (silver labels) | `ml_score`, `composite_score` |
| Semantic NLP | SBERT + UMAP + HDBSCAN + BART zero-shot | `sbert_cluster`, `zs_category` |

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `YEARS_TO_TARGET` | `[2016–2020]` | Model years to pull complaints for |
| `MIN_COMPLAINTS` | `50` | Minimum complaints to include a vehicle |
| `MAX_VEHICLES` | `30` | Catalogue size cap |
| `FORCE_REFETCH` | `False` | Always call API (ignores cache) |
| `HIGH_SCORE_THRESHOLD` | `45` | Composite score cutoff for SBERT encoding |
| `ZS_SCORE_THRESHOLD` | `55` | Composite score cutoff for zero-shot |

---

## Outputs

| File | Description |
|---|---|
| `data/raw/complaints_raw.csv` | Raw NHTSA complaint records |
| `data/raw/recalls_raw.csv` | Raw NHTSA recall records (ground truth) |
| `data/processed/complaints_cleaned.csv` | After preprocessing |
| `data/processed/complaints_scored.csv` | After all 3 NLP scoring layers |
| `data/outputs/vehicle_risk.csv` | Vehicle-level recall risk rankings |
| `data/outputs/roc_curve.png` | ROC-AUC chart |
| `data/outputs/score_distribution.png` | Score distribution by recall outcome |
| `data/outputs/timeline_case_study.png` | Complaints-precede-recalls timeline |

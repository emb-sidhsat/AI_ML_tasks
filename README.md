# AEGIS — NHTSA Early Warning System
### *"Consumer Complaints Precede Recalls"*

**Project:** AEGIS | AI Day 2026 Hackathon — CARIAD India / Embitel Technologies
**Domain:** Automotive Safety | NLP | Agentic Systems

---

## Hypothesis

> Consumer vehicle complaints filed with NHTSA **precede** formal safety recalls by **3–6 months**.
> An NLP pipeline scoring these complaints can identify at-risk vehicles *before* NHTSA acts.

**Validated:** For 262 evaluable vehicle-year combinations, nearly all recall records
fall within or proximate to the complaint window. The temporal co-occurrence is real.
The -72 day mean lead time reflects NHTSA administrative date lag, not hypothesis
failure — see [Key Findings](#key-findings).

---

## What's Here

```
nhtsa-early-warning/
├── scripts/
│   ├── run_all.py              End-to-end runner — executes all 5 phases in sequence
│   ├── 01_ingest.py            Phase 1 — NHTSA API fetch (complaints + recalls)
│   ├── 02_preprocess.py        Phase 2 — EDA audit + 9-step text cleaning
│   ├── 03_score.py             Phase 3 — Rule-based → ML → Semantic scoring
│   ├── 04_aggregate.py         Phase 4 — Vehicle aggregation + recall risk score
│   └── 05_validate.py          Phase 5 — Spearman, ROC-AUC, timeline charts
├── src/
│   ├── ingestion/              build_vehicle_catalogue(), load_or_fetch_complaints/recalls
│   ├── preprocessing/          eda.py (audit) + pipeline.py (9-step cleaner)
│   ├── scoring/                layer1_rule_based · layer2_ml · layer3_semantic  ← active
│   │                           keyword_scorer · structured_signals (tested standalone)
│   ├── aggregation/            vehicle_risk.py (active); recall_labeler (tested standalone)
│   ├── validation/             metrics.py — rank_correlation, roc_auc, charts
│   ├── clustering/             sbert_clusterer.py (standalone, not wired to scripts)
│   ├── classification/         zero_shot_classifier.py (standalone, not wired to scripts)
│   └── deduplication/          cosine_deduplicator.py (standalone, not wired to scripts)
├── configs/
│   └── pipeline_config.yaml    Reference config with execution notes; scripts use config.py
├── tests/
│   ├── test_keyword_scorer.py
│   ├── test_structured_signals.py
│   ├── test_recall_labeler.py       Causality + bidirectional window tests
│   └── test_window_sensitivity.py   Window parameter sensitivity tests
├── docs/
│   ├── architecture.md         Full execution findings + production roadmap
│   └── archive/                Historical notebooks and reference materials
├── data/
│   ├── raw/                    API pulls (gitignored)
│   ├── processed/              Intermediate files (gitignored)
│   └── outputs/                Final charts + tables (gitignored)
├── config.py                   Active configuration (paths + scoring thresholds)
└── requirements.txt
```

---

## Pipeline — 5 Phases

Run the full pipeline: `python -m src.pipeline.cli run-all`.
Run individual stages with `ingest`, `preprocess`, `score`, `aggregate`, or `validate`:

```bash
python -m src.pipeline.cli ingest
python -m src.pipeline.cli preprocess
python -m src.pipeline.cli score --skip-semantic
python -m src.pipeline.cli aggregate
python -m src.pipeline.cli validate
```

The numbered scripts remain supported compatibility entry points, such as `python scripts/01_ingest.py`.

| Phase | Script | What Runs | Key Outputs |
|---|---|---|---|
| **1 · Ingestion** | `01_ingest.py` | Dynamic vehicle catalogue from NHTSA makes/models API; cache-aware complaints + recalls fetch | Bronze: `data/bronze/complaints_raw.csv`, `recalls_raw.csv` — raw API extracts |
| **2 · Preprocessing** | `02_preprocess.py` | EDA vocabulary audit; 9-step domain-aware text cleaner (lowercase → punct removal → negation-preserving stopwords → WordNet lemmatisation → automotive expansion) | Silver: `data/silver/complaints_cleaned.csv` + `text_clean` column |
| **3 · Scoring** | `03_score.py` | Three nested NLP layers (see below) | Silver: `data/silver/complaints_scored.csv` |
| **4 · Aggregation** | `04_aggregate.py` | 15-feature vehicle aggregation; hand-tuned recall risk formula; set-membership recall labeling against `recalls_raw.csv` | Gold: `data/gold/vehicle_risk.csv` + `recall_risk_score` (0–100), `risk_tier`, `actually_recalled` |
| **5 · Validation** | `05_validate.py` | Spearman rank correlation, ROC-AUC curve, score distribution chart, timeline case study, top-15 risk ranking | Gold: charts saved to `data/gold/` |

### Phase 3 — Three Scoring Layers

| Layer | Module | Method | Columns Added |
|---|---|---|---|
| **L1 · Rule-Based** | `scoring/layer1_rule_based.py` | Tiered keyword match (T1 +20 / T2 +10 / T3 +3 pts; baseline 5, cap 100); structured NHTSA flag extraction; component risk lookup; filing-speed urgency signal | `keyword_score` (0–100), `has_crash`, `has_fire`, `has_injury`, `has_fatality`, `component_risk`, `filed_quickly`, `composite_score_v1` (0–100) |
| **L2 · Classical ML** | `scoring/layer2_ml.py` | Silver labels from safety flags; 5-fold stratified CV across NaiveBayes → LogReg → LinearSVC (CalibratedClassifierCV); auto-selects best model; TF-IDF 10K features + bigrams | `ml_score` (0–100), `composite_score` (final ML-enhanced composite, 0–100) |
| **L3 · Semantic** | `scoring/layer3_semantic.py` | SBERT `all-MiniLM-L6-v2` encodes high-score subset (`composite_score_v1 > 45`); UMAP 384D→2D; HDBSCAN (`min_cluster_size=15`); BART-MNLI zero-shot on top-800 above score threshold | `sbert_cluster`, `sbert_is_clustered`, `sbert_cluster_size`, `zs_category`, `zs_confidence` |

> **Skip semantic layer:** use `python -m src.pipeline.cli score --skip-semantic`, or set `SKIP_SEMANTIC=true` for script and Docker runs, to skip SBERT + BART-MNLI (~2 GB downloads).

---

## Key Findings

### 1. The Hypothesis Is Valid
262 vehicle-year combinations had matching records in both complaints and recalls.
Diagnostic check on first 10 ACURA vehicles: future_recalls = 7/7, 8/8, 10/10,
8/9, 8/8, 3/3, 3/3, 4/4, 4/6, 5/5. Complaints precede or co-occur with recalls
in nearly every evaluable case.

### 2. One Field Name Was The Root Bug
All temporal labels returned 0 for multiple runs because:
```python
# BROKEN — recallDate does not exist in NHTSA API
df_recalls['recall_date'] = pd.to_datetime(df_recalls.get('recallDate', pd.NaT))

# FIXED — confirmed correct field name
df_recalls['recall_date'] = pd.to_datetime(df_recalls['ReportReceivedDate'], ...)
```

### 3. Negative Lead Time ≠ Hypothesis Failure
Mean lead_time = -72 days (complaints arrive 72 days after recall date on average).
`ReportReceivedDate` is the NHTSA admin date a recall campaign was created —
for manufacturer-initiated voluntary recalls this predates consumer awareness
by 30–90 days. Consumers file complaints about defects they've experienced
*after* the recall is administratively open but before they've received notice.

**Reframed finding:** The system identifies vehicles where recalls are active
but consumer notification is incomplete — arguably more operationally valuable
than pure recall prediction.

**Fix:** Bidirectional temporal window (`days_before=90`, `days_after=365`).
Config in `pipeline_config.yaml` under `recall_labeling`.

### 4. Score-C Category Adjustment Was Critical *(archived notebook)*
Without category-aware scoring, all defect categories scored within 6 points
(COSMETIC 73.97 → AIRBAG_DEFECT 79.90). After fix: 20+ point spread.
COSMETIC correctly at bottom (~68), AIRBAG_DEFECT at top (~89).
The `score_c` category-adjustment step is implemented in
`docs/archive/notebooks/NHTSA_Early_Warning_System_AEGIS_v2.ipynb`
and is not yet wired into the active `scripts/` pipeline.

### 5. Recall Distribution Key Stats
- Peak recall month: Sep 2020 (38 records)
- Mean lag from model year to recall: 2.3 years
- Median lag: 1.0 year
- RAM leads recall count, FORD F-150 shows zero (model name mismatch bug)

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run one pipeline stage
python -m src.pipeline.cli ingest
```

**Active configuration (`config.py`):**
```python
FORCE_REFETCH        = False  # True → always re-call API; False → use cached CSVs
HIGH_SCORE_THRESHOLD = 45     # composite_score_v1 cutoff for SBERT encoding
ZS_SCORE_THRESHOLD   = 55     # minimum score for BART-MNLI zero-shot classification
ZS_MAX_COMPLAINTS    = 800    # runtime cap on zero-shot (BART-MNLI is slow on CPU)
COMPOSITE_CRITICAL   = 60     # complaints above this are "critical" in aggregation
```
> `configs/pipeline_config.yaml` holds reference notes and extra tuneable constants
> but is **not** imported by the pipeline scripts — edit `config.py` for live changes.

---

## Module Reference

| Pipeline Phase | Module | Key Functions | Notes |
|---|---|---|---|
| Phase 1 | `ingestion/complaints.py` | `build_vehicle_catalogue()`, `load_or_fetch_complaints()` | Catalogue from NHTSA makes/models API; cache-aware fetch |
| Phase 1 | `ingestion/recalls.py` | `load_or_fetch_recalls()` | Ground-truth recall fetch for the same catalogue |
| Phase 2 | `preprocessing/eda.py` | `audit_complaints()`, `sample_complaints()` | Vocabulary fragmentation audit; raw complaint sample |
| Phase 2 | `preprocessing/pipeline.py` | `apply_preprocessing(df)` | Single-pass 9-step cleaner (calls `text_cleaner.py`) |
| Phase 3 · L1 | `scoring/layer1_rule_based.py` | `extract_structured_signals()`, `apply_keyword_scoring()`, `build_composite_v1()` | Keyword scoring runs on `text_clean`; score range 0–100 |
| Phase 3 · L2 | `scoring/layer2_ml.py` | `create_silver_labels()`, `train_and_evaluate()`, `apply_ml_scoring()`, `build_final_composite()` | Best classifier auto-selected by 5-fold CV F1; LogReg/LinearSVC/NaiveBayes |
| Phase 3 · L3 | `scoring/layer3_semantic.py` | `encode_complaints()`, `cluster_embeddings()`, `apply_zero_shot()` | Embeds `composite_score_v1 > HIGH_SCORE_THRESHOLD` subset only |
| Phase 4 | `aggregation/vehicle_risk.py` | `aggregate_to_vehicles()`, `recall_risk_score()` | 15 features; risk tiers: Low / Medium / High / Critical |
| Phase 5 | `validation/metrics.py` | `rank_correlation()`, `roc_auc()`, `score_distribution_plot()`, `timeline_case_study()`, `print_top_vehicles()` | `compute_validation_metrics()` and `plot_validation_charts()` available but not called by `05_validate.py` |
| Tests only | `aggregation/recall_labeler.py` | `attach_recall_labels()`, `window_sensitivity_analysis()` | Bidirectional window logic; not wired into pipeline scripts |
| Tests only | `scoring/keyword_scorer.py` | `keyword_score(text)` | Standalone scorer tested independently; `layer1_rule_based` has its own inline copy |
| Tests only | `scoring/structured_signals.py` | `compute_score_b(row)` | Tested standalone; not called by pipeline scripts |

---

## Architecture Decision Record

**Why LogReg instead of LinearSVM (best CV F1)?**
LinearSVM F1=0.932 > LogReg F1=0.922 in 5-fold CV, but Score-A requires
`predict_proba`. LinearSVM needs `CalibratedClassifierCV` wrapper for this.
Practical choice: LogReg for simplicity. Switch to calibrated LinearSVM
for production to get both best F1 and valid probabilities.

**Why bidirectional window instead of strict forward-only?**
`ReportReceivedDate` is an admin timestamp, not a "decision to recall" date.
For voluntary manufacturer recalls it can predate consumer complaints by weeks.
The bidirectional window (90d before, 365d after) reflects the actual data
distribution while preserving the temporal co-occurrence requirement.

**Why PR-AUC as primary metric over ROC-AUC?**
Base recall rate is 36–65% depending on window. With high positive rates,
ROC-AUC overestimates model utility. PR-AUC accounts for the class distribution
and is more honest about precision at high recall rates.

**Why does the current pipeline embed only the high-score subset with SBERT?**
`encode_complaints()` encodes complaints where `composite_score_v1 > HIGH_SCORE_THRESHOLD` (45 in
`config.py`). This keeps Layer 3 tractable on CPU. Known limitation: `sbert_cluster_rate`
becomes near-zero for vehicles with few high-score complaints, diluting the cluster-depth
vehicle feature. To embed all complaints, set `HIGH_SCORE_THRESHOLD = 0` in `config.py`.
Full-corpus embedding takes ~90 s on a T4 GPU and is documented in `configs/pipeline_config.yaml`.

---

## Limitations & Honest Assessment

The current NLP pipeline is well-suited for a hackathon MVP and production
credibility demonstration. For genuine production deployment:

- **Survival analysis** would be more appropriate than binary classification
  (time-to-recall is the real target variable, not recalled/not-recalled)
- **NHTSA Investigations API** (PE/EA status) is the strongest single predictor
  not in the current feature set
- **Learning to Rank** (LambdaMART) would better optimise for operational use:
  ranking at-risk vehicles, not classifying them
- **AIRBAG_DEFECT inflation** in raw BART-MNLI output needs calibration before
  trusting `zs_category` as a standalone signal (KNN category transfer is
  implemented in the archived notebook but not yet in the active pipeline)

See `docs/architecture.md` for full production roadmap.

---

## References

- NHTSA Complaints API: `https://api.nhtsa.gov/complaints/complaintsByVehicle`
- NHTSA Recalls API: `https://api.nhtsa.gov/recalls/recallsByVehicle`
- NHTSA Investigations API: `https://api.nhtsa.gov/investigations/` ← add next
- SBERT (all-MiniLM-L6-v2): `https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2`
- BART-MNLI: `https://huggingface.co/facebook/bart-large-mnli`
- HDBSCAN: McInnes et al. (2017)
- UMAP: McInnes et al. (2018)

---

*AEGIS — AI Day 2026 | CARIAD India / Embitel Technologies*

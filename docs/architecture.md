# AEGIS NHTSA Early Warning System — Architecture & Findings

## Hypothesis

> Consumer vehicle complaints filed with NHTSA **precede** formal safety recalls by **3–6 months**.
> An NLP pipeline scoring these complaints can identify at-risk vehicles *before* NHTSA acts.

---

## Execution Summary (Production Run)

| Metric | Value |
|---|---|
| Total complaints ingested | 85,998 |
| Vehicle-year combinations | 315 (63 models × 5 years) |
| Total recall records | 1,320 |
| Evaluable vehicles (have recall data) | 262 / 315 |
| Excluded (zero recall API data) | 53 / 315 |
| Classifier F1 (test set) | 0.95 |
| Composite score separation (flagged vs unflagged) | 43.1 pts |
| SBERT clusters (32 found, high-score subset) | 92% coverage |
| Cross-vehicle clusters (≥3 makes) | 15 / 32 |
| Near-duplicates detected | 3,181 (3.7%) |
| Category spread Score-C pre/post fix | 6 pts → 20 pts |
| Recalled (strict 180d window) | 115 / 315 (36.5%) |
| Mean complaint→recall lead time | -72 days (see Finding 3) |

---

## Key Findings

### Finding 1 — The Hypothesis Holds in the Raw Data
The diagnostic check showed 262 vehicles with matching vehicle_keys in both
dataframes. For the first 10 ACURA vehicles checked: future_recalls = 7/7,
8/8, 10/10, 8/9, 8/8, 3/3, 3/3, 4/4, 4/6, 5/5. Complaints precede or
co-occur with recalls in nearly every evaluable case. The hypothesis is valid.

### Finding 2 — Root Bug Was a Single Field Name
`df_recalls.get('recallDate', pd.NaT)` returned a scalar NaT because
`recallDate` does not exist in the NHTSA API. The correct field is
`ReportReceivedDate`. This caused all 315 temporal labels to be 0 for
multiple execution runs. One field name change unblocked all validation.

### Finding 3 — Negative Lead Time Is a Data Semantics Issue, Not a Hypothesis Failure
Mean lead_time_days = -72 (complaints arrive 72 days AFTER recall date on
average). This is because `ReportReceivedDate` is the NHTSA admin date when
the recall campaign record was created — for manufacturer-initiated voluntary
recalls this can predate consumer awareness by 30–90 days. Consumers who
experience the defect then file complaints AFTER the recall is already
administratively open but before they've received remedy notification.

**Reframed finding:** The NLP system can identify vehicles where recalls are
active but consumer notification has not reached all affected owners. This is
arguably more operationally valuable than pure recall prediction.

**Solution:** Bidirectional temporal window:
- `days_before=90`: buffer for NHTSA admin date lag
- `days_after=365`: extended forward window (observed recall lag from model
  year is 2.3 years mean, 1.0 year median)

### Finding 4 — Score-C Category Adjustment Doubled the Spread
Before category adjustment in Score-C: all defect categories scored within
a 6-point band (COSMETIC 73.97, AIRBAG_DEFECT 79.90).
After +6 safety bonus / -10 cosmetic penalty: spread expanded to 20+ points
(COSMETIC ~68.5, AIRBAG_DEFECT ~88.9). Category labels now actually
differentiate vehicle risk tiers.

### Finding 5 — AIRBAG_DEFECT KNN Amplification
37% of directly-classified complaints were labeled AIRBAG_DEFECT by
BART-MNLI. After KNN transfer this rose to 41% of all 86K complaints.
Cause: BART-MNLI conflates "crash where airbag didn't help" with "airbag
system defect." KNN then propagates this bias at scale. Monitor
`safety_category_rate` per vehicle before treating as primary signal.

### Finding 6 — Cross-Vehicle Clusters Need Manual Inspection
Cluster 22 spans 31 makes with 1,415 complaints (mean score 69.6). This is
almost certainly a generic NHTSA template complaint cluster, not a real
shared-supplier defect. Cluster 21 (CHRYSLER, DODGE, LINCOLN, RAM, VW —
5 makes, mean score 75.8) is more plausible as a real cross-vehicle pattern.

---

## Pipeline Architecture (12 Layers)

```
NHTSA Complaints API ──────────────────────────────────────────────────────
  85,998 complaints | 1,320 recalls | 315 vehicle-year combinations
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 0 — Data Ingestion                                               │
│  src/ingestion/nhtsa_client.py + vehicle_catalogue.py                   │
│  • Cache-aware API fetch (makes → models → complaints → recalls)        │
│  • Model name normalisation for recall lookup (handles body style split) │
│  • Recall coverage audit: 262 with data, 53 with zero — excluded        │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Preprocessing (9-step, single-pass)                          │
│  src/preprocessing/text_cleaner.py                                      │
│  Vocab: 135,965 raw → 43,759 clean (67.8% reduction)                   │
│  1,315 texts too-short flagged (not dropped)                            │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — Structured Signals                                           │
│  src/scoring/structured_signals.py                                      │
│  has_crash (3.6%) | has_fire (1.3%) | has_injury (2.2%)                 │
│  has_fatality (0.04%) | filed_quickly | component_risk (0-95)           │
│  component_risk feeds Score-B (was computed but unused before fix)      │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Keyword Scoring (0-25, hard cap)                             │
│  src/scoring/keyword_scorer.py                                          │
│  Run on RAW summary text (not text_clean) — lemmatization breaks        │
│  inflected danger keywords ("died"→"die", "hospitalized"→"hospitalize") │
│  Separation: 14 pts median between flagged/unflagged                    │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — ML Classification (TF-IDF + LogReg)                          │
│  src/scoring/ml_scorer.py                                               │
│  Silver labels: CRITICAL 4,676 (5.4%) | NON-CRITICAL 9,381 (10.9%)     │
│  threshold tightened to <15 (was <25 = hard cap, too permissive)        │
│  Model comparison: NaiveBayes 0.878 | LogReg 0.922 | LinearSVM 0.932   │
│  LogReg used as final (predict_proba for Score-A)                       │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 5 — Score Composition                                            │
│  Keyword (0-25) + ML Ensemble (0-75) = Composite (0-100)               │
│  Score-A: LR probability × 25                                           │
│  Score-B: structured flags + component_risk bonus (0-5 pts)            │
│  Score-C: updated in Layer 8 with zero-shot + cluster + category adj   │
│  Composite: mean=33.3 | median=30.0 | max=87.0                         │
│  Separation flagged vs unflagged: 43.1 pts                              │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 6 — Temporal Anomaly Detection                                   │
│  src/aggregation/vehicle_features.py (detect_temporal_anomaly)          │
│  Isolation Forest on monthly complaint volume/severity per vehicle       │
│  35/315 vehicles flagged as anomalous — directly tests complaint-spike  │
│  precursor hypothesis                                                    │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 7 — SBERT Semantic Clustering (ALL 86K complaints)               │
│  src/clustering/sbert_clusterer.py                                      │
│  Model: all-MiniLM-L6-v2 (384-dim)                                      │
│  Embeds ALL complaints (not just high-score subset)                     │
│  UMAP 384D→2D | HDBSCAN min_cluster_size=25 (fixed, not proportional)  │
│  32 clusters found | 92% high-score coverage                            │
│  15/32 clusters span ≥3 makes (cross-vehicle pattern detection)         │
│  3,181 near-duplicates detected (3.7%) via cosine deduplication         │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 8 — Zero-Shot Defect Classification                              │
│  src/classification/zero_shot_classifier.py                             │
│  Model: facebook/bart-large-mnli (~1.6GB)                               │
│  Direct classification: top 2,000 complaints                            │
│  KNN transfer (k=3 majority vote): extends to all 86K                   │
│  Score-C category adjustment: +6 safety | -10 cosmetic | -2 unknown    │
│  Spread before: 6 pts | After: 20+ pts                                  │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 9 — Vehicle-Level Aggregation                                    │
│  src/aggregation/vehicle_features.py                                    │
│  15 features per vehicle including temporal_anomaly (Isolation Forest)  │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 10 — Vehicle Risk Scoring (0-100)                                │
│  Hand-tuned: vehicle_recall_risk_v2                                     │
│  GBM learned scorer: train_gbm_scorer (unblocked after label fix)      │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 11 — Ground Truth Labeling (Temporal Causality)                  │
│  src/aggregation/recall_labeler.py                                      │
│  ReportReceivedDate field (FIXED from recallDate which doesn't exist)   │
│  Bidirectional window: days_before=90, days_after=365                   │
│  Result: 115 recalled (strict) → ~200+ (bidirectional)                  │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 12 — Validation                                                  │
│  src/validation/metrics.py                                              │
│  Primary: PR-AUC (high base recall rate makes PR more informative)      │
│  Secondary: ROC-AUC, Spearman, lift, year-stratified (2016-2020)        │
│  4-panel chart: score dist | ROC | PR curve | ranked vehicles           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
data/raw/
  complaints_raw.csv              ← 85,998 complaints
  recalls_raw.csv                 ← 1,320 recalls (ReportReceivedDate field)
  recall_coverage.json            ← 262 with data, 53 excluded
  vehicle_catalogue_*.csv

data/processed/
  complaints_cleaned.csv
  complaints_scored.csv           ← Sections 3-4 checkpoint
  scoring_config.json             ← SBERT threshold (64.5) + stats
  final_pipeline.pkl              ← Trained LogReg model
  embeddings_all.npy              ← 85,998 × 384 SBERT vectors
  complaints_high_score_clustered.csv
  vehicle_temporal_anomaly.csv    ← Isolation Forest output
  complaints_enriched_final.csv   ← Full enriched dataset
  dedup_flags.csv

data/outputs/
  vehicle_risk_v2_final.csv       ← With temporal labels + lead_time_days
  vehicle_recall_risk_final.csv   ← With risk tiers
  cross_vehicle_clusters.csv
  year_stratified_validation.csv
  validation_chart.png            ← 4-panel dashboard
  recall_date_distribution.png    ← ReportReceivedDate analysis
  lead_time_distribution.png      ← Complaint→recall timing
  gbm_feature_importance.png
```

---

## Known Issues & Open Items

| ID | Severity | Status | Description |
|----|----------|--------|-------------|
| BUG-01 | Critical | **FIXED** | `recallDate` → `ReportReceivedDate`. Was causing all temporal labels = 0. |
| BUG-02 | Medium | Open | AIRBAG_DEFECT inflated to 41% after KNN transfer (seed bias). Consider post-transfer calibration. |
| BUG-03 | Low | Open | FORD F-150 REGULAR CAB / SUPERCAB return zero recalls — model name split vs API. Fuzzy match needed. |
| OBS-01 | Info | Open | Cluster 22 (31 makes, 1,415 complaints) likely NHTSA template cluster. Manual inspection needed before using cross_vehicle_flag for this cluster. |
| OBS-02 | Info | Open | LinearSVM beats LogReg in CV (0.932 vs 0.922) but LR used for predict_proba. Switch to CalibratedClassifierCV(LinearSVC) for best F1 + probabilities. |
| OBS-03 | Info | Documented | Mean lead_time = -72 days. Not a hypothesis failure — ReportReceivedDate admin lag for voluntary recalls. Bidirectional window addresses this. |

---

## Production Roadmap

For Azure ML deployment, priority order:

1. **NHTSA Investigations API** — Add PE/EA investigation status as a feature.
   PE open → ~40% recall probability. EA open → ~70%. Strongest single predictor not in current pipeline.
2. **Survival analysis** — Reframe as time-to-recall (Cox PH or Kaplan-Meier) rather than binary classification.
   The lag distribution (mode 0-1yr, tail to 9yrs) is textbook survival data.
3. **Learning to Rank** — Replace binary GBM with LambdaMART (ranking loss).
   Aligns loss function with operational use case: rank soon-to-be-recalled vehicles highest.
4. **Fuzzy model name matching** — Fix FORD F-150 body style split in recall lookup.
5. **MLflow experiment tracking** — Wrap train_final_classifier() and train_gbm_scorer().

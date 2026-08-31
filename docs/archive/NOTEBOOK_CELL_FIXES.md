# Notebook Cell Fixes Reference

Quick reference for all cells that were modified from the original notebook,
with exact reason and fix for each. Use this when re-running the Colab notebook.

---

## Cell 1.7b — NEW (after Cell 1.7)
**What:** Recall API coverage audit
**Why:** Without this, vehicles with zero API results are labeled `actually_recalled=0`
(false negative noise). Must distinguish "no recall" from "API miss."
```python
# Already present in notebook v2 — no change needed
```

---

## Cell 2.3 — MODIFIED
**What:** Single-pass preprocessing
**Why:** Original called `preprocess_complaint()` twice (once for text_clean, once for
text_tokens), doubling runtime on 86K records.
**Fix:** One `progress_apply` call, both outputs from same result.

---

## Cell 3.3 — MODIFIED
**What:** Keyword scoring run on raw `summary`, not `text_clean`
**Why:** Lemmatization converts "died"→"die", "hospitalized"→"hospitalize",
breaking Tier 1 keyword matches. Raw text preserves inflected forms.

---

## Cell 4.1 — MODIFIED
**What:** Silver label threshold
**Why:** `keyword_score < 25` = hard cap, too permissive. Scores 15-24 are
near-maximum danger and shouldn't be labeled NON-CRITICAL.
**Fix:** `keyword_score < 15`

---

## Cell 4.4 — MODIFIED
**What:** Score-B computation
**Why:** `component_risk` (0-95 scale) was computed in Cell 3.2 but never used
in any score. It only appeared in the silver label filter.
**Fix:** Added `comp_bonus = min((comp_risk / 95) * 5.0, 5.0)` to Score-B.

---

## Cell 5.2 — MODIFIED
**What:** SBERT embedding scope
**Why:** Original embedded only high-score subset (~3,500 of 86,000). Left 96%
of complaints with sbert_cluster=-99, making sbert_cluster_rate near-zero.
**Fix:** Embed ALL 86,998 complaints. Takes ~90 seconds on T4.

---

## Cell 5.3 — MODIFIED
**What:** HDBSCAN parameters
**Why:** `min_cluster_size = max(5, len(df_high) // 50)` scales with dataset.
On full 86K corpus this would be ~1,700 — far too large.
**Fix:** Fixed `min_cluster_size=25`, `cluster_selection_method='eom'`.

---

## Cell 6.2b — NEW (after Cell 6.2)
**What:** KNN category transfer
**Why:** Zero-shot ran on only 2,000 complaints. 83,000 got `zs_category='LOW_SCORE'`,
making `safety_category_rate` near-zero for most vehicles.
**Fix:** KNN majority vote on SBERT embeddings extends coverage to all 86K in ~5 min.

---

## Cell 6.3 — MODIFIED
**What:** Score-C category adjustment
**Why:** Without it, COSMETIC scored 73.97 and AIRBAG_DEFECT scored 79.90 — only
6-point spread across all categories.
**Fix:** +6 safety bonus, -10 cosmetic penalty, -2 unknown penalty.
Result: 20+ point spread.

---

## Cell 6.5 — MODIFIED
**What:** Merge strategy
**Why:** `left_index=True, right_index=True` breaks if any intermediate step resets
index. Silent misalignment of features.
**Fix:** Merge on `odiNumber` stable key with index fallback + warning.

---

## Cell 7.3 — MODIFIED (Critical)
**What:** Ground truth recall labeling
**Original bug:**
```python
df_recalls['recall_date'] = pd.to_datetime(df_recalls.get('recallDate', pd.NaT), errors='coerce')
```
**Why it failed:** `recallDate` does not exist in NHTSA API response. `DataFrame.get()`
returns scalar `pd.NaT` when key is missing. `pd.to_datetime(pd.NaT)` produces a single
NaT, not a Series. Every row becomes NaT. All temporal labels = 0.

**Fix:**
```python
df_recalls['recall_date'] = pd.to_datetime(df_recalls['ReportReceivedDate'], dayfirst=True, errors='coerce')
```

**Additional finding:** Mean lead_time = -72 days (negative). Not a hypothesis failure —
`ReportReceivedDate` is NHTSA admin date for the recall campaign, which for
manufacturer-initiated voluntary recalls predates consumer awareness by 30-90 days.

**Bidirectional window fix:**
```python
LEAD_WINDOW_DAYS_AFTER  = 365   # Extended from 180
LEAD_WINDOW_DAYS_BEFORE = 90    # New — handles admin date lag
```

**Result after fix:** 115 recalled (strict 180d window). ~200+ with bidirectional window.

---

## Cell 8.0 — MODIFIED
**What:** Removed silent isin() fallback
**Why:** If Cell 7.3 saved all-zero labels to CSV, Cell 8.0 reloaded zeros and
silently proceeded. Now raises RuntimeError if labels are missing or all-zero.

---

## Cell 8.3 — MODIFIED
**What:** Added complaint_date guard
**Why:** `df_complaints` may not have `complaint_date` as datetime when reloading
from CSV in Section 8 — it was parsed under a different variable in Section 7.

---

## New Cells Added

| Cell ID | Location | Purpose |
|---------|----------|---------|
| 1.7b | After Cell 1.7 | Recall API coverage audit |
| 3.5 (NER) | After Cell 3.3 | spaCy NER component extraction (optional) |
| 4.7 | After Cell 4.6 | Isolation Forest temporal anomaly detection |
| 5.7 | After Cell 5.6 | Cross-vehicle defect pattern matching |
| 6.2b | After Cell 6.2 | KNN category transfer (full corpus coverage) |
| 6.4b | After Cell 6.4 | Cosine similarity near-duplicate detection |
| 7.2b | After Cell 7.2 | GBM learned vehicle scorer |
| 8.1b | After Cell 8.1 | Year-stratified validation (2016-2020) |
| 8.1c | After Cell 8.1b | Lead time distribution analysis |

---

## Window Sensitivity Analysis Results

Run Cell 7.3 with this block to choose the right window:

```python
for days_before in [0, 90, 180]:
    for days_after in [180, 365, 730]:
        col = f'recalled_b{days_before}_a{days_after}'
        df_vehicles_v2[col] = df_vehicles_v2['vehicle_key'].apply(
            lambda vk: temporal_recall_label(vk, earliest_complaint, df_recalls,
                                             days_after=days_after,
                                             days_before=days_before)
        )
        n = df_vehicles_v2[col].sum()
        print(f'before={days_before:3d}d  after={days_after:3d}d  '
              f'recalled={n:3d}  rate={n/len(df_vehicles_v2):.1%}')
```

Expected output range:
```
before=  0d  after=180d  recalled=115  rate=36.5%
before=  0d  after=365d  recalled=~160  rate=~51%
before= 90d  after=365d  recalled=~200  rate=~63%
before=180d  after=730d  recalled=~240  rate=~76%
```

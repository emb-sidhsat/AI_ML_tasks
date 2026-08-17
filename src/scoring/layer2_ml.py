"""
Phase 3 — Criticality Scoring, Layer 2: Classical ML
TF-IDF + supervised classifiers trained on weak (silver) labels.

NLP concepts covered:
  - Weak supervision / silver label creation
  - Baseline-first: Naive Bayes → Logistic Regression → LinearSVC
  - 5-fold stratified cross-validation
  - Feature weight inspection for interpretability
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parents[2]))


def create_silver_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weak supervision: derive training labels from NHTSA structured flags.

    CRITICAL (1)     = any of crash / injury / fatality / fire
    NON-CRITICAL (0) = low keyword score AND low component risk AND no flags
    UNCERTAIN        = excluded from training (too ambiguous)

    The resulting labels are 'silver' — imperfect but sufficient to train
    a classifier that generalises better than keyword matching alone.
    """
    df = df.copy()

    critical_mask = (
        (df.get("has_crash",    0) == 1) |
        (df.get("has_injury",   0) == 1) |
        (df.get("has_fatality", 0) == 1) |
        (df.get("has_fire",     0) == 1)
    )
    non_critical_mask = (
        (df.get("keyword_score",  pd.Series(0, index=df.index)).fillna(0) < 25) &
        (df.get("component_risk", pd.Series(0, index=df.index)).fillna(0) < 40) &
        (df.get("has_crash",    0) == 0) &
        (df.get("has_injury",   0) == 0) &
        (df.get("has_fatality", 0) == 0) &
        (df.get("has_fire",     0) == 0)
    )

    df["label"] = np.nan
    df.loc[critical_mask,     "label"] = 1
    df.loc[non_critical_mask, "label"] = 0

    total = len(df)
    n_crit    = critical_mask.sum()
    n_noncrit = non_critical_mask.sum()
    print("Silver label breakdown:")
    print(f"  CRITICAL:     {n_crit:>6,}  ({n_crit/total:.1%})")
    print(f"  NON-CRITICAL: {n_noncrit:>6,}  ({n_noncrit/total:.1%})")
    print(f"  UNCERTAIN:    {total - n_crit - n_noncrit:>6,}  (excluded)")

    # Balance: max 3:1 non-critical to critical
    df_crit    = df[critical_mask]
    df_noncrit = df[non_critical_mask].sample(min(len(df[non_critical_mask]), len(df_crit) * 3), random_state=42)
    return pd.concat([df_crit, df_noncrit]).sample(frac=1, random_state=42).reset_index(drop=True)


def _build_pipelines() -> dict[str, Pipeline]:
    return {
        "NaiveBayes": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10_000, min_df=3)),
            ("clf",   ComplementNB()),
        ]),
        "LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)),
            ("clf",   LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
        ]),
        "LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)),
            ("clf",   CalibratedClassifierCV(LinearSVC(max_iter=2000, class_weight="balanced"))),
        ]),
    }


def train_and_evaluate(df: pd.DataFrame, text_col: str = "text_clean") -> tuple[Pipeline, dict]:
    """
    Train all three classifiers and return the best one.
    Prints a CV F1 comparison and a held-out classification report.
    """
    df_labelled = create_silver_labels(df).dropna(subset=["label"])
    X = df_labelled[text_col].fillna("")
    y = df_labelled["label"].astype(int)

    print(f"\nTraining on {len(df_labelled):,} labelled complaints")
    pipelines = _build_pipelines()
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: dict[str, float] = {}

    print("\n── 5-fold CV F1 (Naive Bayes → LogReg → SVM complexity ladder) ──")
    for name, pipe in pipelines.items():
        scores        = cross_val_score(pipe, X, y, cv=cv, scoring="f1")
        results[name] = scores.mean()
        print(f"  {name:<25}  F1 = {scores.mean():.3f} ± {scores.std():.3f}")

    best_name = max(results, key=results.get)
    print(f"\nBest: {best_name}  (F1={results[best_name]:.3f})")

    best_pipe = pipelines[best_name]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    best_pipe.fit(X_tr, y_tr)
    print(classification_report(y_te, best_pipe.predict(X_te), target_names=["Non-Critical", "Critical"]))

    return best_pipe, results


def inspect_top_features(pipeline: Pipeline, n: int = 20) -> None:
    """Print the top n positive and negative words learned by the classifier."""
    try:
        tfidf  = pipeline.named_steps["tfidf"]
        clf    = pipeline.named_steps["clf"]
        model  = clf.calibrated_classifiers_[0].estimator if hasattr(clf, "calibrated_classifiers_") else clf
        coef   = model.coef_[0]
        terms  = tfidf.get_feature_names_out()
        top_pos = sorted(zip(coef, terms), reverse=True)[:n]
        top_neg = sorted(zip(coef, terms))[:n]
        print("\n── Top features → CRITICAL ──")
        for w, t in top_pos:
            print(f"  {t:<30}  {w:+.3f}")
        print("\n── Top features → NON-CRITICAL ──")
        for w, t in top_neg:
            print(f"  {t:<30}  {w:+.3f}")
    except Exception as e:
        print(f"[inspect_top_features] Could not extract weights: {e}")


def apply_ml_scoring(df: pd.DataFrame, pipeline: Pipeline, text_col: str = "text_clean") -> pd.DataFrame:
    """Add ml_score (0–100) to all complaints using the trained pipeline."""
    df    = df.copy()
    texts = df[text_col].fillna("")
    if hasattr(pipeline, "predict_proba"):
        df["ml_score"] = pipeline.predict_proba(texts)[:, 1] * 100
    else:
        raw             = pipeline.decision_function(texts)
        df["ml_score"] = (raw - raw.min()) / (raw.max() - raw.min()) * 100
    print(f"ml_score range: {df['ml_score'].min():.1f} – {df['ml_score'].max():.1f}")
    return df


def build_final_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    ML-enhanced composite score (final).

    Weights:
      ml_score      50%  — learned patterns (context, negation, structure)
      keyword_score 30%  — deterministic anchor (prevents ML overconfidence)
      Safety flags       — hard boosts that override ambiguous text scores
      component_risk     — small additive
      filed_quickly      — acute event bonus
    """
    df = df.copy()

    def _row_composite(row) -> float:
        s  = row.get("ml_score",      0) * 0.50
        s += row.get("keyword_score", 0) * 0.30
        s += row.get("has_crash",     0) * 45
        s += row.get("has_injury",    0) * 35
        s += row.get("has_fatality",  0) * 50
        s += row.get("has_fire",      0) * 40
        s += max(0, row.get("component_risk", 0) - 30) * 0.05
        s += row.get("filed_quickly", 0) * 5
        return round(min(100.0, s), 2)

    print("Computing final composite scores...")
    df["composite_score"] = df.apply(_row_composite, axis=1)
    print(df["composite_score"].describe().round(1))
    return df

"""
ML-based scoring: silver label creation, model training, and Score-A computation.
Model comparison runs NaiveBayes / LogReg / LinearSVM with 5-fold CV.
Final model is LogisticRegression (needed for predict_proba for Score-A).
NOTE: LinearSVM achieved best CV F1 (0.932) but requires CalibratedClassifierCV for probabilities.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from typing import Tuple


# ---------------------------------------------------------------------------
# Silver labels
# ---------------------------------------------------------------------------

def create_silver_labels(df: pd.DataFrame,
                         keyword_threshold: int = 15,
                         component_threshold: int = 40) -> Tuple[pd.DataFrame, list, list]:
    """
    Weak supervision labeling:
      CRITICAL (1)     : has crash/injury/fatality/fire flag
      NON-CRITICAL (0) : keyword_score < threshold AND component_risk < threshold AND no flag
      UNCERTAIN        : excluded from training

    keyword_threshold set to 15 (not 25 = hard cap) to avoid near-boundary noise.
    """
    critical_mask = (
        (df["has_crash"] == 1) |
        (df["has_injury"] == 1) |
        (df["has_fatality"] == 1) |
        (df["has_fire"] == 1)
    )
    non_critical_mask = (
        (df["keyword_score"] < keyword_threshold) &
        (df["component_risk"] < component_threshold) &
        (~critical_mask)
    )

    df_critical = df[critical_mask].copy()
    df_critical["label"] = 1
    df_non_critical = df[non_critical_mask].copy()
    df_non_critical["label"] = 0

    n_sample = min(len(df_non_critical), len(df_critical) * 3)
    df_train = (
        pd.concat([df_critical, df_non_critical.sample(n_sample, random_state=42)])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    print(f"Silver labels:")
    print(f"  CRITICAL:     {len(df_critical):>6,}  ({len(df_critical)/len(df):.1%})")
    print(f"  NON-CRITICAL: {len(df_non_critical):>6,}  ({len(df_non_critical)/len(df):.1%})")
    print(f"  UNCERTAIN:    {len(df)-len(df_critical)-len(df_non_critical):>6,}  (excluded)")
    print(f"  Training set: {len(df_train):,}  (Critical rate: {df_train['label'].mean():.1%})")

    X = df_train["text_clean"].fillna("").tolist()
    y = df_train["label"].tolist()
    return df_train, X, y


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(X: list, y: list, n_splits: int = 5) -> str:
    """
    5-fold CV comparison of NaiveBayes / LogReg / LinearSVM.
    Returns name of best model.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pipelines = {
        "NaiveBayes (baseline)": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 1), sublinear_tf=True)),
            ("clf", ComplementNB(alpha=0.1)),
        ]),
        "LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
        ]),
        "LinearSVM": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced", max_iter=2000), cv=3
            )),
        ]),
    }
    results = {}
    print("Running 5-fold cross-validation...")
    for name, pipeline in pipelines.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")
        results[name] = scores.mean()
        print(f"  {name}: F1 = {scores.mean():.3f} +/- {scores.std():.3f}")

    best = max(results, key=results.get)
    print(f"Best model: {best}")
    return best


# ---------------------------------------------------------------------------
# Final classifier
# ---------------------------------------------------------------------------

def train_final_classifier(X: list, y: list, model_path: str) -> Pipeline:
    """
    Trains LogisticRegression (chosen for predict_proba availability).
    Prints classification report and top feature weights.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Non-Critical", "Critical"]))

    feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
    coefs = pipeline.named_steps["clf"].coef_[0]
    top_idx = np.argsort(coefs)[::-1][:15]
    print("Top 15 terms -> CRITICAL:")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1:2d}. \"{feature_names[idx]}\" (weight: {coefs[idx]:.3f})")

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved -> {model_path}")
    return pipeline


# ---------------------------------------------------------------------------
# Score-A
# ---------------------------------------------------------------------------

def compute_score_a(df: pd.DataFrame, pipeline: Pipeline,
                    text_col: str = "text_clean") -> pd.DataFrame:
    """Score-A: Defect severity (0-25) from LR class probability."""
    df = df.copy()
    ml_proba = pipeline.predict_proba(df[text_col].fillna("").tolist())[:, 1]
    df["score_a_defect_severity"] = (ml_proba * 25).round(2)
    df["ml_proba"] = ml_proba
    return df

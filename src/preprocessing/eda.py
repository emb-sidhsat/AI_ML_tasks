"""
Phase 2 — EDA: Vocabulary and data quality audit.
Run this before preprocessing to understand what the raw text actually looks like.
"""
from collections import Counter
import pandas as pd


def audit_complaints(df: pd.DataFrame, text_col: str = "summary") -> None:
    print("=" * 60)
    print("COMPLAINT DATA AUDIT")
    print("=" * 60)
    print(f"\nShape:        {df.shape}")
    print(f"Text column:  '{text_col}'")

    null_rate = df[text_col].isna().mean()
    print(f"Null rate:    {null_rate:.1%}")

    lengths = df[text_col].dropna().str.split().apply(len)
    print(f"Word count:   mean={lengths.mean():.0f}  median={lengths.median():.0f}  max={lengths.max()}")

    # Vocabulary fragmentation — the core motivation for preprocessing
    print("\n── Vocabulary fragmentation (why preprocessing matters) ──")
    raw_tokens: list[str] = []
    for text in df[text_col].dropna().head(500).tolist():
        raw_tokens.extend(text.split())
    vocab = Counter(raw_tokens)
    print(f"  Raw vocabulary size (500 docs): {len(vocab):,}")
    for keyword in ["brake", "steer", "engine", "airbag", "fire"]:
        frags = [t for t in vocab if keyword in t.lower()][:6]
        print(f"  '{keyword}' fragments: {frags}")


def sample_complaints(df: pd.DataFrame, n: int = 5, text_col: str = "summary") -> None:
    """Print n raw complaints so you understand what the NLP pipeline is working with."""
    print("\n── Sample raw complaints ──")
    for _, row in df.dropna(subset=[text_col]).head(n).iterrows():
        make  = row.get("make_pulled", "")
        model = row.get("model_pulled", "")
        year  = row.get("year_pulled", "")
        print(f"\n[{make} {model} {year}]")
        print(row[text_col][:400])
        print()

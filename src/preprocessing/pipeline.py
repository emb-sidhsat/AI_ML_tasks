"""
Phase 2 — Preprocessing: Domain-aware NLP pipeline.

Key design decisions vs a generic pipeline:
  - Negations (not/no/never) are KEPT — "brakes NOT working" ≠ "brakes working"
  - Automotive abbreviations are expanded before punctuation removal
  - POS-aware lemmatisation (verbs: driving→drive; nouns: brakes→brake)
  - Returns intermediate steps so each transform is inspectable
"""
import re
import sys
import nltk
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[2]))
import config

# NLTK data — downloaded once, silent on subsequent runs
for _pkg in ("stopwords", "punkt", "wordnet", "punkt_tab", "averaged_perceptron_tagger_eng"):
    nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# ── Stop-word set with negations preserved ─────────────────────────────────
_STOP_WORDS      = set(stopwords.words("english"))
_KEEP_NEGATIONS  = {"not", "no", "never", "without", "none", "nobody"}
_DOMAIN_SW       = _STOP_WORDS - _KEEP_NEGATIONS

# ── Automotive abbreviation expansion ──────────────────────────────────────
# Applied BEFORE punctuation removal so "ABS" isn't split by surrounding text.
AUTOMOTIVE_ABBREVIATIONS: dict[str, str] = {
    r"\babs\b":   "antilock braking system",
    r"\baeb\b":   "automatic emergency braking",
    r"\bfcw\b":   "forward collision warning",
    r"\becu\b":   "electronic control unit",
    r"\blks\b":   "lane keeping system",
    r"\bacc\b":   "adaptive cruise control",
    r"\bepas\b":  "electric power assisted steering",
    r"\btpms\b":  "tire pressure monitoring system",
    r"\bdtc\b":   "diagnostic trouble code",
    r"\bvin\b":   "vehicle identification number",
    r"\bmy\b":    "model year",
}

_LEMMATIZER = WordNetLemmatizer()


def _wordnet_pos(treebank_tag: str) -> str:
    """Map NLTK POS tag to WordNet POS for accurate lemmatisation."""
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def _lemmatize_with_pos(tokens: list[str]) -> list[str]:
    tagged = pos_tag(tokens)
    return [_LEMMATIZER.lemmatize(word, _wordnet_pos(tag)) for word, tag in tagged]


def preprocess_complaint(text: str) -> dict:
    """
    Full domain-aware preprocessing pipeline.
    Returns {'cleaned': str, 'tokens': list, 'steps': dict}.
    'steps' contains intermediate text at each stage for inspection.
    """
    if not text or not isinstance(text, str):
        return {"cleaned": "", "tokens": [], "steps": {}}

    steps: dict = {"0_original": text[:100]}

    # 1 — Lowercase
    s = text.lower()
    steps["1_lowercase"] = s[:100]

    # 2 — Expand abbreviations (before punctuation removal)
    for pattern, expansion in AUTOMOTIVE_ABBREVIATIONS.items():
        s = re.sub(pattern, expansion, s, flags=re.IGNORECASE)
    steps["2_abbr_expanded"] = s[:100]

    # 3 — Remove URLs / emails
    s = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", s)

    # 4 — Replace punctuation with space (not delete — preserves word boundaries)
    s = re.sub(r"[^\w\s]", " ", s)
    steps["4_punct_removed"] = s[:100]

    # 5 — Collapse repeated characters ("BRAAAKES" → "brakes")
    s = re.sub(r"(.)\1{2,}", r"\1", s)

    # 6 — Normalise whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # 7 — Tokenise
    tokens = word_tokenize(s)
    steps["7_tokenized"] = tokens[:10]

    # 8 — Remove stop words (negations survive)
    tokens = [t for t in tokens if t not in _DOMAIN_SW and len(t) > 1]
    steps["8_stopwords_removed"] = tokens[:10]

    # 9 — POS-aware lemmatisation
    tokens = _lemmatize_with_pos(tokens)
    steps["9_lemmatized"] = tokens[:10]

    cleaned = " ".join(tokens)
    return {"cleaned": cleaned, "tokens": tokens, "steps": steps}


def apply_preprocessing(df: pd.DataFrame, text_col: str = "summary") -> pd.DataFrame:
    """Apply the full pipeline to every row; adds a 'text_clean' column."""
    print(f"Preprocessing {len(df):,} complaints (text_col='{text_col}')...")
    tqdm.pandas()
    df = df.copy()
    df["text_clean"] = (
        df[text_col]
        .fillna("")
        .progress_apply(lambda x: preprocess_complaint(x)["cleaned"])
    )
    empty_pct = (df["text_clean"] == "").mean()
    print(f"  Done. Empty after cleaning: {empty_pct:.1%}")
    return df

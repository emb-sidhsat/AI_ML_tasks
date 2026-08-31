"""
9-step NLP preprocessing pipeline for NHTSA complaint text.
Single-pass: one call returns both cleaned text and token list.
"""

import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from typing import List

for pkg in ["stopwords", "punkt", "wordnet", "punkt_tab", "averaged_perceptron_tagger_eng"]:
    nltk.download(pkg, quiet=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEEP_NEGATIONS = {"not", "no", "never", "without", "none", "nobody"}
STOP_WORDS = set(stopwords.words("english")) - KEEP_NEGATIONS

AUTOMOTIVE_ABBREVIATIONS = {
    r"\babs\b":   "antilock braking system",
    r"\baeb\b":   "automatic emergency braking",
    r"\bfcw\b":   "forward collision warning",
    r"\becu\b":   "electronic control unit",
    r"\blks\b":   "lane keeping system",
    r"\bacc\b":   "adaptive cruise control",
    r"\bepas\b":  "electric power assisted steering",
    r"\btpms\b":  "tire pressure monitoring system",
    r"\bdtc\b":   "diagnostic trouble code",
    # NHTSA-specific prefixes — strip entirely
    r"^tl\*\s*": "",
    r"^the contact\s+": "",
}

lemmatizer = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_wordnet_pos(tag: str) -> str:
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN


def _lemmatize_with_pos(tokens: List[str]) -> List[str]:
    tagged = pos_tag(tokens)
    return [lemmatizer.lemmatize(w, _get_wordnet_pos(t)) for w, t in tagged]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def preprocess_complaint(text: str, debug: bool = False) -> dict:
    """
    9-step cleaning pipeline. Returns:
      cleaned : str   — space-joined lemmatized tokens
      tokens  : list  — individual lemmatized tokens
      steps   : dict  — intermediate states (only populated if debug=True)
    """
    if not text or not isinstance(text, str):
        return {"cleaned": "", "tokens": [], "steps": {}}

    steps = {}
    if debug:
        steps["0_original"] = text[:80]

    s = text.lower()
    if debug: steps["1_lowercase"] = s[:80]

    for pattern, expansion in AUTOMOTIVE_ABBREVIATIONS.items():
        s = re.sub(pattern, expansion, s, flags=re.IGNORECASE)
    if debug: steps["2_abbr_expanded"] = s[:80]

    s = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", s)
    if debug: steps["3_urls_removed"] = s[:80]

    s = re.sub(r"[^\w\s]", " ", s)
    if debug: steps["4_punct_removed"] = s[:80]

    s = re.sub(r"(.)\1{2,}", r"\1", s)
    if debug: steps["5_repeated_chars"] = s[:80]

    s = re.sub(r"\s+", " ", s).strip()
    if debug: steps["6_whitespace"] = s[:80]

    tokens = word_tokenize(s)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    lemmatized = _lemmatize_with_pos(tokens)
    if debug: steps["7_lemmatized"] = lemmatized[:10]

    return {"cleaned": " ".join(lemmatized), "tokens": lemmatized, "steps": steps}


def apply_preprocessing(df, text_col: str = "summary", show_progress: bool = True):
    """
    Apply preprocessing to a full DataFrame. Single-pass (one apply call).
    Adds: text_clean, text_tokens, word_count_clean, too_short.
    """
    from tqdm import tqdm
    tqdm.pandas()

    apply_fn = df[text_col].fillna("").progress_apply if show_progress else df[text_col].fillna("").apply
    results = apply_fn(preprocess_complaint)

    df = df.copy()
    df["text_clean"] = results.apply(lambda x: x["cleaned"])
    df["text_tokens"] = results.apply(lambda x: x["tokens"])
    df["word_count_clean"] = df["text_clean"].str.split().str.len()
    df["too_short"] = df["word_count_clean"] < 5
    return df

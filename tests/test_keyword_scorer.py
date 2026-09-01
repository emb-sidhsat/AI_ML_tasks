"""Unit tests for keyword scoring logic."""
import pytest
from src.scoring.keyword_scorer import keyword_score


@pytest.mark.parametrize("text,lo,hi", [
    ("fire death fatality brake failure",         20, 25),  # T1 dominated, cap=25
    ("crash accident collision injury",            5, 25),  # T2 heavy
    ("failed failure malfunction broken defect",   5, 18),  # T3 only
    ("the radio cuts out sometimes",               5,  6),  # no keywords
    ("",                                           5,  5),  # empty → baseline
])
def test_score_ranges(text, lo, hi):
    result = keyword_score(text)
    assert lo <= result["score"] <= hi

def test_baseline_score_on_empty():
    result = keyword_score("")
    assert result["score"] == 5.0
    assert result["keywords_found"] == []


def test_tier1_keyword_fires():
    result = keyword_score("The vehicle caught fire and driver died")
    assert result["score"] > 20
    assert any("[T1]" in k for k in result["keywords_found"])


def test_hard_cap():
    text = "fire explode death fatality crash injury dangerous stall suddenly collision airbag"
    result = keyword_score(text)
    assert result["score"] == 25.0


def test_cosmetic_scores_low():
    result = keyword_score("The cup holder is broken and the trim rattles")
    assert result["score"] < 15


def test_tier_summary_format():
    result = keyword_score("vehicle stalled suddenly and crashed")
    assert "T1:" in result["tier_summary"]
    assert "T2:" in result["tier_summary"]
    assert "T3:" in result["tier_summary"]


def test_raw_text_preserves_inflections():
    # "died" must match on raw text (lemmatization converts it to "die")
    result = keyword_score("My wife died in this crash")
    assert any("died" in k for k in result["keywords_found"])

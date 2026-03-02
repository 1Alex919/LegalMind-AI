"""Tests for evaluation metrics."""

import pytest

from src.evaluation.metrics import (
    mean_reciprocal_rank,
    retrieval_hit_rate,
    _is_match,
    _token_containment,
    _jaccard_similarity,
)


def test_hit_rate_perfect() -> None:
    retrieved = [["the answer is here"], ["another answer"]]
    ground_truth = ["the answer is here", "another answer"]
    assert retrieval_hit_rate(retrieved, ground_truth) == pytest.approx(1.0)


def test_hit_rate_zero() -> None:
    retrieved = [["completely unrelated xyz"], ["totally different abc"]]
    ground_truth = ["the real answer about contracts", "another real answer about law"]
    assert retrieval_hit_rate(retrieved, ground_truth) == pytest.approx(0.0)


def test_hit_rate_partial() -> None:
    retrieved = [["the answer is here", "noise"], ["irrelevant text"]]
    ground_truth = ["the answer is here", "not found anywhere in retrieved"]
    assert retrieval_hit_rate(retrieved, ground_truth) == pytest.approx(0.5)


def test_hit_rate_empty() -> None:
    assert retrieval_hit_rate([], []) == pytest.approx(0.0)


def test_hit_rate_fuzzy_match() -> None:
    """Token containment should match paraphrased text."""
    retrieved = [
        [
            "This Agreement is between County of Orange and Microsoft Corporation "
            "and its affiliates as indicated in the signature block below."
        ]
    ]
    ground_truth = [
        "This Non-Disclosure Agreement ('Agreement') is between County of Orange "
        "('County') and Microsoft Corporation and its affiliates ('Microsoft')."
    ]
    assert retrieval_hit_rate(retrieved, ground_truth) == pytest.approx(1.0)


def test_mrr_first_position() -> None:
    retrieved = [["correct answer"]]
    ground_truth = ["correct answer"]
    assert mean_reciprocal_rank(retrieved, ground_truth) == pytest.approx(1.0)


def test_mrr_second_position() -> None:
    retrieved = [["wrong unrelated text", "correct answer"]]
    ground_truth = ["correct answer"]
    assert mean_reciprocal_rank(retrieved, ground_truth) == pytest.approx(0.5)


def test_mrr_empty() -> None:
    assert mean_reciprocal_rank([], []) == pytest.approx(0.0)


def test_is_match_substring() -> None:
    assert _is_match("the answer is here in this long chunk", "answer is here")
    assert _is_match("short", "this is short and something else")


def test_is_match_token_containment() -> None:
    """70%+ of truth tokens in chunk should match."""
    chunk = "The Agreement expires one hundred eighty days from the signature date"
    truth = "This Agreement expires one hundred and eighty days from the later signature dates"
    assert _is_match(chunk, truth)


def test_token_containment_score() -> None:
    chunk = "County Orange Microsoft Corporation affiliates Agreement"
    truth = "County of Orange and Microsoft Corporation"
    score = _token_containment(chunk, truth)
    assert score >= 0.7


def test_jaccard_similarity() -> None:
    assert _jaccard_similarity("hello world foo", "hello world bar") > 0.3
    assert _jaccard_similarity("completely different", "nothing similar here") < 0.3

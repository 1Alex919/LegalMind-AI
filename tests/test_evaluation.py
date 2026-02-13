"""Tests for evaluation metrics."""

import pytest

from src.evaluation.metrics import mean_reciprocal_rank, retrieval_hit_rate


def test_hit_rate_perfect() -> None:
    retrieved = [["the answer is here"], ["another answer"]]
    ground_truth = ["the answer is here", "another answer"]
    assert retrieval_hit_rate(retrieved, ground_truth) == pytest.approx(1.0)


def test_hit_rate_zero() -> None:
    retrieved = [["irrelevant text"], ["more irrelevant"]]
    ground_truth = ["the real answer", "another real answer"]
    assert retrieval_hit_rate(retrieved, ground_truth) == pytest.approx(0.0)


def test_hit_rate_partial() -> None:
    retrieved = [["the answer is here", "noise"], ["irrelevant text"]]
    ground_truth = ["the answer is here", "not found"]
    assert retrieval_hit_rate(retrieved, ground_truth) == pytest.approx(0.5)


def test_hit_rate_empty() -> None:
    assert retrieval_hit_rate([], []) == pytest.approx(0.0)


def test_mrr_first_position() -> None:
    retrieved = [["correct answer"]]
    ground_truth = ["correct answer"]
    assert mean_reciprocal_rank(retrieved, ground_truth) == pytest.approx(1.0)


def test_mrr_second_position() -> None:
    retrieved = [["wrong", "correct answer"]]
    ground_truth = ["correct answer"]
    assert mean_reciprocal_rank(retrieved, ground_truth) == pytest.approx(0.5)


def test_mrr_empty() -> None:
    assert mean_reciprocal_rank([], []) == pytest.approx(0.0)

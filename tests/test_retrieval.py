"""Tests for RAG retrieval pipeline."""

import numpy as np
import pytest

from src.retrieval.hybrid_search import HybridSearch, SearchResult


def test_normalize_scores() -> None:
    hs = HybridSearch.__new__(HybridSearch)
    scores = [0.1, 0.5, 0.9]
    normalized = hs._normalize_scores(scores)
    assert len(normalized) == 3
    assert min(normalized) == pytest.approx(0.0)
    assert max(normalized) == pytest.approx(1.0)


def test_normalize_scores_identical() -> None:
    hs = HybridSearch.__new__(HybridSearch)
    scores = [0.5, 0.5, 0.5]
    normalized = hs._normalize_scores(scores)
    assert all(s == pytest.approx(1.0) for s in normalized)


def test_normalize_scores_empty() -> None:
    hs = HybridSearch.__new__(HybridSearch)
    assert hs._normalize_scores([]) == []


def test_search_result_dataclass() -> None:
    result = SearchResult(
        chunk_id="test-id",
        text="Some contract text",
        score=0.85,
        metadata={"page": 1, "source": "test.pdf"},
    )
    assert result.chunk_id == "test-id"
    assert result.score == 0.85
    assert result.metadata["page"] == 1

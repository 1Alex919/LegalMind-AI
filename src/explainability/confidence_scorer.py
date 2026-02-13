"""Calculate answer confidence based on retrieval scores."""

import numpy as np
from loguru import logger

from src.retrieval.hybrid_search import SearchResult


def calculate_confidence(
    search_results: list[SearchResult],
    llm_confidence: float | None = None,
) -> float:
    """Calculate overall confidence score from retrieval and LLM signals.

    Combines:
    - Mean retrieval score (how relevant the sources are)
    - Score spread (consistency of retrieval results)
    - LLM self-reported confidence (if available)
    """
    if not search_results:
        return 0.0

    scores = [r.score for r in search_results]
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    # Higher mean = better retrieval. Normalized to [0, 1].
    retrieval_confidence = min(mean_score, 1.0)

    # Lower spread = more consistent results = higher confidence
    consistency_bonus = max(0, 0.1 - std_score) * 2  # up to +0.2

    combined = retrieval_confidence + consistency_bonus

    # Blend with LLM's own confidence if available
    if llm_confidence is not None:
        combined = 0.6 * combined + 0.4 * llm_confidence

    final = round(min(max(combined, 0.0), 1.0), 3)

    logger.debug(
        f"Confidence: retrieval={retrieval_confidence:.3f}, "
        f"consistency_bonus={consistency_bonus:.3f}, "
        f"llm={llm_confidence}, final={final}"
    )
    return final

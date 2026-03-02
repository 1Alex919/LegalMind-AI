"""Custom evaluation metrics for RAG pipeline."""

import re

from loguru import logger


def _tokenize(text: str) -> set[str]:
    """Tokenize text into a set of lowercase words (3+ chars)."""
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) >= 3}


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Calculate Jaccard similarity between two texts based on token overlap."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _token_containment(chunk: str, truth: str) -> float:
    """What fraction of ground truth tokens appear in the chunk.

    This is asymmetric: we care that the truth is *contained* in the chunk,
    not that the chunk is contained in the truth. A large chunk containing
    all truth tokens scores 1.0 even if the chunk has extra tokens.
    """
    truth_tokens = _tokenize(truth)
    if not truth_tokens:
        return 0.0
    chunk_tokens = _tokenize(chunk)
    overlap = truth_tokens & chunk_tokens
    return len(overlap) / len(truth_tokens)


def _is_match(chunk: str, truth: str, threshold: float = 0.3) -> bool:
    """Check if a chunk matches the ground truth using multiple strategies.

    Strategies (any match counts):
    1. Substring containment (exact match)
    2. Token containment >= 0.7 (most of truth's key words are in chunk)
    3. Jaccard token overlap >= threshold
    """
    chunk_lower = chunk.lower()
    truth_lower = truth.lower()

    # Strategy 1: Substring containment
    if truth_lower in chunk_lower or chunk_lower in truth_lower:
        return True

    # Strategy 2: Token containment (are most ground truth words in the chunk?)
    if _token_containment(chunk, truth) >= 0.7:
        return True

    # Strategy 3: Jaccard token overlap
    if _jaccard_similarity(chunk, truth) >= threshold:
        return True

    return False


def retrieval_hit_rate(
    retrieved_contexts: list[list[str]],
    ground_truth_contexts: list[str],
    threshold: float = 0.3,
) -> float:
    """Calculate how often the correct context appears in retrieved results.

    Uses substring matching, token containment, and Jaccard overlap.

    Args:
        retrieved_contexts: List of retrieved context lists per query.
        ground_truth_contexts: List of ground truth context strings.
        threshold: Jaccard similarity threshold for fuzzy matching.

    Returns:
        Hit rate (0 to 1).
    """
    if not retrieved_contexts:
        return 0.0

    hits = 0
    for i, (retrieved, truth) in enumerate(
        zip(retrieved_contexts, ground_truth_contexts)
    ):
        matched = False
        for chunk in retrieved:
            if _is_match(chunk, truth, threshold):
                matched = True
                break
        if matched:
            hits += 1
        else:
            # Log miss for debugging
            best_score = 0.0
            for chunk in retrieved:
                score = _token_containment(chunk, truth)
                best_score = max(best_score, score)
            logger.debug(
                f"Hit rate miss sample {i+1}: best_containment={best_score:.2f}, "
                f"truth='{truth[:60]}...'"
            )

    rate = hits / len(retrieved_contexts)
    logger.info(f"Retrieval hit rate: {rate:.3f} ({hits}/{len(retrieved_contexts)})")
    return rate


def mean_reciprocal_rank(
    retrieved_contexts: list[list[str]],
    ground_truth_contexts: list[str],
    threshold: float = 0.3,
) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    Uses substring matching, token containment, and Jaccard overlap.

    Args:
        retrieved_contexts: List of retrieved context lists per query.
        ground_truth_contexts: List of ground truth context strings.
        threshold: Jaccard similarity threshold for fuzzy matching.

    Returns:
        MRR score (0 to 1).
    """
    if not retrieved_contexts:
        return 0.0

    rr_sum = 0.0
    for retrieved, truth in zip(retrieved_contexts, ground_truth_contexts):
        for rank, chunk in enumerate(retrieved, 1):
            if _is_match(chunk, truth, threshold):
                rr_sum += 1.0 / rank
                break

    mrr = rr_sum / len(retrieved_contexts)
    logger.info(f"MRR: {mrr:.3f}")
    return mrr

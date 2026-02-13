"""Custom evaluation metrics for RAG pipeline."""

from loguru import logger


def retrieval_hit_rate(
    retrieved_contexts: list[list[str]],
    ground_truth_contexts: list[str],
) -> float:
    """Calculate how often the correct context appears in retrieved results.

    Args:
        retrieved_contexts: List of retrieved context lists per query.
        ground_truth_contexts: List of ground truth context strings.

    Returns:
        Hit rate (0 to 1).
    """
    if not retrieved_contexts:
        return 0.0

    hits = 0
    for retrieved, truth in zip(retrieved_contexts, ground_truth_contexts):
        truth_lower = truth.lower()
        for chunk in retrieved:
            if truth_lower in chunk.lower() or chunk.lower() in truth_lower:
                hits += 1
                break

    rate = hits / len(retrieved_contexts)
    logger.info(f"Retrieval hit rate: {rate:.3f} ({hits}/{len(retrieved_contexts)})")
    return rate


def mean_reciprocal_rank(
    retrieved_contexts: list[list[str]],
    ground_truth_contexts: list[str],
) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    Returns:
        MRR score (0 to 1).
    """
    if not retrieved_contexts:
        return 0.0

    rr_sum = 0.0
    for retrieved, truth in zip(retrieved_contexts, ground_truth_contexts):
        truth_lower = truth.lower()
        for rank, chunk in enumerate(retrieved, 1):
            if truth_lower in chunk.lower() or chunk.lower() in truth_lower:
                rr_sum += 1.0 / rank
                break

    mrr = rr_sum / len(retrieved_contexts)
    logger.info(f"MRR: {mrr:.3f}")
    return mrr

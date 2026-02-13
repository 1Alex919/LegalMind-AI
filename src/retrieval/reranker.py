"""FlashRank reranking for search results."""

from flashrank import Ranker, RerankRequest
from loguru import logger

from config.settings import settings
from src.retrieval.hybrid_search import SearchResult


class ReRanker:
    """Rerank search results using FlashRank for better precision."""

    def __init__(self) -> None:
        self.model_name = settings.RERANKER_MODEL
        self.top_n = settings.RERANKER_TOP_N
        self._ranker: Ranker | None = None

    def _get_ranker(self) -> Ranker:
        if self._ranker is None:
            logger.info(f"Loading reranker model: {self.model_name}")
            self._ranker = Ranker(model_name=self.model_name)
        return self._ranker

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int | None = None,
    ) -> list[SearchResult]:
        """Rerank search results for the given query.

        Falls back to original results if reranking fails.
        """
        top_n = top_n or self.top_n

        if not results:
            return []

        try:
            ranker = self._get_ranker()

            passages = [
                {"id": r.chunk_id, "text": r.text, "meta": r.metadata}
                for r in results
            ]

            rerank_request = RerankRequest(query=query, passages=passages)
            reranked = ranker.rerank(rerank_request)

            reranked_results = []
            for item in reranked[:top_n]:
                reranked_results.append(
                    SearchResult(
                        chunk_id=item["id"],
                        text=item["text"],
                        score=float(item["score"]),
                        metadata=item["meta"],
                    )
                )

            logger.info(
                f"Reranked {len(results)} results to top {len(reranked_results)}"
            )
            return reranked_results

        except Exception as e:
            logger.warning(f"Reranking failed, using hybrid results: {e}")
            return results[:top_n]

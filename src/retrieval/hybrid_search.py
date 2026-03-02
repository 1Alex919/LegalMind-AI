"""Hybrid search combining BM25 (sparse) and vector (dense) search."""

from dataclasses import dataclass

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from config.settings import settings
from src.ingestion.embeddings import (
    get_chroma_client,
    get_or_create_collection,
    get_query_embedding_function,
)


@dataclass
class SearchResult:
    """A single search result with score and metadata."""

    chunk_id: str
    text: str
    score: float
    metadata: dict


class HybridSearch:
    """Combines BM25 keyword search with ChromaDB vector search."""

    def __init__(self) -> None:
        self.client = get_chroma_client()
        self.collection = get_or_create_collection(self.client)
        self.embeddings_fn = get_query_embedding_function()
        self.alpha = settings.HYBRID_ALPHA  # vector weight
        self._bm25: BM25Okapi | None = None
        self._corpus_ids: list[str] = []
        self._corpus_texts: list[str] = []
        self._corpus_meta: list[dict] = []

    def _build_bm25_index(self) -> None:
        """Build BM25 index from all documents in the collection."""
        all_docs = self.collection.get(include=["documents", "metadatas"])
        if not all_docs["documents"]:
            logger.warning("No documents in collection for BM25 index.")
            return

        self._corpus_ids = all_docs["ids"]
        self._corpus_texts = all_docs["documents"]
        self._corpus_meta = all_docs["metadatas"]

        tokenized = [doc.lower().split() for doc in self._corpus_texts]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built with {len(self._corpus_texts)} documents")

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Min-max normalize scores to [0, 1]."""
        if not scores:
            return []
        arr = np.array(scores)
        min_s, max_s = arr.min(), arr.max()
        if max_s - min_s == 0:
            return [1.0] * len(scores)
        return ((arr - min_s) / (max_s - min_s)).tolist()

    def _bm25_search(self, query: str, k: int) -> list[SearchResult]:
        """Perform BM25 keyword search."""
        if self._bm25 is None:
            self._build_bm25_index()
        if self._bm25 is None:
            return []

        tokenized_query = query.lower().split()
        raw_scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(raw_scores)[::-1][:k]

        scores_for_top = [raw_scores[i] for i in top_indices]
        norm_scores = self._normalize_scores(scores_for_top)

        results = []
        for idx, norm_score in zip(top_indices, norm_scores):
            results.append(
                SearchResult(
                    chunk_id=self._corpus_ids[idx],
                    text=self._corpus_texts[idx],
                    score=norm_score,
                    metadata=self._corpus_meta[idx],
                )
            )
        return results

    def _vector_search(self, query: str, k: int) -> list[SearchResult]:
        """Perform vector similarity search via ChromaDB."""
        query_embedding = self.embeddings_fn.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        # ChromaDB returns cosine distances; convert to similarity
        distances = results["distances"][0]
        similarities = [1 - d for d in distances]
        norm_scores = self._normalize_scores(similarities)

        search_results = []
        for i, chunk_id in enumerate(results["ids"][0]):
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=results["documents"][0][i],
                    score=norm_scores[i],
                    metadata=results["metadatas"][0][i],
                )
            )
        return search_results

    @staticmethod
    def _rrf_score(rank: int, k: int = 60) -> float:
        """Reciprocal Rank Fusion score: 1 / (k + rank)."""
        return 1.0 / (k + rank)

    def search(self, query: str, k: int | None = None) -> list[SearchResult]:
        """Perform hybrid search combining BM25 and vector scores.

        Uses Reciprocal Rank Fusion (RRF) for robust score merging:
        final_score = alpha * rrf_vector + (1 - alpha) * rrf_bm25
        """
        k = k or settings.RETRIEVAL_TOP_K
        fetch_k = k * 3  # fetch more candidates for fusion

        logger.info(f"Hybrid search: '{query[:60]}...' (k={k})")

        vector_results = self._vector_search(query, fetch_k)
        bm25_results = self._bm25_search(query, fetch_k)

        # Build RRF rank maps (chunk_id -> rank position, 1-indexed)
        vector_ranks = {r.chunk_id: i + 1 for i, r in enumerate(vector_results)}
        bm25_ranks = {r.chunk_id: i + 1 for i, r in enumerate(bm25_results)}

        # Collect all chunk data
        chunk_data: dict[str, dict] = {}
        for r in vector_results:
            chunk_data[r.chunk_id] = {"text": r.text, "metadata": r.metadata}
        for r in bm25_results:
            if r.chunk_id not in chunk_data:
                chunk_data[r.chunk_id] = {"text": r.text, "metadata": r.metadata}

        # Calculate RRF combined scores
        default_rank = fetch_k + 1  # penalty rank for missing results
        combined: list[SearchResult] = []
        for chunk_id, data in chunk_data.items():
            v_rank = vector_ranks.get(chunk_id, default_rank)
            b_rank = bm25_ranks.get(chunk_id, default_rank)
            final_score = (
                self.alpha * self._rrf_score(v_rank)
                + (1 - self.alpha) * self._rrf_score(b_rank)
            )
            combined.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=data["text"],
                    score=final_score,
                    metadata=data["metadata"],
                )
            )

        combined.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            f"Hybrid search returned {len(combined)} candidates, "
            f"returning top {k}"
        )
        return combined[:k]

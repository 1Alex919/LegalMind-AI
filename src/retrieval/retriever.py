"""Main retrieval orchestrator combining hybrid search, reranking, and query expansion."""

import time
from dataclasses import dataclass, field

from loguru import logger

from config.settings import settings
from src.ingestion.embeddings import get_chroma_client, get_or_create_collection
from src.retrieval.hybrid_search import HybridSearch, SearchResult
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.reranker import ReRanker


@dataclass
class RetrievalResult:
    """Final retrieval result with metrics."""

    results: list[SearchResult]
    expanded_queries: list[str]
    latency_ms: float
    total_candidates: int
    parent_context: dict[str, str] = field(default_factory=dict)


class Retriever:
    """Main retrieval interface with hybrid search, reranking, and query expansion."""

    def __init__(
        self,
        use_reranking: bool = True,
        use_query_expansion: bool = False,
        use_query_translation: bool = False,
    ) -> None:
        self.hybrid_search = HybridSearch()
        self.reranker = ReRanker() if use_reranking else None
        self.query_expander = QueryExpander() if (use_query_expansion or use_query_translation) else None
        self.use_query_translation = use_query_translation

    def _get_parent_context(self, results: list[SearchResult]) -> dict[str, str]:
        """Fetch parent chunk text for child chunks (parent document retrieval)."""
        parent_ids = set()
        for r in results:
            pid = r.metadata.get("parent_id")
            if pid:
                parent_ids.add(pid)

        if not parent_ids:
            return {}

        client = get_chroma_client()
        parent_collection = get_or_create_collection(client, name="legalmind_parents")

        parent_context = {}
        for pid in parent_ids:
            try:
                result = parent_collection.get(ids=[pid], include=["documents"])
                if result["documents"]:
                    parent_context[pid] = result["documents"][0]
            except Exception:
                continue

        logger.info(f"Fetched {len(parent_context)} parent contexts")
        return parent_context

    def _detect_doc_language(self) -> str | None:
        """Detect language of stored documents by sampling a chunk."""
        try:
            collection = self.hybrid_search.collection
            sample = collection.peek(limit=1)
            if not sample["documents"]:
                return None
            text = sample["documents"][0][:200]

            # Simple heuristic: check Unicode script of alphabetic chars
            latin = sum(1 for c in text if c.isalpha() and ord(c) < 256)
            total_alpha = sum(1 for c in text if c.isalpha())
            if total_alpha == 0:
                return None

            # If mostly non-Latin (Cyrillic, Slovak diacritics still Latin)
            # Use GPT for accurate detection
            if not self.query_expander:
                return None

            response = self.query_expander.client.chat.completions.create(
                model=self.query_expander.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Detect the language of this text. Reply with only the language name in English (e.g. 'English', 'Slovak', 'Russian').",
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=10,
            )
            lang = (response.choices[0].message.content or "").strip()
            logger.info(f"Detected document language: {lang}")
            return lang
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
            return None

    def _detect_query_language(self, query: str) -> str | None:
        """Detect language of the query using simple heuristics."""
        # Cyrillic characters → likely Russian/Ukrainian
        cyrillic = sum(1 for c in query if "\u0400" <= c <= "\u04ff")
        latin = sum(1 for c in query if c.isalpha() and ord(c) < 256)

        if cyrillic > latin:
            return "Russian"
        if latin > 0:
            return None  # Could be English, Slovak, etc. — skip detection
        return None

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        use_hyde: bool = False,
        use_multi_query: bool = False,
    ) -> RetrievalResult:
        """Execute the full retrieval pipeline.

        1. (Optional) Translate query to document language
        2. (Optional) Expand query via HyDE / multi-query
        3. Hybrid search (BM25 + vector)
        4. (Optional) Rerank results
        5. Fetch parent context for expanded context
        """
        start = time.perf_counter()
        k = k or settings.RETRIEVAL_TOP_K

        # Step 0: Cross-language query translation
        search_query = query
        if self.use_query_translation and self.query_expander:
            query_lang = self._detect_query_language(query)
            if query_lang:  # query is in a detectable non-ambiguous language
                doc_lang = self._detect_doc_language()
                if doc_lang and doc_lang.lower() != query_lang.lower():
                    translated = self.query_expander.translate_query(query, doc_lang)
                    search_query = translated
                    logger.info(f"Cross-language: {query_lang} → {doc_lang}")

        # Step 1: Query expansion
        queries = [search_query]
        if self.query_expander and (use_hyde or use_multi_query):
            queries = self.query_expander.expand(
                search_query, use_hyde=use_hyde, use_multi=use_multi_query
            )

        # Step 2: Hybrid search (merge results from all query variations)
        all_results: dict[str, SearchResult] = {}
        for q in queries:
            for r in self.hybrid_search.search(q, k=k * 2):
                if r.chunk_id not in all_results or r.score > all_results[r.chunk_id].score:
                    all_results[r.chunk_id] = r

        candidates = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        total_candidates = len(candidates)

        # Step 3: Rerank
        if self.reranker and candidates:
            results = self.reranker.rerank(query, candidates, top_n=k)
        else:
            results = candidates[:k]

        # Step 4: Parent context
        parent_context = self._get_parent_context(results)

        latency = (time.perf_counter() - start) * 1000

        logger.info(
            f"Retrieval complete: {len(results)} results from "
            f"{total_candidates} candidates in {latency:.0f}ms"
        )

        return RetrievalResult(
            results=results,
            expanded_queries=queries,
            latency_ms=latency,
            total_candidates=total_candidates,
            parent_context=parent_context,
        )

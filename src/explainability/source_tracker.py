"""Track which document chunks were used to generate answers."""

from pydantic import BaseModel

from src.retrieval.hybrid_search import SearchResult


class TrackedSource(BaseModel):
    """A source with tracking information for explainability."""

    text: str
    page: int | None = None
    relevance_score: float = 0.0
    chunk_id: str = ""
    highlighted: bool = True


class ExplainableResponse(BaseModel):
    """Full response with explainability data."""

    answer: str
    confidence: float
    sources: list[TrackedSource]
    reasoning: str
    agent_used: str
    retrieval_latency_ms: float = 0.0
    total_chunks_retrieved: int = 0


def track_sources(
    search_results: list[SearchResult],
    top_n: int = 5,
) -> list[TrackedSource]:
    """Convert search results into tracked sources for explainability."""
    tracked = []
    for r in search_results[:top_n]:
        tracked.append(
            TrackedSource(
                text=r.text[:500],  # Truncate for display
                page=r.metadata.get("page"),
                relevance_score=round(r.score, 3),
                chunk_id=r.chunk_id,
            )
        )
    return tracked


def build_explainable_response(
    answer: str,
    confidence: float,
    sources: list[TrackedSource],
    agent_used: str,
    reasoning_steps: list[str],
    retrieval_latency_ms: float = 0.0,
    total_chunks: int = 0,
) -> ExplainableResponse:
    """Build a full explainable response from components."""
    reasoning = " -> ".join(reasoning_steps)

    return ExplainableResponse(
        answer=answer,
        confidence=round(confidence, 3),
        sources=sources,
        reasoning=reasoning,
        agent_used=agent_used,
        retrieval_latency_ms=round(retrieval_latency_ms, 1),
        total_chunks_retrieved=total_chunks,
    )

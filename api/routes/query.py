"""Q&A endpoint."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from api.routes.upload import get_document_registry
from api.schemas import QueryRequest, QueryResponse, SourceItem
from src.agents.orchestrator import run

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest) -> QueryResponse:
    """Ask a question about an uploaded document."""
    registry = get_document_registry()
    if request.document_id not in registry:
        raise HTTPException(status_code=404, detail="Document not found")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        state = run(
            task_type="qa",
            question=request.question,
            document_id=request.document_id,
        )
        result = state.get("result", {})

        sources = [SourceItem(**s) for s in result.get("sources", [])]

        return QueryResponse(
            answer=result.get("answer", "I couldn't generate an answer."),
            confidence=result.get("confidence", 0.0),
            sources=sources,
            reasoning=f"Used QA Agent with {len(sources)} sources",
            document_id=request.document_id,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

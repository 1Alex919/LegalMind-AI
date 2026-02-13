"""Risk and summary analysis endpoints."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from api.routes.upload import get_document_registry
from api.schemas import RiskItem, RiskRequest, RiskResponse, SummaryRequest, SummaryResponse
from src.agents.orchestrator import run

router = APIRouter()


@router.post("/analyze/risks", response_model=RiskResponse)
async def analyze_risks(request: RiskRequest) -> RiskResponse:
    """Run risk analysis on an uploaded document."""
    registry = get_document_registry()
    if request.document_id not in registry:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        state = run(task_type="risk", document_id=request.document_id)
        result = state.get("result", {})

        risks = [RiskItem(**r) for r in result.get("risks", [])]

        return RiskResponse(
            risks=risks,
            summary=result.get("summary", ""),
            total_risks=result.get("total_risks", len(risks)),
            document_id=request.document_id,
        )

    except Exception as e:
        logger.error(f"Risk analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@router.post("/analyze/summary", response_model=SummaryResponse)
async def analyze_summary(request: SummaryRequest) -> SummaryResponse:
    """Generate a summary of an uploaded document."""
    registry = get_document_registry()
    if request.document_id not in registry:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        state = run(task_type="summary", document_id=request.document_id)
        result = state.get("result", {})

        return SummaryResponse(
            summary=result.get("summary", ""),
            key_points=result.get("key_points", []),
            parties=result.get("parties", []),
            contract_type=result.get("contract_type", "Unknown"),
            document_id=request.document_id,
        )

    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summary failed: {e}")

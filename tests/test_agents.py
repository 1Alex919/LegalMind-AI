"""Tests for multi-agent system."""

import pytest

from src.agents.qa_agent import QAResult, Source
from src.agents.risk_agent import Risk, RiskAnalysisResult
from src.agents.summary_agent import SummaryResult


def test_risk_model_creation() -> None:
    risk = Risk(
        risk_type="liability",
        severity="high",
        clause_text="The contractor shall be liable...",
        explanation="Unlimited liability.",
        recommendation="Add a cap.",
        page=3,
    )
    assert risk.severity == "high"
    assert risk.page == 3


def test_risk_analysis_result() -> None:
    result = RiskAnalysisResult(
        risks=[
            Risk(
                risk_type="termination",
                severity="medium",
                clause_text="Either party may terminate...",
                explanation="Short notice period.",
                recommendation="Extend to 60 days.",
            )
        ],
        summary="1 risk found.",
        total_risks=1,
    )
    assert result.total_risks == 1
    assert result.risks[0].risk_type == "termination"


def test_qa_result_model() -> None:
    result = QAResult(
        answer="The termination period is 30 days.",
        confidence=0.92,
        sources=[
            Source(text="Either party may terminate...", page=5, relevance_score=0.89)
        ],
    )
    assert result.confidence == 0.92
    assert len(result.sources) == 1


def test_summary_result_model() -> None:
    result = SummaryResult(
        summary="This is an NDA between two parties.",
        key_points=["2-year term", "Mutual obligations"],
        parties=["Company A", "Company B"],
        contract_type="NDA",
    )
    assert result.contract_type == "NDA"
    assert len(result.key_points) == 2

"""Risk detection agent for identifying contract risks."""

from typing import Any

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel

from config.settings import settings


class Risk(BaseModel):
    """A single identified risk in a contract."""

    risk_type: str
    severity: str  # low, medium, high, critical
    clause_text: str
    explanation: str
    recommendation: str
    page: int | None = None


class RiskAnalysisResult(BaseModel):
    """Structured result from risk analysis."""

    risks: list[Risk]
    summary: str
    total_risks: int


RISK_SYSTEM_PROMPT = """You are a legal risk analysis expert. Analyze the provided contract clauses \
and identify potential risks. For each risk, provide:
- risk_type: category (e.g., liability, termination, IP, confidentiality, indemnification, non-compete, payment, data_privacy)
- severity: one of "low", "medium", "high", "critical"
- clause_text: the exact clause text that poses the risk
- explanation: why this is risky
- recommendation: how to mitigate it

Be thorough but concise. Focus on actionable insights.

Respond in JSON format:
{
  "risks": [
    {
      "risk_type": "...",
      "severity": "...",
      "clause_text": "...",
      "explanation": "...",
      "recommendation": "...",
      "page": null
    }
  ],
  "summary": "Brief summary of overall risk profile",
  "total_risks": 0
}"""


class RiskAgent:
    """Agent that identifies potential risks in contract text."""

    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_CHAT_MODEL

    def analyze(self, context: str, page_numbers: list[int] | None = None) -> RiskAnalysisResult:
        """Analyze contract text for risks."""
        logger.info("Risk agent analyzing contract...")

        user_msg = f"Analyze the following contract sections for legal risks:\n\n{context}"
        if page_numbers:
            user_msg += f"\n\nSource pages: {page_numbers}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RISK_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=2000,
        )

        raw = response.choices[0].message.content or "{}"

        import json

        data = json.loads(raw)
        result = RiskAnalysisResult(
            risks=[Risk(**r) for r in data.get("risks", [])],
            summary=data.get("summary", ""),
            total_risks=data.get("total_risks", len(data.get("risks", []))),
        )

        logger.info(f"Risk agent found {result.total_risks} risks")
        return result

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph-compatible call interface."""
        context = state.get("context", "")
        pages = state.get("page_numbers", [])
        result = self.analyze(context, pages)
        return {**state, "result": result.model_dump(), "agent": "risk"}

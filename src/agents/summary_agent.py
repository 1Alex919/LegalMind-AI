"""Summary agent for generating concise contract summaries."""

from typing import Any

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel

from config.settings import settings


class SummaryResult(BaseModel):
    """Structured result from summary agent."""

    summary: str
    key_points: list[str]
    parties: list[str]
    contract_type: str


SUMMARY_SYSTEM_PROMPT = """You are a legal document summarization expert. Create concise summaries \
of legal contracts that cover:
- Key parties and their roles
- Main obligations and rights
- Important dates and deadlines
- Financial terms
- Notable clauses or unusual provisions

Respond in JSON format:
{
  "summary": "5-7 sentence summary of the contract",
  "key_points": ["point 1", "point 2", "..."],
  "parties": ["Party A", "Party B"],
  "contract_type": "NDA / Employment / SaaS / etc."
}"""


class SummaryAgent:
    """Agent that generates concise contract summaries."""

    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_CHAT_MODEL

    def summarize(self, context: str) -> SummaryResult:
        """Generate a summary of contract text."""
        logger.info("Summary agent generating summary...")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Summarize the following contract:\n\n{context}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1000,
        )

        raw = response.choices[0].message.content or "{}"

        import json

        data = json.loads(raw)
        result = SummaryResult(
            summary=data.get("summary", ""),
            key_points=data.get("key_points", []),
            parties=data.get("parties", []),
            contract_type=data.get("contract_type", "Unknown"),
        )

        logger.info(f"Summary agent: {result.contract_type}, {len(result.key_points)} key points")
        return result

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph-compatible call interface."""
        context = state.get("context", "")
        result = self.summarize(context)
        return {**state, "result": result.model_dump(), "agent": "summary"}

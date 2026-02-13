"""Q&A agent for answering questions about contracts with source citations."""

from typing import Any

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel

from config.settings import settings


class Source(BaseModel):
    """A source citation for an answer."""

    text: str
    page: int | str | None = None
    relevance_score: float = 0.0


class QAResult(BaseModel):
    """Structured result from Q&A agent."""

    answer: str
    confidence: float
    sources: list[Source]


QA_SYSTEM_PROMPT = """You are a legal document Q&A assistant. Answer questions about contracts \
based strictly on the provided context. Always:
- Cite the specific source sections used
- State your confidence level (0.0 to 1.0)
- Say "I don't have enough information in the provided context" if the \
answer isn't clearly supported by the sources
- Use precise legal terminology where appropriate

Respond in JSON format:
{
  "answer": "Your detailed answer here",
  "confidence": 0.85,
  "sources": [
    {
      "text": "Exact quoted text from the context",
      "page": null,
      "relevance_score": 0.9
    }
  ]
}"""


class QAAgent:
    """Agent that answers questions about contracts with citations."""

    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_CHAT_MODEL

    def answer(
        self,
        question: str,
        context: str,
        page_numbers: list[int] | None = None,
    ) -> QAResult:
        """Answer a question based on contract context."""
        logger.info(f"QA agent answering: '{question[:60]}...'")

        user_msg = (
            f"Context from the contract:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Provide a clear answer with source citations."
        )
        if page_numbers:
            user_msg += f"\nSource pages: {page_numbers}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1500,
        )

        raw = response.choices[0].message.content or "{}"

        import json

        data = json.loads(raw)
        result = QAResult(
            answer=data.get("answer", "I couldn't generate an answer."),
            confidence=data.get("confidence", 0.0),
            sources=[Source(**s) for s in data.get("sources", [])],
        )

        logger.info(f"QA agent confidence: {result.confidence:.2f}")
        return result

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph-compatible call interface."""
        question = state.get("question", "")
        context = state.get("context", "")
        pages = state.get("page_numbers", [])
        result = self.answer(question, context, pages)
        return {**state, "result": result.model_dump(), "agent": "qa"}

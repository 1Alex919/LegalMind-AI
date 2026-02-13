"""Synthetic test data generation for evaluation."""

import json
from pathlib import Path

from loguru import logger
from openai import OpenAI

from config.settings import settings

SYNTH_PROMPT = """Given the following contract text, generate {n} question-answer pairs \
that could be used to evaluate a legal document Q&A system.

For each pair, provide:
- question: A natural question someone might ask about this contract
- answer: The correct answer based on the contract text
- context: The specific passage from the contract that supports the answer

Return as JSON:
{{
  "samples": [
    {{
      "question": "...",
      "answer": "...",
      "context": "..."
    }}
  ]
}}

Contract text:
{text}"""


def generate_synthetic_testset(
    contract_text: str,
    n: int = 10,
    output_path: str | Path | None = None,
) -> list[dict]:
    """Generate synthetic evaluation data from contract text.

    Args:
        contract_text: Full contract text to generate questions from.
        n: Number of question-answer pairs to generate.
        output_path: Optional path to save the testset JSON.

    Returns:
        List of dicts with 'question', 'answer', 'context' keys.
    """
    logger.info(f"Generating {n} synthetic test samples...")

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # Truncate to fit context window
    max_chars = 12000
    truncated = contract_text[:max_chars]

    response = client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at creating evaluation datasets for legal AI systems.",
            },
            {
                "role": "user",
                "content": SYNTH_PROMPT.format(n=n, text=truncated),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=3000,
    )

    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)
    samples = data.get("samples", [])

    logger.info(f"Generated {len(samples)} synthetic samples")

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"samples": samples}, indent=2))
        logger.info(f"Saved testset to {path}")

    return samples

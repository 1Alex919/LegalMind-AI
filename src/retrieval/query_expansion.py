"""Query expansion with HyDE and multi-query generation."""

from loguru import logger
from openai import OpenAI

from config.settings import settings


class QueryExpander:
    """Expand queries using HyDE and multi-query techniques."""

    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_CHAT_MODEL

    def hyde(self, query: str) -> str:
        """Hypothetical Document Embeddings: generate a hypothetical answer
        that can be used for better semantic search."""
        logger.info(f"HyDE expansion for: '{query[:60]}...'")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Given the question, write a short hypothetical passage "
                        "that would answer this question in the context of a "
                        "legal contract. The passage should sound like it comes "
                        "from an actual contract clause."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=200,
        )

        hypothetical = response.choices[0].message.content or ""
        logger.debug(f"HyDE generated: '{hypothetical[:80]}...'")
        return hypothetical

    def multi_query(self, query: str) -> list[str]:
        """Generate multiple query variations for broader retrieval."""
        logger.info(f"Multi-query expansion for: '{query[:60]}...'")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate 3 alternative versions of this question that "
                        "could help retrieve relevant information from a legal "
                        "contract. Each version should approach the question from "
                        "a different angle.\n\n"
                        "Return only the 3 questions, one per line."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=300,
        )

        text = response.choices[0].message.content or ""
        queries = [q.strip().lstrip("0123456789.-) ") for q in text.strip().split("\n") if q.strip()]
        queries = queries[:3]

        logger.info(f"Generated {len(queries)} query variations")
        return queries

    def translate_query(self, query: str, target_language: str) -> str:
        """Translate query to the target language for cross-language retrieval."""
        logger.info(f"Translating query to '{target_language}': '{query[:60]}...'")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Translate the following text to {target_language}. "
                        "Keep legal terminology accurate. "
                        "Return only the translation, nothing else."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=300,
        )

        translated = response.choices[0].message.content or query
        logger.info(f"Translated: '{translated[:80]}...'")
        return translated.strip()

    def expand(self, query: str, use_hyde: bool = True, use_multi: bool = True) -> list[str]:
        """Expand query using all enabled techniques.

        Returns list of queries to search (original + expansions).
        """
        queries = [query]

        if use_hyde:
            queries.append(self.hyde(query))

        if use_multi:
            queries.extend(self.multi_query(query))

        logger.info(f"Total queries after expansion: {len(queries)}")
        return queries

"""LLM provider with OpenAI primary and Ollama fallback."""

from typing import Any

from loguru import logger
from openai import OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings
from src.llm.rate_limiter import RateLimiter


class LLMProvider:
    """Unified LLM interface with automatic fallback from OpenAI to Ollama."""

    def __init__(self) -> None:
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.rate_limiter = RateLimiter(max_requests=50, window_seconds=60)
        self._active_provider = "openai"
        self._ollama_available: bool | None = None

    def _check_ollama(self) -> bool:
        """Check if Ollama is running locally."""
        if self._ollama_available is not None:
            return self._ollama_available

        try:
            import httpx

            resp = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=2)
            self._ollama_available = resp.status_code == 200
        except Exception:
            self._ollama_available = False

        if self._ollama_available:
            logger.info("Ollama detected as available fallback")
        return self._ollama_available

    def _call_ollama(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> str:
        """Call Ollama as fallback provider."""
        logger.info(f"Using Ollama ({settings.OLLAMA_MODEL}) as fallback")
        # Use OpenAI-compatible endpoint
        ollama_client = OpenAI(
            base_url=f"{settings.OLLAMA_BASE_URL}/v1",
            api_key="ollama",
        )
        response = ollama_client.chat.completions.create(
            model=settings.OLLAMA_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    def _call_openai(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1000,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> str:
        """Call OpenAI with retry logic."""
        self.rate_limiter.wait_if_needed()

        create_kwargs: dict[str, Any] = {
            "model": settings.OPENAI_CHAT_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            create_kwargs["response_format"] = response_format

        response = self.openai_client.chat.completions.create(**create_kwargs)
        self.rate_limiter.record_request()
        self._active_provider = "openai"

        return response.choices[0].message.content or ""

    def invoke(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        max_tokens: int = 1000,
        response_format: dict | None = None,
    ) -> str:
        """Send a prompt to the LLM with automatic fallback.

        Tries OpenAI first, falls back to Ollama on rate limit errors.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            return self._call_openai(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except RateLimitError:
            logger.warning("OpenAI rate limit exceeded after retries")
            if self._check_ollama():
                self._active_provider = "ollama"
                return self._call_ollama(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            raise

    @property
    def active_provider(self) -> str:
        return self._active_provider

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "active_provider": self._active_provider,
            "total_requests": self.rate_limiter.total_requests,
            "current_window_count": self.rate_limiter.current_count,
            "ollama_available": self._ollama_available,
        }

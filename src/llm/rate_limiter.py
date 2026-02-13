"""Request throttling for LLM API calls."""

import time
from collections import deque

from loguru import logger


class RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, max_requests: int = 50, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._total_requests = 0

    def _clean_window(self) -> None:
        now = time.time()
        while self._timestamps and now - self._timestamps[0] > self.window_seconds:
            self._timestamps.popleft()

    @property
    def current_count(self) -> int:
        self._clean_window()
        return len(self._timestamps)

    @property
    def is_rate_limited(self) -> bool:
        return self.current_count >= self.max_requests

    def wait_if_needed(self) -> None:
        """Block until a request slot is available."""
        self._clean_window()
        if len(self._timestamps) >= self.max_requests:
            wait_time = self._timestamps[0] + self.window_seconds - time.time()
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self._clean_window()

    def record_request(self) -> None:
        """Record a new request."""
        self._timestamps.append(time.time())
        self._total_requests += 1

    @property
    def total_requests(self) -> int:
        return self._total_requests

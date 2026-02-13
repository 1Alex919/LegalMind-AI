"""Log agent decision-making paths for transparency."""

import time
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class ReasoningStep:
    """A single step in the reasoning trace."""

    step_name: str
    details: str
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ReasoningTracer:
    """Tracks the reasoning process through the pipeline for explainability."""

    def __init__(self) -> None:
        self.steps: list[ReasoningStep] = []
        self._step_start: float | None = None
        self._current_step: str = ""

    def start_step(self, step_name: str) -> None:
        """Begin timing a reasoning step."""
        self._current_step = step_name
        self._step_start = time.perf_counter()
        logger.debug(f"Reasoning step started: {step_name}")

    def end_step(self, details: str = "") -> None:
        """End the current step and record it."""
        if self._step_start is None:
            return

        duration = (time.perf_counter() - self._step_start) * 1000
        step = ReasoningStep(
            step_name=self._current_step,
            details=details,
            duration_ms=round(duration, 1),
        )
        self.steps.append(step)
        logger.debug(
            f"Reasoning step complete: {self._current_step} ({duration:.1f}ms)"
        )
        self._step_start = None

    def add_step(self, step_name: str, details: str = "") -> None:
        """Add a completed step directly (no timing)."""
        self.steps.append(ReasoningStep(step_name=step_name, details=details))

    @property
    def step_names(self) -> list[str]:
        return [s.step_name for s in self.steps]

    @property
    def trace_summary(self) -> str:
        """Human-readable trace summary."""
        parts = []
        for s in self.steps:
            if s.duration_ms > 0:
                parts.append(f"{s.step_name} ({s.duration_ms:.0f}ms)")
            else:
                parts.append(s.step_name)
        return " -> ".join(parts)

    @property
    def total_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.steps)

    def to_dict(self) -> dict:
        return {
            "steps": [
                {
                    "name": s.step_name,
                    "details": s.details,
                    "duration_ms": s.duration_ms,
                }
                for s in self.steps
            ],
            "summary": self.trace_summary,
            "total_duration_ms": round(self.total_duration_ms, 1),
        }

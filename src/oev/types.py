"""Public input and output types."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any


MAX_OPTIONS = 25


@dataclass(frozen=True)
class Option:
    id: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("option id must be a nonempty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("option description must be a nonempty string")


@dataclass(frozen=True)
class Decision:
    state: str | dict[str, Any] | list[Any]
    question: str
    options: tuple[Option, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.state, (str, dict, list)) or not self.state:
            raise ValueError("state must be a nonempty string, object, or array")
        try:
            json.dumps(self.state, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("state must be finite JSON data") from exc
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a nonempty string")
        if not isinstance(self.options, tuple) or not 2 <= len(self.options) <= MAX_OPTIONS:
            raise ValueError(f"options must be a tuple of 2 to {MAX_OPTIONS} entries")
        if any(not isinstance(option, Option) for option in self.options):
            raise ValueError("every option must be an Option")
        if len({option.id for option in self.options}) != len(self.options):
            raise ValueError("option ids must be unique")


@dataclass(frozen=True)
class DecisionResult:
    selected_id: str
    probabilities: dict[str, float]
    logits: dict[str, float]
    input_tokens: int
    elapsed_seconds: float
    prompt_sha256: str
    model: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_id": self.selected_id,
            "probabilities": self.probabilities,
            "logits": self.logits,
            "input_tokens": self.input_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "prompt_sha256": self.prompt_sha256,
            "model": self.model,
            "probability_status": "conditional on listed options; not calibrated confidence",
        }


def conditional_softmax(logits: list[float]) -> list[float]:
    if len(logits) < 2 or any(not math.isfinite(value) for value in logits):
        raise ValueError("at least two finite logits are required")
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]

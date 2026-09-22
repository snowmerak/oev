"""System One questions evaluated through oev's option probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from threading import BoundedSemaphore
from typing import Any, Mapping

from .engine import DecisionEngine
from .types import Decision, MAX_OPTIONS, Option


@dataclass(frozen=True)
class _Question:
    name: str
    kind: str
    keys: tuple[str, ...]
    legend: dict[str, Any] | None
    decision: Decision | None


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _content(value: Any, field: str, *, nullable: bool = True) -> Any:
    if value is None and nullable:
        return value
    if not isinstance(value, (str, dict, list)):
        raise ValueError(f"{field} must be a string, object, or array")
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain finite JSON data") from exc
    return value


def _instructions(question: Mapping[str, Any], fallback: str) -> str:
    value = _content(question.get("instructions"), "instructions")
    return _render(value) if value not in (None, "", {}, []) else fallback


def _option_text(name: str, description: Any) -> str:
    return name if description in (None, "") else f"{name}: {_render(description)}"


def _prepare_question(name: str, question: Any, state: Any) -> _Question:
    if not isinstance(name, str) or not name:
        raise ValueError("question names must be nonempty strings")
    if not isinstance(question, dict):
        raise ValueError(f"questions.{name} must be an object")
    kind = question.get("type")
    criteria = question.get("criteria")

    if kind == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= MAX_OPTIONS:
            raise ValueError(f"questions.{name}.criteria must have 1 to {MAX_OPTIONS} choices")
        keys = tuple(criteria)
        options = []
        for key, description in criteria.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"questions.{name}.criteria keys must be nonempty strings")
            _content(description, f"questions.{name}.criteria.{key}")
            options.append(Option(key, _option_text(key, description)))
        instructions = _instructions(question, "Choose the option that best applies to the state.")
        legend = None
    elif kind == "noul":
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
            raise ValueError(f"questions.{name}.criteria may contain only true and false")
        for key, description in criteria.items():
            _content(description, f"questions.{name}.criteria.{key}")
        keys = ("false", "true")
        options = [
            Option("false", _option_text("No", criteria.get("false"))),
            Option("true", _option_text("Yes", criteria.get("true"))),
        ]
        instructions = _instructions(question, "Does the state satisfy the true criterion?")
        if question.get("instructions") in (None, "", {}, []) and not criteria:
            raise ValueError(f"questions.{name} needs instructions or criteria")
        legend = None
    elif kind == "score":
        if not isinstance(criteria, list) or not 1 <= len(criteria) <= MAX_OPTIONS:
            raise ValueError(f"questions.{name}.criteria must have 1 to {MAX_OPTIONS} levels")
        keys = tuple(str(index) for index in range(len(criteria)))
        options = []
        for index, description in enumerate(criteria):
            _content(description, f"questions.{name}.criteria[{index}]", nullable=False)
            options.append(Option(str(index), f"Level {index}: {_render(description)}"))
        instructions = _instructions(question, "Rate the state using the ordered levels.")
        legend = dict(zip(keys, criteria))
    else:
        raise ValueError(f"questions.{name}.type must be choice, noul, or score")

    # A single listed choice or level has a deterministic answer and needs no model pass.
    decision = Decision(state, instructions, tuple(options)) if len(options) > 1 else None
    return _Question(name, kind, keys, legend, decision)


def _confidence(probabilities: list[float]) -> float:
    count = len(probabilities)
    return 1.0 if count == 1 else (max(probabilities) - 1 / count) / (1 - 1 / count)


def _score_confidence(probabilities: list[float]) -> float:
    # Distance from the modal level, matching Kev's documented approximation.
    count = len(probabilities)
    if count == 1:
        return 1.0
    mode = max(range(count), key=probabilities.__getitem__)
    return 1 - sum(p * abs(index - mode) for index, p in enumerate(probabilities)) / (count - 1)


class SystemOneService:
    """Evaluate System One requests with one oev decision per nontrivial question."""

    def __init__(
        self,
        engine: DecisionEngine,
        model: str,
        *,
        release_date: str | None = None,
        max_concurrency: int = 4,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        self.engine = engine
        self.model = model
        self.release_date = release_date or date.today().isoformat()
        self.max_concurrency = max_concurrency
        self._inference_slots = BoundedSemaphore(max_concurrency)

    def models(self) -> dict[str, Any]:
        return {
            "models": [
                {
                    "name": self.model,
                    "description": (
                        f"oev System One: choice, noul, score; up to {MAX_OPTIONS} choices or levels. "
                        "Probabilities are conditional on listed options and are not calibrated."
                    ),
                    "release_date": self.release_date,
                }
            ]
        }

    def evaluate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise ValueError("request must be an object")
        if request.get("model") != self.model:
            raise ValueError(f"model must be {self.model!r}")
        state = request.get("state")
        _content(state, "state", nullable=False)
        if not state:
            raise ValueError("state must be nonempty")
        questions = request.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty object")
        prepared = [_prepare_question(name, question, state) for name, question in questions.items()]

        answers: dict[str, dict[str, Any]] = {}
        input_tokens = 0
        for question in prepared:
            if question.decision is None:
                probabilities = [1.0]
            else:
                with self._inference_slots:
                    result = self.engine.decide(question.decision)
                probabilities = [result.probabilities[key] for key in question.keys]
                input_tokens += result.input_tokens
            distribution = dict(zip(question.keys, probabilities))

            if question.kind == "noul":
                answers[question.name] = {"type": "noul", "noul": distribution["true"]}
            elif question.kind == "choice":
                winner = max(question.keys, key=distribution.__getitem__)
                answers[question.name] = {
                    "type": "choice",
                    "choice": winner,
                    "confidence": _confidence(probabilities),
                    "probabilities": distribution,
                }
            else:
                answers[question.name] = {
                    "type": "score",
                    "score": sum(index * p for index, p in enumerate(probabilities)),
                    "confidence": _score_confidence(probabilities),
                    "legend": question.legend,
                    "probabilities": distribution,
                }

        return {
            "model": self.model,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }

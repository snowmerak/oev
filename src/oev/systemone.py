"""System One questions evaluated through oev's option probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from threading import BoundedSemaphore, Event
from typing import Any, Iterator, Mapping

from .engine import DecisionEngine
from .generation import GenerationEngine, GenerationOptions, GenerationResult, PreparedGeneration, ReasoningDelta, template_options
from .output_parsers import detect_tool_parser, parser_name, template_text
from .reasoning import reasoning_metadata
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
    """Share one backend and inference limit across decisions and text generation."""

    capabilities = frozenset({"decision", "generation"})

    def __init__(
        self,
        engine: DecisionEngine,
        model: str,
        *,
        release_date: str | None = None,
        max_concurrency: int = 4,
        tool_call_parser: str = "auto",
        reasoning_parser: str = "auto",
        chat_template_kwargs: dict | None = None,
        chat_template: str | None = None,
        default_max_output_tokens: int = 8192,
        generation_defaults: dict | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        self.engine = engine
        self.model = model
        self.release_date = release_date or date.today().isoformat()
        self.max_concurrency = max_concurrency
        self.generation_defaults = dict(generation_defaults or {})
        self._inference_slots = BoundedSemaphore(max_concurrency)
        self._generation_options = {
            "tool_call_parser": tool_call_parser, "reasoning_parser": reasoning_parser,
            "chat_template_kwargs": chat_template_kwargs, "chat_template": chat_template,
            "default_max_output_tokens": default_max_output_tokens,
        }

    def _generation_engine(self):
        return GenerationEngine(self.engine.backend, max_input_tokens=self.engine.max_input_tokens,
                                **self._generation_options)

    def prepare_generation(
        self, messages: list[dict[str, str]], options: GenerationOptions,
    ) -> PreparedGeneration:
        return self._generation_engine().prepare(messages, options)

    def generate(self, prepared: PreparedGeneration) -> GenerationResult:
        with self._inference_slots:
            return self._generation_engine().generate(prepared)

    def stream_generation(
        self, prepared: PreparedGeneration, cancel: Event,
    ) -> Iterator[str | ReasoningDelta | GenerationResult]:
        # A disconnected request can leave the queue without taking a slot.
        while not cancel.is_set():
            if self._inference_slots.acquire(timeout=0.1):
                break
        else:
            return
        try:
            yield from self._generation_engine().stream(prepared, cancel)
        finally:
            self._inference_slots.release()

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

    def model_metadata(self) -> dict[str, Any]:
        backend = self.engine.backend
        context = getattr(backend, "context_window", None)
        model = getattr(backend, "model", None)
        if context is None and model is not None:
            context = getattr(model.config.get_text_config(), "max_position_embeddings", None)
        metadata = {"task": "generation", "max_input_tokens": self.engine.max_input_tokens,
                    "max_output_tokens": min(32768, context - 1) if context else 32768,
                    "context_length": min(context, self.engine.max_input_tokens) if context else self.engine.max_input_tokens}
        metadata["default_max_output_tokens"] = min(self._generation_options["default_max_output_tokens"], context - 1) if context else self._generation_options["default_max_output_tokens"]
        if self.generation_defaults:
            metadata["generation_defaults"] = self.generation_defaults.copy()
        if context:
            metadata["context_window"] = context
        template = self._generation_options["chat_template"] or template_text(backend.tokenizer)
        profile = parser_name(self._generation_options["tool_call_parser"])
        metadata["tool_call_parser"] = detect_tool_parser(template) if profile == "auto" else profile
        reasoning = parser_name(self._generation_options["reasoning_parser"], reasoning=True)
        defaults = template_options("none", self._generation_options["chat_template_kwargs"] or {}, {}, False)
        capability = reasoning_metadata(template, reasoning, defaults)
        if capability is not None:
            metadata["capabilities"] = {"reasoning": capability}
        return metadata

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
        cached_input_tokens = 0
        for question in prepared:
            if question.decision is None:
                probabilities = [1.0]
            else:
                with self._inference_slots:
                    result = self.engine.decide(question.decision)
                probabilities = [result.probabilities[key] for key in question.keys]
                input_tokens += result.input_tokens
                cached_input_tokens += result.cached_input_tokens
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

        usage = {"input_tokens": input_tokens, "output_tokens": 0}
        if cached_input_tokens:
            usage["input_tokens_details"] = {"cached_tokens": cached_input_tokens}
        return {
            "model": self.model,
            "answers": answers,
            "usage": usage,
        }

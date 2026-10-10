"""Text generation using the same backend as typed decisions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Event
from typing import Iterator, Protocol

from .prompt import Tokenizer


@dataclass(frozen=True)
class GenerationOptions:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    stop: tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 4096:
            raise ValueError("max output tokens must be between 1 and 4096")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if len(self.stop) > 4 or any(not isinstance(item, str) or not item for item in self.stop):
            raise ValueError("stop must contain at most four nonempty strings")


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str


@dataclass(frozen=True)
class PreparedGeneration:
    input_ids: list[int]
    options: GenerationOptions


class GenerationBackend(Protocol):
    tokenizer: Tokenizer
    model_name: str

    def generate_tokens(
        self, input_ids: list[int], options: GenerationOptions, cancel: Event,
    ) -> Iterator[int]: ...


def eos_token_ids(model) -> set[int]:
    value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = getattr(model.config.get_text_config(), "eos_token_id", None)
    return set(value if isinstance(value, (list, tuple)) else [] if value is None else [value])


def sample_tokens(forward, model, input_ids, options, cancel):
    """Keep the KV cache local; enter inference mode separately for each step."""
    import torch

    cache = None
    current_ids = input_ids
    history = list(input_ids)
    eos = eos_token_ids(model)
    for index in range(options.max_new_tokens):
        if cancel.is_set():
            return
        with torch.inference_mode():
            logits, cache = forward(current_ids, cache, len(input_ids) + index)
            logits = logits.float()
            if not torch.isfinite(logits).all():
                raise RuntimeError("model returned nonfinite generation logits")
            if options.temperature == 0:
                token = logits.argmax().item()
            else:
                logits = logits / options.temperature
                if options.top_p < 1:
                    sorted_logits, order = torch.sort(logits, descending=True)
                    remove = sorted_logits.softmax(-1).cumsum(-1) > options.top_p
                    remove[1:] = remove[:-1].clone()
                    remove[0] = False
                    logits[order[remove]] = -float("inf")
                token = torch.multinomial(logits.softmax(-1), 1).item()
        # Do not carry thread-local inference mode across an iterator yield.
        yield token
        if token in eos:
            return
        history.append(token)
        current_ids = [token] if cache is not None else history


class GenerationEngine:
    def __init__(self, backend: GenerationBackend, *, max_input_tokens: int = 4096):
        self.backend = backend
        self.max_input_tokens = max_input_tokens

    def prepare(self, messages: list[dict[str, str]], options: GenerationOptions) -> PreparedGeneration:
        if not callable(getattr(self.backend, "generate_tokens", None)):
            raise NotImplementedError("this backend does not support text generation")
        if not messages:
            raise ValueError("at least one message is required")
        normalized = []
        for message in messages:
            role, content = message.get("role"), message.get("content")
            if role not in {"system", "developer", "user", "assistant"}:
                raise ValueError("messages must use system, developer, user, or assistant roles")
            if not isinstance(content, str) or not content:
                raise ValueError("message content must be nonempty text")
            normalized.append({"role": "system" if role == "developer" else role, "content": content})
        prompt = self.backend.tokenizer.apply_chat_template(
            normalized, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = self.backend.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids or len(ids) > self.max_input_tokens:
            raise ValueError(f"prompt has {len(ids)} tokens; limit is {self.max_input_tokens}; truncation is disabled")
        model = getattr(self.backend, "model", None)
        context = getattr(model.config.get_text_config(), "max_position_embeddings", None) if model is not None else None
        if context is not None and len(ids) + options.max_new_tokens > context:
            raise ValueError(f"prompt plus max output tokens exceeds model context limit {context}")
        return PreparedGeneration(ids, options)

    def stream(self, prepared: PreparedGeneration, cancel: Event) -> Iterator[str | GenerationResult]:
        tokens: list[int] = []
        emitted = ""
        text = ""
        reason = "length"
        stops = prepared.options.stop
        holdback = max((len(item) - 1 for item in stops), default=0)
        model = getattr(self.backend, "model", None)
        eos = eos_token_ids(model) if model is not None else set()
        source = self.backend.generate_tokens(prepared.input_ids, prepared.options, cancel)
        try:
            for token in source:
                if cancel.is_set():
                    return
                tokens.append(token)
                text = self.backend.tokenizer.decode(
                    tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                )
                matches = [text.find(stop) for stop in stops if stop in text]
                if matches:
                    text = text[:min(matches)]
                    reason = "stop"
                    break
                if token in eos:
                    reason = "stop"
                    break
                # Buffer incomplete words/UTF-8 and possible stop prefixes.
                ready = len(text) if text.endswith("\n") else text.rfind(" ") + 1
                replacement = text.find("\ufffd")
                if replacement >= 0:
                    ready = min(ready, replacement)
                ready = min(ready, max(0, len(text) - holdback))
                if ready > len(emitted):
                    yield text[len(emitted):ready]
                    emitted = text[:ready]
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()
        if cancel.is_set():
            return
        if len(tokens) < prepared.options.max_new_tokens:
            reason = "stop"
        if len(text) > len(emitted):
            yield text[len(emitted):]
        yield GenerationResult(text, len(prepared.input_ids), len(tokens), reason)

    def generate(self, prepared: PreparedGeneration) -> GenerationResult:
        for event in self.stream(prepared, Event()):
            if isinstance(event, GenerationResult):
                return event
        raise RuntimeError("generation ended without a result")

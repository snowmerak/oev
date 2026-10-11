"""Text generation using the same backend as typed decisions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import math
import json
import re
from threading import Event
from typing import Iterator, Protocol

from .prompt import Tokenizer
from .prompt_cache import CacheUsage, tracked_tokens
from .output_parsers import (
    REASONING_MARKERS, TOOL_STARTS, detect_tool_parser, parser_name, template_text,
)


@dataclass(frozen=True)
class GenerationOptions:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    stop: tuple[str, ...] = ()
    seed: int | None = None
    top_k: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    logit_bias: dict[int, float] = field(default_factory=dict)
    logprobs: bool = False
    top_logprobs: int = 0
    tools: tuple[dict, ...] = ()
    tool_choice: str = "auto"
    parallel_tool_calls: bool = True
    reasoning_effort: str = "none"
    response_format: dict = field(default_factory=lambda: {"type": "text"})
    tool_call_parser: str = "auto"
    reasoning_parser: str = "auto"
    chat_template_kwargs: dict = field(default_factory=dict)
    chat_template: str | None = None
    reasoning_effort_explicit: bool = False

    def __post_init__(self):
        parser_name(self.tool_call_parser)
        parser_name(self.reasoning_parser, reasoning=True)
        validate_template_kwargs(self.chat_template_kwargs)
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 32768:
            raise ValueError("max output tokens must be between 1 and 32768")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if len(self.stop) > 4 or any(not isinstance(item, str) or not item for item in self.stop):
            raise ValueError("stop must contain at most four nonempty strings")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**64):
            raise ValueError("seed must be an unsigned 64-bit integer")
        if type(self.top_k) is not int or self.top_k < 0 or not math.isfinite(self.min_p) or not 0 <= self.min_p <= 1:
            raise ValueError("invalid top_k or min_p")
        if any(not math.isfinite(value) or not -2 <= value <= 2 for value in (self.presence_penalty, self.frequency_penalty)):
            raise ValueError("presence/frequency penalties must be between -2 and 2")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if any(type(token) is not int or token < 0 or not math.isfinite(bias) or not -100 <= bias <= 100 for token, bias in self.logit_bias.items()):
            raise ValueError("logit_bias must map nonnegative token IDs to biases between -100 and 100")
        if type(self.top_logprobs) is not int or not 0 <= self.top_logprobs <= 20 or (self.top_logprobs and not self.logprobs):
            raise ValueError("top_logprobs requires logprobs=true and must be between 0 and 20")


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    cached_input_tokens: int = 0
    raw_text: str | None = None
    token_logprobs: tuple[dict, ...] = ()
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class PreparedGeneration:
    input_ids: list[int]
    options: GenerationOptions
    reasoning_prefix: str = ""


def validate_template_kwargs(values):
    if not isinstance(values, dict):
        raise ValueError("chat_template_kwargs must be an object")
    reserved = {"messages", "tools", "functions", "tokenize", "add_generation_prompt", "chat_template",
                "return_dict", "return_tensors"}
    if reserved.intersection(values):
        raise ValueError("chat_template_kwargs cannot override messages, tools or template rendering")
    if any(not isinstance(key, str) for key in values):
        raise ValueError("chat_template_kwargs keys must be strings")
    for key in ("enable_thinking", "thinking"):
        if key in values and type(values[key]) is not bool:
            raise ValueError(f"chat_template_kwargs.{key} must be a boolean")
    if "reasoning_effort" in values and not isinstance(values["reasoning_effort"], str):
        raise ValueError("chat_template_kwargs.reasoning_effort must be a string")
    if "enable_thinking" in values and "thinking" in values and values["enable_thinking"] != values["thinking"]:
        raise ValueError("chat_template_kwargs contains conflicting thinking controls")
    json.dumps(values, allow_nan=False)


def template_options(effort, defaults, overrides, explicit):
    def apply(target, values):
        if "reasoning_effort" in values:
            enabled = values["reasoning_effort"] not in {"none", "minimal", "off"}
            target.update(enable_thinking=enabled, thinking=enabled)
        target.update(values)
        if "enable_thinking" in values:
            target["thinking"] = values["enable_thinking"]
        elif "thinking" in values:
            target["enable_thinking"] = values["thinking"]
    enabled = effort not in {"none", "minimal"}
    requested = {"reasoning_effort": effort, "enable_thinking": enabled, "thinking": enabled}
    kwargs = requested.copy()
    apply(kwargs, defaults)
    if explicit:
        kwargs.update(requested)
    apply(kwargs, overrides)
    return kwargs


class GenerationBackend(Protocol):
    tokenizer: Tokenizer
    model_name: str

    def generate_tokens(
        self, input_ids: list[int], options: GenerationOptions, cancel: Event,
    ) -> Iterator[int]: ...


@lru_cache(maxsize=1)
def byte_decoder():
    alphabet = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    missing = [byte for byte in range(256) if byte not in alphabet]
    mapping = {chr(byte): byte for byte in alphabet}
    mapping.update({chr(256 + index): byte for index, byte in enumerate(missing)})
    return mapping


def token_record(tokenizer, token_id, probability):
    token_id = int(token_id)
    value = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    data = value.encode("utf-8")
    llama = getattr(tokenizer, "llama", None)
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    if llama is not None:
        data = llama.detokenize([token_id], special=True)
    elif converter is not None:
        piece = converter(token_id)
        if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece):
            data = bytes([int(piece[3:5], 16)])
        else:
            decoder = getattr(getattr(tokenizer, "backend_tokenizer", None), "decoder", None)
            state = json.loads(decoder.__getstate__()) if decoder is not None else {}
            if state.get("type") == "ByteLevel":
                # Reverse the GPT byte-to-Unicode alphabet, including partial
                # UTF-8 tokens whose individual display strings contain U+FFFD.
                mapping = byte_decoder()
                if all(char in mapping for char in piece):
                    data = bytes(mapping[char] for char in piece)
    return {"token": value, "logprob": probability, "bytes": list(data)}


def reasoning_token_count(tokens, tokenizer, prefix):
    count = 0
    # Marker token sequences are stable special-token boundaries in Qwen and
    # Gemma. Do not re-tokenize the reasoning prose to estimate usage.
    boundaries = [tokenizer.encode(marker, add_special_tokens=False) for marker in TOOL_STARTS]
    for _, opening, closing in REASONING_MARKERS:
        start_marker = tokenizer.encode(opening, add_special_tokens=False)
        end_marker = tokenizer.encode(closing, add_special_tokens=False)
        if not start_marker or not end_marker:
            continue
        active = prefix == opening
        index = 0
        while index < len(tokens):
            if tokens[index:index + len(start_marker)] == start_marker:
                active = True
                index += len(start_marker)
            elif tokens[index:index + len(end_marker)] == end_marker:
                active = False
                index += len(end_marker)
            else:
                boundary = next((marker for marker in boundaries if marker and
                                 tokens[index:index + len(marker)] == marker), None) if active else None
                if boundary:
                    active = False
                    index += len(boundary)
                    continue
                count += int(active)
                index += 1
    return count


def eos_token_ids(model) -> set[int]:
    value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = getattr(model.config.get_text_config(), "eos_token_id", None)
    return set(value if isinstance(value, (list, tuple)) else [] if value is None else [value])


def sample_tokens(forward, model, input_ids, options, cancel):
    """Keep the KV cache local; enter inference mode separately for each step."""
    import torch
    from .sampling import extended_sampling, sample_logits
    if extended_sampling(options):
        import numpy as np
        rng = np.random.default_rng(options.seed)
    generated = []

    cache = None
    current_ids = input_ids
    history = list(input_ids)
    eos = eos_token_ids(model)
    for index in range(options.max_new_tokens):
        if cancel.is_set():
            return
        with torch.inference_mode():
            output = forward(current_ids, cache, len(input_ids) + index)
            if output is None:
                return
            logits, cache = output
            logits = logits.float()
            if not torch.isfinite(logits).all():
                raise RuntimeError("model returned nonfinite generation logits")
            if extended_sampling(options):
                token = sample_logits(logits.cpu().numpy(), options, generated, rng, input_ids)
            elif options.temperature == 0:
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
        generated.append(int(token))
        current_ids = [token] if cache is not None else history


class GenerationEngine:
    def __init__(self, backend: GenerationBackend, *, max_input_tokens: int = 4096,
                 tool_call_parser="auto", reasoning_parser="auto", chat_template_kwargs=None,
                 chat_template=None):
        self.backend = backend
        self.max_input_tokens = max_input_tokens
        self.tool_call_parser = parser_name(tool_call_parser)
        self.reasoning_parser = parser_name(reasoning_parser, reasoning=True)
        self.chat_template_kwargs = chat_template_kwargs or {}
        validate_template_kwargs(self.chat_template_kwargs)
        self.chat_template = chat_template

    def prepare(self, messages: list[dict], options: GenerationOptions) -> PreparedGeneration:
        if not callable(getattr(self.backend, "generate_tokens", None)):
            raise NotImplementedError("this backend does not support text generation")
        if not messages:
            raise ValueError("at least one message is required")
        normalized = []
        for message in messages:
            role, content = message.get("role"), message.get("content")
            if role not in {"system", "developer", "user", "assistant", "tool"}:
                raise ValueError("invalid message role")
            if not isinstance(content, str) or (not content and not message.get("tool_calls") and role != "tool"):
                raise ValueError("message content must be nonempty text")
            normalized.append({**message, "role": "system" if role == "developer" else role, "content": content})
        while len(normalized) > 1 and normalized[0]["role"] == normalized[1]["role"] == "system":
            normalized[0]["content"] += "\n\n" + normalized.pop(1)["content"]
        from .completion import prompt_messages
        template = self.chat_template or options.chat_template or template_text(self.backend.tokenizer)
        tool_parser = self.tool_call_parser if options.tool_call_parser == "auto" else parser_name(options.tool_call_parser)
        if tool_parser == "auto":
            tool_parser = detect_tool_parser(template)
        reasoning_parser = self.reasoning_parser if options.reasoning_parser == "auto" else parser_name(options.reasoning_parser, reasoning=True)
        kwargs = template_options(options.reasoning_effort, self.chat_template_kwargs,
                                  options.chat_template_kwargs,
                                  options.reasoning_effort_explicit or options.reasoning_effort != "none")
        thinking = kwargs["enable_thinking"]
        if reasoning_parser == "auto" and (thinking or (options.tools and options.tool_choice != "none")):
            for family, opening, _ in REASONING_MARKERS:
                if opening in template or (family == "gemma4" and "<|channel>thought" in template):
                    reasoning_parser = family
                    break
        effort = kwargs["reasoning_effort"] if thinking else "none"
        if thinking and effort in {"none", "minimal", "off"}:
            effort = "low"
        options = replace(options, tool_call_parser=tool_parser, reasoning_parser=reasoning_parser,
                          chat_template=self.chat_template or options.chat_template,
                          reasoning_effort=effort)
        normalized = prompt_messages(normalized, options, self.backend.tokenizer)
        if options.tools and options.tool_choice != "none":
            if tool_parser == "none":
                raise ValueError("tool parsing is disabled for this model")
            kwargs["tools"] = list(options.tools)
        if options.chat_template:
            kwargs["chat_template"] = options.chat_template
        prompt = self.backend.tokenizer.apply_chat_template(
            normalized, tokenize=False, add_generation_prompt=True, **kwargs,
        )
        ids = self.backend.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids or len(ids) > self.max_input_tokens:
            raise ValueError(f"prompt has {len(ids)} tokens; limit is {self.max_input_tokens}; truncation is disabled")
        model = getattr(self.backend, "model", None)
        vocabulary = getattr(self.backend.tokenizer, "vocab_size", None)
        if model is not None:
            vocabulary = getattr(model.config.get_text_config(), "vocab_size", vocabulary)
        llama = getattr(self.backend, "llama", None)
        if llama is not None:
            vocabulary = llama.n_vocab()
        if vocabulary is not None and any(token >= vocabulary for token in options.logit_bias):
            raise ValueError("logit_bias token ID is outside model vocabulary")
        context = getattr(self.backend, "context_window", None)
        if context is None and model is not None:
            context = getattr(model.config.get_text_config(), "max_position_embeddings", None)
        if context is not None and len(ids) + options.max_new_tokens > context:
            raise ValueError(f"prompt plus max output tokens exceeds model context limit {context}")
        prefix = next((opening for _, opening, _ in REASONING_MARKERS if prompt.rstrip().endswith(opening.rstrip())), "")
        return PreparedGeneration(ids, options, prefix)

    def stream(self, prepared: PreparedGeneration, cancel: Event) -> Iterator[str | GenerationResult]:
        tokens: list[int] = []
        token_logprobs = []
        emitted = ""
        text = ""
        reason = "length"
        stops = prepared.options.stop
        holdback = max((len(item) - 1 for item in stops), default=0)
        model = getattr(self.backend, "model", None)
        eos = eos_token_ids(model) if model is not None else set()
        is_end_token = getattr(self.backend, "is_end_token", None)
        source = self.backend.generate_tokens(prepared.input_ids, prepared.options, cancel)
        usage = CacheUsage()
        try:
            for token in tracked_tokens(source, usage):
                if cancel.is_set():
                    return
                tokens.append(token)
                end_token = token in eos or (is_end_token is not None and is_end_token(token))
                if hasattr(token, "logprob") and not end_token:
                    record = lambda t, p: token_record(self.backend.tokenizer, t, p)
                    token_logprobs.append({**record(token, token.logprob), "top_logprobs": [record(t, p) for t, p in token.top]})
                text = self.backend.tokenizer.decode(
                    tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                )
                matches = [text.find(stop) for stop in stops if stop in text]
                if matches:
                    text = text[:min(matches)]
                    reason = "stop"
                    break
                if end_token:
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
        # Gemma's tool-call closing marker is also an EOS token. Keep control
        # markers for protocol parsing; display text already excludes EOS.
        raw_text = prepared.reasoning_prefix + self.backend.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if stops:
            matches = [raw_text.find(stop) for stop in stops if stop in raw_text]
            if matches:
                raw_text = raw_text[:min(matches)]
        content_tokens = tokens[:-1] if tokens and (tokens[-1] in eos or (is_end_token and is_end_token(tokens[-1]))) else tokens
        reasoning_tokens = reasoning_token_count(content_tokens, self.backend.tokenizer, prepared.reasoning_prefix) if prepared.options.reasoning_effort != "none" else 0
        yield GenerationResult(text, len(prepared.input_ids), len(tokens), reason, usage.cached_tokens, raw_text, tuple(token_logprobs), reasoning_tokens)

    def generate(self, prepared: PreparedGeneration) -> GenerationResult:
        for event in self.stream(prepared, Event()):
            if isinstance(event, GenerationResult):
                return event
        raise RuntimeError("generation ended without a result")

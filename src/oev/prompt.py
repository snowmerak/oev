"""Prompt and answer-token validation shared by every backend."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from .types import Decision

LETTERS = "ABCDEFGHIJKLMNOPQRST"
SYSTEM_PROMPT = (
    "Apply the question to the supplied state. Choose exactly one listed option. "
    "Reply with only its uppercase letter, without explanation or reasoning."
)


class Tokenizer(Protocol):
    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str: ...
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, token_ids: list[int], **kwargs: object) -> str: ...


def prepare(tokenizer: Tokenizer, decision: Decision, max_input_tokens: int) -> tuple[list[int], list[int], str]:
    if max_input_tokens < 1:
        raise ValueError("max_input_tokens must be positive")
    payload = {
        "state": decision.state,
        "question": decision.question,
        "options": [
            {"letter": LETTERS[i], "description": option.description}
            for i, option in enumerate(decision.options)
        ],
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_input_tokens:
        raise ValueError(f"prompt has {len(ids)} tokens; limit is {max_input_tokens}; truncation is disabled")
    slots: list[int] = []
    for letter in LETTERS[: len(decision.options)]:
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(token_ids) != 1 or tokenizer.decode(token_ids) != letter:
            raise ValueError(f"answer letter {letter!r} is not one exact token for this tokenizer")
        if tokenizer.encode(prompt + letter, add_special_tokens=False) != ids + token_ids:
            raise ValueError(f"answer letter {letter!r} merges with the prompt boundary")
        slots.append(token_ids[0])
    if len(slots) != len(set(slots)):
        raise ValueError("answer letters map to duplicate token ids")
    return ids, slots, hashlib.sha256(prompt.encode("utf-8")).hexdigest()

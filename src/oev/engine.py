"""Model-independent decision API."""

from __future__ import annotations

from typing import Protocol
import time

from .prompt import Tokenizer, prepare
from .types import Decision, DecisionResult, conditional_softmax


class LogitBackend(Protocol):
    tokenizer: Tokenizer
    model_name: str

    def selected_logits(self, input_ids: list[int], answer_token_ids: list[int]) -> list[float]: ...


class DecisionEngine:
    def __init__(self, backend: LogitBackend, *, max_input_tokens: int = 4096):
        self.backend = backend
        self.max_input_tokens = max_input_tokens

    @classmethod
    def from_pretrained(
        cls,
        model: str = "google/gemma-4-E2B-it",
        *,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        max_input_tokens: int = 4096,
    ) -> "DecisionEngine":
        if device == "hybrid-cuda":
            from .hybrid import HybridGemma4Backend

            if dtype not in {"auto", "bfloat16"}:
                raise ValueError("hybrid-cuda requires bfloat16")
            backend = HybridGemma4Backend.load(model, revision=revision)
        else:
            from .hf import HuggingFaceBackend

            backend = HuggingFaceBackend.load(model, revision=revision, device=device, dtype=dtype)

        return cls(backend, max_input_tokens=max_input_tokens)

    def decide(self, decision: Decision) -> DecisionResult:
        started = time.perf_counter()
        ids, slots, prompt_hash = prepare(self.backend.tokenizer, decision, self.max_input_tokens)
        scores = self.backend.selected_logits(ids, slots)
        if len(scores) != len(decision.options):
            raise RuntimeError("backend returned the wrong number of logits")
        probabilities = conditional_softmax(scores)
        option_ids = [option.id for option in decision.options]
        winner = max(range(len(scores)), key=lambda i: scores[i])
        return DecisionResult(
            selected_id=option_ids[winner],
            probabilities=dict(zip(option_ids, probabilities)),
            logits=dict(zip(option_ids, scores)),
            input_tokens=len(ids),
            elapsed_seconds=time.perf_counter() - started,
            prompt_sha256=prompt_hash,
            model=self.backend.model_name,
        )

import math

import pytest

from oev import Decision, DecisionEngine, Option
from oev.prompt import prepare
from oev.types import conditional_softmax


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is False
        return "\n".join(message["content"] for message in messages) + "\nANSWER: "

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


class CountingBackend:
    def __init__(self):
        self.tokenizer = CharacterTokenizer()
        self.model_name = "fake-model"
        self.calls = []

    def selected_logits(self, input_ids, answer_token_ids):
        self.calls.append((input_ids, answer_token_ids))
        return [0.0, 2.0, -1.0][: len(answer_token_ids)]


def test_one_readout_maps_scores_to_runtime_options():
    backend = CountingBackend()
    engine = DecisionEngine(backend)
    decision = Decision(
        state={"ticket": "refund", "days": 9},
        question="Which queue?",
        options=(
            Option("account", "Account support"),
            Option("billing", "Billing support"),
            Option("other", "Other support"),
        ),
    )
    result = engine.decide(decision)
    assert result.selected_id == "billing"
    assert result.logits == {"account": 0.0, "billing": 2.0, "other": -1.0}
    assert math.isclose(sum(result.probabilities.values()), 1.0)
    assert len(backend.calls) == 1
    assert backend.calls[0][1] == [ord("A"), ord("B"), ord("C")]
    assert result.input_tokens == len(backend.calls[0][0])
    assert len(result.prompt_sha256) == 64


def test_rejects_ambiguous_answer_tokenization():
    class MergingTokenizer(CharacterTokenizer):
        def encode(self, text, add_special_tokens=False):
            if text.endswith("ANSWER: A"):
                return super().encode(text[:-1])[:-1] + [999]
            return super().encode(text)

    decision = Decision("state", "question", (Option("a", "one"), Option("b", "two")))
    with pytest.raises(ValueError, match="merges with the prompt boundary"):
        prepare(MergingTokenizer(), decision, 4096)


def test_rejects_bad_inputs_and_nonfinite_scores():
    with pytest.raises(ValueError, match="unique"):
        Decision("state", "question", (Option("same", "one"), Option("same", "two")))
    with pytest.raises(ValueError, match="finite"):
        conditional_softmax([0.0, float("nan")])
    decision = Decision("state", "question", (Option("a", "one"), Option("b", "two")))
    with pytest.raises(ValueError, match="truncation is disabled"):
        prepare(CharacterTokenizer(), decision, 2)


def test_softmax_handles_large_logits():
    assert conditional_softmax([10000.0, 9999.0]) == pytest.approx([0.7310586, 0.2689414])


def test_hybrid_device_uses_the_gemma_backend(monkeypatch):
    from oev.hybrid import HybridGemma4Backend

    backend = CountingBackend()
    monkeypatch.setattr(
        HybridGemma4Backend,
        "load",
        lambda source, *, revision: backend,
    )
    engine = DecisionEngine.from_pretrained(
        "gemma", revision="commit", device="hybrid-cuda", dtype="bfloat16"
    )
    assert engine.backend is backend
    with pytest.raises(ValueError, match="requires bfloat16"):
        DecisionEngine.from_pretrained("gemma", device="hybrid-cuda", dtype="float32")

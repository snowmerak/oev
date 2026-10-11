"""Hugging Face causal-model backend for Gemma 4 and compatible models."""

from __future__ import annotations

import inspect
from copy import deepcopy
from contextlib import nullcontext
from threading import Lock

from .prompt_cache import PrefixCache, state_bytes


class HuggingFaceBackend:
    def __init__(self, model, tokenizer, model_name: str):
        import torch

        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_name = model_name
        self._decompression_lock = Lock()
        self._cache_lock = Lock()
        self.prompt_cache = PrefixCache()
        self._keep_last_only = "logits_to_keep" in inspect.signature(model.forward).parameters
        # Gemma 4's causal head is a linear projection followed only by optional
        # tanh softcapping. Selecting its weight rows avoids projecting 262k tokens.
        self._selective_gemma4 = (
            type(model).__name__ in {
                "Gemma4ForCausalLM",
                "Gemma4ForConditionalGeneration",
                "Gemma4UnifiedForCausalLM",
                "Gemma4UnifiedForConditionalGeneration",
            }
            and isinstance(model.lm_head, torch.nn.Linear)
        )

    @classmethod
    def load(
        cls, source: str, *, revision: str | None = None, device: str = "auto", dtype: str = "auto",
        local_files_only: bool = False,
    ) -> "HuggingFaceBackend":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the Hugging Face backend with: uv sync --extra hf") from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable")
        if dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be auto, float32, float16, or bfloat16")
        common = {"revision": revision, "trust_remote_code": False, "local_files_only": local_files_only}
        tokenizer = AutoTokenizer.from_pretrained(source, **common)
        model = AutoModelForCausalLM.from_pretrained(
            source,
            dtype="auto" if dtype == "auto" else getattr(torch, dtype),
            device_map=device,
            low_cpu_mem_usage=True,
            **common,
        )
        resolved_revision = revision or getattr(model.config, "_commit_hash", None)
        model_name = f"{source}@{resolved_revision}" if resolved_revision else source
        return cls(model, tokenizer, model_name)

    def configure_prompt_cache(self, max_bytes):
        with self._cache_lock:
            self.prompt_cache = PrefixCache(max_bytes)

    def _cached_prefill(self, input_ids, answer_token_ids=None, cancel=None):
        start, snapshot = self.prompt_cache.lookup(input_ids)
        cache = deepcopy(snapshot) if snapshot is not None else None
        end = len(input_ids) - 1
        while start < end:
            if cancel is not None and cancel.is_set():
                return None
            stop = min(((start // self.prompt_cache.chunk_tokens) + 1) * self.prompt_cache.chunk_tokens, end)
            # Prefill only needs decoder state; avoid Gemma's vocabulary head.
            _, cache = self._forward_step(input_ids[start:stop], cache, stop, [])
            if cache is None:
                raise RuntimeError("this HF model does not return a reusable KV cache")
            self.prompt_cache.store(input_ids[:stop], cache, state_bytes(cache), deepcopy)
            start = stop
        if cancel is not None and cancel.is_set():
            return None
        return self._forward_step(input_ids[-1:], cache, len(input_ids), answer_token_ids)

    def selected_logits(self, input_ids: list[int], answer_token_ids: list[int]) -> list[float]:
        import torch
        import torch.nn.functional as F

        if self.prompt_cache.max_bytes:
            with self._cache_lock, torch.inference_mode():
                logits, _ = self._cached_prefill(input_ids, answer_token_ids)
                return logits.float().cpu().tolist()

        device = next(self.model.parameters()).device
        tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
        kwargs = {
            "input_ids": tokens,
            "attention_mask": torch.ones_like(tokens),
            "use_cache": False,
            "return_dict": True,
        }
        if self._keep_last_only:
            kwargs["logits_to_keep"] = 1
        # CT's first forward mutates shared weights. Waiting callers must check
        # the hooks again after acquiring the lock; later forwards can overlap.
        initialization = self._decompression_lock if hasattr(self.model, "ct_decompress_hook") else nullcontext()
        with initialization, torch.inference_mode():
            # Compressed-tensors decompresses packed weights in a root pre-hook
            # on the first forward. Calling only the decoder would bypass it.
            if self._selective_gemma4 and not self.model._forward_pre_hooks:
                kwargs.pop("logits_to_keep", None)
                hidden = self.model.model(**kwargs).last_hidden_state[:, -1, :]
                slots = torch.tensor(answer_token_ids, dtype=torch.long, device=device)
                weight = self.model.lm_head.weight.index_select(0, slots)
                bias = self.model.lm_head.bias
                if bias is not None:
                    bias = bias.index_select(0, slots)
                selected_tensor = F.linear(hidden, weight, bias)[0]
                text_config = self.model.config.get_text_config()
                cap = text_config.final_logit_softcapping
                if cap is not None:
                    selected_tensor = torch.tanh(selected_tensor / cap) * cap
            else:
                logits = self.model(**kwargs).logits[0, -1]
                selected_tensor = logits[answer_token_ids]
            selected = selected_tensor.float().cpu().tolist()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        return selected

    def generate_tokens(self, input_ids, options, cancel):
        from .generation import sample_tokens

        if not self.prompt_cache.max_bytes:
            yield from sample_tokens(self._generation_step, self.model, input_ids, options, cancel)
            return
        while not cancel.is_set():
            if self._cache_lock.acquire(timeout=0.1):
                break
        else:
            return
        try:
            def forward(ids, cache, length):
                if cache is None:
                    return self._cached_prefill(ids, cancel=cancel)
                return self._forward_step(ids, cache, length)

            yield from sample_tokens(forward, self.model, input_ids, options, cancel)
        finally:
            self._cache_lock.release()

    def _generation_step(self, input_ids, cache, sequence_length):
        return self._forward_step(input_ids, cache, sequence_length)

    def _forward_step(self, input_ids, cache, sequence_length, answer_token_ids=None):
        import torch
        import torch.nn.functional as F

        device = next(self.model.parameters()).device
        kwargs = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, sequence_length), dtype=torch.long, device=device),
            "past_key_values": cache,
            "use_cache": True,
            "return_dict": True,
        }
        if self._keep_last_only:
            kwargs["logits_to_keep"] = 1
        initialization = self._decompression_lock if hasattr(self.model, "ct_decompress_hook") else nullcontext()
        with initialization:
            if answer_token_ids is not None and self._selective_gemma4 and not self.model._forward_pre_hooks:
                kwargs.pop("logits_to_keep", None)
                output = self.model.model(**kwargs)
                slots = torch.tensor(answer_token_ids, dtype=torch.long, device=device)
                bias = self.model.lm_head.bias
                selected = F.linear(output.last_hidden_state[:, -1, :],
                                    self.model.lm_head.weight.index_select(0, slots),
                                    None if bias is None else bias.index_select(0, slots))[0]
                cap = self.model.config.get_text_config().final_logit_softcapping
                if cap is not None:
                    selected = torch.tanh(selected / cap) * cap
                return selected, output.past_key_values
            output = self.model(**kwargs)
        logits = output.logits[0, -1]
        return (logits if answer_token_ids is None else logits[answer_token_ids]), output.past_key_values

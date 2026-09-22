"""Hugging Face causal-model backend for Gemma 4 and compatible models."""

from __future__ import annotations

import inspect


class HuggingFaceBackend:
    def __init__(self, model, tokenizer, model_name: str):
        import torch

        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_name = model_name
        self._keep_last_only = "logits_to_keep" in inspect.signature(model.forward).parameters
        # Gemma 4's causal head is a linear projection followed only by optional
        # tanh softcapping. Selecting its weight rows avoids projecting 262k tokens.
        self._selective_gemma4 = (
            type(model).__name__ in {"Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"}
            and isinstance(model.lm_head, torch.nn.Linear)
        )

    @classmethod
    def load(
        cls, source: str, *, revision: str | None = None, device: str = "auto", dtype: str = "auto"
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
        common = {"revision": revision, "trust_remote_code": False}
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

    def selected_logits(self, input_ids: list[int], answer_token_ids: list[int]) -> list[float]:
        import torch
        import torch.nn.functional as F

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
        with torch.inference_mode():
            if self._selective_gemma4:
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

"""Gemma 4 backend with CUDA decoder layers and CPU embeddings/output head."""

from __future__ import annotations


class HybridGemma4Backend:
    def __init__(self, model, tokenizer, model_name: str):
        import torch

        if type(model).__name__ != "Gemma4ForConditionalGeneration":
            raise ValueError("hybrid-cuda requires a Gemma4ForConditionalGeneration checkpoint")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.text_model = model.model.language_model
        for name in (
            "layers",
            "per_layer_model_projection",
            "per_layer_projection_norm",
            "norm",
            "rotary_emb",
        ):
            getattr(self.text_model, name).to("cuda")
        self.cuda_weight_bytes = torch.cuda.memory_allocated()

    @classmethod
    def load(cls, source: str, *, revision: str | None = None) -> "HybridGemma4Backend":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the Hugging Face backend with: uv sync --extra hf") from exc

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("hybrid-cuda requires a CUDA GPU with bfloat16 support")
        common = {"revision": revision, "trust_remote_code": False}
        tokenizer = AutoTokenizer.from_pretrained(source, **common)
        model = AutoModelForCausalLM.from_pretrained(
            source, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True, **common
        )
        resolved_revision = revision or getattr(model.config, "_commit_hash", None)
        model_name = f"{source}@{resolved_revision}" if resolved_revision else source
        return cls(model, tokenizer, model_name)

    def selected_logits(self, input_ids: list[int], answer_token_ids: list[int]) -> list[float]:
        import torch
        import torch.nn.functional as F

        tokens = torch.tensor([input_ids], dtype=torch.long)
        slots = torch.tensor(answer_token_ids, dtype=torch.long)
        with torch.inference_mode():
            embeddings = self.text_model.embed_tokens(tokens)
            per_layer_inputs = self.text_model.get_per_layer_inputs(tokens, embeddings)
            hidden = self.text_model(
                inputs_embeds=embeddings.to("cuda"),
                per_layer_inputs=per_layer_inputs.to("cuda"),
                attention_mask=torch.ones_like(tokens, device="cuda"),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, -1, :]
            weight = self.model.lm_head.weight.index_select(0, slots)
            bias = self.model.lm_head.bias
            if bias is not None:
                bias = bias.index_select(0, slots)
            selected = F.linear(hidden.cpu(), weight, bias)[0]
            cap = self.model.config.get_text_config().final_logit_softcapping
            if cap is not None:
                selected = torch.tanh(selected / cap) * cap
            return selected.float().tolist()

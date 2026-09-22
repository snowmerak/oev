"""Run Gemma 4 text decisions with CPU embeddings and CUDA decoder layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from oev.engine import DecisionEngine
from oev.io import decision_from_record, iter_jsonl


class HybridGemma4Backend:
    def __init__(self, model, tokenizer, model_name: str):
        import torch

        if type(model).__name__ != "Gemma4ForConditionalGeneration":
            raise ValueError("this runner requires a Gemma4ForConditionalGeneration checkpoint")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("this runner requires a CUDA GPU with bfloat16 support")
    common = {"revision": args.revision, "trust_remote_code": False}
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True, **common
    )
    backend = HybridGemma4Backend(model, tokenizer, f"{args.model}@{args.revision}")
    engine = DecisionEngine(backend, max_input_tokens=args.max_input_tokens)
    print(
        f"loaded in {time.perf_counter() - started:.2f}s; "
        f"CUDA weights {backend.cuda_weight_bytes / 1024**3:.2f} GiB",
        file=sys.stderr,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for number, row in enumerate(iter_jsonl(args.input), 1):
            result = engine.decide(decision_from_record(row)).as_dict()
            result["id"] = row.get("id", str(number))
            print(json.dumps(result, ensure_ascii=False), file=out, flush=True)
            print(
                f"{number}: {result['id']}: {result['selected_id']} "
                f"({result['elapsed_seconds']:.3f}s)",
                file=sys.stderr,
            )
    print(
        f"peak CUDA allocation {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

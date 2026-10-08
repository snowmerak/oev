"""Experimental CT W4A16 probe with temporary per-Linear BF16 weights.

This avoids the default runtime's whole-model decompression. It is a diagnostic
runner, not a fused INT4 kernel or a change to oev's default model loader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import MethodType


MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"
REVISION = "1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee"


def _packed_linear_forward(self, inputs):
    import torch
    import torch.nn.functional as F

    shifts = torch.arange(8, dtype=torch.int32, device=self.weight_packed.device) * 4
    unpacked = ((self.weight_packed.unsqueeze(-1) >> shifts) & 15) - 8
    weight = unpacked.to(self.weight_scale.dtype).reshape(self.out_features, -1, 32)
    weight = (weight * self.weight_scale.unsqueeze(-1)).reshape(self.out_features, self.in_features)
    return F.linear(inputs, weight, self.bias)


def enable_on_demand(model) -> int:
    """Validate every packed layer before replacing its inference-only forward."""
    import torch

    modules = []
    for name, module in model.named_modules():
        if not hasattr(module, "weight_packed"):
            continue
        scheme = module.quantization_scheme
        weights = scheme.weights
        if not (
            isinstance(module, torch.nn.Linear)
            and scheme.format == "pack-quantized"
            and weights.num_bits == 4
            and weights.type == "int"
            and weights.symmetric
            and weights.group_size == 32
            and weights.actorder is None
            and scheme.input_activations is None
            and scheme.output_activations is None
            and module.in_features % 32 == 0
            and module.weight_packed.shape == (module.out_features, module.in_features // 8)
            and module.weight_scale.shape == (module.out_features, module.in_features // 32)
        ):
            raise ValueError(f"unsupported packed quantization in {name!r}; expected symmetric W4A16 group 32")
        modules.append(module)
    if not modules or not hasattr(model, "ct_decompress_hook"):
        raise ValueError("expected a compressed CT model with its initial decompression hook")
    for module in modules:
        module.forward = MethodType(_packed_linear_forward, module)
    model.ct_decompress_hook.remove()
    delattr(model, "ct_decompress_hook")
    return len(modules)


def main() -> None:
    import torch
    from oev import DecisionEngine
    from oev.io import decision_from_record, iter_jsonl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu", "mps"))
    parser.add_argument("--input", type=Path, default=Path("examples/decisions.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    args = parser.parse_args()
    args.events.parent.mkdir(parents=True, exist_ok=True)

    def record(event: str, **details) -> None:
        text = json.dumps({"event": event, **details}, ensure_ascii=False)
        with args.events.open("a", encoding="utf-8") as stream:
            print(text, file=stream)
        print(text, flush=True)

    started = time.perf_counter()
    record("load_start", device=args.device)
    engine = DecisionEngine.from_pretrained(
        args.model, revision=args.revision, device=args.device, dtype="bfloat16",
    )
    record("loaded", seconds=time.perf_counter() - started, model=engine.backend.model_name)
    record("on_demand_installed", packed_modules=enable_on_demand(engine.backend.model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for number, row in enumerate(iter_jsonl(args.input), 1):
            record("inference_start", id=row.get("id", str(number)))
            result = engine.decide(decision_from_record(row)).as_dict()
            result["id"] = row.get("id", str(number))
            print(json.dumps(result, ensure_ascii=False), file=stream, flush=True)
            memory = {}
            if args.device == "cuda":
                memory = {
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                    "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
                }
            record("inference_complete", id=result["id"], selected_id=result["selected_id"],
                   seconds=result["elapsed_seconds"], **memory)
    record("complete", seconds=time.perf_counter() - started)


if __name__ == "__main__":
    main()

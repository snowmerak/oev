"""Run Gemma 4 text decisions with CPU embeddings and CUDA decoder layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from oev.engine import DecisionEngine
from oev.hybrid import HybridGemma4Backend
from oev.io import decision_from_record, iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    args = parser.parse_args()

    import torch
    started = time.perf_counter()
    backend = HybridGemma4Backend.load(args.model, revision=args.revision)
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

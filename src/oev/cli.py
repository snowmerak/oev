"""Run decisions from a JSONL file."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

from .engine import DecisionEngine
from .io import decision_from_record, iter_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read option logits from one model forward pass per decision")
    parser.add_argument("--model", default="google/gemma-4-E2B-it", help="Hugging Face model ID or local path")
    parser.add_argument("--revision", help="Hugging Face commit revision for reproducible runs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--input", type=Path, default=Path("examples/decisions.jsonl"))
    parser.add_argument("--output", type=Path, help="Write JSONL results here (stdout by default)")
    args = parser.parse_args(argv)

    loaded_at = time.perf_counter()
    engine = DecisionEngine.from_pretrained(
        args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
    )
    print(f"loaded {args.model} in {time.perf_counter() - loaded_at:.2f}s", file=sys.stderr)

    with ExitStack() as stack:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            out = stack.enter_context(args.output.open("w", encoding="utf-8"))
        else:
            out = sys.stdout
        for record_number, row in enumerate(iter_jsonl(args.input), 1):
            result = engine.decide(decision_from_record(row)).as_dict()
            result["id"] = row.get("id", str(record_number))
            print(json.dumps(result, ensure_ascii=False), file=out, flush=True)
            print(
                f"{result['id']}: {result['selected_id']} ({result['elapsed_seconds']:.3f}s)",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

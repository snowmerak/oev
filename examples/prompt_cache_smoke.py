"""Opt-in real-model cache experiments and production-backend benchmarks.

Run with the project's locked HF/GGUF environment:
    python examples/prompt_cache_smoke.py --models results/prompt-cache/models.json
Or supply cached sources with --hf-model, --gguf-model, --gemma-tokenizer.
On a Mac, add --mlx-model /path/to/mlx-checkpoint --backend mlx.
Add --production to measure oev's bounded cache implementation (HF/GGUF/MLX).
Results include cache restoration overhead. These are local microbenchmarks,
not HTTP throughput or end-to-end serving benchmarks.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import platform
import statistics
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Lock

import numpy as np

from oev.prompt import prepare
from oev.types import Decision, Option, conditional_softmax


def prompts(tokenizer):
    state = {
        "ticket": "The customer was charged twice and requests a refund.",
        "history": [f"Order {i}: delivered, customer service replied." for i in range(48)],
    }
    options = (Option("billing", "payments and refunds"), Option("shipping", "deliveries"),
               Option("security", "account access"))
    queries = ["Which team should handle this ticket?", "Which team should handle the refund?"]
    prepared = [prepare(tokenizer, Decision(state, question, options), 2048) for question in queries]
    changed = prepare(tokenizer, Decision({**state, "ticket": "My account was hacked."}, queries[0], options), 2048)
    reordered = prepare(tokenizer, Decision(state, queries[0], options[::-1]), 2048)
    cases = [("original", *prepared[0][:2]), ("new_question", *prepared[1][:2]),
             ("reordered_options", *reordered[:2]), ("changed_state", *changed[:2])]
    prefix = []
    for a, b in zip(prepared[0][0], prepared[1][0]):
        if a != b:
            break
        prefix.append(a)
    assert prefix and len(prefix) < len(prepared[0][0])
    return prefix, cases


class HFProbe:
    def __init__(self, source, gemma=False):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.source = source
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        if gemma:
            from transformers import Gemma4ForCausalLM, Gemma4TextConfig

            # The cached real E2B checkpoint exceeds this laptop's VRAM. Exercise
            # actual Gemma4 sliding-cache code with small deterministic random
            # weights; report this as structural evidence, not pretrained-model
            # latency or decision quality.
            torch.manual_seed(17)
            config = Gemma4TextConfig(
                vocab_size=self.tokenizer.vocab_size, hidden_size=128, intermediate_size=256,
                num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                head_dim=32, global_head_dim=32, hidden_size_per_layer_input=0,
                sliding_window=512, layer_types=["sliding_attention", "full_attention"] * 2,
            )
            config._attn_implementation = "sdpa"
            self.model = Gemma4ForCausalLM(config).to(self.device).eval()
            self.source = f"random tiny Gemma4; real tokenizer: {source}"
        else:
            config = AutoConfig.from_pretrained(source, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                source, config=config, local_files_only=True, dtype=self.dtype,
                device_map=self.device, attn_implementation="sdpa",
            ).eval()
        self.checkpoint = None
        self.prefix = []
        self.lock = Lock()
        self.gemma = gemma
        print(f"Loaded HF model on {self.device}: {type(self.model).__name__}", flush=True)

    def context(self):
        return self.torch.inference_mode()

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def step(self, ids, cache, total):
        torch = self.torch
        output = self.model(
            input_ids=torch.tensor([ids], device=self.device),
            attention_mask=torch.ones((1, total), dtype=torch.long, device=self.device),
            past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1,
        )
        return output.logits[0, -1].float().cpu().numpy(), output.past_key_values

    def seed(self, prefix):
        self.prefix = list(prefix)
        with self.context():
            _, self.checkpoint = self.step(prefix, None, len(prefix))
        self.sync()

    def restore(self, ids, cached):
        hit = len(self.prefix) if cached and ids[:len(self.prefix)] == self.prefix else 0
        return copy.deepcopy(self.checkpoint) if hit else None, hit

    def cache_bytes(self):
        with self.context():
            tensors = []
            for layer in self.checkpoint.layers:
                tensors.extend(t for t in (getattr(layer, "keys", None), getattr(layer, "values", None))
                               if t is not None)
            return sum(t.numel() * t.element_size() for t in tensors)


class GGUFProbe:
    def __init__(self, source):
        from oev.gguf import GGUFBackend

        self.backend = GGUFBackend.load(source, device="cpu", context_length=2048)
        self.llama = self.backend.llama
        self.tokenizer = self.backend.tokenizer
        self.checkpoint = None
        self.prefix = []
        self.lock = self.backend._lock
        self.device = "cpu"
        self.source = source

    def context(self):
        return nullcontext()

    def sync(self):
        pass

    def step(self, ids, cache, total):
        self.llama.eval(ids)
        pointer = self.backend._logits()
        if not pointer:
            raise RuntimeError("native context returned no logits")
        return np.ctypeslib.as_array(pointer, shape=(self.llama.n_vocab(),)).copy(), None

    def seed(self, prefix):
        self.prefix = list(prefix)
        self.llama.reset()
        self.llama.eval(prefix)
        self.checkpoint = self.llama.save_state()
        self.llama.reset()

    def restore(self, ids, cached):
        self.llama.reset()
        hit = len(self.prefix) if cached and ids[:len(self.prefix)] == self.prefix else 0
        if hit:
            self.llama.load_state(self.checkpoint)
        return None, hit

    def cache_bytes(self):
        return self.checkpoint.llama_state_size + self.checkpoint.scores.nbytes + self.checkpoint.input_ids.nbytes


class MLXProbe:
    def __init__(self, source):
        from oev.mlx import MLXBackend

        self.backend = MLXBackend.load(source, local_files_only=True)
        self.tokenizer = self.backend.tokenizer
        self.lock = self.backend._lock
        self.mx = self.backend._mx
        self.checkpoint = None
        self.prefix = []
        self.source = source
        self.device = "metal"

    def context(self):
        return self.mx.stream(self.backend._stream)

    def sync(self):
        self.mx.synchronize(self.backend._stream)

    def step(self, ids, cache, total):
        cache = self.backend._make_cache(self.backend.mlx_model) if cache is None else cache
        logits = self.backend._prefill(ids, cache)[0].astype(self.mx.float32)
        self.mx.eval(logits)
        return np.array(logits.tolist(), dtype=np.float32), cache

    def seed(self, prefix):
        self.prefix = list(prefix)
        with self.context():
            _, self.checkpoint = self.step(prefix, None, len(prefix))
        self.sync()

    def restore(self, ids, cached):
        hit = len(self.prefix) if cached and ids[:len(self.prefix)] == self.prefix else 0
        return copy.deepcopy(self.checkpoint) if hit else None, hit

    def cache_bytes(self):
        return sum(entry.nbytes for entry in self.checkpoint)


def read(probe, ids, cached, generate=0):
    # One stored boundary is deliberate: SWA states aren't cropped arbitrarily.
    with probe.lock, probe.context():
        probe.sync()
        start = time.perf_counter()
        cache, hit = probe.restore(ids, cached)
        # Restored native states need a decode to refresh the last logits.
        # The test checkpoint is strictly shorter than all matching test prompts.
        logits, cache = probe.step(ids[hit:], cache, len(ids))
        probe.sync()
        prefill_ms = 1000 * (time.perf_counter() - start)
        first_logits = logits.copy()
        generated = []
        for index in range(generate):
            token = int(logits.argmax())
            generated.append(token)
            if index + 1 < generate:
                logits, cache = probe.step([token], cache, len(ids) + index + 1)
        probe.sync()
    return {"logits": first_logits, "tokens": generated, "ms": prefill_ms, "hit": hit}


def snapshot_smoke(probe, repeats):
    prefix, cases = prompts(probe.tokenizer)
    probe.seed(prefix)
    before = probe.cache_bytes()
    sliding_window = getattr(getattr(probe, "model", None), "config", None)
    sliding_window = getattr(sliding_window, "sliding_window", None)
    if getattr(probe, "gemma", False):
        assert len(prefix) > sliding_window, "SWA test must exceed the actual window"
        assert probe.checkpoint.layers[0].keys.shape[-2] < len(prefix)
    # Warm kernels before measuring; restoration is included in the timed path.
    read(probe, cases[0][1], False)
    read(probe, cases[0][1], True)
    measurements = []
    references = {}
    for name, ids, slots in cases:
        cold = read(probe, ids, False, generate=4)
        warm = read(probe, ids, True, generate=4)
        delta = float(np.max(np.abs(cold["logits"][slots] - warm["logits"][slots])))
        cold_probs = np.array(conditional_softmax(cold["logits"][slots].tolist()))
        warm_probs = np.array(conditional_softmax(warm["logits"][slots].tolist()))
        probability_delta = float(np.max(np.abs(cold_probs - warm_probs)))
        assert np.isfinite(warm["logits"]).all()
        assert warm["hit"] == (0 if name == "changed_state" else len(prefix))
        assert cold_probs.argmax() == warm_probs.argmax(), f"{name}: winner changed"
        assert probability_delta < 0.02, f"{name}: probability error {probability_delta}"
        assert cold["tokens"] == warm["tokens"], f"{name}: greedy generation changed"
        references[name] = cold["tokens"]
        measurements.append({"case": name, "input_tokens": len(ids), "reused_tokens": warm["hit"],
                             "evaluated_tokens": len(ids) - warm["hit"], "max_logit_delta": delta,
                             "max_probability_delta": probability_delta,
                             "greedy_tokens_equal": True})
    ids = cases[1][1]
    timings = {"cold_ms": [], "cached_ms": []}
    for index in range(repeats):
        order = [False, True] if index % 2 == 0 else [True, False]
        for cached in order:
            result = read(probe, ids, cached)
            timings["cached_ms" if cached else "cold_ms"].append(result["ms"])
    cold_median = statistics.median(timings["cold_ms"])
    cached_median = statistics.median(timings["cached_ms"])
    assert probe.cache_bytes() == before
    # A generation must not mutate the saved prefix used by later worker threads.
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [(name, workers.submit(read, probe, ids, True, 4)) for name, ids, _ in cases]
        for name, future in futures:
            assert future.result(timeout=60)["tokens"] == references[name], f"{name}: worker result changed"
    return {"status": "passed", "device": probe.device, "source": probe.source,
            "prefix_tokens": len(prefix), "checkpoint_bytes": before, "checks": measurements,
            "timings": timings, "cold_median_ms": cold_median, "cached_median_ms": cached_median,
            "prefill_speedup": cold_median / cached_median, "worker_reuse_passed": True,
            "weights": "random" if getattr(probe, "gemma", False) else "pretrained",
            "sliding_window": sliding_window}


def paged_smoke(probe):
    from transformers import ContinuousBatchingConfig, GenerationConfig

    _, cases = prompts(probe.tokenizer)
    all_results = []
    for sharing in (False, True):
        config = ContinuousBatchingConfig(
            page_size=64, num_blocks=64, max_batch_tokens=128, max_requests_per_batch=2,
            max_blocks_per_request=0, allow_block_sharing=sharing, auto_switch_to_flash=False,
            use_cuda_graph=False, use_async_batching=False, default_compile_level=0,
        )
        generation = GenerationConfig(max_new_tokens=4, do_sample=False, eos_token_id=None, pad_token_id=0)
        manager = probe.model.init_continuous_batching(
            generation_config=generation, continuous_batching_config=config,
        )
        try:
            manager.warmup()
            manager.start()
            cache = manager.batch_processor.cache
            original = cache.search_prefix_match
            hits = {}

            def search(request_id, ids):
                reused = original(request_id, ids)
                hits.setdefault(request_id, []).append(reused)
                return reused

            cache.search_prefix_match = search
            runs = []
            for name, ids, _ in cases:
                start = time.perf_counter()
                request_id = manager.add_request(ids, request_id=name)
                deadline = time.monotonic() + 60
                while True:
                    result = manager.get_result(request_id=request_id, timeout=0.1)
                    if result is not None:
                        break
                    if not manager.is_running() or time.monotonic() > deadline:
                        raise RuntimeError(f"paged request {name} did not finish")
                    time.sleep(0.001)
                assert result.error is None, result.error
                runs.append({"case": name, "input_tokens": len(ids),
                             "prefix_hits": hits.get(name, []), "tokens": result.generated_tokens,
                             "total_ms": 1000 * (time.perf_counter() - start)})
            all_results.append({"block_sharing_requested": sharing,
                                "prefix_sharing_enabled": cache.use_prefix_sharing, "runs": runs})
        finally:
            manager.stop(hard_stop=True, timeout=15)
            manager.destroy()
    assert [r["tokens"] for r in all_results[0]["runs"]] == [r["tokens"] for r in all_results[1]["runs"]]
    assert any(all_results[1]["runs"][1]["prefix_hits"]), "no prefix reused"
    return {"status": "passed", "device": probe.device, "attention": "paged|sdpa", "runs": all_results}


def production_smoke(name, source, budget, repeats):
    from threading import Event
    from oev import DecisionEngine, GenerationOptions
    from oev.prompt_cache import CacheUsage, track_cache

    options = {"device": "cpu", "context_length": 2048} if name == "gguf" else {"device": "auto"}
    if name == "hf":
        options["dtype"] = "float32"
    backend = DecisionEngine.from_pretrained(source, backend=name, local_files_only=True,
                                             max_input_tokens=2048, **options).backend
    try:
        _, cases = prompts(backend.tokenizer)
        references = [(backend.selected_logits(ids, slots),
                       list(backend.generate_tokens(ids, GenerationOptions(4), Event())))
                      for _, ids, slots in cases]
        backend.configure_prompt_cache(budget)
        checks = []
        for index, (case, ids, slots) in enumerate(cases):
            # First call can be a miss; the second must reuse this exact prompt.
            backend.selected_logits(ids, slots)
            usage = CacheUsage()
            with track_cache(usage):
                scores = backend.selected_logits(ids, slots)
            tokens = list(backend.generate_tokens(ids, GenerationOptions(4), Event()))
            delta = max(abs(a - b) for a, b in zip(scores, references[index][0]))
            probability_delta = max(abs(a - b) for a, b in zip(
                conditional_softmax(scores), conditional_softmax(references[index][0])))
            assert probability_delta < 0.02 and tokens == references[index][1]
            assert usage.cached_tokens > 0, "budget cannot hold a reusable snapshot"
            assert backend.prompt_cache.size_bytes <= budget
            checks.append({"case": case, "cached_tokens": usage.cached_tokens,
                           "max_logit_delta": delta, "max_probability_delta": probability_delta,
                           "greedy_tokens_equal": True})
        # Compare a new question sharing the original prompt's state. Seed
        # construction is outside warm timing; restoration and readout are inside.
        _, original, slots = cases[0]
        _, changed, changed_slots = cases[1]
        timings = {"cold_ms": [], "cached_ms": [], "cached_tokens": []}
        for _ in range(repeats):
            backend.configure_prompt_cache(0)
            started = time.perf_counter()
            backend.selected_logits(changed, changed_slots)
            timings["cold_ms"].append(1000 * (time.perf_counter() - started))
            backend.configure_prompt_cache(budget)
            backend.selected_logits(original, slots)
            usage = CacheUsage()
            started = time.perf_counter()
            with track_cache(usage):
                backend.selected_logits(changed, changed_slots)
            timings["cached_ms"].append(1000 * (time.perf_counter() - started))
            timings["cached_tokens"].append(usage.cached_tokens)
        cold, warm = statistics.median(timings["cold_ms"]), statistics.median(timings["cached_ms"])
        return {"status": "passed", "source": source, "budget_bytes": budget,
                "retained_bytes": backend.prompt_cache.size_bytes, "checks": checks,
                "timings": timings, "cold_median_ms": cold, "cached_median_ms": warm,
                "prefill_speedup": cold / warm}
    finally:
        if name == "gguf":
            backend.llama.close()


def main():
    from oev.prompt_cache import parse_cache_size
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("results/prompt-cache/models.json"))
    parser.add_argument("--backend", choices=["all", "hf", "paged", "gemma", "gguf", "mlx"], default="all")
    parser.add_argument("--mlx-model")
    parser.add_argument("--hf-model", help="cached HF causal model directory or repository ID")
    parser.add_argument("--gguf-model", help="local causal-model GGUF file")
    parser.add_argument("--gemma-tokenizer", help="cached Gemma4 tokenizer for the tiny random SWA test")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--production", action="store_true", help="measure oev's bounded cache, rather than standalone snapshots")
    parser.add_argument("--prompt-cache-max-size", "--prompt-cache-max-bytes", type=parse_cache_size, default="512MiB")
    parser.add_argument("--report", type=Path, default=Path("results/prompt-cache/report.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.production and args.backend in {"gemma", "paged"}:
        parser.error("--production supports hf, gguf, mlx, or all")
    if args.prompt_cache_max_size < 1:
        parser.error("--prompt-cache-max-size must be positive")
    models = json.loads(args.models.read_text(encoding="utf-8")) if args.models.exists() else {}
    models.update({key: value for key, value in {
        "hf": args.hf_model, "gguf": args.gguf_model, "gemma": args.gemma_tokenizer,
    }.items() if value})
    from importlib.metadata import PackageNotFoundError, version
    versions = {}
    for package in ("torch", "transformers", "llama-cpp-python", "mlx-lm"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    report = {"platform": platform.platform(), "versions": versions,
              "scope": "oev bounded production cache" if args.production else "standalone cache experiment; Gemma4 uses random tiny weights", "results": {}}
    for name in ("hf", "paged", "gemma", "gguf", "mlx"):
        if args.backend not in {"all", name}:
            continue
        if args.production and name in {"paged", "gemma"}:
            continue
        probe = None
        try:
            required = "gemma" if name == "gemma" else "hf" if name == "paged" else name
            if name == "mlx" and (platform.system() != "Darwin" or not args.mlx_model):
                result = {"status": "skipped", "reason": "Apple Silicon Mac and --mlx-model required"}
            elif name != "mlx" and not models.get(required):
                result = {"status": "skipped", "reason": f"cached {required} source required"}
            else:
                print(f"START {name}", flush=True)
                if args.production:
                    result = production_smoke(name, args.mlx_model if name == "mlx" else models[name],
                                              args.prompt_cache_max_size, args.repeats)
                elif name == "gguf":
                    probe = GGUFProbe(models["gguf"])
                elif name == "mlx":
                    probe = MLXProbe(args.mlx_model)
                else:
                    probe = HFProbe(models["gemma" if name == "gemma" else "hf"], gemma=name == "gemma")
                if not args.production:
                    result = paged_smoke(probe) if name == "paged" else snapshot_smoke(probe, args.repeats)
        except Exception as error:
            traceback.print_exc()
            result = {"status": "failed", "error": str(error), "type": type(error).__name__}
        finally:
            if isinstance(probe, GGUFProbe):
                probe.llama.close()
            probe = None
            gc.collect()
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        report["results"][name] = result
        print(json.dumps({name: result}, ensure_ascii=False), flush=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(any(r["status"] == "failed" for r in report["results"].values()))


if __name__ == "__main__":
    raise SystemExit(main())

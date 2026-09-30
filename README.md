# oev

`oev` is a small Python library for choosing one option with a language model. It builds a prompt from `state`, `question`, and `options`, runs the model once, and reads the next-token logits for option labels `a`–`z` and `0`–`9`. It does not generate an answer sentence or JSON. The direct logit readout approach is inspired by [SemIf](https://github.com/TheoLeeCJ/SemIf).

The default model is `google/gemma-4-E2B-it`. Use `--model` to select another compatible Hugging Face causal LM or a local checkpoint in Transformers format. GGUF files are not currently supported.

## Quick start

These examples use Python 3.11 and `uv`.

```powershell
uv sync --locked --extra hf --extra test
uv run --locked --extra hf oev --model google/gemma-4-E2B-it --device cpu --dtype bfloat16 --input examples/decisions.jsonl --output results/demo.jsonl
```

Model weights are downloaded on the first run if they are not cached. CPU inference works, but Gemma 4 E2B needs substantial memory and processing time. To check the CLI and model loading with a smaller model:

```powershell
uv run --locked --extra hf oev --model HuggingFaceTB/SmolLM2-135M-Instruct --device cpu --input examples/decisions.jsonl --output results/smoke.jsonl
```

This small model is intended for smoke tests. Its classification quality does not represent Gemma 4 E2B. `uv.lock` selects a PyTorch distribution based on the operating system.

| Operating system | PyTorch distribution | Available `--device` values |
| --- | --- | --- |
| Windows | CUDA 13.0 build | `cuda`, `hybrid-cuda`, or `cpu` |
| macOS | PyPI build | `mps` on supported hardware, otherwise `cpu` |
| Linux | PyPI build | `cuda` or `hybrid-cuda` when CUDA is available, otherwise `cpu` |

The lockfile does not distinguish between machines with and without a GPU on the same operating system. Windows machines without a GPU still install the CUDA build and can run with `--device cpu`. `--device auto` selects CUDA, MPS, then CPU in order of availability; it does not check VRAM capacity. [Google's memory table](https://ai.google.dev/gemma/docs/core) lists approximately 11.4 GB for Gemma 4 E2B BF16 inference. On a 6 GB GPU, the Gemma 4-specific `--device hybrid-cuda --dtype bfloat16` mode keeps embeddings and the output head on the CPU while placing decoder layers on the GPU. See the [hybrid evaluation results](reports/gemma4-e2b-stress-challenges-gpu-hybrid.md).

## Input and output

Input uses JSONL: one JSON object per line.

```json
{"id":"route-1","state":"The account is still locked after a password reset.","question":"Which team should handle this?","options":[{"id":"account","description":"Account access support"},{"id":"billing","description":"Billing support"}]}
```

| Input field | Meaning |
| --- | --- |
| `id` | Record ID used to associate the result with its input. Defaults to the record's position in the file. |
| `state` | A string, JSON object, or array containing the facts needed for the decision. |
| `question` | The instruction describing how to choose. |
| `options` | 2–36 objects with `id` and `description`, mapped in order to `a`–`z`, then `0`–`9`. |

Results are also written as JSONL. The main fields are:

| Output field | Meaning |
| --- | --- |
| `selected_id` | ID of the option with the highest logit. |
| `logits` | Raw logits for the option label tokens. |
| `probabilities` | Softmax computed only over the listed option logits. |
| `input_tokens`, `elapsed_seconds` | Prompt token count and processing time for the decision. |
| `prompt_sha256`, `model` | Prompt hash and model identifier, including the checkpoint revision when available. |

`probabilities` are **not probabilities of correctness or calibrated confidence**. In an evaluation where the correct sentiment was absent from the options, the model incorrectly classified a positive review as neutral with a conditional probability above 0.9999.

In the core decision API, option IDs are used only to map results and are not included in the prompt. The model sees each description and its option label. Changing IDs while preserving descriptions and order leaves the model input unchanged. The System One API described below includes IDs in the option descriptions it constructs.

## Python API

```python
from oev import Decision, DecisionEngine, Option

engine = DecisionEngine.from_pretrained(
    "google/gemma-4-E2B-it", device="cpu", dtype="bfloat16"
)
result = engine.decide(Decision(
    state="The account is still locked after a password reset.",
    question="Which team should handle this?",
    options=(
        Option("account", "Account access support"),
        Option("billing", "Billing support"),
    ),
))
print(result.selected_id, result.logits, result.elapsed_seconds)
```

`DecisionEngine` separates prompt construction and result mapping from the backend that returns option logits. `from_pretrained` loads the Hugging Face backend and accepts a compatible model ID or local path. To use another runtime, implement the `LogitBackend` protocol with `tokenizer`, `model_name`, and `selected_logits(input_ids, answer_token_ids)`, then pass it to `DecisionEngine(backend)`.

## System One API

The HTTP server exposes `POST /v1/systemone` and `GET /v1/models` using System One request and response formats. `GET /.skill` serves Markdown instructions that an LLM can use to call the API. The server binds to `127.0.0.1:8000` by default.

```powershell
uv sync --locked --extra hf --extra serve
$env:OEV_HOST = "127.0.0.1"
$env:OEV_PORT = "8000"
$env:OEV_MAX_CONCURRENCY = "4"
uv run --locked --extra hf --extra serve oev-serve --model google/gemma-4-E2B-it --device hybrid-cuda --dtype bfloat16
```

`OEV_HOST`, `OEV_PORT`, and `OEV_MAX_CONCURRENCY` default to `127.0.0.1`, `8000`, and `4`. The `--host`, `--port`, and `--max-concurrency` arguments override these environment variables. Each server process loads the model once and allows up to four simultaneous inferences by default. Questions within one request run sequentially, and requests wait when the concurrency limit is reached. Concurrent inference does not guarantee higher throughput and may increase GPU memory use depending on input length.

The default device is `auto`. The example explicitly selects `hybrid-cuda` for a 6 GB CUDA GPU. This mode requires Gemma 4 and a CUDA GPU with bfloat16 support.

```json
{
  "model": "google/gemma-4-E2B-it",
  "state": {"message": "I would like a refund."},
  "questions": {
    "route": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "Billing team", "support": "Customer support team"}},
    "urgent": {"type": "noul", "instructions": "Does this require urgent action?"},
    "priority": {"type": "score", "instructions": "How urgent is this?", "criteria": ["Can wait", "This week", "Today"]}
  }
}
```

When using the System One Python SDK, set `base_url="http://127.0.0.1:8000"` and use the model name served by the API. The local server does not validate API keys, but you can supply an arbitrary local key if the SDK constructor requires one.

All questions in a request share the same `state`, and oev runs the model once per question. `choice` and `score` accept up to 36 options or levels. `noul` evaluates `false` and `true` and returns `p(true)`. `score` returns the probability-weighted average of the level indices. A single option or level produces a deterministic result without running the model.

`confidence` summarizes how concentrated the probability distribution is. For `choice`, it uses `(max probability − 1/K) / (1 − 1/K)`; for `score`, it uses the average distance from the most probable level. These values are not calibrated probabilities of correctness, so thresholds for automated actions should be validated on actual application data. `usage.input_tokens` sums the prompt tokens across questions, and `usage.output_tokens` is zero because no tokens are generated. The default server provides no authentication; use authentication and TLS in front of it when exposing it externally.

## Computation and limits

The Gemma 4 backend projects the final hidden state **only onto output weight rows corresponding to the option tokens**. It skips the final projection over the full vocabulary but still performs the model forward pass over the input prompt. For other compatible models, it uses `logits_to_keep=1` when supported to compute only the last position before reading the option logits.

Before inference, oev checks that each label is exactly one token and does not merge with the end of the prompt. It raises an error if either condition fails. Input is never truncated automatically; the default limit is 4,096 tokens. File processing runs one decision at a time and reuses the loaded model. SemIf's prefix cache optimization is not implemented yet.

## Reproducing evaluations

The original E2B evaluation results and reports used uppercase option labels. Their scripts and fixtures retain uppercase labels as position metadata. The newer [E4B HTTP evaluation](reports/macstudio-e4b-stress-shuffles-20261001.md) used the current lowercase labels and matched the expected answer in all 220 permutation cases, with a median HTTP response time of 0.355 seconds. That set contains 2–20 options per case; it does not evaluate 36-option accuracy.

CPU results and per-option logits are available in the [13-case baseline report](reports/gemma4-e2b-cpu.md) and [stress evaluation summary](reports/gemma4-e2b-stress-summary.md). The [16-case GPU hybrid report](reports/gemma4-e2b-stress-challenges-gpu-hybrid.md) covers execution using a 6 GB GPU together with CPU RAM. These are small, manually constructed datasets; their accuracy rates should not be treated as general performance estimates.

The original stress cases are in `examples/stress-challenges.jsonl` and `examples/stress-semantic20-hard.jsonl`. The following command uses a fixed seed to generate 20 option permutations for each of 11 base cases, producing 220 cases plus follow-up cases.

The `expected_id`, `expected_letter`, and `scenario` fields are scoring metadata. Core oev inference uses only `state`, `question`, and `options`.

```powershell
uv run --locked --extra hf python scripts/build_stress_suite.py
```

To run and score the 16 main stress cases:

```powershell
uv run --locked --extra hf oev --model google/gemma-4-E2B-it --device cpu --dtype bfloat16 --input examples/stress-challenges.jsonl --output results/gemma4-e2b-stress-challenges-cpu.jsonl
uv run --locked --extra hf python scripts/score_stress.py --input examples/stress-challenges.jsonl --output results/gemma4-e2b-stress-challenges-cpu.jsonl --report reports/gemma4-e2b-stress-challenges-cpu.md --details
```

`--details` includes every option's logit in the report. The scoring script can also read partial output files during a run and report the completed and correct counts. In the recorded CPU run, 15 of these 16 cases matched the expected answer.

The original CPU permutation run completed 20 different orders of the duplicate-charge case. To run the full 220-case set locally on a GPU:

```powershell
uv run --locked --extra hf oev --model google/gemma-4-E2B-it --device cuda --dtype bfloat16 --input examples/stress-shuffles.jsonl --output results/gemma4-e2b-stress-shuffles-gpu.jsonl
uv run --locked --extra hf python scripts/score_stress.py --input examples/stress-shuffles.jsonl --output results/gemma4-e2b-stress-shuffles-gpu.jsonl --report reports/gemma4-e2b-stress-shuffles-gpu.md
```

These commands require CUDA-enabled PyTorch and a GPU with bfloat16 support. Cases with two or three options have fewer than 20 possible permutations, so some inputs repeat. Four-option cases use 20 distinct orders, and 20-option cases place the correct answer once at each position. Cases with multiple intents specify a priority or manual triage rule to define one expected answer.

## HTTP benchmark

To run the 220-case permutation evaluation against a running System One server, use the command below. Replace the base URL with your server address. The `test` extra provides `httpx`. The benchmark excludes three warm-up requests, reuses the connection, and sends requests sequentially. It records accuracy and median/p95 response times, including network time. Output files are not overwritten; use new paths for each run. The server API includes option IDs in the model input, so its prompts differ from the local evaluation prompts.

```powershell
uv run --locked --extra test python scripts/benchmark_systemone.py --base-url http://macstudio:12821 --input examples/stress-shuffles.jsonl --output results/http-shuffles.jsonl --report reports/http-shuffles.md
```

## Validation

```powershell
uv run --locked --extra hf --extra serve --extra test pytest -q
uv lock --check
```

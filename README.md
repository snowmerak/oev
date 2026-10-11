# oev

`oev` is a small Python library for choosing one option with a language model. Its decision path builds a prompt from `state`, `question`, and `options`, runs the model once, and reads the next-token logits for option labels `a`–`z` and `0`–`9`. The direct logit readout approach is inspired by [SemIf](https://github.com/TheoLeeCJ/SemIf). The server also supports text generation through OpenAI-compatible Responses and Chat Completions endpoints, using the same loaded model as typed decisions.

The default model is `google/gemma-4-E2B-it`. Use `--model` to select a compatible Hugging Face causal LM, a local Transformers checkpoint, a GGUF file, or an MLX checkpoint on Apple Silicon.

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

Compressed-tensors QAT checkpoints also require the `qat` extra:

```powershell
uv sync --locked --extra hf --extra qat
uv run --locked --extra hf --extra qat oev --model google/gemma-4-E4B-it-qat-w4a16-ct --revision 6cd26aaa2357fb2bad8c51699a7558a4d1a965bb --device cuda --dtype bfloat16 --input examples/decisions.jsonl --output results/qat-demo.jsonl
```

The installed Transformers/compressed-tensors runtime expands packed W4A16 weights for inference on the first forward pass. Checkpoint size therefore does not represent runtime GPU memory use. oev serializes this initial decompression and can use selective output projection afterward. In `hybrid-cuda` mode, supported Gemma 4 QAT models are decompressed on the CPU during loading, before decoder layers are moved to CUDA; sufficient CPU RAM and GPU memory for the expanded weights are still required.

Gemma 4 12B QAT loads on the tested 6 GB Windows GPU, but its default whole-model decompression runs out of memory even with shared GPU memory. The experimental [CT probe](scripts/probe_ct_w4a16.py) keeps symmetric group-32 W4 weights packed and temporarily expands each Linear layer for BF16 computation. It uses ordinary BF16 matrix multiplication and leaves the default loader unchanged. The probe defaults to the tested 12B checkpoint and pinned revision; see the [12B execution report](reports/gemma4-12b-qat-w4a16-cuda-20261008.md).

```powershell
uv run --locked --extra hf --extra qat python scripts/probe_ct_w4a16.py --device cuda --input examples/decisions.jsonl --output results/12b-qat.jsonl --events results/12b-qat-events.jsonl
```

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

`OEV_HOST`, `OEV_PORT`, and `OEV_MAX_CONCURRENCY` default to `127.0.0.1`, `8000`, and `4`. The `--host`, `--port`, and `--max-concurrency` arguments override these environment variables. Each server process loads each configured model once and allows up to four simultaneous inferences **per model** by default, shared across that model's decisions and text generation. A generation holds one slot until it finishes or cancellation is observed. Questions within one decision request run sequentially, and requests wait when their model's concurrency limit is reached. Concurrent inference does not guarantee higher throughput and may increase GPU memory use depending on input length and generation KV caches.

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

## Multiple resident models

Repeat `--model` to keep multiple models loaded in the same process. Every request selects a loaded model through its required `model` field, across System One, Responses, and Chat Completions, including streaming. The loading options apply to all models unless overridden in a config file.

```powershell
uv run --locked --extra hf --extra serve oev-serve --model google/gemma-4-E2B-it --model google/gemma-4-E4B-it --device cpu --dtype bfloat16
```

Model IDs or local paths are served as their names by default. To assign aliases, supply one `--served-model` per `--model`, in the same order:

```powershell
uv run --locked --extra hf --extra serve oev-serve --model google/gemma-4-E2B-it --served-model fast --model google/gemma-4-E4B-it --served-model large --device cpu --dtype bfloat16
```

For different devices, revisions, or limits, use `--models-config` with a nonempty JSON array. All model configuration examples are collected in [examples/models.json](examples/models.json), grouped by backend and identified by descriptive `served_model` names:

| `served_model` | Example |
| --- | --- |
| `hf-repo-hybrid-cuda` | HF repository on hybrid CUDA/CPU |
| `hf-repo-cpu` | HF repository on CPU |
| `hf-local-cpu` | Offline Transformers directory on CPU |
| `gguf-local-cpu` | Local GGUF on CPU |
| `gguf-local-metal-flash-attn` | Local GGUF on Metal with Flash Attention requested |
| `mlx-local-small-metal`, `mlx-local-large-metal` | Two offline MLX directories on Metal |

Copy the entries you want into `config.json` as a JSON array, and replace the example model paths. Keep only the entries supported by your machine and needed for this server: every entry is loaded, including separate instances when two entries point to the same checkpoint. JSON has no comments; the served names explain each example and are also the model names clients send in API requests.

For example, a custom config serving a hybrid-CUDA model alongside a CPU model can contain:

```json
[
  {"model": "google/gemma-4-E2B-it", "served_model": "fast", "device": "hybrid-cuda", "dtype": "bfloat16", "max_concurrency": 1},
  {"model": "google/gemma-4-E4B-it", "served_model": "large", "device": "cpu", "dtype": "bfloat16", "max_input_tokens": 8192, "max_concurrency": 2}
]
```

```powershell
uv run --locked --extra hf --extra serve oev-serve --models-config config.json
```

Each entry requires `model` and may set `served_model`, `revision`, `device`, `dtype`, `max_input_tokens`, `max_concurrency`, `backend`, `local_files_only`, `context_length`, `gpu_layers`, `chat_template`, and `flash_attn`. Omitted loading options inherit CLI defaults, including `OEV_MAX_CONCURRENCY`; explicit entry values override them. `--models-config` cannot be combined with `--model` or `--served-model`. Served names must be unique; aliases let you load the same checkpoint with different revisions or devices as separate instances.

`GET /v1/models` lists all served names in configuration order in both `models` and `data`. For example, use `model="fast"` for a decision and `model="large"` for a generated reply. An unknown name returns `422` for System One or `404` with `model_not_found` for generation, before any inference.

All models finish loading before HTTP starts and remain resident until the process exits. A loading failure prevents the server from starting. Models have independent inference limits, so one model's occupied slots do not block another; total concurrency can reach the sum of their limits. RAM/VRAM must accommodate all loaded weights and active KV caches together. There is no automatic eviction or runtime load/unload API.

You can also register already-loaded Python services:

```python
from oev import DecisionEngine, ModelRegistry, SystemOneService
from oev.serve import create_app

registry = ModelRegistry([
    SystemOneService(DecisionEngine.from_pretrained("google/gemma-4-E2B-it", device="cpu"), "fast", max_concurrency=1),
    SystemOneService(DecisionEngine.from_pretrained("google/gemma-4-E4B-it", device="cpu"), "large", max_concurrency=2),
])
app = create_app(registry)
```

`create_app(single_service)` remains supported. The registry retains the supplied services and routes to them without loading additional model copies.

## Downloaded models and offline loading

The downloader does not determine the backend: the downloaded **file format** does.

| Downloaded format | Model source | Backend |
| --- | --- | --- |
| Transformers checkpoint | Directory with config, tokenizer/chat template, and all weight files; or a cached HF repository ID | `hf` |
| GGUF text model | Path to a specific `.gguf` file, including an LM Studio download | `gguf` (auto-detected by file extension) |
| MLX checkpoint | Complete MLX directory or HF repository ID | `mlx` (Apple Silicon macOS) |

### `hf download`

[HF's CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) can download a complete repository into a directory:

```powershell
hf download google/gemma-4-E2B-it --local-dir ./models/gemma4
uv run --locked --extra hf --extra serve oev-serve --model ./models/gemma4 --served-model gemma-local --device cpu --dtype bfloat16 --local-files-only
```

A local checkpoint directory automatically uses `local_files_only=True`, including `hybrid-cuda`; missing files fail instead of being fetched. You may also pass the snapshot directory printed by `hf download`, or pass the original repository ID with `--local-files-only` to use the HF cache. A directory needs the complete tokenizer, model config, and all weight shards. Downloading only a `.safetensors` file is insufficient. `~` is expanded in local model paths.

### `lms get` and other GGUF downloads

[`lms get`](https://lmstudio.ai/docs/cli/local-models/get) can download GGUF models; [`lms ls --json`](https://lmstudio.ai/docs/cli/local-models/ls) lists downloaded models. Locate the actual `.gguf` file under your configured LM Studio model directory and pass its local path. Remote LM Link model identifiers are not local file paths. GGUF files downloaded with `hf download` work the same way.

```powershell
uv sync --locked --extra gguf --extra serve
uv run --locked --extra gguf --extra serve oev-serve --model "C:/path/to/model.Q4_K_M.gguf" --served-model gguf-local --device cpu --context-length 8192
```

GGUF support uses the optional [llama-cpp-python runtime](https://llama-cpp-python.readthedocs.io/en/latest/), which must support the model's architecture. Installing this extra may require a native compiler; GPU offload requires a build for your GPU backend. It loads quantized weights directly without converting them to Transformers weights. Selection, Responses, Chat Completions, and SSE all reuse that loaded instance.

GGUF `device=auto` defaults to CPU. Set `gpu_layers`/`--gpu-layers` to the number of layers to offload, or `-1` for all, with a GPU-capable runtime. `device=cuda` or `mps` requests all layers by default; the installed native runtime determines the GPU backend. `device=cpu` requires zero offloaded layers. `context_length` defaults to 8,192 and includes prompt plus requested output. The GGUF's quantization determines its weight type, so `dtype` must remain `auto`. `revision` is supported by HF and MLX; `hybrid-cuda` is HF-only. `context_length`, `gpu_layers`, `chat_template`, and enabling `flash_attn` are GGUF-only options.

#### GGUF Flash Attention

Flash Attention is included in the existing `gguf` dependency (`llama-cpp-python>=0.3.36,<0.4`). The separate PyTorch/CUDA `flash-attn` package is not required by this backend. Use `--flash-attn` to request native Flash Attention; `--no-flash-attn` explicitly disables it. Both `oev` and `oev-serve` accept these flags, and the default remains disabled. The Python loader accepts `DecisionEngine.from_pretrained(path, backend="gguf", flash_attn=True)`.

For mixed model configs, set `"flash_attn": true` on the GGUF entry. It must be a JSON boolean, and omitted values inherit the CLI default. HF and MLX entries reject enabling it; when using a global `--flash-attn` with mixed backends, explicitly set `"flash_attn": false` on other entries. This option is set when loading the model, so restart the server after changing it. Actual kernel availability and activation depend on the model and installed native build; the flag requests activation rather than guaranteeing compatibility or a speedup.

On Mac Studio, a Metal-capable runtime and `device=mps` enable GPU offload. If your installed runtime lacks GPU offload, [build llama-cpp-python with Metal](https://llama-cpp-python.readthedocs.io/en/latest/#supported-backends) using an arm64 Python and Xcode command-line tools:

```bash
CMAKE_ARGS="-DGGML_METAL=on" uv sync --locked --extra gguf --extra serve --reinstall-package llama-cpp-python --no-binary-package llama-cpp-python
```

Copy the `gguf-local-metal-flash-attn` entry from [examples/models.json](examples/models.json) into `config.json` as a JSON array and edit its model path, then run:

```bash
uv run --locked --extra gguf --extra serve oev-serve --models-config config.json
# Or load one model directly:
uv run --locked --extra gguf --extra serve oev-serve --model "/path/to/model.gguf" --device mps --flash-attn
```

That entry selects Metal with Flash Attention. Enabling Flash Attention alone leaves your selected device and GPU offload settings unchanged; a CPU config stays on CPU.

The model's embedded Jinja chat template is used with `enable_thinking=False`. If absent, provide `--chat-template path/to/chat.jinja` (or `chat_template` in JSON) with the correct template. The same exact single-token answer-label and prompt-boundary validation applies to GGUF decisions; incompatible tokenizers fail clearly. All native end-of-generation tokens, including end-of-turn markers, stop generation and count toward usage without appearing in output text.

One GGUF instance owns one mutable native KV context. Requests to that instance are serialized internally, even if `max_concurrency` is larger than one; different models can still run independently. Cancellation is observed between prompt batches and generation forwards, and closing a stream resets the context and releases its lock. GGUF selection computes the native full vocabulary logits and reads the requested answer tokens; HF's Gemma-specific projection optimization does not apply to this backend.

To mix offline HF and GGUF models, copy the `hf-local-cpu` and `gguf-local-cpu` entries from [examples/models.json](examples/models.json) into `config.json` as a JSON array, edit their paths, then run:

```powershell
uv run --locked --extra hf --extra gguf --extra serve oev-serve --models-config config.json
```

Relative paths are resolved from the server's working directory. Local sources are loaded directly without LM Studio running. Model directories are not scanned automatically; pass the actual downloaded path.

The Python loader accepts the same settings:

```python
engine = DecisionEngine.from_pretrained("C:/path/to/model.gguf", backend="gguf", context_length=8192)
offline = DecisionEngine.from_pretrained("./models/gemma4", local_files_only=True, device="cpu")
```

Tests exercise an actual saved/reloaded tiny Transformers checkpoint offline and GGUF routing, native-context serialization, cancellation, stop tokens, and error handling with a simulated native runtime. To exercise your installed llama.cpp runtime and an actual downloaded text GGUF:

```powershell
$env:OEV_TEST_GGUF_PATH = "C:/path/to/model.gguf"
uv run --locked --extra gguf --extra serve --extra test python -m pytest -q tests/test_local_models.py::test_real_downloaded_gguf_when_explicitly_configured
```

Without that environment variable, the real GGUF test is skipped; simulated tests do not establish compatibility with every GGUF architecture.

To exercise the real GGUF runtime with Flash Attention requested on Metal, use these settings on your Mac. The test checks selection and repeated generation using the native runtime; it does not measure performance or verify which kernel the runtime ultimately selected:

```bash
export OEV_TEST_GGUF_PATH="/path/to/model.gguf"
export OEV_TEST_GGUF_DEVICE=mps
export OEV_TEST_GGUF_FLASH_ATTN=1
uv run --locked --extra gguf --extra serve --extra test python -m pytest -q tests/test_local_models.py::test_real_downloaded_gguf_when_explicitly_configured
```

### MLX on Mac Studio

The optional [mlx-lm runtime](https://github.com/ml-explore/mlx-lm) loads MLX weights directly on Apple Silicon macOS. A complete model directory downloaded with `hf download` or LM Studio can be used as-is, including quantized weights:

```bash
git pull
uv sync --locked --extra mlx --extra serve
uv run --locked --extra mlx --extra serve oev-serve --backend mlx --model "/path/to/mlx-model" --served-model mlx-local --device auto --local-files-only --max-concurrency 1
```

For an HF repository, use `--backend mlx --model owner/model`; add `--local-files-only` to use a previously downloaded HF cache, and `--revision` to choose a revision. Online repository loading downloads the snapshot. Automatic backend selection recognizes `mlx-community/` repository IDs and local configs with MLX `quantization.bits` and `quantization.group_size`; use an explicit backend for other sources, including unquantized local MLX checkpoints. A single weight file is insufficient: the directory must include model config, tokenizer/chat template, and every weight shard.

`device=auto` and `mps` use Metal; `cpu` selects MLX CPU execution. Keep `dtype=auto` to use the checkpoint's stored dtype and quantization. Selection, Responses, Chat Completions, and SSE share the same resident model. Each request gets a fresh KV cache, and inference is serialized per model even with a larger `max_concurrency`. Cancellation is checked between prompt chunks and generation forwards; pending work is synchronized before releasing the model lock. Separate registered models retain independent weights and execution streams. Each model uses an MLX stream that can be evaluated across HTTP worker threads under that lock, including lazy constants created while loading; a thread-local loading stream would cause `There is no Stream(...) in current thread` failures.

The checkpoint's tokenizer chat template and EOS/end-of-turn token IDs are used. Prompts obey `max_input_tokens`; prompt plus requested output also obeys `max_position_embeddings` when declared in the config. MLX computes final-token vocabulary logits and selects the answer rows; HF's Gemma-specific projection optimization does not apply. Available architectures depend on the installed `mlx-lm` version.

To keep multiple MLX models loaded, copy the `mlx-local-small-metal` and `mlx-local-large-metal` entries from [examples/models.json](examples/models.json) into `config.json` as a JSON array, edit their paths, then run:

```bash
uv run --locked --extra mlx --extra serve oev-serve --models-config config.json
```

Windows tests use a simulated MLX runtime to cover loading, lazy initialization stream affinity across threads, cache isolation, sampling, API responses, SSE, serialization, cancellation, and failures. The following opt-in test runs real Metal inference on your Mac, first checking Responses JSON/SSE and System One on HTTP worker threads, then comparing greedy output with `mlx-lm.generate_step`, checking selection logits against a full forward, and checking repeated requests for cache contamination:

```bash
export OEV_TEST_MLX_PATH="/path/to/mlx-model"
uv run --locked --extra mlx --extra serve --extra test python -m pytest -q tests/test_mlx.py
```

The real MLX test is skipped without that variable or outside macOS. Real MLX/Metal execution has not been verified on Windows.

## Responses and Chat Completions

The same `oev-serve` process exposes `POST /v1/responses` and `POST /v1/chat/completions`. These implement the text-generation subset of the [OpenAI API formats](https://developers.openai.com/api/docs/guides/migrate-to-responses). Generation uses the selected model's existing decision backend without another model load. HF, `hybrid-cuda`, GGUF, and MLX backends support generation; each request owns an isolated KV context. Decision requests use the backend's next-token readout.

`GET /v1/models` includes the OpenAI-compatible `object` and `data` fields alongside the existing System One `models` array. Both lists describe the same set of served model names.

With an OpenAI Python SDK installed in your client environment:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
model = client.models.list().data[0].id

response = client.responses.create(
    model=model,
    instructions="Answer briefly in Korean.",
    input="What is the difference between a choice and a score?",
    max_output_tokens=128,
    store=False,
)
print(response.output_text)

completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Say hello in Korean."}],
    max_completion_tokens=128,
)
print(completion.choices[0].message.content)
```

Both endpoints accept `stream=True` for SSE output. Chat Completions sends `chat.completion.chunk` objects and ends with `[DONE]`; `stream_options={"include_usage": True}` adds a final usage chunk. Responses sends lifecycle events, `response.output_text.delta`, and a terminal `response.completed` or `response.incomplete` event.

```python
with client.responses.create(
    model=model, input="Say hello in Korean.", stream=True, max_output_tokens=128,
) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
```

The supported request fields are:

| Endpoint | Fields |
| --- | --- |
| Both | `model`, `temperature` (0–2; default 0 for greedy decoding), `top_p` (greater than 0 and at most 1), `stream`, `store` (false only) |
| Chat Completions | `messages`, `max_completion_tokens` or `max_tokens`, `stop` (up to four strings), `n` (1 only), `stream_options.include_usage`, `response_format` (text only) |
| Responses | `input` (string or message array), `instructions`, `max_output_tokens`, `text.format` (text only) |

Messages support `system`, `developer`, `user`, and `assistant` roles. Developer messages are passed to model templates as system messages. Content can be a string or text parts (`text` for Chat Completions; `input_text`/`output_text` for Responses). For another turn, resend the history; Responses output message items can be appended to the next `input` array.

Output limits default to 256 and accept 1–4,096 tokens. Prompts obey `--max-input-tokens`, and prompt plus requested output must fit the model context. Input is never truncated. Token usage includes generated stop/EOS tokens, even when they are omitted from displayed text. A length-limited completion reports `finish_reason="length"`; Responses reports `status="incomplete"` with `incomplete_details.reason="max_output_tokens"`.

These endpoints are stateless: response storage, `previous_response_id`, tool calls, multimodal input/output, log probabilities, and JSON/schema output constraints are unsupported. Unsupported fields and invalid generation inputs return HTTP `400` with an OpenAI-style error object; an unknown model returns `404`. System One keeps its existing `422` validation behavior. Generation failures return `500`, or a streaming error event after SSE starts. Client disconnection cancels a stream between model forwards; its execution slot remains occupied until the worker exits.

The Python API can generate using an existing decision backend too:

```python
from oev import GenerationEngine, GenerationOptions

generation = GenerationEngine(engine.backend, max_input_tokens=engine.max_input_tokens)
prepared = generation.prepare(
    [{"role": "user", "content": "Say hello in Korean."}],
    GenerationOptions(max_new_tokens=128),
)
print(generation.generate(prepared).text)
```

## Computation and limits

The HF Gemma 4 backend, including Gemma4Unified models such as 12B, projects the final hidden state **only onto output weight rows corresponding to the option tokens**. It skips the final projection over the full vocabulary but still performs the model forward pass over the input prompt. For other compatible HF models, it uses `logits_to_keep=1` when supported to compute only the last position before reading the option logits.

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

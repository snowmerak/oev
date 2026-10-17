# Multimodal embedding verification — 2026-10-17

The `/v1/embeddings` extension accepts inline image/audio data URLs, text/media combinations, and heterogeneous batches. Implementations use Sentence Transformers for HF, MLX-VLM for MLX EmbeddingGemma2, and typed-content inputs to a resident llama-server with a matching multimodal projector for GGUF.

## Real model checks

| Runtime | Model and revision | Result |
| --- | --- | --- |
| HF CPU | `google/embeddinggemma-2` at `914f7f89142e33e77833254d9c9b90c3cef7303b` | Passed |
| GGUF CPU | `ggml-org/embeddinggemma-2-GGUF` at `bfcd298762cc34d0357ece5ebdd31791a3a374d8`, Q8_0 model and Q8_0 multimodal projector | Passed |
| MLX Metal | MLX-VLM 0.7.8 adapter | Simulated runtime tests passed; real Apple Silicon execution pending |

Both real runtimes passed HTTP text, image, audio, and combined text/image inputs; mixed batch versus individual input agreement; 768-dimensional finite unit vectors; 128-dimensional truncation followed by normalization; and differing image content producing different vectors. GGUF used the official llama.cpp b11552 Windows CPU distribution, verified against its release SHA-256 digest before extraction.

The HF check also submits a 40-second recording with an 800-token limit. It is rejected using the complete expanded token count, with no 30-second audio truncation or saved-processor placeholder clipping. Audio truncation is disabled explicitly, and the processor's audio placeholder ceiling covers the embedding context window.

Environment: Windows, CPython 3.11, PyTorch 2.14.0+cu130, Transformers 5.19.0, Sentence Transformers 6.1.0, SoundFile 0.13.1. All real model inference in this report used CPU. These are correctness checks with synthetic media, not retrieval-quality or throughput benchmarks.

## Automated checks

Final full regression result: **677 passed, 24 skipped**. The real HF and GGUF multimodal tests also passed in separate opt-in runs. Skips include unavailable platform-specific runtimes and checkpoints not selected by environment variables.

The regression suite covers compatibility with existing text embedding requests, request validation, data URL decoding, modality order, downmixing/resampling, byte/sample budgets, token limits, prefixes, output normalization, base64 output, wrong-backend rejection, model catalog capabilities, GGUF projector configuration/native errors, and MLX stream ownership across HTTP worker threads. Real model checks are opt-in through the environment variables documented in [the usage guide](../docs/multimodal-embeddings.md).

The installed Windows CUDA runtime crashes in `torch.cuda.is_current_stream_capturing()` even during CPU masking checks. CPU verification therefore ran with that probe overridden to return `False` in the test process only; application code and dependency pins were not changed to hide this environment issue:

```powershell
uv run --locked --extra hf --extra embeddings --extra serve --extra test python -c "import torch; torch.cuda.is_current_stream_capturing=lambda: False; import pytest; raise SystemExit(pytest.main(['-q']))"
```

`uv lock --check` and `git diff --check` passed. The dependency audit reports the existing `diskcache==5.6.3` unsafe-pickle advisory (CVE-2025-69872, reported under two identifiers); it was already present in the base lockfile, and the audit lists no fixed version. This change adds no diskcache usage.

Mac Studio verification must run `test_real_multimodal_embedding_checkpoint[mlx]` against the deployed full MLX checkpoint after syncing the updated `mlx-embeddings` extra. Windows simulated tests cover caller behavior and cannot prove Metal execution or memory use.

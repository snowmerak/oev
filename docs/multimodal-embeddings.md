# Image and audio embeddings

`POST /v1/embeddings` accepts text, image, audio, and combined inputs with EmbeddingGemma2 on HF, MLX-VLM, and native llama-server. Each input produces one vector in the model's shared embedding space. Existing string inputs, output formats, dimension reduction, and normalization remain supported.

## Requests

`input` can be a string, a modality object, or a batch of 1–128 strings and objects. Objects contain `text`, `image`, `audio`, or a combination. Text must be a nonempty string. Media values are base64 data URLs, or nonempty lists of data URLs. The object key order determines modality order; a combined object still produces one vector.

```json
{
  "model": "embed",
  "input": [
    "A dog barking",
    {"image": "data:image/png;base64,..."},
    {"audio": "data:audio/wav;base64,..."},
    {"text": "A dog barking", "image": "data:image/png;base64,..."}
  ],
  "dimensions": 256,
  "input_type": "document",
  "encoding_format": "float"
}
```

Use an HTTP client to send the extension:

```python
import base64
from pathlib import Path
import httpx

def media(path, mime):
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")

response = httpx.post("http://localhost:8000/v1/embeddings", json={
    "model": "embed",  # Your configured served_model
    "input": [
        "A dog barking",
        {"image": media("dog.png", "image/png")},
        {"audio": media("barking.wav", "audio/wav")},
        {"text": "A dog barking", "image": media("dog.png", "image/png")},
    ],
    "dimensions": 256,
}, timeout=120)
response.raise_for_status()
vectors = [item["embedding"] for item in response.json()["data"]]
```

Media must be inline: remote URLs and server file paths are rejected. A plain string remains text even if it resembles a media URL or local path. `/v1/models` exposes `input_modalities` for each embedding entry, based on the loaded checkpoint/runtime; text-only models reject image/audio input with HTTP 400.

Images are decoded to RGB. Audio uses libsndfile-supported formats, including WAV, FLAC, and OGG, then is downmixed to mono and resampled to 16 kHz. These inputs go through the model's own media encoders and learned embedding head. No OCR, captions, speech transcription, or alternative model is substituted. The existing `input_type=query/document` prefixes apply to all modalities, and prompt/media tokens participate in usage and pooling. Output vectors are L2-normalized after optional dimension reduction; `encoding_format=base64` returns little-endian float32 as before.

Limits are 20 MiB decoded file bytes per media item, 64 MiB per batch, 16 media items per input, and 16 million image pixels. Audio must be nonempty and at most 330 seconds long, with at most 8 channels, a maximum 192 kHz source sample rate, and a decoded budget of 32 million samples counting channels. Model token limits can be lower. Invalid media and token overflow return HTTP 400 without silently truncating the input.

## HF

```bash
uv sync --locked --extra embeddings --extra serve
uv run --locked --extra embeddings --extra serve oev-serve \
  --task embedding --backend hf --model google/embeddinggemma-2 \
  --served-model embed --device cpu --dtype float32
```

HF now retains the checkpoint's vision and audio towers, increasing memory use compared with the previous text-only loader. Complete downloaded checkpoints need media weights, processor configs, tokenizer, and chat template. `dtype=auto` selects float32 for EmbeddingGemma2; float16 remains rejected. The native Sentence Transformers processor counts expanded tokens before inference, using the same prompt and no-truncation policy as encoding.

## MLX on Mac Studio

```bash
uv sync --locked --extra mlx-embeddings --extra serve
uv run --locked --extra mlx-embeddings --extra serve oev-serve \
  --task embedding --backend mlx --model mlx-community/embeddinggemma-2-bf16 \
  --served-model embed --device mps
```

Local MLX directories and `--local-files-only` work too. EmbeddingGemma2 uses MLX-VLM 0.7.8+, its native encoder, trained projection, and all configured towers. Other supported MLX text architectures continue using mlx-embeddings. The Transformers 5.19+ processor prepares media in NumPy, and the native model evaluates MLX arrays; it does not load the HF/PyTorch model. Keep `dtype=auto` to retain checkpoint precision and quantization. Model loading, forward execution, and synchronization use one shared MLX stream under a per-model lock, including HTTP worker threads. Expanded tokens are checked before model inference, with an 8,192-token model ceiling.

## GGUF

Use a recent [official llama-server](https://github.com/ggml-org/llama.cpp/releases/tag/b11552), the embedding GGUF, and its matching combined vision/audio projector from [ggml-org/embeddinggemma-2-GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF). Keep the executable and its backend libraries together. Add both paths to an embedding entry in your models config:

```json
[
  {
    "model": "/path/to/embeddinggemma-2-Q8_0.gguf",
    "served_model": "embed",
    "task": "embedding",
    "backend": "gguf",
    "device": "mps",
    "pooling": "mean",
    "gguf_server_path": "/path/to/llama-server",
    "mmproj_path": "/path/to/mmproj-embeddinggemma-2-Q8_0.gguf",
    "context_length": 8192,
    "max_input_tokens": 4096,
    "embedding_query_prefix": "task: search result | query: ",
    "embedding_document_prefix": "title: none | text: "
  }
]
```

On Windows, use `device=cpu` or a CUDA-capable native distribution with `device=cuda`.

```bash
uv sync --locked --extra embedding-media --extra serve
uv run --locked --extra embedding-media --extra serve oev-serve --models-config config.json
```

`mmproj_path` requires `task=embedding`, `backend=gguf`, and `gguf_server_path`. The worker receives `--mmproj` and reports available media modalities through `/props`. oev sends typed `content` entries inside each native `/v1/embeddings` input, preserves batch indices, and uses native token usage. Images are encoded as PNG and normalized audio as WAV for native decoding. The worker's context/batch size is capped at the smaller of `context_length` and `max_input_tokens`, so the native media-token expansion cannot bypass the service limit. llama-server rejects expanded inputs exceeding that budget. Pure Python llama-cpp-python embedding contexts continue supporting text only; there is no silent fallback for media. Model/projector paths and token budgets require a server restart to change.

## Verification

```bash
uv run --locked --extra embeddings --extra serve --extra test python -m pytest -q \
  tests/test_embeddings.py tests/test_multimodal_embeddings.py
```

The real-model test is opt-in. Set `OEV_TEST_EMBEDDING_HF_PATH`, `OEV_TEST_EMBEDDING_MLX_PATH`, or `OEV_TEST_EMBEDDING_GGUF_PATH` to the complete local checkpoint. GGUF additionally needs `OEV_TEST_GGUF_SERVER_PATH` and `OEV_TEST_EMBEDDING_MMPROJ_PATH`. For MLX run on Apple Silicon and include `--extra mlx-embeddings`, optionally setting `OEV_TEST_EMBEDDING_DEVICE=mps`. Then run:

```bash
uv run --locked --extra embeddings --extra serve --extra test python -m pytest -q \
  tests/test_multimodal_embeddings.py::test_real_multimodal_embedding_checkpoint
```

The test checks real image/audio/mixed HTTP inference, batch/single agreement, normalization, reduced dimensions, and different image content producing different embeddings. Simulated MLX tests verify loading, preprocessing, stream ownership across threads, token checks, and failure synchronization; they do not establish Metal execution on Windows.

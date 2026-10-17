---
name: oev-system-one
description: Use the oev HTTP API for typed decisions, text generation, and text/image/audio embeddings. Discover model capabilities before selecting an endpoint; use multimodal embeddings for semantic search and similarity across text, images, and audio.
---

# oev HTTP API

Use the same origin that served this document for every endpoint below. The built-in local server does not require an API key, though a deployment proxy may add authentication. Discover the server's model aliases and capabilities before making an inference request.

The server provides three workflows:

| Goal | Endpoint | Model selection |
| --- | --- | --- |
| Select an option, estimate a soft boolean, or return an ordinal score | `POST /v1/systemone` | `task=generation` |
| Generate text or function calls | `POST /v1/responses` or `POST /v1/chat/completions` | `task=generation` |
| Produce vectors for semantic search, retrieval, clustering, or similarity | `POST /v1/embeddings` | `task=embedding`; check `input_modalities` |

## When to use it

Use System One when the task can be expressed as:

- selecting one item from an explicit set (`choice`);
- estimating how strongly a true/false criterion applies (`noul`); or
- locating the state on an ordered scale (`score`).

System One reads next-token logits without generating an explanation. Do not use that endpoint for open-ended generation, fact retrieval, tool execution, or calibrated risk estimates. Supply all relevant facts in `state` and make the criteria mutually distinguishable.

Use embeddings to index content or compare its meaning. A multimodal embedding model can map text, images, audio, or a combination into the same vector space. For example, embed photos and recordings, then retrieve them with an embedded text query. Embeddings return vectors; they do not return captions, transcripts, answers, or similarity scores automatically.

## Workflow

1. Send `GET /v1/models` and inspect the entries in `data` for their `id`, `task`, and capabilities.
2. Select a model matching the operation. For media embeddings, every required modality must appear in `input_modalities`.
3. Follow the embedding workflow below, or send structured questions to System One and match each entry in `answers` to the same key in `questions`.
4. Interpret the result according to its endpoint: compare embedding vectors for similarity; treat System One probabilities as relative signals over the supplied criteria, not calibrated correctness probabilities.

For open-ended text, this server also exposes `POST /v1/responses` and `POST /v1/chat/completions`, using the selected model's existing loaded backend. Both support text messages, function tools, reasoning, JSON output validation, and SSE streaming. Responses accepts `model`, `input`, optional `instructions`, and `max_output_tokens`; Chat Completions accepts `model`, `messages`, and `max_completion_tokens`. Omitted output limits use the model default (8,192), capped by remaining context, including reasoning tokens. A limit reached before completion produces `status="incomplete"` with `incomplete_details.reason="max_output_tokens"` in Responses or `finish_reason="length"` in Chat; truncated turns withhold executable tool calls. Reasoning streams separately from answers. Resend conversation history or use opt-in stored Responses history. Images/audio on generation endpoints and grammar-constrained decoding are unsupported; image/audio embeddings use `/v1/embeddings`. `GET /v1/models` also returns an OpenAI-compatible `data` array with context/output budgets and native reasoning capabilities; the existing `models` array below remains available. Generation and embedding validation use HTTP `400`; System One uses HTTP `422`.

The server may keep multiple models loaded. Both model lists contain every served name in configuration order, but models cannot serve every endpoint. Choose `task=generation` for generation and System One, or `task=embedding` for embeddings. Within one generation model, generation and decisions share its backend and concurrency limit. Embedding inference is serialized per model, while different models remain independent. An unknown name returns HTTP `422` for System One or HTTP `404` with `model_not_found` for generation/embeddings. Choosing the wrong model task returns HTTP `400` with `unsupported_operation` for generation/embeddings, or HTTP `422` for System One.

Served models can come from HF repositories, downloaded Transformers directories, local GGUF files, or MLX checkpoints on Apple Silicon, depending on server configuration. Select the served alias from `/v1/models`; clients do not send filesystem paths unless that path itself is the served name. GGUF and MLX models serialize inference internally while other registered models remain independent.

## Discover the model

```http
GET /v1/models
```

Example catalog, with additional generation metadata omitted:

```json
{
  "models": [
    {
      "name": "google/gemma-4-E2B-it",
      "description": "oev System One: choice, noul, score; up to 36 choices or levels. Probabilities are conditional on listed options and are not calibrated.",
      "release_date": "2026-09-23"
    },
    {
      "name": "embeddinggemma2",
      "description": "oev audio/image/text embeddings; 768 dimensions.",
      "release_date": "2026-10-17"
    }
  ],
  "object": "list",
  "data": [
    {
      "id": "google/gemma-4-E2B-it",
      "object": "model",
      "created": 0,
      "owned_by": "oev",
      "task": "generation"
    },
    {
      "id": "embeddinggemma2",
      "object": "model",
      "created": 0,
      "owned_by": "oev",
      "task": "embedding",
      "input_modalities": ["audio", "image", "text"]
    }
  ]
}
```

Copy the selected entry's `name` from `models` (or its `id` from `data`) into the request's `model` field. Do not assume the example name is configured on every server.

## Generate embeddings

```http
POST /v1/embeddings
Content-Type: application/json
```

Choose an entry with `task="embedding"`. For image/audio inputs, require `image`/`audio` in that entry's `input_modalities`; a text-only embedding model cannot process media. If no listed model supports the required input, report that limitation rather than sending media to a generation model.

### Text and query/document prompts

```json
{
  "model": "embeddinggemma2",
  "input": "A dog barking",
  "input_type": "query",
  "encoding_format": "float"
}
```

`input_type` is an optional oev extension: `query` or `document` applies the model's configured task prefix. A missing prefix produces HTTP `400`. Without `input_type`, oev applies no default task prompt, so manually supplied prefixes are preserved. Do not add the same prefix manually and through `input_type`. For media, omit `input_type` unless the application intentionally uses a configured task prompt.

### Images, audio, combined inputs, and batches

The `input` field accepts:

- One nonempty text string.
- One object containing `text`, `image`, `audio`, or a combination. `text` must be a nonempty string. Each media value must be a base64 data URL, or a nonempty list of data URLs.
- A batch of 1–128 strings and/or modality objects. The response has one vector per batch item, in the same order.

Each combined object produces **one vector for all its content**. To obtain separate vectors for an image and a recording, send two objects in a batch. Within a combined object, the current API uses key order for modality order; preserve that order when serializing it.

Example request shape; the `...` values below are placeholders and must be replaced with complete base64-encoded file bytes:

```json
{
  "model": "embeddinggemma2",
  "input": [
    "A dog barking",
    {"image": "data:image/png;base64,..."},
    {"audio": "data:audio/wav;base64,..."},
    {"text": "A dog barking", "image": "data:image/png;base64,..."}
  ],
  "dimensions": 256,
  "encoding_format": "float"
}
```

Use a MIME type matching the file, such as `image/jpeg`, `image/png`, or `audio/wav`. Images are decoded to RGB. Supported audio formats include WAV, FLAC, and OGG; oev decodes the recording, downmixes it to mono, and resamples it to 16 kHz. The media goes through the embedding model's native encoders.

Media must be inline data URLs. Do not send remote URLs or server-side file paths in `image`/`audio`; they are rejected. A plain string such as `"dog.png"` is embedded as literal text and does not load a file. Read the file on the client and encode its bytes before calling the API.

The modality keys follow the model's native dictionary convention, as used by EmbeddingGemma2, but their use inside this HTTP endpoint is an **oev extension**. This is the text-string subset of the OpenAI Embeddings API plus multimodal inputs, not complete OpenAI API compatibility. Do not send Chat Completions `image_url`/`input_audio` content parts to this endpoint. Token-ID arrays and video inputs are unsupported.

### Complete client example

This example uses Python 3's standard library and two existing client files, `dog.png` and `barking.wav`. Replace `BASE_URL` with the origin that served this skill, and include any proxy-required authentication headers in both requests.

```python
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

BASE_URL = "http://localhost:8000"  # Replace with this skill's server origin.

with urlopen(BASE_URL + "/v1/models", timeout=15) as response:
    catalog = json.load(response)
required = {"text", "image", "audio"}
candidates = [
    item for item in catalog["data"]
    if item.get("task") == "embedding"
    and required.issubset(item.get("input_modalities", []))
]
if not candidates:
    raise RuntimeError("No served embedding model supports text, image, and audio")
model = candidates[0]["id"]

def media(path, mime):
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"

image = media("dog.png", "image/png")
audio = media("barking.wav", "audio/wav")
payload = {
    "model": model,
    "input": [
        "A dog barking",
        {"image": image},
        {"audio": audio},
        {"text": "A dog barking", "image": image},
    ],
    "encoding_format": "float",
}
request = Request(
    BASE_URL + "/v1/embeddings",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=120) as response:
    result = json.load(response)
vectors = [item["embedding"] for item in result["data"]]
print(model, len(vectors), len(vectors[0]), result["usage"])
```

Success returns four vectors: text, image, audio, and combined text/image, respectively. EmbeddingGemma2 returns 768 dimensions by default. The selected served model determines the actual width.

### Embedding response and similarity

Responses have `object="list"`, `model`, `data`, and `usage`. Every `data` item has `object="embedding"`, its zero-based `index`, and an `embedding` vector. `usage.prompt_tokens` and `usage.total_tokens` report the same input-token total, including prefixes, special tokens, and expanded media tokens.

Output vectors are L2-normalized. Compare vectors from the **same model and dimension**; their dot product is cosine similarity. For retrieval, index one vector per candidate, embed the query, and rank candidates by descending similarity. A similarity score is not a probability of correctness, and thresholds must be evaluated on application data.

Optional `dimensions` keeps that many leading components and normalizes the result again. Use reduced dimensions with models trained for Matryoshka embeddings, such as EmbeddingGemma2, and apply the same dimension to queries and indexed content. `encoding_format="base64"` returns each embedding as a base64 string of little-endian float32 values; decode the bytes before comparing vectors. The optional `user` field is accepted as a client identifier.

### Embedding limits and errors

- File bytes: at most 20 MiB decoded per media item and 64 MiB per batch; at most 16 media items in one input object.
- Images: at most 16 million pixels per image.
- Audio: nonempty, at most 330 seconds, 8 channels, 192 kHz source sample rate, and 32 million decoded samples counting channels. The model's token budget may impose a lower duration limit.
- Text and expanded media must fit the configured token limit. Inputs are not silently truncated; reduce the input or split it into separate embeddings when the server reports overflow.
- Invalid input, malformed media, unsupported modalities, missing task prefixes, and token overflow return HTTP `400` with `error.message`.
- An unknown model returns HTTP `404` with `error.code="model_not_found"`. A generation model sent to the embedding endpoint returns HTTP `400` with `error.code="unsupported_operation"`.
- An inference failure returns HTTP `500` with `error.code="embedding_error"`. Do not interpret an error as an empty or zero embedding.

## Evaluate questions

```http
POST /v1/systemone
Content-Type: application/json
```

```json
{
  "model": "google/gemma-4-E2B-it",
  "state": {
    "customer_message": "I was charged twice and need the duplicate reversed today."
  },
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Choose the team that should handle this request.",
      "criteria": {
        "billing": "Payments, duplicate charges, and refunds",
        "account": "Login, profile, and account access"
      }
    },
    "duplicate_charge": {
      "type": "noul",
      "instructions": "Does the message report a duplicate charge?",
      "criteria": {
        "true": "The same purchase was charged more than once",
        "false": "There is no duplicate charge"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "Rate the requested response urgency.",
      "criteria": ["Can wait", "Handle this week", "Handle today"]
    }
  }
}
```

The response has this shape:

```json
{
  "model": "google/gemma-4-E2B-it",
  "answers": {
    "route": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.84,
      "probabilities": {"billing": 0.92, "account": 0.08}
    },
    "duplicate_charge": {
      "type": "noul",
      "noul": 0.95
    },
    "urgency": {
      "type": "score",
      "score": 1.8,
      "confidence": 0.9,
      "legend": {
        "0": "Can wait",
        "1": "Handle this week",
        "2": "Handle today"
      },
      "probabilities": {"0": 0.05, "1": 0.1, "2": 0.85}
    }
  },
  "usage": {"input_tokens": 387, "output_tokens": 0}
}
```

The numbers above illustrate the response format; actual values depend on the configured model and input.

## Request schema

The top-level object requires:

- `model`: the exact string returned by `GET /v1/models`;
- `state`: a nonempty string, JSON object, or JSON array containing the facts used by every question;
- `questions`: a nonempty object whose keys are stable names chosen by the caller.

Each question has a `type`, optional `instructions`, and type-specific `criteria`. `instructions` may be a string, object, array, or `null`. Prefer a direct question that explains the decision boundary.

### `choice`

Use `criteria` as an object with 1 to 36 entries. Each key is a nonempty option identifier and each value is a string, object, array, or `null` description. The returned `choice` is exactly one of those keys. The model sees both the key and its description, so use short, meaningful keys and descriptions that state the distinction.

```json
{
  "type": "choice",
  "instructions": "Which queue owns this case?",
  "criteria": {
    "billing": "Charges and refunds",
    "support": "Product usage and troubleshooting"
  }
}
```

`probabilities` is a distribution over only the listed choices. `confidence` summarizes how concentrated that distribution is relative to a uniform distribution.

### `noul`

Use this for a soft boolean. Provide nonempty `instructions`, `criteria`, or both. If present, `criteria` may contain only `true` and `false`; their values may be strings, objects, arrays, or `null`.

```json
{
  "type": "noul",
  "instructions": "Does the state require immediate human review?",
  "criteria": {
    "true": "Delay can cause material harm",
    "false": "Normal automated handling is safe"
  }
}
```

The returned `noul` is the conditional probability assigned to `true`, from `0.0` to `1.0`. It is not a thresholded boolean and not a calibrated probability of real-world truth; choose any automation threshold using representative evaluation data.

### `score`

Use `criteria` as an ordered array of 1 to 36 non-null levels, from lowest at index `0` to highest at index `N-1`. Each level may be a string, object, or array.

```json
{
  "type": "score",
  "instructions": "Rate severity.",
  "criteria": ["Low", "Medium", "High", "Critical"]
}
```

The returned `score` is the probability-weighted mean of the zero-based level indices and can be fractional. `legend` maps those index strings back to the supplied levels. `confidence` is based on probability-weighted distance from the most likely level.

## Constraints and interpretation

- A request may contain multiple questions; all use the same `state` and are evaluated independently.
- Except for single-option `choice` and single-level `score`, each question requires one model inference. Those single-entry cases return deterministic results without inference.
- Input is never truncated. An overlong rendered prompt produces HTTP `422`.
- Invalid schemas, unknown model names, empty state, and invalid criteria also produce HTTP `422` with details in the response body.
- `probabilities` compare only the options you supplied. They cannot indicate that every option is wrong or that important context is missing.
- `confidence` is a concentration summary, not a calibrated confidence or safety guarantee.
- `usage.input_tokens` is summed across inferred questions. The API generates no tokens, so `usage.output_tokens` is always `0`.

For the machine-readable HTTP schema, use `GET /openapi.json`. Interactive documentation is available at `GET /docs`.

---
name: oev-system-one
description: Use the oev System One HTTP API for fast closed-set choices, yes/no likelihoods, and ordinal scores from a shared state. Use when an agent needs a typed decision instead of generated prose.
---

# oev System One API

Use this API to apply one or more structured questions to the same state. The server reads next-token logits from a local language model; it does not generate an explanation or free-form text.

## When to use it

Use System One when the task can be expressed as:

- selecting one item from an explicit set (`choice`);
- estimating how strongly a true/false criterion applies (`noul`); or
- locating the state on an ordered scale (`score`).

Do not use it for open-ended generation, fact retrieval, tool execution, or calibrated risk estimates. Supply all relevant facts in `state` and make the criteria mutually distinguishable.

## Workflow

1. Send `GET /v1/models` to discover the exact model name accepted by this server.
2. Send `POST /v1/systemone` with that model name, one nonempty `state`, and one or more named questions.
3. Match each entry in `answers` to the same key in `questions`.
4. Treat probabilities and confidence values as relative signals over the supplied criteria, not as calibrated correctness probabilities.

Use the same origin that served this document. The built-in local server does not require an API key, though a deployment proxy may add authentication.

## Discover the model

```http
GET /v1/models
```

```json
{
  "models": [
    {
      "name": "google/gemma-4-E2B-it",
      "description": "oev System One: choice, noul, score; up to 25 choices or levels. Probabilities are conditional on listed options and are not calibrated.",
      "release_date": "2026-09-23"
    }
  ]
}
```

Copy `models[0].name` into the request's `model` field. Do not assume the example name is configured on every server.

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

Use `criteria` as an object with 1 to 25 entries. Each key is a nonempty option identifier and each value is a string, object, array, or `null` description. The returned `choice` is exactly one of those keys. The model sees both the key and its description, so use short, meaningful keys and descriptions that state the distinction.

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

Use `criteria` as an ordered array of 1 to 25 non-null levels, from lowest at index `0` to highest at index `N-1`. Each level may be a string, object, or array.

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

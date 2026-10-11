# Responses truncation and reasoning investigation

Investigated the full request/template/inference/parser/SSE/q-consumer path on
2026-10-12. The Mac server used `qwen3.8-27b`, the Dirk/Sharp GGUF checkpoint.
Its catalog still exposed the earlier reasoning metadata, so it was not treated
as a deployment of the current workspace fixes.

## Reproduced causes

| Request to the existing Mac server | Observed result |
| --- | --- |
| Solve a three-modulus integer problem, `effort=xhigh`, output limit omitted | `response.incomplete`, `max_output_tokens`; all 256 generated tokens were reasoning. First thought arrived at 29.44 s, terminal response at 29.45 s. |
| Required `read_file`, `effort=xhigh`, limit 16 | Stream error: required call missing. No reasoning delivered. The final validator treated output-budget exhaustion as a completed invalid call. |
| Calculate 123 × 456, only `enable_thinking=true`, limit 1024 | Completed answer, 866 output tokens, zero reasoning. The automatically inserted `reasoning_effort=none` disabled thinking inside Sharp despite the explicit flag. |
| Read README.md with a function tool, `effort=xhigh`, default/2048 limits | Both completed with 43 output tokens, 15 reasoning tokens, valid function arguments. The problem was not universal tool-parser failure. |

The larger 2048-token version of the modulus problem exceeded the client's
180-second read timeout. It does not establish that raising the budget alone
would produce a completed answer.

q's Responses decoder read a top-level message/error for `response.incomplete`.
The actual reason is nested in `response.incomplete_details`, explaining the
empty error after the colon. The official [incomplete event schema](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_incomplete_event.py)
contains a response and sequence number, not a top-level error message.
[Response.max_output_tokens](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response.py)
includes both reasoning and visible output.

Studio's frontend also cleared `thinkingDraft` as soon as a run reached a
terminal state. Its session projection did not include visible thought history,
so even received reasoning disappeared after completion or reload.

Sharp consumes `enable_thinking` and `reasoning_effort` together. Its native
levels and aliases are described in the [checkpoint model card](https://huggingface.co/peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF).
Globally synthesizing an effort confused that native precedence with the
server's parsing controls.

## Resulting behavior

- Missing HTTP output limits use a configurable per-model default of 8192,
  capped by remaining context. Explicit limits remain unchanged. Responses and
  `/v1/models` expose the applied limit and model default respectively.
- Omitted effort stays undefined in the native template. Explicit thinking
  toggles clear stale disabled levels. `minimal` enables thinking; it does not
  act as `none`. Rendered pre-opened thought channels are detected separately
  from a caller's effort, and their generated token usage is counted.
- Native thought and plain answer channels stream during inference, with
  split markers and unfinished UTF-8 held back. Thoughts preceding tools,
  structured output or logprobs also stream while those outputs await validation.
  Final text must match already delivered deltas; reasoning item IDs are stable.
- Output-budget exhaustion remains incomplete. Partial thoughts/text are
  retained; executable calls are withheld for the entire truncated turn.
  Required-call checks and strict schema failures still apply to completed turns.
- A companion local q change reads nested failure details and budget usage,
  accepts native thought events, and displays one reasoning channel per item
  when both native text and summary events are present. It also persists visible
  thought history separately from model context and displays it after completion,
  failure and session reload. Its publication requires
  separate confirmation because q is a different repository.

## Verification

The complete oev suite passed: **663 passed, 8 skipped**. The q client,
agentloop, workspace, app and Studio Go packages passed. Its Svelte check and
production build passed; two Playwright cases verified completed and failed
thought history at desktop/mobile widths, including session reload. The rendered
screenshots were inspected for retained reasoning and usable layout.

Regression tests exercise the nine native reasoning marker forms, split markers,
pre-opened/computed channels, literal JSON/argument markers, output budgets,
exact-budget EOG, truncated required calls, and final/delta consistency. Real HTTP
tests gate the backend before completion: they fail if thoughts are buffered
until EOF and check preserved thoughts after a later inference failure. A final
document that differs from delivered deltas is rejected before it can be stored
and retrieved as a completed response.

Two downloaded Qwen3-0.6B models were resident concurrently: HF CUDA BF16 and
official Q8_0 GGUF CPU (`llama-cpp-python 0.3.36`). Their actual HTTP results were:

| Runtime / case | First thought | Terminal | Result |
| --- | ---: | ---: | --- |
| HF / reasoning enabled | 0.546 s | 3.968 s | completed; 100 output, 89 reasoning tokens; 82 thought deltas, 5 answer deltas |
| GGUF / reasoning enabled | 0.453 s | 2.594 s | completed; 101 output, 89 reasoning tokens; 82 thought deltas, 6 answer deltas |
| HF / required call, limit 12 | 0.187 s | 0.500 s | incomplete; partial thoughts, no executable call |
| GGUF / required call, limit 12 | 0.484 s | 0.672 s | incomplete; partial thoughts, no executable call |
| HF / only thinking toggle | 0.109 s | 3.593 s | completed; 89 reasoning tokens |
| GGUF / only thinking toggle | 0.156 s | 2.437 s | completed; 89 reasoning tokens |
| HF / thinking disabled | — | 0.375 s | completed; 9 answer deltas, no thoughts |
| GGUF / thinking disabled | — | 0.281 s | completed; 9 answer deltas, no thoughts |

These timings establish delivery before completion, not a throughput benchmark.
For every actual-model request, concatenated answer/thought deltas exactly matched
the terminal document. Both runtimes also passed function calling, result replay,
Responses SSE, JSON schema validation and seeded logprob generation. The actual
q client and agentloop consumer passed both Chat and Responses roundtrips against
the modified server, including native thought delivery and tool-result followup.

The new code has not been deployed to the Mac Studio 27B server. Real MLX/Metal
execution remains unverified from Windows; shared protocol behavior and simulated
MLX runtime coverage are included in the regression suite. Reaching an explicit
output limit is still a valid incomplete outcome; the server does not retry or
increase that limit automatically.

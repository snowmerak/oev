# Chat templates, tools, and reasoning

oev uses each checkpoint's native Jinja chat template and interprets its decoded
output through a shared, dependency-free protocol layer. HF, GGUF and MLX all
use this layer. vLLM and SGLang are research references, not runtime dependencies.
Installing their GPU inference engines is unnecessary for these parsers.

## Research snapshot

Reviewed on 2026-10-12, including parser registries, implementations and upstream
regression tests:

- [vLLM tool calling](https://docs.vllm.ai/en/latest/features/tool_calling/),
  [tool parsers](https://github.com/vllm-project/vllm/tree/f26d9f432ced7b1087d02d29bffed2249df1a18e/vllm/tool_parsers),
  [unified parsers](https://github.com/vllm-project/vllm/tree/f26d9f432ced7b1087d02d29bffed2249df1a18e/vllm/parser),
  [parser tests](https://github.com/vllm-project/vllm/tree/f26d9f432ced7b1087d02d29bffed2249df1a18e/tests/tool_parsers).
- [SGLang tool parsers](https://github.com/sgl-project/sglang/tree/d62b9c2480f3b4b2ff16344c90e6a73e4b1d09fa/python/sglang/srt/function_call),
  [reasoning parsers](https://github.com/sgl-project/sglang/blob/d62b9c2480f3b4b2ff16344c90e6a73e4b1d09fa/python/sglang/srt/parser/reasoning_parser.py),
  [reasoning documentation](https://docs.sglang.io/docs/advanced_features/separate_reasoning).

The main findings incorporated here are preserving special tokens for parsing,
separating rendering from output parsing, using parameter schemas for XML type
conversion, preserving string whitespace, recognizing an implicit reasoning end
at a tool call, and replaying native assistant/tool history through the template.
Parsers are implemented independently against these protocol descriptions.

## Supported tool protocols

Names are compatible aliases, not claims that every checkpoint variant in a
model family has been tested. Explicit wrappers are recognized automatically.
A selected parser additionally enables its ambiguous, unwrapped variants.

| Family / parser names | Recognized output |
| --- | --- |
| Hermes, Qwen 2.5 / original Qwen3, `hermes`, `qwen25`, `json` | `<tool_call>{"name":...,"arguments":...}</tool_call>`; complete JSON envelopes/arrays; nested OpenAI `function` envelopes |
| Qwen3-Coder / 3.5 / 3.8, `qwen3_xml`, `qwen3_coder`, `qwen3`; MiMo, Step3.5 | `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`; bare function blocks with an explicit profile |
| GLM 4.5 / 4.7, `glm`, `glm45`, `glm47` | Tool block with function name and `<arg_key>` / `<arg_value>` pairs, or a JSON argument object |
| Gemma 4, `gemma4` | `<|tool_call>call:NAME{...}<tool_call|>`; omitted `call:` prefix; `<turn|>` closer; nested values and delimited/JSON keys |
| FunctionGemma, `functiongemma` | `<start_function_call>call:NAME{KEY:<escape>VALUE<escape>}<end_function_call>` |
| MiniMax M2, `minimax_m2`, `minimax-m2` | `<minimax:tool_call><invoke name="NAME"><parameter name="KEY">VALUE</parameter></invoke></minimax:tool_call>` |
| DeepSeek V3 / V3.1, `deepseekv3`, `deepseekv31` | Unicode tool section/call/separator tokens, JSON arguments, V3 JSON fences |
| DeepSeek V3.2, `deepseekv32`, `deepseek_dsml` | DSML function/invoke/parameter tags; explicit `string` flags; direct JSON invoke bodies and empty self-closing invocations |
| Kimi K2 / K2.5, `kimi_k2` | Tool section and call tokens; `functions.NAME:INDEX` and JSON arguments |
| Llama JSON, `llama3`, `llama3_json` | `<|python_tag|>` JSON function envelope |
| Mistral, `mistral` | `[TOOL_CALLS]` JSON array; compact `[TOOL_CALLS]NAME[ARGS]{...}` |
| Granite, `granite`, `granite4`, `granite-20b-fc` | JSON tool blocks, `<|tools_prefix|>` arrays, `<function_call>` envelopes |
| Jamba, `jamba` | `<tool_calls>` JSON arrays |
| Dots, `dots` | `<dots_function_call>` JSON envelopes |
| Phi-4 Mini, `phi4_mini_json` | `functools[...]` JSON calls |
| xLAM, `xlam` | Complete JSON function arrays; fenced JSON envelopes with an explicit JSON profile |
| InternLM, `internlm2`, `interns1` | `<|action_start|><|plugin|>` JSON call and action-end token |
| Pythonic / Llama4, `pythonic`, `llama4_pythonic` | Lists of named calls with literal keyword arguments; explicit selection required |
| Olmo3, `olmo3` | Newline-separated literal calls within `<function_calls>`; JSON booleans/null accepted |
| LFM2, `lfm2` | `<|tool_call_start|>` Python-style or JSON call arrays |
| Step3, `step3` | Unicode call-section tokens and `steptml:invoke` / parameter tags |
| GPT-OSS, `gpt-oss`, `openai`, `harmony` | Harmony assistant/tool-recipient headers, JSON body, `<|ghissue|>` / `<|call|>` endings |

XML string parameters preserve their contents, removing only one leading and
trailing template newline. Typed numbers, booleans, arrays, objects and nullable
values are converted using the tool schema, including local references and
composed schemas. JSON argument strings may contain parameter-looking tags.
Python-style calls use AST literal parsing and never execute model output.

Unknown functions, duplicate JSON keys/parameters, malformed or incomplete
calls, invalid strict schemas, required/named-call violations and forbidden
parallel calls return `invalid_model_output`. Some upstream parsers repair
incomplete tags, infer missing function names, or forward unknown tools; oev
does not. Calls are validated as a complete result before SSE emission.

## Reasoning

Supported delimiters include `<think>` / `<thinking>` (Qwen, DeepSeek,
GLM and MiniMax), Gemma thought channels, Kimi's `◁think▷`, Apertus inner
prefix/suffix, Mistral `[THINK]`, Cohere thinking tokens, Granite thinking/
response tokens, and Harmony analysis/final channels. Configurable aliases
include `qwen3`, `deepseek-r1`, `deepseek-v3`, `glm45`, `gemma4`,
`kimi_k2`, `apertus2509`, `mistral`, `cohere_command4`, `granite`
and `gpt-oss`. `none` disables separation.

Chat returns `reasoning_content`. Assistant history also accepts the
`reasoning` and `thinking` aliases; conflicting values are rejected.
Responses preserves native reasoning items and emits `response.reasoning_text.delta`.
Visible thought summaries default to `auto`, emitting the
`response.reasoning_summary_text.delta` events consumed by q.
Explicit `reasoning.summary=null` disables the summary events while retaining
native reasoning content. These summaries contain the model's visible thoughts;
oev does not generate a separate summary.

Reasoning usage counts generated token ranges, including pre-opened thought
blocks, and stops before tool arguments. Effort is passed to the template along
with `enable_thinking` and `thinking`. Actual level semantics depend on
the checkpoint; models with only a toggle do not gain calibrated reasoning
budgets. `none` disables thinking; `minimal` enables it (Sharp maps it to `low`).
An omitted effort is left undefined in the native template rather than injecting
an unsupported level. Server thinking flags default to false. Enabling a thinking
flag explicitly removes an earlier `none`/`off` effort so the template can use its
own native level. Parsing also detects thought prefixes opened by the rendered
prompt, even when the caller did not request an effort.

Plain text and leading native thought/answer channels stream incrementally on
HF, GGUF and MLX. Split control markers and unfinished UTF-8 are held back;
thoughts never become answer deltas. Chat sends `reasoning_content`; Responses
sends native reasoning and, when requested, mirrored summary deltas. Final output
must match the text already streamed, and item IDs remain stable until completion.
Already delivered thoughts survive a later inference or validation failure.

Tool calls, structured answers and logprobs remain buffered for final validation;
their preceding thoughts still stream. Per-tool argument deltas during inference
and grammar-constrained decoding are not implemented. Late reasoning after an
answer has started is not a supported native channel order.

A request without an output limit uses `default_max_output_tokens` (8,192), capped
by the remaining native context. Explicit limits are honored, including the
reasoning tokens consumed before an answer or call. Exhaustion emits
`response.incomplete` with `response.incomplete_details.reason="max_output_tokens"`,
as defined by the [Responses event schema](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_incomplete_event.py).
It preserves partial prose/thoughts and withholds proposed calls; required-call
validation applies to completed turns. Clients must read the nested details
rather than expecting an error message at the event's top level. See the
[investigation and verification report](../reports/responses-reasoning-20261012.md).

## Configuration

No additional flags are needed for explicit supported output delimiters.
Use per-model configuration to select unwrapped formats or customize a template:

```json
{
  "model": "./models/qwen3.8-27b.gguf",
  "served_model": "qwen3.8-27b",
  "backend": "gguf",
  "default_max_output_tokens": 8192,
  "tool_call_parser": "qwen3_coder",
  "reasoning_parser": "qwen3",
  "chat_template_kwargs": {
    "terse": false
  }
}
```

CLI equivalents are `--tool-call-parser`, `--reasoning-parser` and
`--chat-template-kwargs '{"terse":false}'`. `--chat-template` and the
`chat_template` config field accept a Jinja file for HF, GGUF and MLX
generation. HF/MLX typed decisions retain their own native template; a GGUF
override is loaded into its shared tokenizer.

Both generation endpoints accept `chat_template_kwargs`. Precedence is
model defaults, an explicitly supplied API reasoning effort, then request
template kwargs. Callers cannot replace messages, tool definitions, the
template source or rendering controls through this field. Leading consecutive
system/developer instructions are merged for templates requiring one system turn.

`GET /v1/models` advertises `max_input_tokens`, `max_output_tokens`, `default_max_output_tokens`,
the selected tool protocol, task, and discoverable reasoning controls.
Reasoning output parsing does not establish effort support. Templates using only
thinking flags advertise `control="toggle"` and `default_enabled`, without an
effort list. Templates with no discoverable control advertise `fixed`; unknown
effort ranges omit control/levels. Closed literal comparisons and canonical
level assignments in the native Jinja template determine `supported_efforts`.
Comments, history fields and unrelated enums are excluded. For Sharp's
Qwen3.8 template the distinct controls are `none`, `low`, `medium`, `xhigh`;
`minimal`, `high`, and `max` are aliases, not additional native levels.
The API still accepts its existing effort enum for compatibility; acceptance
and template toggling do not imply support for distinct native effort levels.
`context_length` is the conservative minimum of the native context window
and the server input limit, so q's budget stays within the accepted input size.
`context_window` preserves the actual configured/native window. A request's
prompt plus output must still fit that native window. Unknown reasoning
capabilities are omitted; an explicitly disabled parser advertises unsupported.

## Remaining model variants and verification

Parser registries also contain specialized formats for Cohere action/Melody
tool calls, Kimi K3, MiniMax M3, DeepSeek V4/V4.1, Hunyuan's suffixed tokens,
Inkling, Muse Glimmer and other newer families. They are not implied by this
support table; unknown parser names fail configuration validation. New variants
should add a protocol vector, schema conversion checks and HTTP replay tests
rather than relying on a model-name guess.

Regression tests cover 26 representative tool output forms, both generation
endpoints, streaming and non-streaming, tool-result replay, reasoning delimiters,
schema conversion, literal payload markers, invalid calls and model settings.
The Mac Studio Qwen3.8-27B GGUF server reproduced the original failure. Its
actual generated XML was then successfully normalized by the new local parser.
That server was not upgraded during verification. MLX inference cannot be
executed on Windows; shared protocol handling is covered through simulated
backends. Test vectors do not claim inference support for every listed model.

Local Qwen3-0.6B inference checks passed on HF/CUDA and GGUF/CPU for tool calls,
tool-result follow-up, Responses SSE, structured output and reasoning usage.
The native Qwen3.5, Dirk-Qwen3.8 and Gemma4 templates rendered matching tool and
reasoning histories through HF and llama.cpp's Jinja formatter. q's actual Go
client consumed both APIs, tool-result replay, reasoning stream events and model
metadata against a local HTTP server replaying the captured native output form.

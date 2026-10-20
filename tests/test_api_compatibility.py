import json

from fastapi.testclient import TestClient
import numpy as np
import pytest

from oev.generation import GenerationOptions, GenerationResult
from oev.completion import parse_completion
from oev.response_store import ResponseStore
from oev.sampling import sample_logits
from oev.serve import create_app
from test_chat import Backend, CharacterTokenizer, chat_request, service_for, sse_events


FUNCTION = {"name": "weather", "description": "Get weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False,
}, "strict": True}
TOOL = {"type": "function", "function": FUNCTION}
CALL = '<tool_call>{"name":"weather","arguments":{"city":"Seoul"}}</tool_call>'
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


class ThinkingTokenizer(CharacterTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return json.dumps(messages, ensure_ascii=False) + "\nassistant: "


def client_for(text="Hello world!"):
    backend = Backend(text)
    backend.tokenizer = ThinkingTokenizer()
    service, _ = service_for(backend, max_input_tokens=8192)
    return TestClient(create_app(service)), backend


@pytest.mark.parametrize("stream", [False, True])
def test_q_agent_chat_tools_and_result_roundtrip(stream):
    client, backend = client_for(CALL)
    body = chat_request(tools=[TOOL], tool_choice="auto", parallel_tool_calls=True,
                        reasoning_effort="high", conversation_id="q-session", stream=stream)
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    if stream:
        events = sse_events(response)
        assert events[-1] == "[DONE]"
        assert "<tool_call>" not in response.text
        delta = next(event["choices"][0]["delta"] for event in events[:-1] if event["choices"][0]["delta"].get("tool_calls"))
        call = delta["tool_calls"][0]
        assert events[-2]["choices"][0]["finish_reason"] == "tool_calls"
        assistant = {"role": "assistant", "content": None, "tool_calls": [call]}
    else:
        choice = response.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assistant = choice["message"]
        call = assistant["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"city": "Seoul"}
    assert backend.tokenizer.kwargs["enable_thinking"] is True
    assert backend.tokenizer.kwargs["tools"][0]["function"]["name"] == "weather"
    backend.text = "Sunny"
    body["messages"] += [assistant, {"role": "tool", "tool_call_id": call["id"], "content": "Sunny, 20C"}]
    reply = client.post("/v1/chat/completions", json=body)
    assert reply.status_code == 200, reply.text
    prompt = backend.tokenizer.decode(backend.calls[-1][1])
    assert '"arguments": {"city": "Seoul"}' in prompt
    assert '"name": "weather"' in prompt and "Sunny, 20C" in prompt


@pytest.mark.parametrize("stream", [False, True])
def test_parallel_streamed_tool_calls_replay_with_indices(stream):
    second_call = '<tool_call>{"name":"weather","arguments":{"city":"Busan"}}</tool_call>'
    client, backend = client_for(CALL + second_call)
    body = chat_request(tools=[TOOL], stream=True, max_completion_tokens=512)
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    calls = [call for event in sse_events(response)[:-1]
             for call in event["choices"][0]["delta"].get("tool_calls", [])]
    assert [call["index"] for call in calls] == [0, 1]
    body["messages"] += [{"role": "assistant", "content": None, "tool_calls": calls}]
    body["messages"] += [{"role": "tool", "tool_call_id": call["id"], "content": city}
                         for call, city in zip(calls, ("Sunny Seoul", "Rainy Busan"))]
    body.update(stream=stream, store=True)
    backend.text = "Seoul is sunny; Busan is rainy."
    reply = client.post("/v1/chat/completions", json=body)
    assert reply.status_code == 200, reply.text
    prompt = backend.tokenizer.decode(backend.calls[-1][1])
    history = json.loads(prompt.removesuffix("\nassistant: "))
    replayed = next(message["tool_calls"] for message in history if message.get("tool_calls"))
    assert [call["id"] for call in replayed] == [call["id"] for call in calls]
    assert all("index" not in call for call in replayed)
    assert [call["function"]["arguments"]["city"] for call in replayed] == ["Seoul", "Busan"]
    assert "Sunny Seoul" in prompt and "Rainy Busan" in prompt


@pytest.mark.parametrize("index", [-1, True, 0.5, "0"])
def test_invalid_tool_call_indices_fail_before_inference(index):
    client, backend = client_for()
    call = {"id": "call_a", "index": index, "type": "function",
            "function": {"name": "weather", "arguments": '{"city":"Seoul"}'}}
    body = chat_request(messages=[{"role": "assistant", "content": None, "tool_calls": [call]},
                                  {"role": "tool", "tool_call_id": "call_a", "content": "Sunny"}])
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == "messages.0.tool_calls.0.index"
    assert not backend.calls


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("stored", [False, True])
def test_responses_tools_output_replay_and_previous_id(stream, stored):
    client, backend = client_for(CALL)
    body = {"model": "one-model", "input": "Weather?", "tools": [{"type": "function", **FUNCTION}],
            "tool_choice": {"type": "function", "name": "weather"}, "parallel_tool_calls": False,
            "store": stored, "stream": stream, "cache_affinity_id": "q-cache"}
    response = client.post("/v1/responses", json=body)
    assert response.status_code == 200, response.text
    if stream:
        events = sse_events(response)
        result = events[-1]["response"]
        assert events[-1]["type"] == "response.completed"
        assert [event["sequence_number"] for event in events] == list(range(len(events)))
        assert next(e["delta"] for e in events if e["type"] == "response.function_call_arguments.delta") == '{"city":"Seoul"}'
    else:
        result = response.json()
    call = result["output"][0]
    assert call["type"] == "function_call"
    assert call["name"] == "weather"
    assert client.get("/v1/responses/" + result["id"]).status_code == (200 if stored else 404)
    output = {"type": "function_call_output", "call_id": call["call_id"], "output": "20C"}
    body["tool_choice"] = "auto"
    if stored:
        body.update(previous_response_id=result["id"], input=[output])
    else:
        body["input"] = [{"role": "user", "content": "Weather?"}, *result["output"], output]
    backend.text = "Sunny"
    reply = client.post("/v1/responses", json=body)
    assert reply.status_code == 200, reply.text
    assert "20C" in backend.tokenizer.decode(backend.calls[-1][1])


@pytest.mark.parametrize("raw", [
    '<|tool_call>call:weather{city:<|"|>Seoul<|"|>}<tool_call|><|tool_response>',
    CALL + '<|im_end|>',
    '{"name":"weather","arguments":{"city":"Seoul"}}',
    '<|python_tag|>{"name":"weather","parameters":{"city":"Seoul"}}<|eot_id|>',
    '[TOOL_CALLS] [{"name":"weather","arguments":{"city":"Seoul"}}]</s>',
])
def test_native_call_markers_are_not_displayed(raw):
    options = GenerationOptions(tools=(TOOL,), tool_choice="required")
    result = parse_completion(GenerationResult("special tokens removed", 5, 10, "stop", raw_text=raw), options)
    assert result.text == ""
    assert json.loads(result.calls[0]["function"]["arguments"]) == {"city": "Seoul"}


def test_gemma_nested_arguments_and_reasoning():
    function = {"type": "function", "function": {"name": "complex"}}
    raw = '<|channel>thought\nChoose a tool\n<channel|><|tool_call>call:complex{obj:{x:<|"|>a,b:[]{}<|"|>},items:[true,null,3,-1.5]}<tool_call|><|tool_response>'
    result = parse_completion(GenerationResult("", 5, 10, "stop", raw_text=raw), GenerationOptions(tools=(function,)))
    assert result.reasoning == "Choose a tool" and result.text == ""
    assert json.loads(result.calls[0]["function"]["arguments"]) == {"obj": {"x": "a,b:[]{}"}, "items": [True, None, 3, -1.5]}


@pytest.mark.parametrize("raw,options", [
    ("No call", {"tool_choice": "required"}),
    (CALL + CALL, {"parallel_tool_calls": False}),
    (CALL.replace('"weather"', '"unknown"'), {}),
    (CALL.replace('"Seoul"', '4'), {}),
    (CALL[:-10], {}),
])
@pytest.mark.parametrize("stream", [False, True])
def test_invalid_model_calls_never_become_executable_calls(raw, options, stream):
    client, _ = client_for(raw)
    response = client.post("/v1/chat/completions", json=chat_request(tools=[TOOL], stream=stream, **options))
    if stream:
        assert sse_events(response)[-1]["error"]["code"] == "invalid_model_output"
        assert '"finish_reason": "tool_calls"' not in response.text
    else:
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "invalid_model_output"


@pytest.mark.parametrize("effort", ["low", "high"])
@pytest.mark.parametrize("stream", [False, True])
def test_reasoning_separate_from_answer_and_replay(effort, stream):
    client, _ = client_for("<think>Reason first</think>Final")
    response = client.post("/v1/responses", json={"model": "one-model", "input": "Hello",
        "stream": stream, "reasoning": {"effort": effort, "summary": "auto"}})
    assert response.status_code == 200
    result = sse_events(response)[-1]["response"] if stream else response.json()
    assert [item["type"] for item in result["output"]] == ["reasoning", "message"]
    assert result["output"][1]["content"][0]["text"] == "Final"
    replay = [{"role": "user", "content": "Hello"}, *result["output"], {"role": "user", "content": "Again"}]
    assert client.post("/v1/responses", json={"model": "one-model", "input": replay}).status_code == 200


def test_unfinished_reasoning_is_not_a_final_answer():
    result = parse_completion(GenerationResult("Thought", 2, 2, "length", raw_text="<think>Thought"), GenerationOptions(reasoning_effort="high"))
    assert result.reasoning == "Thought" and result.text == ""


def test_plain_text_literals_are_not_treated_as_reasoning():
    raw = "Here is <think>a tag</think>."
    result = parse_completion(GenerationResult(raw, 1, 1, "stop", raw_text=raw), GenerationOptions())
    assert result.text == raw and result.reasoning == ""


def test_reasoning_usage_counts_generated_tokens_and_preopened_prompt():
    from oev.generation import reasoning_token_count
    tokenizer = CharacterTokenizer()
    assert reasoning_token_count(tokenizer.encode("<think>Thought</think>Answer"), tokenizer, "") == 7
    assert reasoning_token_count(tokenizer.encode("Thought</think>Answer"), tokenizer, "<think>") == 7
    assert reasoning_token_count(tokenizer.encode("<|channel>thought\nThought<channel|>Answer"), tokenizer, "") == 7


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_structured_output_schema_checked_before_success(path, stream, valid):
    client, _ = client_for('{"ok":true}' if valid else '{"ok":42}')
    fmt = {"type": "json_schema", "name": "result", "schema": SCHEMA, "strict": True}
    if path.endswith("completions"):
        body = chat_request(stream=stream, response_format={"type": "json_schema", "json_schema": {key: value for key, value in fmt.items() if key != "type"}})
    else:
        body = {"model": "one-model", "input": "Return JSON", "stream": stream, "text": {"format": fmt}}
    response = client.post(path, json=body)
    if valid:
        assert response.status_code == 200
        assert "invalid_model_output" not in response.text
    elif stream:
        error = sse_events(response)[-1]
        assert error.get("error", error)["code"] == "invalid_model_output"
    else:
        assert response.status_code == 502


def test_json_object_and_schema_local_refs():
    client, _ = client_for('{"ok":true}')
    assert client.post("/v1/chat/completions", json=chat_request(response_format={"type": "json_object"})).status_code == 200
    schema = {"$defs": {"answer": SCHEMA}, "$ref": "#/$defs/answer"}
    assert client.post("/v1/responses", json={"model": "one-model", "input": "JSON", "text": {
        "format": {"type": "json_schema", "name": "x", "schema": schema}}}).status_code == 200
    schema["$ref"] = "https://example.com/schema"
    assert client.post("/v1/responses", json={"model": "one-model", "input": "JSON", "text": {
        "format": {"type": "json_schema", "name": "x", "schema": schema}}}).status_code == 400
    schema["$ref"] = "#/missing"
    response = client.post("/v1/responses", json={"model": "one-model", "input": "JSON", "text": {
        "format": {"type": "json_schema", "name": "x", "schema": schema}}})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_model_output"


@pytest.mark.parametrize("stream", [False, True])
def test_multiple_choices_seed_usage_and_stored_chat(stream):
    client, backend = client_for("Hi")
    response = client.post("/v1/chat/completions", json=chat_request(
        n=3, seed=10, stream=stream, store=True, stream_options={"include_usage": True} if stream else None))
    assert response.status_code == 200, response.text
    if stream:
        events = sse_events(response)
        result = client.get("/v1/chat/completions/" + events[0]["id"]).json()
        assert events[-2]["usage"]["completion_tokens"] == 6
        assert {e["choices"][0]["index"] for e in events[:-1] if e["choices"]} == {0, 1, 2}
    else:
        result = response.json()
    assert [call[2].seed for call in backend.calls] == [10, 11, 12]
    assert len(result["choices"]) == 3 and result["usage"]["completion_tokens"] == 6
    assert result["usage"]["prompt_tokens"] == len(backend.calls[0][1])
    assert client.delete("/v1/chat/completions/" + result["id"]).json()["deleted"]
    assert client.get("/v1/chat/completions/" + result["id"]).status_code == 404


def test_stored_history_pagination_instructions_and_delete():
    client, backend = client_for()
    first = client.post("/v1/responses", json={"model": "one-model", "input": "first", "instructions": "OLD", "store": True}).json()
    second = client.post("/v1/responses", json={"model": "one-model", "input": "second", "instructions": "NEW", "store": True,
                                              "previous_response_id": first["id"]}).json()
    prompt = backend.tokenizer.decode(backend.calls[-1][1])
    assert "NEW" in prompt and "OLD" not in prompt and "first" in prompt and "second" in prompt
    url = "/v1/responses/" + second["id"] + "/input_items"
    page = client.get(url, params={"limit": 1, "order": "asc"}).json()
    assert page["has_more"] and page["data"][0]["content"] == [{"type": "input_text", "text": "first"}]
    next_page = client.get(url, params={"after": page["last_id"], "order": "asc"}).json()
    assert len(next_page["data"]) == 2
    assert client.delete("/v1/responses/" + first["id"]).status_code == 200
    assert client.get("/v1/responses/" + first["id"]).status_code == 404
    assert client.get("/v1/responses/" + second["id"]).status_code == 200


@pytest.mark.parametrize("field", [
    {"tool_choice": "required"}, {"tools": [{"type": "web_search"}]},
    {"tool_choice": {"type": "function", "function": None}},
    {"tools": [TOOL, TOOL]}, {"tool_choice": {"type": "function", "function": {"name": "unknown"}}, "tools": [TOOL]},
    {"top_logprobs": 3}, {"seed": -1}, {"presence_penalty": 3}, {"repetition_penalty": 0},
    {"logit_bias": {"bad": 1}}, {"logit_bias": {"1": 101}}, {"metadata": {"x": "a" * 513}},
    {"messages": [{"role": "tool", "tool_call_id": "missing", "content": "Hi"}]},
    {"messages": [{"role": "assistant", "content": None, "tool_calls": [{"id": "a", "function": {"name": "x", "arguments": "not json"}}]}]},
])
def test_bad_requests_fail_before_inference(field):
    client, backend = client_for()
    response = client.post("/v1/chat/completions", json=chat_request(**field))
    assert response.status_code == 400, response.text
    assert not backend.calls


def test_common_fields_sampling_and_larger_output_limit():
    client, backend = client_for("OK")
    response = client.post("/v1/chat/completions", json=chat_request(
        max_completion_tokens=8192, seed=42, top_k=10, min_p=.1, presence_penalty=.2,
        frequency_penalty=.3, repetition_penalty=1.1, logit_bias={"7": -10},
        metadata={"project": "q"}, user="u", safety_identifier="s", prompt_cache_key="p",
        prompt_cache_retention="in-memory", service_tier="auto", conversation_id="q"))
    assert response.status_code == 200, response.text
    options = backend.calls[-1][2]
    assert options.max_new_tokens == 8192 and options.seed == 42 and options.logit_bias == {7: -10}


def test_nullable_optional_fields_use_their_defaults():
    client, _ = client_for()
    assert client.post("/v1/chat/completions", json=chat_request(
        temperature=None, top_p=None, tools=None, tool_choice=None, metadata=None,
        logprobs=None, response_format=None, reasoning_effort=None, n=None)).status_code == 200
    assert client.post("/v1/responses", json={"model": "one-model", "input": "Hello", "text": None,
        "reasoning": {"effort": None}, "include": None, "max_output_tokens": None}).status_code == 200


def test_unknown_option_error_names_the_rejected_field():
    client, backend = client_for()
    response = client.post("/v1/chat/completions", json=chat_request(unknown_option=True))
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "unknown_option"
    assert "unknown_option" in response.json()["error"]["message"]
    assert not backend.calls


def test_text_only_template_receives_function_history_as_text():
    from oev.completion import prompt_messages
    class TextTokenizer:
        chat_template = "{% for m in messages %}{{m.role}}: {{m.content}}{% endfor %}"
    messages = [{"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "weather", "arguments": '{"city":"Seoul"}'}}]},
                {"role": "tool", "tool_call_id": "c", "content": "Sunny"}]
    result = prompt_messages(messages, GenerationOptions(tools=(TOOL,)), TextTokenizer())
    assert "<tool_call>" in result[-2]["content"]
    assert result[-1]["role"] == "user" and "weather returned" in result[-1]["content"]
    assert messages[-1]["role"] == "tool" and messages[-2]["content"] == ""


def test_response_store_bounds_expiry_and_copy(monkeypatch):
    store = ResponseStore(max_entries=2, max_bytes=500, ttl=10)
    for key in ("a", "b", "c"):
        store.put({"id": key, "object": "response"}, [{"content": "X"}])
    with pytest.raises(KeyError):
        store.get("a")
    body, history = store.get("b")
    history[0]["content"] = "modified"
    assert store.get("b")[1][0]["content"] == "X"
    assert store.bytes <= 500
    with pytest.raises(ValueError):
        store.put({"id": "d"}, ["x" * 501])
    monkeypatch.setattr("oev.response_store.time.monotonic", lambda: float("inf"))
    with pytest.raises(KeyError):
        store.get("b")
    assert store.bytes == 0


def test_sampling_bias_penalties_and_request_local_seed():
    rng = np.random.default_rng(0)
    assert sample_logits([0, 2, 1], GenerationOptions(logit_bias={0: 5}), [], rng) == 0
    assert sample_logits([0, 2, 1], GenerationOptions(presence_penalty=2), [1], rng) == 2
    assert sample_logits([0, 2, 1], GenerationOptions(frequency_penalty=1), [1, 1], rng) == 2
    assert sample_logits([0, 2, 1], GenerationOptions(repetition_penalty=3), [], rng, [1]) == 2
    options = GenerationOptions(temperature=1, seed=10)
    def sample():
        rng = np.random.default_rng(options.seed)
        return [int(sample_logits([0, 1, 2], options, [], rng)) for _ in range(20)]
    assert sample() == sample()
    assert len(set(sample())) > 1


def test_sampling_filters_and_logprobs():
    options = GenerationOptions(temperature=1, top_k=2, min_p=.8, logprobs=True, top_logprobs=3)
    token = sample_logits([0, 1, 4], options, [], np.random.default_rng(1))
    assert token == 2 and token.logprob == 0 and token.top == ((2, 0),)
    token = sample_logits([0, 0, 0], GenerationOptions(temperature=1, top_k=1), [], np.random.default_rng(2))
    assert token == 0
    with pytest.raises(ValueError, match="outside model vocabulary"):
        sample_logits([0, 1], GenerationOptions(logit_bias={4: 1}), [], np.random.default_rng(0))


def test_http_logprobs_match_backend_selected_tokens():
    class Logged(Backend):
        def generate_tokens(self, ids, options, cancel):
            logits = np.full(256, -10.)
            logits[ord("A")] = 10
            for _ in range(options.max_new_tokens):
                yield sample_logits(logits, options, [], np.random.default_rng(options.seed))
    service, _ = service_for(Logged())
    response = TestClient(create_app(service)).post("/v1/chat/completions", json=chat_request(logprobs=True, top_logprobs=2, max_tokens=2))
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"]["content"] == "AA"
    assert len(choice["logprobs"]["content"]) == 2
    assert choice["logprobs"]["content"][0]["token"] == "A"
    assert len(choice["logprobs"]["content"][0]["top_logprobs"]) == 2

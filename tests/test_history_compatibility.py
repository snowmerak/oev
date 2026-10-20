"""Round-trip incomplete output and SDK-shaped history through the HTTP API."""

import json

import pytest

from oev.api_requests import ResponsesRequest
from oev.generation import GenerationEngine, GenerationOptions
from test_api_compatibility import client_for
from test_chat import chat_request, sse_events


def prompt_messages(backend):
    prompt = backend.tokenizer.decode(backend.calls[-1][1])
    return json.loads(prompt.removesuffix("\nassistant: "))


def response_result(response, stream):
    assert response.status_code == 200, response.text
    return sse_events(response)[-1]["response"] if stream else response.json()


@pytest.mark.parametrize("stream", [False, True])
def test_reasoning_only_chat_cutoff_can_continue(stream):
    client, backend = client_for("<think>Still reasoning without an answer")
    body = chat_request(reasoning_effort="high", max_completion_tokens=16, stream=stream)
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    if stream:
        frames = sse_events(response)[:-1]
        thought = "".join(frame["choices"][0]["delta"].get("reasoning_content", "")
                          for frame in frames if frame["choices"])
        assert frames[-1]["choices"][0]["finish_reason"] == "length"
        assistant = {"role": "assistant", "content": "", "reasoning_content": thought}
    else:
        choice = response.json()["choices"][0]
        assert choice["finish_reason"] == "length"
        assistant = choice["message"]
        thought = assistant["reasoning_content"]
    assert thought and not assistant["content"] and not assistant.get("tool_calls")
    backend.text = "Answer"
    reply = client.post("/v1/chat/completions", json=chat_request(messages=[
        *body["messages"], assistant, {"role": "user", "content": "Continue"},
    ], stream=stream))
    assert reply.status_code == 200, reply.text
    if stream:
        assert sse_events(reply)[-1] == "[DONE]"
    history = prompt_messages(backend)
    replayed = next(message for message in history if message["role"] == "assistant")
    assert replayed["reasoning_content"] == replayed["reasoning"] == replayed["thinking"] == thought


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("stored", [False, True])
def test_reasoning_only_responses_cutoff_can_continue(stream, stored):
    client, backend = client_for("<think>Still reasoning without an answer")
    first = response_result(client.post("/v1/responses", json={
        "model": "one-model", "input": "Hi", "reasoning": {"effort": "high"},
        "max_output_tokens": 16, "stream": stream, "store": stored,
    }), stream)
    assert first["status"] == "incomplete"
    assert first["incomplete_details"]["reason"] == "max_output_tokens"
    thought = next(item["content"][0]["text"] for item in first["output"] if item["type"] == "reasoning")
    assert thought and all(item["type"] != "function_call" for item in first["output"])
    body = {"model": "one-model", "input": "Continue", "stream": stream}
    if stored:
        body["previous_response_id"] = first["id"]
    else:
        body["input"] = [{"role": "user", "content": "Hi"}, *first["output"],
                         {"role": "user", "content": "Continue"}]
    backend.text = "Answer"
    final = response_result(client.post("/v1/responses", json=body), stream)
    assert final["status"] == "completed"
    replayed = next(message for message in prompt_messages(backend) if message["role"] == "assistant")
    assert replayed["reasoning_content"] == thought


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_empty_assistant_result_can_be_replayed(api):
    client, backend = client_for("")
    if api == "chat":
        first = client.post("/v1/chat/completions", json=chat_request()).json()
        body = chat_request(messages=[*chat_request()["messages"], first["choices"][0]["message"],
                                      {"role": "user", "content": "Continue"}])
        path = "/v1/chat/completions"
    else:
        first = client.post("/v1/responses", json={"model": "one-model", "input": "Hi", "store": True}).json()
        body = {"model": "one-model", "input": "Continue", "previous_response_id": first["id"]}
        path = "/v1/responses"
    backend.text = "Answer"
    reply = client.post(path, json=body)
    assert reply.status_code == 200, reply.text


def test_chat_sdk_null_metadata_and_reasoning_aliases_are_normalized():
    client, backend = client_for()
    assistant = {"role": "assistant", "content": "Hello", "refusal": None,
                 "tool_calls": None, "annotations": None, "audio": None, "function_call": None,
                 "reasoning_content": "Plan", "reasoning": None, "thinking": None}
    response = client.post("/v1/chat/completions", json=chat_request(messages=[
        {"role": "user", "content": "Hi"}, assistant, {"role": "user", "content": "Continue"},
    ]))
    assert response.status_code == 200, response.text
    replayed = prompt_messages(backend)[1]
    assert replayed["reasoning_content"] == "Plan"
    assert not {"audio", "function_call", "annotations", "tool_calls"}.intersection(replayed)
    assistant["annotations"] = []
    assert client.post("/v1/chat/completions", json=chat_request(messages=[assistant])).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("audio", {"id": "audio_1"}), ("function_call", {"name": "weather", "arguments": "{}"}),
    ("annotations", [{"type": "url_citation"}]), ("unknown", None),
])
def test_meaningful_unsupported_chat_metadata_still_fails(field, value):
    client, backend = client_for()
    response = client.post("/v1/chat/completions", json=chat_request(messages=[
        {"role": "assistant", "content": "Hello", field: value},
    ]))
    assert response.status_code == 400
    assert response.json()["error"]["param"] == f"messages.0.{field}"
    assert not backend.calls


@pytest.mark.parametrize("native_thought", [False, True])
def test_responses_sdk_nulls_preserve_visible_reasoning_and_calls(native_thought):
    client, backend = client_for()
    reason = {"type": "reasoning", "id": "rs_1", "encrypted_content": None,
              "summary": [{"type": "summary_text", "text": "Summary plan"}], "content": None}
    if native_thought:
        reason.update(summary=None, content=[{"type": "reasoning_text", "text": "Native plan"}])
    items = [{"role": "user", "content": "Hi"}, reason,
             {"type": "function_call", "call_id": "call_a", "name": "weather",
              "arguments": '{"city":"Seoul"}', "async": None, "caller": None, "namespace": None},
             {"type": "function_call_output", "call_id": "call_a", "output": "Sunny",
              "caller": None, "namespace": None, "name": None},
             {"role": "user", "content": [{"type": "input_text", "text": "Continue",
                                             "annotations": None, "logprobs": None}]}]
    response = client.post("/v1/responses", json={"model": "one-model", "input": items})
    assert response.status_code == 200, response.text
    history = prompt_messages(backend)
    assistant = next(message for message in history if message["role"] == "assistant")
    assert assistant["reasoning_content"] == ("Native plan" if native_thought else "Summary plan")
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"city": "Seoul"}
    assert next(message for message in history if message["role"] == "tool")["content"] == "Sunny"


@pytest.mark.parametrize("field,value", [("encrypted_content", "opaque"), ("namespace", "math"),
                                         ("caller", {"type": "program"}), ("async", True)])
def test_non_null_responses_extensions_still_fail_with_useful_paths(field, value):
    client, backend = client_for()
    item = ({"type": "reasoning", "summary": [], field: value} if field == "encrypted_content" else
            {"type": "function_call", "call_id": "call_a", "name": "weather", "arguments": "{}", field: value})
    response = client.post("/v1/responses", json={"model": "one-model", "input": [item]})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == f"input.0.{field}"
    assert field in response.json()["error"]["message"]
    assert not backend.calls


@pytest.mark.parametrize("assistant", [
    {"role": "assistant", "content": None, "refusal": "REFUSAL_SENTINEL"},
    {"role": "assistant", "content": "Fallback", "refusal": "REFUSAL_SENTINEL"},
    {"role": "assistant", "content": [{"type": "text", "text": "Fallback"},
                                        {"type": "refusal", "refusal": "REFUSAL_SENTINEL"}]},
])
def test_chat_refusal_history_reaches_model(assistant):
    client, backend = client_for()
    response = client.post("/v1/chat/completions", json=chat_request(messages=[
        {"role": "user", "content": "Hi"}, assistant, {"role": "user", "content": "Continue"},
    ]))
    assert response.status_code == 200, response.text
    assert "REFUSAL_SENTINEL" in prompt_messages(backend)[1]["content"]


def test_responses_refusal_and_phase_survive_stored_history():
    client, backend = client_for()
    items = [{"role": "user", "content": "Hi"},
             {"role": "assistant", "phase": "commentary", "content": [
                 {"type": "output_text", "text": "Fallback"},
                 {"type": "refusal", "refusal": "REFUSAL_SENTINEL"}]},
             {"role": "user", "content": "Another question"}]
    first = response_result(client.post("/v1/responses", json={
        "model": "one-model", "input": items, "store": True,
    }), False)
    reply = client.post("/v1/responses", json={
        "model": "one-model", "input": "Continue", "previous_response_id": first["id"],
    })
    assert reply.status_code == 200, reply.text
    assistant = prompt_messages(backend)[1]
    assert assistant["phase"] == "commentary"
    assert assistant["content"] == "Fallback\nREFUSAL_SENTINEL"
    stored = client.get(f"/v1/responses/{first['id']}/input_items", params={"order": "asc"}).json()
    assert stored["data"][1]["phase"] == "commentary"
    assert stored["data"][1]["content"][1] == {"type": "refusal", "refusal": "REFUSAL_SENTINEL"}


@pytest.mark.parametrize("role", ["user", "system", "developer", "tool"])
def test_refusal_parts_require_assistant_role(role):
    client, backend = client_for()
    message = {"role": role, "content": [{"type": "refusal", "refusal": "No"}]}
    if role == "tool":
        message["tool_call_id"] = "call_a"
    assert client.post("/v1/chat/completions", json=chat_request(messages=[message])).status_code == 400
    if role != "tool":
        assert client.post("/v1/responses", json={"model": "one-model", "input": [message]}).status_code == 400
    assert not backend.calls


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_input_items_normalize_message_strings(role):
    client, _ = client_for()
    first = client.post("/v1/responses", json={"model": "one-model", "store": True,
                                              "input": [{"role": role, "content": "Hello"}]}).json()
    page = client.get(f"/v1/responses/{first['id']}/input_items").json()
    item = page["data"][0]
    assert item["id"] and item["status"] == "completed"
    part = item["content"][0]
    assert part["type"] == ("output_text" if role == "assistant" else "input_text")
    assert part["text"] == "Hello"


def test_input_items_shorthand_and_pagination_stay_stable():
    client, _ = client_for()
    first = client.post("/v1/responses", json={"model": "one-model", "input": "Hello", "store": True}).json()
    url = f"/v1/responses/{first['id']}/input_items"
    one = client.get(url, params={"order": "asc", "limit": 1}).json()
    two = client.get(url, params={"order": "asc", "limit": 1}).json()
    assert one == two
    assert one["data"][0]["content"] == [{"type": "input_text", "text": "Hello"}]


def test_input_item_query_errors_use_openai_envelope():
    client, _ = client_for()
    response = client.get("/v1/responses/any/input_items", params={"limit": "abc"})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "limit"
    systemone = client.post("/v1/systemone", json={})
    assert systemone.status_code == 422 and "detail" in systemone.json()


@pytest.mark.parametrize("item,param", [
    ({"role": "user", "content": [{"type": "input_text", "text": "Hi", "unknown": True}]}, "input.0.content.0.unknown"),
    ({"type": "function_call", "call_id": "a", "name": "weather", "arguments": "{}", "namespace": "x"}, "input.0.namespace"),
    ({"role": "invalid", "content": "Hi"}, "input.0.role"),
])
def test_nested_response_errors_name_actual_fields(item, param):
    client, backend = client_for()
    response = client.post("/v1/responses", json={"model": "one-model", "input": [item]})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == param
    assert not backend.calls


def test_standalone_reasoning_item_is_not_silently_discarded():
    body = ResponsesRequest.model_validate({"model": "one-model", "input": [
        {"role": "user", "content": "Hi"},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Plan"}]},
        {"role": "user", "content": "Continue"},
    ]})
    messages = body.messages()
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[1]["reasoning_content"] == "Plan"


@pytest.mark.parametrize("native_history", [False, True, "needs_answer"])
def test_reasoning_only_history_survives_native_and_text_only_templates(native_history):
    from jinja2 import Environment
    from test_chat import Backend

    template = ("{% for m in messages %}{{ m.role }}: "
                "{% if native_history and m.reasoning_content is defined "
                "and (native_history != 'needs_answer' or m.content) %}"
                "<think>{{ m.reasoning_content }}</think>{% endif %}{{ m.content }}\n"
                "{% endfor %}assistant: ")

    class NativeTokenizer:
        chat_template = template

        def apply_chat_template(self, messages, **kwargs):
            return Environment().from_string(template).render(messages=messages, native_history=native_history)

        def encode(self, text, **kwargs):
            return list(text.encode())

        def decode(self, tokens, **kwargs):
            return bytes(tokens).decode()

    backend = Backend()
    backend.tokenizer = NativeTokenizer()
    messages = [{"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "", "reasoning_content": "Partial plan\nNext step"},
                {"role": "user", "content": "Continue"}]
    prepared = GenerationEngine(backend).prepare(messages, GenerationOptions(8))
    prompt = backend.tokenizer.decode(prepared.input_ids)
    assert prompt.count("Partial plan\nNext step") == 1
    assert ("<think>Partial plan" in prompt) is (native_history is True)
    assert "oev_reasoning_" not in prompt
    assert GenerationEngine(backend).prepare(messages, GenerationOptions(8)).input_ids == prepared.input_ids
    assert messages[1]["content"] == "" and messages[1]["reasoning_content"] == "Partial plan\nNext step"


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_adjacent_text_parts_keep_their_original_concatenation(api):
    client, backend = client_for()
    kind = "text" if api == "chat" else "input_text"
    message = {"role": "user", "content": [{"type": kind, "text": "Hel"}, {"type": kind, "text": "lo"}]}
    if api == "chat":
        response = client.post("/v1/chat/completions", json=chat_request(messages=[message]))
    else:
        response = client.post("/v1/responses", json={"model": "one-model", "input": [message]})
    assert response.status_code == 200, response.text
    assert prompt_messages(backend)[0]["content"] == "Hello"

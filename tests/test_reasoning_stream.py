"""Exercise output budgets, native controls and thought delivery before EOF."""

from contextlib import contextmanager
import json
import socket
from threading import Event, Thread
import time

from fastapi.testclient import TestClient
import httpx
from jinja2 import Environment
import pytest
import uvicorn

from oev.api_requests import ResponsesRequest
from oev.completion import parse_completion
from oev.generation import GenerationEngine, GenerationOptions, GenerationResult, reasoning_token_count
from oev.output_parsers import REASONING_MARKERS, OutputAccumulator, ReasoningAccumulator
from oev.serve import ModelConfig, create_app
from oev.systemone import SystemOneService
from oev.engine import DecisionEngine
from test_api_compatibility import ThinkingTokenizer, TOOL
from test_chat import Backend, chat_request, service_for, sse_events


def test_omitted_budget_uses_model_default_and_remaining_context():
    backend = Backend()
    backend.tokenizer = ThinkingTokenizer()
    backend.context_window = 100
    engine = GenerationEngine(backend, default_max_output_tokens=70)
    body = ResponsesRequest(model="one-model", input="Hi")
    prepared = engine.prepare(body.messages(), body.options())
    assert prepared.options.max_new_tokens == 100 - len(prepared.input_ids)
    explicit = body.model_copy(update={"max_output_tokens": 3})
    assert engine.prepare(explicit.messages(), explicit.options()).options.max_new_tokens == 3
    with pytest.raises(ValueError, match="context limit"):
        engine.prepare(body.messages(), GenerationOptions(max_new_tokens=70))
    service = SystemOneService(DecisionEngine(backend), "one-model", default_max_output_tokens=70)
    client = TestClient(create_app(service))
    result = client.post("/v1/responses", json={"model": "one-model", "input": "Hi"}).json()
    assert result["max_output_tokens"] == prepared.options.max_new_tokens
    assert client.get("/v1/models").json()["data"][0]["default_max_output_tokens"] == 70


@pytest.mark.parametrize("value", [0, True, 32769])
def test_invalid_model_default_budget(value):
    with pytest.raises(ValueError):
        ModelConfig(model="local", default_max_output_tokens=value)
    with pytest.raises(ValueError):
        GenerationEngine(Backend(), default_max_output_tokens=value)


@pytest.mark.parametrize("fields", [
    {"chat_template_kwargs": {"enable_thinking": True}},
    {"reasoning": {"effort": "none"}, "chat_template_kwargs": {"thinking": True}},
    {"reasoning": {"effort": "minimal"}},
])
def test_native_sharp_toggle_and_minimal_do_not_disable_thinking(fields):
    # Sharp consumes both flags: a stale 'none' effort overrides enable_thinking.
    template = """{% set enabled = enable_thinking %}
    {% if reasoning_effort is defined and reasoning_effort in ['none', 'off'] %}
      {% set enabled = false %}
    {% endif %}{{ '<think>' if enabled else '' }}"""

    class NativeTokenizer(ThinkingTokenizer):
        chat_template = template

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return Environment().from_string(template).render(**kwargs).strip()

    backend = Backend("Thought</think>Answer")
    backend.tokenizer = NativeTokenizer()
    body = ResponsesRequest(model="one-model", input="Hi", **fields)
    engine = GenerationEngine(backend)
    prepared = engine.prepare(body.messages(), body.options())
    assert backend.tokenizer.decode(prepared.input_ids) == "<think>"
    assert prepared.reasoning_prefix == "<think>"
    result = engine.generate(prepared)
    assert parse_completion(result, prepared.options).reasoning == "Thought"
    assert result.reasoning_tokens == len("Thought")


@pytest.mark.parametrize("family,opening,closing", REASONING_MARKERS)
@pytest.mark.parametrize("preopened", [False, True])
def test_split_native_markers_stream_without_duplicates(family, opening, closing, preopened):
    source = opening + "First thought.\nSecond thought." + closing + "Answer"
    accumulator = ReasoningAccumulator(family, opening if preopened else "")
    start = len(opening) if preopened else 1
    text = "".join(accumulator.advance(source[:end]) for end in range(start, len(source) + 1))
    assert text == "First thought.\nSecond thought."
    assert "Answer" not in text


def test_reasoning_stream_holds_stop_prefix_and_tool_argument_literals():
    raw = '<think>Plan</think><tool_call>{"name":"weather","arguments":{"city":"<think>literal</think>"}}</tool_call>'
    accumulator = ReasoningAccumulator("think")
    assert "".join(accumulator.advance(raw[:end]) for end in range(1, len(raw) + 1)) == "Plan"
    accumulator = ReasoningAccumulator("think", stops=("END",))
    source = "<think>Plan EN"
    assert "".join(accumulator.advance(source[:end]) for end in range(1, len(source) + 1)) == "Plan"
    assert ReasoningAccumulator("think").advance('{"value":"<think>literal"}') == ""


@pytest.mark.parametrize("family,opening,closing", REASONING_MARKERS)
def test_live_answer_channel_follows_thought_without_control_marker_leaks(family, opening, closing):
    prefix = "<|start|>assistant" if family == "harmony" else ""
    final_header = {"harmony": "<|start|>assistant<|channel|>final<|message|>",
                    "gemma4": "<|channel>final\n", "cohere": "<|START_RESPONSE|>"}.get(family, "")
    ending = "<|END_RESPONSE|>" if family == "cohere" else "<|im_end|>"
    raw = prefix + opening + "Plan" + closing + final_header + "Answer" + ending
    accumulator = OutputAccumulator(family)
    thoughts, answer = "", ""
    for end in range(1, len(raw) + 1):
        text, thought = accumulator.advance(raw[:end])
        thoughts += thought
        answer += text
    assert thoughts == "Plan"
    assert answer == "Answer"


def test_native_thinking_off_answer_still_streams_incrementally():
    accumulator = OutputAccumulator("think")
    first, _ = accumulator.advance("Hello ")
    second, _ = accumulator.advance("Hello world!")
    assert first == "Hello" and second == " world!"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("path", ["/v1/responses", "/v1/chat/completions"])
@pytest.mark.parametrize("raw", [
    "<think>Plan more",
    "<think>Plan</think><tool_call>{\"name\":\"weather\",\"arguments\":{",
    "<think>Plan</think><tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Seoul\"}}</tool_call><tool_call>",
])
def test_truncated_required_tools_preserve_thoughts_without_executable_calls(path, stream, raw):
    backend = Backend(raw)
    backend.tokenizer = ThinkingTokenizer()
    backend.tokenizer.chat_template = "<think> tools"
    client = TestClient(create_app(service_for(backend)[0]))
    if path.endswith("responses"):
        body = {"model": "one-model", "input": "Hi", "tools": [{"type": "function", **TOOL["function"]}],
                "max_output_tokens": len(raw)}
    else:
        body = chat_request(tools=[TOOL], max_tokens=len(raw))
    body.update(stream=stream, tool_choice="required")
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    if path.endswith("responses"):
        events = sse_events(response) if stream else []
        document = events[-1]["response"] if stream else response.json()
        assert document["status"] == "incomplete"
        assert document["completed_at"] is None
        assert document["incomplete_details"] == {"reason": "max_output_tokens"}
        assert not any(item["type"] == "function_call" for item in document["output"])
        reasoning = document["output"][0]["content"][0]["text"]
        if stream:
            assert events[-1]["type"] == "response.incomplete"
            assert "".join(e["delta"] for e in events if e["type"] == "response.reasoning_summary_text.delta") == reasoning
            added = next(e["item"]["id"] for e in events if e["type"] == "response.output_item.added")
            assert added == document["output"][0]["id"]
    else:
        events = sse_events(response)[:-1] if stream else []
        choice = events[-1]["choices"][0] if stream else response.json()["choices"][0]
        assert choice["finish_reason"] == "length"
        reasoning = "".join(e["choices"][0]["delta"].get("reasoning_content", "") for e in events) if stream else choice["message"]["reasoning_content"]
        assert "tool_calls" not in response.text
    assert reasoning == ("Plan more" if raw.endswith("more") else "Plan")
    assert "<tool_call>" not in response.text


@pytest.mark.parametrize("raw", ['{"answer":"a <think>literal', '{"name":"a <thi'])
def test_partial_json_answer_is_not_reclassified_as_tool_or_reasoning(raw):
    result = parse_completion(GenerationResult(raw, 1, len(raw), "length", raw_text=raw),
                              GenerationOptions(tools=(TOOL,), response_format={"type": "json_object"}))
    assert result.text == raw and result.reasoning == "" and not result.calls


def test_reasoning_usage_excludes_markers_inside_tool_arguments():
    raw = '<think>Plan</think><tool_call>{"name":"weather","arguments":{"city":"<think>literal</think>"}}</tool_call>'
    tokenizer = ThinkingTokenizer()
    assert reasoning_token_count(tokenizer.encode(raw), tokenizer, "") == 4


def test_rendered_reasoning_prefix_detects_a_computed_template_channel():
    class ComputedTokenizer(ThinkingTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return "<think>"
    backend = Backend("Plan</think>Answer")
    backend.tokenizer = ComputedTokenizer()
    engine = GenerationEngine(backend)
    prepared = engine.prepare([{"role": "user", "content": "Hi"}], GenerationOptions())
    assert prepared.options.reasoning_parser == "think"
    assert parse_completion(engine.generate(prepared), prepared.options).reasoning == "Plan"


@contextmanager
def running_server(service):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(create_app(service), log_level="error"))
    thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        assert not thread.is_alive()


@pytest.mark.parametrize("path", ["/v1/responses", "/v1/chat/completions"])
@pytest.mark.parametrize("fail_after_thought", [False, True])
def test_http_thought_arrives_before_generation_finishes_even_on_failure(path, fail_after_thought):
    release, finished = Event(), Event()

    class GatedBackend(Backend):
        def generate_tokens(self, ids, options, cancel):
            try:
                for char in "<think>Visible thought ":
                    yield ord(char)
                while not release.wait(0.01):
                    if cancel.is_set():
                        return
                if fail_after_thought:
                    raise RuntimeError("backend detail must remain private")
                for char in "ends</think>Answer":
                    yield ord(char)
            finally:
                finished.set()

    backend = GatedBackend()
    backend.tokenizer = ThinkingTokenizer()
    backend.tokenizer.chat_template = "<think>"
    body = {"model": "one-model", "input": "Hi", "stream": True} if path.endswith("responses") else chat_request(stream=True)
    with running_server(service_for(backend)[0]) as url, httpx.Client(base_url=url, timeout=5) as client:
        events, thought = [], ""
        try:
            with client.stream("POST", path, json=body) as response:
                for line in response.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    events.append(event)
                    delta = event.get("delta", "") if event.get("type") == "response.reasoning_summary_text.delta" else ""
                    if path.endswith("completions") and event.get("choices"):
                        delta = event["choices"][0]["delta"].get("reasoning_content", "")
                    thought += delta
                    if delta and not release.is_set():
                        assert not finished.is_set(), "reasoning was buffered until backend EOF"
                        release.set()
        finally:
            release.set()
        assert finished.wait(5)
        assert thought == ("Visible thought" if fail_after_thought else "Visible thought ends")
        if fail_after_thought:
            assert "backend detail" not in json.dumps(events)
            assert events[-1].get("type") == "error" or "error" in events[-1]
        elif path.endswith("responses"):
            assert events[-1]["type"] == "response.completed"
            assert events[-1]["response"]["output"][0]["content"][0]["text"] == thought
        else:
            assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_eog_at_exact_budget_is_completed():
    class EOGBackend(Backend):
        def is_end_token(self, token):
            return token == ord("!")

    backend = EOGBackend("Hello!")
    prepared = GenerationEngine(backend).prepare([{"role": "user", "content": "Hi"}], GenerationOptions(max_new_tokens=6))
    events = list(GenerationEngine(backend).stream(prepared, Event()))
    assert isinstance(events[-1], GenerationResult)
    assert events[-1].finish_reason == "stop"


@pytest.mark.parametrize("path", ["/v1/responses", "/v1/chat/completions"])
def test_reopened_thought_after_answer_is_not_silently_reordered(path):
    backend = Backend("<think>Plan</think>Answer<think>More</think>")
    backend.tokenizer = ThinkingTokenizer()
    backend.tokenizer.chat_template = "<think>"
    body = {"model": "one-model", "input": "Hi", "stream": True} if path.endswith("responses") else chat_request(stream=True)
    response = TestClient(create_app(service_for(backend)[0])).post(path, json=body)
    assert "Model opened reasoning after its answer" in response.text
    assert "response.completed" not in response.text and "[DONE]" not in response.text


def test_inconsistent_stream_cannot_be_retrieved_as_a_completed_response(monkeypatch):
    service, _ = service_for()

    def inconsistent_output(prepared, cancel):
        yield "Already delivered"
        yield GenerationResult("Different final text", 1, 1, "stop")

    monkeypatch.setattr(service, "stream_generation", inconsistent_output)
    client = TestClient(create_app(service))
    response = client.post("/v1/responses", json={
        "model": "one-model", "input": "Hi", "stream": True, "store": True,
    })
    events = sse_events(response)
    identifier = events[0]["response"]["id"]
    assert events[-1]["type"] == "error"
    assert "Final answer differs" in events[-1]["message"]
    assert client.get("/v1/responses/" + identifier).status_code == 404

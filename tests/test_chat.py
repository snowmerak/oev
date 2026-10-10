import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import socket
from threading import Event, Thread
import time

from fastapi.testclient import TestClient
import httpx
import pytest
import uvicorn

from oev import DecisionEngine, GenerationEngine, GenerationOptions, SystemOneService
from oev.chat import generation_events
from oev.serve import create_app


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return json.dumps(messages, ensure_ascii=False) + "\nassistant: "

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, tokens, **kwargs):
        return "".join(map(chr, tokens))


class Backend:
    tokenizer = CharacterTokenizer()
    model_name = "one-model"

    def __init__(self, text="Hello world!", fail=False):
        self.text = text
        self.fail = fail
        self.calls = []
        self.closed = Event()

    def selected_logits(self, ids, slots):
        self.calls.append(("decision", ids))
        return [0.0, 1.0]

    def generate_tokens(self, ids, options, cancel):
        self.calls.append(("generation", ids, options))
        try:
            for char in self.text[:options.max_new_tokens]:
                if cancel.is_set():
                    return
                yield ord(char)
            if self.fail:
                raise RuntimeError("private backend details")
        finally:
            self.closed.set()


def service_for(backend=None, max_concurrency=4, max_input_tokens=4096):
    backend = backend or Backend()
    return SystemOneService(
        DecisionEngine(backend, max_input_tokens=max_input_tokens),
        "one-model", max_concurrency=max_concurrency,
    ), backend


def chat_request(**fields):
    return {"model": "one-model", "messages": [{"role": "user", "content": "Hello"}], **fields}


def decision_request():
    return {"model": "one-model", "state": "state", "questions": {
        "route": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
    }}


def sse_events(response):
    events = []
    for block in response.text.strip().split("\n\n"):
        data = next(line[6:] for line in block.splitlines() if line.startswith("data: "))
        events.append(data if data == "[DONE]" else json.loads(data))
    return events


def test_three_apis_share_backend_and_preserve_systemone_contract():
    service, backend = service_for()
    client = TestClient(create_app(service))
    response = client.post("/v1/chat/completions", json=chat_request(max_completion_tokens=20))
    assert response.status_code == 200
    chat = response.json()
    assert chat["object"] == "chat.completion"
    assert chat["choices"][0]["message"]["content"] == "Hello world!"
    assert chat["choices"][0]["finish_reason"] == "stop"
    assert chat["usage"]["completion_tokens"] == 12
    response = client.post("/v1/responses", json={"model": "one-model", "input": "Hello"})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "completed"
    assert result["output"][0]["content"][0]["text"] == "Hello world!"
    assert result["usage"]["output_tokens"] == 12
    assert result["store"] is False
    assert client.post("/v1/systemone", json=decision_request()).json()["answers"]["route"]["choice"] == "b"
    assert [call[0] for call in backend.calls] == ["generation", "generation", "decision"]
    models = client.get("/v1/models").json()
    assert models["data"][0]["id"] == models["models"][0]["name"] == service.model


def test_text_parts_instructions_and_response_output_replay():
    service, backend = service_for()
    client = TestClient(create_app(service))
    assert client.post("/v1/chat/completions", json=chat_request(messages=[
        {"role": "developer", "content": "Be concise"},
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
    ])).status_code == 200
    prompt = backend.tokenizer.decode(backend.calls[-1][1])
    assert '"role": "system"' in prompt
    body = {"model": "one-model", "instructions": "Be concise", "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
    ]}
    response = client.post("/v1/responses", json=body).json()
    body["input"].extend(response["output"])
    body["input"].append({"role": "user", "content": "Again"})
    assert client.post("/v1/responses", json=body).status_code == 200
    prompt = backend.tokenizer.decode(backend.calls[-1][1])
    assert "Be concise" in prompt and "Hello world!" in prompt and "Again" in prompt


def test_chat_stream_deltas_finish_and_usage():
    service, _ = service_for()
    client = TestClient(create_app(service))
    response = client.post("/v1/chat/completions", json=chat_request(
        stream=True, max_tokens=5, stream_options={"include_usage": True},
    ))
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response)
    assert events[-1] == "[DONE]"
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    chunks = events[:-1]
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks[:-1]) == "Hello"
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["completion_tokens"] == 5


@pytest.mark.parametrize(("limit", "terminal", "status"), [
    (5, "response.incomplete", "incomplete"), (20, "response.completed", "completed"),
])
def test_responses_stream_lifecycle_and_final_text(limit, terminal, status):
    service, _ = service_for()
    response = TestClient(create_app(service)).post("/v1/responses", json={
        "model": "one-model", "input": "Hello", "stream": True, "max_output_tokens": limit,
    })
    events = sse_events(response)
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    assert [event["type"] for event in events[:4]] == [
        "response.created", "response.in_progress", "response.output_item.added", "response.content_part.added",
    ]
    text = "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta")
    final = events[-1]
    assert final["type"] == terminal
    assert final["response"]["status"] == status
    assert text == final["response"]["output"][0]["content"][0]["text"] == "Hello world!"[:limit]
    assert final["response"]["usage"]["output_tokens"] == min(12, limit)


@pytest.mark.parametrize("fields", [
    {"tools": [{"type": "function"}]}, {"n": 2}, {"temperature": 3},
    {"messages": []}, {"messages": [{"role": "user", "content": ""}]},
    {"max_tokens": 0}, {"max_tokens": 3, "max_completion_tokens": 4},
    {"stop": ""}, {"stop": ["1", "2", "3", "4", "5"]},
    {"stream_options": {"include_usage": True}}, {"store": True},
    {"response_format": {"type": "json_object"}},
    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
])
def test_unsupported_or_invalid_requests_fail_before_generation(fields):
    service, backend = service_for()
    response = TestClient(create_app(service)).post("/v1/chat/completions", json=chat_request(**fields))
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not backend.calls


def test_unknown_model_and_context_limits_and_stateless_responses():
    service, backend = service_for(max_input_tokens=10)
    client = TestClient(create_app(service))
    assert client.post("/v1/chat/completions", json=chat_request(model="wrong")).status_code == 404
    response = client.post("/v1/chat/completions", json=chat_request(stream=True))
    assert response.status_code == 400
    assert "truncation is disabled" in response.json()["error"]["message"]
    assert client.post("/v1/responses", json={
        "model": "one-model", "input": "Hello", "previous_response_id": "resp_x",
    }).status_code == 400
    assert not backend.calls


def test_stop_sequences_never_leak_into_streamed_text():
    service, backend = service_for(Backend("안녕 hello END ignored"))
    response = TestClient(create_app(service)).post("/v1/chat/completions", json=chat_request(
        stream=True, stop=["END", "ignored"], stream_options={"include_usage": True},
    ))
    events = sse_events(response)[:-1]
    text = "".join(event["choices"][0]["delta"].get("content", "") for event in events if event["choices"])
    assert text == "안녕 hello "
    assert events[-2]["choices"][0]["finish_reason"] == "stop"
    assert backend.closed.is_set()


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_generation_failures_are_reported_without_backend_details(path, stream):
    service, _ = service_for(Backend(fail=True))
    body = chat_request(stream=stream) if path.endswith("completions") else {
        "model": "one-model", "input": "Hello", "stream": stream,
    }
    response = TestClient(create_app(service)).post(path, json=body)
    assert "private backend details" not in response.text
    if stream:
        assert response.status_code == 200
        events = sse_events(response)
        error = events[-1]
        assert error.get("error", error)["code"] == "generation_error"
        assert "response.completed" not in response.text
    else:
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "generation_error"


def test_decision_and_generation_share_concurrency_slots():
    entered, release, decision_entered = Event(), Event(), Event()

    class BlockingBackend(Backend):
        def generate_tokens(self, ids, options, cancel):
            entered.set()
            assert release.wait(timeout=5)
            yield from super().generate_tokens(ids, options, cancel)

        def selected_logits(self, ids, slots):
            decision_entered.set()
            return super().selected_logits(ids, slots)

    service, _ = service_for(BlockingBackend(), max_concurrency=1)
    prepared = service.prepare_generation([{"role": "user", "content": "Hello"}], GenerationOptions())
    with ThreadPoolExecutor(max_workers=2) as pool:
        generation = pool.submit(service.generate, prepared)
        try:
            assert entered.wait(timeout=5)
            decision = pool.submit(service.evaluate, decision_request())
            assert not decision_entered.wait(timeout=0.1)
        finally:
            release.set()
        assert generation.result(timeout=5).text == "Hello world!"
        assert decision.result(timeout=5)["answers"]["route"]["choice"] == "b"


def test_closing_stream_cancels_worker_and_releases_slot():
    stopped = Event()

    class EndlessBackend(Backend):
        def generate_tokens(self, ids, options, cancel):
            try:
                while not cancel.is_set():
                    yield ord(" ")
            finally:
                stopped.set()

    class ConnectedRequest:
        async def is_disconnected(self):
            return False

    service, _ = service_for(EndlessBackend(), max_concurrency=1)
    prepared = service.prepare_generation([{"role": "user", "content": "Hello"}], GenerationOptions())

    async def consume():
        stream = generation_events(service, prepared, ConnectedRequest())
        await anext(stream)
        await stream.aclose()

    asyncio.run(consume())
    assert stopped.wait(timeout=5)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(service.evaluate, decision_request()).result(timeout=5)["answers"]["route"]["choice"] == "b"


def test_generation_backend_is_optional_for_existing_decision_backends():
    class DecisionOnly:
        tokenizer = CharacterTokenizer()
        model_name = "decision-only"

        def selected_logits(self, ids, slots):
            return [0, 1]

    service, _ = service_for(DecisionOnly())
    client = TestClient(create_app(service))
    assert client.post("/v1/chat/completions", json=chat_request()).status_code == 501
    assert client.post("/v1/systemone", json=decision_request()).status_code == 200


def test_http_disconnect_cancels_generation_and_keeps_decisions_available():
    stopped = Event()

    class EndlessBackend(Backend):
        def generate_tokens(self, ids, options, cancel):
            try:
                while not cancel.is_set():
                    yield ord(" ")
            finally:
                stopped.set()

    service, _ = service_for(EndlessBackend(), max_concurrency=1)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(service), log_level="error"))
    thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5) as client:
            with client.stream("POST", "/v1/chat/completions", json=chat_request(stream=True)) as response:
                assert response.status_code == 200
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        chunk = json.loads(line[6:])
                        if chunk["choices"][0]["delta"].get("content"):
                            break
            assert stopped.wait(timeout=5)
            response = client.post("/v1/systemone", json=decision_request())
            assert response.status_code == 200
            assert response.json()["answers"]["route"]["choice"] == "b"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
    assert not thread.is_alive()

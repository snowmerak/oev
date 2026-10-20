"""Model JSON defaults reach both APIs without overriding explicit requests."""

import json

from fastapi.testclient import TestClient
import pytest

from oev import DecisionEngine, serve
from test_chat import Backend, sse_events


DEFAULTS = {
    "temperature": .7, "top_p": .9, "top_k": 40, "min_p": .05,
    "presence_penalty": .2, "frequency_penalty": .3, "repetition_penalty": 1.1,
    "seed": 42, "logit_bias": {"1": .4},
}


@pytest.fixture
def configured_server(monkeypatch, tmp_path):
    backends = {name: Backend() for name in ("alpha", "beta", "legacy")}
    launched = []
    monkeypatch.setattr(serve.DecisionEngine, "from_pretrained",
                        lambda model, **kwargs: DecisionEngine(backends[model]))
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: launched.append(app))
    config = tmp_path / "models.json"
    config.write_text(json.dumps([
        {"model": "alpha", "generation_defaults": DEFAULTS},
        {"model": "beta", "generation_defaults": {"temperature": .3, "min_p": .02}},
        {"model": "legacy"},
    ]), encoding="utf-8")
    serve.main(["--models-config", str(config)])
    return TestClient(launched[0]), backends


def request_body(path, **fields):
    input_field = {"messages": [{"role": "user", "content": "Hi"}]} if path.endswith("completions") else {"input": "Hi"}
    return {"model": "alpha", **input_field, **fields}


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_json_defaults_and_overrides_reach_inference(configured_server, path, stream):
    client, backends = configured_server
    for overrides in ({}, {"temperature": None, "top_p": None, "top_k": None}, {
        "temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0,
        "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1,
        "seed": None, "logit_bias": {},
    }):
        response = client.post(path, json=request_body(path, stream=stream, **overrides))
        assert response.status_code == 200, response.text
        if stream:
            chunks = sse_events(response)
        applied = backends["alpha"].calls[-1][2]
        for key, default in DEFAULTS.items():
            expected = overrides.get(key, default)
            if expected is None and key != "seed":
                expected = default
            if key == "logit_bias":
                expected = {int(token): bias for token, bias in expected.items()}
            assert getattr(applied, key) == expected
        if path.endswith("responses"):
            values = [event["response"] for event in chunks if "response" in event] if stream else [response.json()]
            assert values
            for value in values:
                assert value["temperature"] == applied.temperature
                assert value["top_p"] == applied.top_p


def test_model_defaults_are_independent_and_legacy_settings_are_preserved(configured_server):
    client, backends = configured_server
    for model, temperature, min_p in (("alpha", .7, .05), ("beta", .3, .02), ("legacy", 0, 0)):
        response = client.post("/v1/responses", json=request_body("/v1/responses", model=model))
        assert response.status_code == 200
        options = backends[model].calls[-1][2]
        assert options.temperature == temperature
        assert options.min_p == min_p
    catalog = {item["id"]: item for item in client.get("/v1/models").json()["data"]}
    assert catalog["alpha"]["generation_defaults"] == DEFAULTS
    assert "generation_defaults" not in catalog["legacy"]


@pytest.mark.parametrize("stream", [False, True])
def test_configured_seed_advances_for_multiple_chat_choices(configured_server, stream):
    client, backends = configured_server
    response = client.post("/v1/chat/completions", json=request_body("/v1/chat/completions", n=2, stream=stream))
    assert response.status_code == 200
    assert [call[2].seed for call in backends["alpha"].calls] == [42, 43]


def test_generation_defaults_do_not_change_systemone(configured_server):
    client, backends = configured_server
    for model in ("alpha", "legacy"):
        result = client.post("/v1/systemone", json={"model": model, "state": "Hi", "questions": {
            "route": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
        }})
        assert result.status_code == 200
        assert result.json()["answers"]["route"]["choice"] == "b"
        assert [call[0] for call in backends[model].calls] == ["decision"]


@pytest.mark.parametrize("defaults", [
    {"temperature": 2.1}, {"temperature": float("nan")}, {"temperature": None},
    {"top_p": 0}, {"top_k": True}, {"min_p": 1.1}, {"repetition_penalty": 0},
    {"presence_penalty": 3}, {"frequency_penalty": -3}, {"seed": -1},
    {"logit_bias": {"word": 1}}, {"logit_bias": {"1": 101}}, {"do_sample": True},
])
def test_invalid_defaults_fail_before_loading_models(monkeypatch, tmp_path, defaults):
    loaded = []
    monkeypatch.setattr(serve.DecisionEngine, "from_pretrained", lambda *args, **kwargs: loaded.append(args))
    config = tmp_path / "models.json"
    config.write_text(json.dumps([{"model": "alpha", "generation_defaults": defaults}]), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        serve.main(["--models-config", str(config)])
    assert exc.value.code == 2
    assert not loaded


def test_embedding_models_reject_generation_defaults(tmp_path):
    config = tmp_path / "models.json"
    config.write_text(json.dumps([{
        "model": "embedding", "task": "embedding", "generation_defaults": {"temperature": .7},
    }]), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        serve.main(["--models-config", str(config)])
    assert exc.value.code == 2

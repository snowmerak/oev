from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event

from fastapi.testclient import TestClient
import pytest

from oev import DecisionEngine, GenerationEngine, GenerationOptions, ModelRegistry, SystemOneService
from oev import serve


class CharacterTokenizer:
    def __init__(self, prefix):
        self.prefix = prefix

    def apply_chat_template(self, messages, **kwargs):
        return self.prefix + "\n" + "\n".join(message["content"] for message in messages) + "\nassistant: "

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, tokens, **kwargs):
        return "".join(map(chr, tokens))


class Backend:
    def __init__(self, name, scores):
        self.model_name = name
        self.tokenizer = CharacterTokenizer(name)
        self.scores = scores
        self.calls = []

    def selected_logits(self, ids, slots):
        self.calls.append(("decision", ids))
        return self.scores

    def generate_tokens(self, ids, options, cancel):
        self.calls.append(("generation", ids))
        for char in self.model_name[:options.max_new_tokens]:
            if cancel.is_set():
                return
            yield ord(char)


def services():
    backends = [Backend("alpha", [2, 0]), Backend("beta", [0, 2])]
    loaded = [SystemOneService(DecisionEngine(backend), backend.model_name, max_concurrency=1)
              for backend in backends]
    return ModelRegistry(loaded), loaded, backends


def decision_request(model):
    return {"model": model, "state": "Hello", "questions": {
        "route": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
    }}


def generation_request(path, model, stream=False):
    if path.endswith("completions"):
        return {"model": model, "messages": [{"role": "user", "content": "Hello"}], "stream": stream}
    return {"model": model, "input": "Hello", "stream": stream}


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


def test_catalog_and_decisions_route_to_the_selected_model():
    registry, loaded, backends = services()
    client = TestClient(serve.create_app(registry))
    catalog = client.get("/v1/models").json()
    assert catalog == registry.models()
    assert [item["id"] for item in catalog["data"]] == ["alpha", "beta"]
    assert [item["name"] for item in catalog["models"]] == ["alpha", "beta"]
    for name, expected in [("alpha", "a"), ("beta", "b"), ("alpha", "a")]:
        response = client.post("/v1/systemone", json=decision_request(name))
        assert response.status_code == 200
        assert response.json()["model"] == name
        assert response.json()["answers"]["route"]["choice"] == expected
    assert registry.resolve("alpha") is loaded[0]
    assert registry.resolve("beta") is loaded[1]
    assert len(backends[0].calls) == 2 and len(backends[1].calls) == 1
    for backend in backends:
        assert all(backend.tokenizer.decode(ids).startswith(backend.model_name) for _, ids in backend.calls)


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_generation_and_streaming_route_to_each_models_backend_and_tokenizer(path, stream):
    registry, _, backends = services()
    client = TestClient(serve.create_app(registry))
    for name in ["beta", "alpha", "beta"]:
        response = client.post(path, json=generation_request(path, name, stream))
        assert response.status_code == 200
        if stream:
            chunks = events(response)
            if path.endswith("completions"):
                assert all(chunk["model"] == name for chunk in chunks)
                text = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
            else:
                assert chunks[0]["response"]["model"] == chunks[-1]["response"]["model"] == name
                text = "".join(chunk["delta"] for chunk in chunks if chunk["type"] == "response.output_text.delta")
        else:
            body = response.json()
            assert body["model"] == name
            text = body["choices"][0]["message"]["content"] if path.endswith("completions") else body["output"][0]["content"][0]["text"]
        assert text == name
    assert len(backends[0].calls) == 1 and len(backends[1].calls) == 2
    for backend in backends:
        assert all(backend.tokenizer.decode(ids).startswith(backend.model_name) for _, ids in backend.calls)


def test_unknown_models_keep_endpoint_error_contracts_without_inference():
    registry, _, backends = services()
    client = TestClient(serve.create_app(registry))
    assert client.post("/v1/systemone", json=decision_request("missing")).status_code == 422
    for path in ["/v1/chat/completions", "/v1/responses"]:
        for stream in [False, True]:
            response = client.post(path, json=generation_request(path, "missing", stream))
            assert response.status_code == 404
            assert response.json()["error"]["code"] == "model_not_found"
            assert response.json()["error"]["param"] == "model"
    assert not any(backend.calls for backend in backends)


def test_model_specific_input_limits():
    registry, loaded, backends = services()
    loaded[0].engine.max_input_tokens = 1
    client = TestClient(serve.create_app(registry))
    for path in ["/v1/systemone", "/v1/chat/completions", "/v1/responses"]:
        for name in ["alpha", "beta"]:
            body = decision_request(name) if path.endswith("systemone") else generation_request(path, name)
            response = client.post(path, json=body)
            expected = (422 if path.endswith("systemone") else 400) if name == "alpha" else 200
            assert response.status_code == expected
    assert not backends[0].calls
    assert len(backends[1].calls) == 3


def test_model_limits_are_independent_while_decisions_and_generation_share_one_models_slot():
    entered, release, decision_entered = Event(), Event(), Event()

    class BlockingBackend(Backend):
        def generate_tokens(self, ids, options, cancel):
            entered.set()
            assert release.wait(timeout=5)
            yield from super().generate_tokens(ids, options, cancel)

        def selected_logits(self, ids, slots):
            decision_entered.set()
            return super().selected_logits(ids, slots)

    alpha = SystemOneService(DecisionEngine(BlockingBackend("alpha", [2, 0])), "alpha", max_concurrency=1)
    beta = SystemOneService(DecisionEngine(Backend("beta", [0, 2])), "beta", max_concurrency=1)
    client = TestClient(serve.create_app(ModelRegistry([alpha, beta])))
    with ThreadPoolExecutor(max_workers=3) as pool:
        generation = pool.submit(client.post, "/v1/responses", json=generation_request("/v1/responses", "alpha"))
        try:
            assert entered.wait(timeout=5)
            decision = pool.submit(client.post, "/v1/systemone", json=decision_request("alpha"))
            assert not decision_entered.wait(timeout=0.1)
            other = pool.submit(client.post, "/v1/chat/completions", json=generation_request("/v1/chat/completions", "beta"))
            assert other.result(timeout=5).json()["choices"][0]["message"]["content"] == "beta"
            other = pool.submit(client.post, "/v1/systemone", json=decision_request("beta"))
            assert other.result(timeout=5).json()["answers"]["route"]["choice"] == "b"
        finally:
            release.set()
        assert generation.result(timeout=5).status_code == 200
        assert decision.result(timeout=5).json()["answers"]["route"]["choice"] == "a"


def test_registry_rejects_empty_or_duplicate_names():
    _, loaded, _ = services()
    with pytest.raises(ValueError, match="at least one"):
        ModelRegistry([])
    with pytest.raises(ValueError, match="duplicate"):
        ModelRegistry([loaded[0], loaded[0]])


def test_generation_capability_is_checked_only_on_the_selected_model():
    class DecisionOnly:
        tokenizer = CharacterTokenizer("selection")
        model_name = "selection"

        def selected_logits(self, ids, slots):
            return [2, 0]

    registry, loaded, _ = services()
    registry = ModelRegistry([*loaded, SystemOneService(DecisionEngine(DecisionOnly()), "selection")])
    client = TestClient(serve.create_app(registry))
    for path in ["/v1/chat/completions", "/v1/responses"]:
        assert client.post(path, json=generation_request(path, "selection")).status_code == 501
        assert client.post(path, json=generation_request(path, "beta")).status_code == 200
    assert client.post("/v1/systemone", json=decision_request("selection")).status_code == 200


def capture_server(monkeypatch):
    calls, launched = [], []

    def load(model, **kwargs):
        calls.append((model, kwargs))
        return DecisionEngine(Backend(model, [2, 0]), max_input_tokens=kwargs["max_input_tokens"])

    monkeypatch.setattr(serve.DecisionEngine, "from_pretrained", load)
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: launched.append((app, kwargs)))
    return calls, launched


def test_cli_repeated_models_load_once_and_serve_aliases(monkeypatch):
    calls, launched = capture_server(monkeypatch)
    serve.main(["--model", "repo/first", "--model", "repo/second",
                "--served-model", "alpha", "--served-model", "beta", "--device", "cpu", "--max-concurrency", "2"])
    assert [model for model, _ in calls] == ["repo/first", "repo/second"]
    assert all(options["device"] == "cpu" for _, options in calls)
    client = TestClient(launched[0][0])
    for name in ["alpha", "beta", "alpha"]:
        assert client.post("/v1/responses", json=generation_request("/v1/responses", name)).status_code == 200
    assert [item["id"] for item in client.get("/v1/models").json()["data"]] == ["alpha", "beta"]
    assert len(calls) == 2


def test_cli_repeated_models_default_to_checkpoint_names(monkeypatch):
    calls, launched = capture_server(monkeypatch)
    serve.main(["--model", "repo/first", "--model", "repo/second"])
    client = TestClient(launched[0][0])
    assert [item["id"] for item in client.get("/v1/models").json()["data"]] == ["repo/first", "repo/second"]
    assert len(calls) == 2


def test_cli_without_models_keeps_the_default_single_model(monkeypatch):
    calls, launched = capture_server(monkeypatch)
    serve.main(["--served-model", "local"])
    assert calls[0][0] == "google/gemma-4-E2B-it"
    assert TestClient(launched[0][0]).get("/v1/models").json()["data"][0]["id"] == "local"


def test_cli_config_loads_all_models_with_per_model_options_and_env_defaults(monkeypatch, tmp_path):
    calls, launched = capture_server(monkeypatch)
    monkeypatch.setenv("OEV_MAX_CONCURRENCY", "3")
    config = tmp_path / "models.json"
    config.write_text(json.dumps([
        {"model": "repo/first", "served_model": "alpha", "device": "cpu", "dtype": "float32", "max_input_tokens": 100, "max_concurrency": 1},
        {"model": "repo/second", "served_model": "beta", "device": "hybrid-cuda", "revision": "rev-2"},
    ]), encoding="utf-8-sig")
    registries = []
    monkeypatch.setattr(serve, "create_app", lambda registry: registries.append(registry) or registry)
    serve.main(["--models-config", str(config), "--dtype", "bfloat16", "--revision", "rev-1"])
    assert calls == [
        ("repo/first", {"revision": "rev-1", "device": "cpu", "dtype": "float32", "max_input_tokens": 100,
                        "backend": "auto", "local_files_only": False, "context_length": 8192,
                        "gpu_layers": None, "chat_template": None, "flash_attn": False}),
        ("repo/second", {"revision": "rev-2", "device": "hybrid-cuda", "dtype": "bfloat16", "max_input_tokens": 4096,
                         "backend": "auto", "local_files_only": False, "context_length": 8192,
                         "gpu_layers": None, "chat_template": None, "flash_attn": False}),
    ]
    assert registries[0].resolve("alpha").max_concurrency == 1
    assert registries[0].resolve("beta").max_concurrency == 3
    assert len(launched) == 1


@pytest.mark.parametrize("entries", [
    [], {}, ["repo/first"], [{"device": "cpu"}], [{"model": " "}],
    [{"model": "first", "unknown_option": True}],
    [{"model": "first", "max_concurrency": 0}], [{"model": "first", "max_input_tokens": True}],
    [{"model": "first", "device": "hybrid-cuda", "dtype": "float32"}],
    [{"model": "first", "served_model": "same"}, {"model": "second", "served_model": "same"}],
])
def test_invalid_config_is_rejected_before_any_model_load(monkeypatch, tmp_path, entries):
    calls, launched = capture_server(monkeypatch)
    config = tmp_path / "models.json"
    config.write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        serve.main(["--models-config", str(config)])
    assert exc.value.code == 2
    assert not calls and not launched


@pytest.mark.parametrize("args", [
    ["--model", "same", "--model", "same"],
    ["--model", "first", "--model", "second", "--served-model", "only-one"],
    ["--model", "first", "--models-config", "unused.json"],
    ["--served-model", "alias", "--models-config", "unused.json"],
    ["--max-input-tokens", "0"],
])
def test_invalid_cli_is_rejected_before_loading(monkeypatch, args):
    calls, launched = capture_server(monkeypatch)
    with pytest.raises(SystemExit):
        serve.main(args)
    assert not calls and not launched


@pytest.mark.parametrize("content", [None, "not json"])
def test_missing_or_malformed_config_is_rejected_before_loading(monkeypatch, tmp_path, content):
    calls, launched = capture_server(monkeypatch)
    config = tmp_path / "models.json"
    if content is not None:
        config.write_text(content, encoding="utf-8")
    with pytest.raises(SystemExit):
        serve.main(["--models-config", str(config)])
    assert not calls and not launched


def test_failure_loading_second_model_does_not_start_a_partial_server(monkeypatch):
    calls, launched = capture_server(monkeypatch)
    original = serve.DecisionEngine.from_pretrained

    def load(model, **kwargs):
        if model == "second":
            raise RuntimeError("out of memory")
        return original(model, **kwargs)

    monkeypatch.setattr(serve.DecisionEngine, "from_pretrained", load)
    with pytest.raises(RuntimeError, match="out of memory"):
        serve.main(["--model", "first", "--model", "second"])
    assert len(calls) == 1
    assert not launched


def test_two_resident_hf_models_generate_and_select_without_reloading(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from oev.hf import HuggingFaceBackend

    loaded, launched = [], []

    def load(name, **kwargs):
        # Build two distinct real models locally, with no checkpoint download.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7 if name == "alpha" else 42)
            model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
                vocab_size=128, n_positions=1024, n_ctx=1024,
                n_embd=16, n_layer=1, n_head=2, eos_token_id=127, pad_token_id=0,
            )).eval()
        backend = HuggingFaceBackend(model, CharacterTokenizer(name), name)
        engine = DecisionEngine(backend, max_input_tokens=kwargs["max_input_tokens"])
        loaded.append(engine)
        return engine

    monkeypatch.setattr(serve.DecisionEngine, "from_pretrained", load)
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: launched.append(app))
    serve.main(["--model", "alpha", "--model", "beta", "--device", "cpu"])
    assert len(loaded) == 2
    assert loaded[0].backend.model is not loaded[1].backend.model
    counts = [[], []]
    handles = [engine.backend.model.register_forward_pre_hook(
        lambda module, args, index=index: counts[index].append(True),
    ) for index, engine in enumerate(loaded)]
    client = TestClient(launched[0])
    try:
        for index, name in [(0, "alpha"), (1, "beta"), (0, "alpha")]:
            options = GenerationOptions(max_new_tokens=4)
            reference = GenerationEngine(loaded[index].backend).prepare(
                [{"role": "user", "content": "Hello"}], options,
            )
            expected = loaded[index].backend.tokenizer.decode(
                list(loaded[index].backend.generate_tokens(reference.input_ids, options, Event())),
            )
            before = [len(count) for count in counts]
            response = client.post("/v1/chat/completions", json={
                **generation_request("/v1/chat/completions", name), "max_completion_tokens": 4,
            })
            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"] == expected
            decision = client.post("/v1/systemone", json=decision_request(name))
            assert decision.status_code == 200
            assert decision.json()["model"] == name
            assert len(counts[index]) > before[index]
            assert len(counts[1 - index]) == before[1 - index]
        assert len(loaded) == 2
    finally:
        for handle in handles:
            handle.remove()

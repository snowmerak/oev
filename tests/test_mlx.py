from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
import sys
from threading import Event, get_ident, local
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest

from oev import Decision, DecisionEngine, GenerationEngine, GenerationOptions, ModelRegistry, Option, SystemOneService
from oev import serve
from oev.engine import model_backend
from oev.mlx import MLXBackend
from oev.prompt import prepare


class Tokenizer:
    has_chat_template = True
    eos_token_ids = {254, 255}
    eos_token_id = 254

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "\n".join(item["content"] for item in messages) + "\nassistant: "

    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))

    def decode(self, tokens, **kwargs):
        tokens = [token for token in tokens if not kwargs.get("skip_special_tokens") or token not in self.eos_token_ids]
        return bytes(tokens).decode("utf-8", errors="replace")


class Model:
    def __init__(self, sequence, mx):
        self.sequence = sequence
        self.calls = []
        self.mx = mx
        # Model initialization may leave lazy constants on the loading stream.
        self.initialization_stream = mx.current.stream

    def __call__(self, ids, cache):
        self.mx.check_stream(self.initialization_stream)
        state = cache[0]
        if ids.shape[1] == 1:
            state.step += 1
        state.offset += ids.shape[1]
        state.state = np.array([state.offset])
        self.calls.append((ids.tolist(), cache))
        logits = np.zeros((1, ids.shape[1], 256), dtype=np.float32)
        logits[..., ord("b")] = 3
        logits[..., self.sequence[min(max(state.step, 0), len(self.sequence) - 1)]] = 10
        return logits


class MX:
    cpu, gpu = "cpu", "gpu"
    uint32, float32 = np.uint32, np.float32
    array, argmax, all, isfinite = staticmethod(np.array), staticmethod(np.argmax), staticmethod(np.all), staticmethod(np.isfinite)

    def __init__(self):
        self.metal = SimpleNamespace(is_available=lambda: True)
        self.current = local()
        self.synchronizations = []
        self.evaluations = []

    def new_thread_local_stream(self, device):
        return SimpleNamespace(device=device, per_thread={})

    def new_thread_unsafe_stream(self, device):
        return SimpleNamespace(device=device, owner=get_ident(), shared=True)

    def resolve_stream(self, stream):
        if hasattr(stream, "per_thread"):
            if get_ident() not in stream.per_thread:
                stream.per_thread[get_ident()] = SimpleNamespace(device=stream.device, owner=get_ident(), shared=False)
            return stream.per_thread[get_ident()]
        return stream

    def check_stream(self, stream):
        stream = self.resolve_stream(stream)
        if not stream.shared and stream.owner != get_ident():
            raise RuntimeError("There is no Stream(gpu, 1) in current thread.")

    @contextmanager
    def stream(self, stream):
        stream = self.resolve_stream(stream)
        self.check_stream(stream)
        previous = getattr(self.current, "stream", None)
        self.current.stream = stream
        try:
            yield
        finally:
            self.current.stream = previous

    def eval(self, values):
        assert getattr(self.current, "stream", None) is not None
        self.evaluations.append(values)

    def synchronize(self, stream):
        self.check_stream(stream)
        self.synchronizations.append(stream)

    @staticmethod
    def logsumexp(values, axis, keepdims):
        maximum = values.max(axis=axis, keepdims=True)
        return maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=keepdims))


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    mx = MX()
    loaded, caches, samplers, logprobs = [], [], [], []
    tokenizer = Tokenizer()

    def load(source, **kwargs):
        assert kwargs == {
            "lazy": False, "return_config": True, "trust_remote_code": False,
            "tokenizer_config": {"local_files_only": True, "trust_remote_code": False},
        }
        assert mx.current.stream is not None
        model = Model([ord("O"), ord("K"), ord(" "), 255], mx)
        loaded.append((source, model))
        return model, tokenizer, {"text_config": {"max_position_embeddings": 8192}}

    def make_cache(model):
        cache = [SimpleNamespace(step=-1, offset=0, state=np.array([0]))]
        caches.append(cache)
        return cache

    def make_sampler(**kwargs):
        samplers.append(kwargs)

        def sample(values):
            logprobs.append(values)
            return values.argmax(axis=-1)

        return sample

    monkeypatch.setattr("oev.mlx.platform.system", lambda: "Darwin")
    monkeypatch.setattr("oev.mlx.platform.machine", lambda: "arm64")
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=load))
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", SimpleNamespace(make_prompt_cache=make_cache))
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", SimpleNamespace(make_sampler=make_sampler))
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", SimpleNamespace(DEFAULT_ALLOW_PATTERNS=["*.json", "model*.safetensors"]))
    (tmp_path / "config.json").write_text(json.dumps({"quantization": {"bits": 4, "group_size": 64}}), encoding="utf-8")
    return SimpleNamespace(path=tmp_path, mx=mx, loaded=loaded, caches=caches, samplers=samplers, logprobs=logprobs, tokenizer=tokenizer)


def test_auto_detection_loads_mlx_weights_once_with_metal_and_eager_evaluation(runtime):
    engine = DecisionEngine.from_pretrained(str(runtime.path))
    backend = engine.backend
    assert isinstance(backend, MLXBackend)
    assert backend.mlx_model is runtime.loaded[0][1]
    assert backend._stream.device == "gpu"
    assert backend._stream.shared
    assert backend.context_window == 8192
    assert runtime.mx.synchronizations == [backend._stream]
    assert runtime.mx.current.stream is None


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_generation_sse_and_selection_share_mlx_model_with_isolated_caches(runtime, path, stream):
    engine = DecisionEngine.from_pretrained(str(runtime.path), backend="mlx")
    client = TestClient(serve.create_app(SystemOneService(engine, "mac")))
    body = {"model": "mac", "stream": stream}
    body.update({"messages": [{"role": "user", "content": "Hello"}]} if path.endswith("completions") else {"input": "Hello"})
    for _ in range(2):
        response = client.post(path, json=body)
        assert response.status_code == 200
        if stream:
            chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            if path.endswith("completions"):
                assert "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks) == "OK "
                assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
            else:
                assert chunks[-1]["type"] == "response.completed"
                assert chunks[-1]["response"]["output"][0]["content"][0]["text"] == "OK "
        else:
            value = response.json()
            if path.endswith("completions"):
                assert value["choices"][0]["message"]["content"] == "OK "
                assert value["usage"]["completion_tokens"] == 4
            else:
                assert value["output"][0]["content"][0]["text"] == "OK "
                assert value["usage"]["output_tokens"] == 4
        response = client.post("/v1/systemone", json={
            "model": "mac", "state": "Hello", "questions": {
                "route": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
            },
        })
        assert response.status_code == 200
        assert response.json()["answers"]["route"]["choice"] == "b"
    assert len(runtime.loaded) == 1
    assert len(runtime.caches) == 4
    assert len({id(cache) for cache in runtime.caches}) == 4


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_loaded_model_can_select_and_generate_from_different_threads(runtime, device):
    backend = MLXBackend.load(str(runtime.path), device=device)
    caller = get_ident()

    def work():
        scores = backend.selected_logits([1, 2, 3], [ord("b")])
        tokens = list(backend.generate_tokens([1, 2, 3], GenerationOptions(4), Event()))
        return get_ident(), scores, tokens

    with ThreadPoolExecutor(max_workers=1) as pool:
        thread, scores, tokens = pool.submit(work).result(timeout=5)
    assert thread != caller
    assert scores == [3]
    assert tokens == [ord("O"), ord("K"), ord(" "), 255]
    assert all(stream is backend._stream for stream in runtime.mx.synchronizations)


def test_thread_local_loading_stream_reproduces_lazy_constant_failure(runtime, monkeypatch):
    # Reproduce the old loader while the context itself resolves per thread.
    monkeypatch.setattr(runtime.mx, "new_thread_unsafe_stream", runtime.mx.new_thread_local_stream)
    backend = MLXBackend.load(str(runtime.path))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(backend.selected_logits, [1, 2, 3], [ord("b")])
        with pytest.raises(RuntimeError, match="no Stream.*current thread"):
            future.result(timeout=5)
    assert backend._lock.acquire(blocking=False)
    backend._lock.release()


def test_cached_decoding_uses_single_tokens_and_stream_scope_ends_before_yield(runtime):
    backend = MLXBackend.load(str(runtime.path))
    stream = backend.generate_tokens([1, 2, 3], GenerationOptions(4), Event())
    assert next(stream) == ord("O")
    assert runtime.mx.current.stream is None
    assert list(stream) == [ord("K"), ord(" "), 255]
    calls = backend.mlx_model.calls
    assert [ids for ids, _ in calls] == [[[1, 2]], [[3]], [[ord("O")]], [[ord("K")]], [[ord(" ")]]]
    assert all(cache is runtime.caches[0] for _, cache in calls)


def test_temperature_is_applied_before_top_p_without_other_filters(runtime):
    backend = MLXBackend.load(str(runtime.path))
    assert list(backend.generate_tokens([1, 2, 3], GenerationOptions(1, temperature=2, top_p=0.8), Event())) == [ord("O")]
    assert runtime.samplers == [{"temp": 1.0, "top_p": 0.8}]
    assert runtime.logprobs[0][0, ord("O")] - runtime.logprobs[0][0, ord("b")] == pytest.approx(3.5)


def test_stop_closing_and_context_limits(runtime):
    backend = MLXBackend.load(str(runtime.path))
    generation = GenerationEngine(backend)
    prepared = generation.prepare([{"role": "user", "content": "Hi"}], GenerationOptions(4, stop=("K",)))
    result = generation.generate(prepared)
    assert result.text == "O" and result.finish_reason == "stop"
    assert backend._lock.acquire(blocking=False)
    backend._lock.release()
    backend.context_window = 2
    with pytest.raises(ValueError, match="context limit"):
        generation.prepare([{"role": "user", "content": "Hi"}], GenerationOptions(4))
    with pytest.raises(ValueError, match="context limit"):
        backend.selected_logits([1, 2, 3], [1])


def test_model_lock_serializes_selection_until_generation_closes(runtime):
    backend = MLXBackend.load(str(runtime.path))
    stream = backend.generate_tokens([1, 2], GenerationOptions(4), Event())
    next(stream)
    started = Event()

    def decide():
        started.set()
        return backend.selected_logits([3, 4], [ord("b")])

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(decide)
        try:
            assert started.wait(timeout=5)
            assert not future.done()
            assert len(runtime.caches) == 1
        finally:
            stream.close()
        assert future.result(timeout=5) == [3]
    assert runtime.mx.current.stream is None


def test_cancel_waiting_for_model_does_not_make_a_cache(runtime):
    backend = MLXBackend.load(str(runtime.path))
    cancel = Event()
    backend._lock.acquire()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: list(backend.generate_tokens([1, 2], GenerationOptions(4), cancel)))
            cancel.set()
            assert future.result(timeout=5) == []
        assert not runtime.caches
    finally:
        backend._lock.release()


def test_cancel_prefill_stops_between_bounded_chunks(runtime, monkeypatch):
    backend = MLXBackend.load(str(runtime.path))
    cancel = Event()
    original = runtime.mx.eval

    def cancel_after_chunk(values):
        original(values)
        cancel.set()

    monkeypatch.setattr(runtime.mx, "eval", cancel_after_chunk)
    assert list(backend.generate_tokens([1] * 1026, GenerationOptions(4), cancel)) == []
    assert len(backend.mlx_model.calls) == 1
    assert len(backend.mlx_model.calls[0][0][0]) == 512
    assert backend._lock.acquire(blocking=False)
    backend._lock.release()


def test_failed_forward_and_sync_still_release_model_lock(runtime, monkeypatch):
    backend = MLXBackend.load(str(runtime.path))

    def fail(*args, **kwargs):
        raise RuntimeError("Metal failed")

    monkeypatch.setattr(backend, "_prefill", fail)
    monkeypatch.setattr(runtime.mx, "synchronize", fail)
    with pytest.raises(RuntimeError, match="Metal failed"):
        list(backend.generate_tokens([1, 2], GenerationOptions(4), Event()))
    assert backend._lock.acquire(blocking=False)
    backend._lock.release()
    assert runtime.mx.current.stream is None


def test_mlx_cache_only_repositories_resolve_snapshot_without_network(runtime, monkeypatch):
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        return str(runtime.path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    backend = MLXBackend.load("someone/mlx-model", revision="commit", local_files_only=True, device="cpu")
    assert calls == [{"repo_id": "someone/mlx-model", "revision": "commit", "local_files_only": True,
                      "allow_patterns": ["*.json", "model*.safetensors"]}]
    assert backend.model_name == "someone/mlx-model@commit"
    assert backend._stream.device == "cpu"
    assert runtime.loaded[0][0] == str(runtime.path)


@pytest.mark.parametrize("options", [{"device": "cuda"}, {"device": "hybrid-cuda"}, {"dtype": "bfloat16"}, {"gpu_layers": 1}])
def test_incompatible_mlx_options_fail_before_loading(runtime, options):
    with pytest.raises(ValueError):
        DecisionEngine.from_pretrained(str(runtime.path), backend="mlx", **options)
    assert not runtime.loaded


def test_windows_has_an_actionable_platform_error_before_import(monkeypatch):
    monkeypatch.setattr("oev.mlx.platform.system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="Apple Silicon macOS"):
        MLXBackend.load("mlx-community/model")


def test_auto_routes_unambiguous_mlx_sources_and_explicit_hf_is_preserved(runtime):
    assert model_backend("mlx-community/model") == "mlx"
    assert model_backend(str(runtime.path)) == "mlx"
    assert model_backend(str(runtime.path), "hf") == "hf"
    assert model_backend("other-owner/model") == "hf"
    assert model_backend("other-owner/model", "mlx") == "mlx"


def test_multi_model_registry_keeps_mlx_models_resident_and_independent(runtime):
    first = MLXBackend.load(str(runtime.path))
    second = MLXBackend.load(str(runtime.path))
    second.mlx_model.sequence = [ord("B"), 255]
    registry = ModelRegistry([
        SystemOneService(DecisionEngine(first), "first"), SystemOneService(DecisionEngine(second), "second"),
    ])
    client = TestClient(serve.create_app(registry))
    for name, expected in [("first", "OK "), ("second", "B"), ("first", "OK ")]:
        result = client.post("/v1/responses", json={"model": name, "input": "Hi"}).json()
        assert result["output"][0]["content"][0]["text"] == expected
    assert len(runtime.loaded) == 2
    assert first._stream is not second._stream


def test_server_loads_mlx_config_and_rejects_invalid_defaults_before_loading(runtime, monkeypatch, tmp_path):
    launched = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: launched.append(app))
    config = tmp_path / "models.json"
    config.write_text(json.dumps([{"model": str(runtime.path), "served_model": "mac", "backend": "mlx"}]), encoding="utf-8")
    serve.main(["--models-config", str(config)])
    assert TestClient(launched[0]).get("/v1/models").json()["data"][0]["id"] == "mac"
    with pytest.raises(SystemExit):
        serve.main(["--models-config", str(config), "--dtype", "bfloat16"])
    assert len(runtime.loaded) == len(launched) == 1


def test_real_mlx_when_explicitly_configured():
    source = os.environ.get("OEV_TEST_MLX_PATH")
    if sys.platform != "darwin" or not source:
        pytest.skip("run on Apple Silicon with OEV_TEST_MLX_PATH pointing to a local MLX checkpoint")
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    backend = MLXBackend.load(source, local_files_only=True)
    # Make the first forwards on HTTP worker threads, before a main-thread
    # forward could materialize lazy initialization constants and mask the bug.
    with TestClient(serve.create_app(SystemOneService(DecisionEngine(backend), "mlx-real"))) as client:
        for stream in [False, True]:
            response = client.post("/v1/responses", json={
                "model": "mlx-real", "input": "Say hello.", "max_output_tokens": 4, "stream": stream,
            })
            assert response.status_code == 200, response.text
            if stream:
                events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                assert not any(event["type"] == "error" for event in events), events
                assert events[-1]["type"] in {"response.completed", "response.incomplete"}
            else:
                assert response.json()["usage"]["output_tokens"] > 0
            response = client.post("/v1/systemone", json={
                "model": "mlx-real", "state": "Hello", "questions": {
                    "greeting": {"type": "choice", "criteria": {"hello": "Hello", "bye": "Bye"}},
                },
            })
            assert response.status_code == 200, response.text
            assert set(response.json()["answers"]["greeting"]["probabilities"]) == {"hello", "bye"}
    generation = GenerationEngine(backend)
    prepared = generation.prepare([{"role": "user", "content": "Say hello."}], GenerationOptions(4))
    reference = generate_step(mx.array(prepared.input_ids), backend.mlx_model, stream=backend._stream,
                              max_tokens=4, prompt_cache=make_prompt_cache(backend.mlx_model), prefill_step_size=512)
    expected = []
    try:
        for token, _ in reference:
            expected.append(int(token))
            if backend.is_end_token(token):
                break
    finally:
        reference.close()
        mx.synchronize(backend._stream)
    assert list(backend.generate_tokens(prepared.input_ids, prepared.options, Event())) == expected
    first = generation.generate(prepared)
    assert generation.generate(prepared) == first
    ids, slots, _ = prepare(backend.tokenizer, Decision("Hello", "Which greeting?", (Option("a", "Hello"), Option("b", "Bye"))), 4096)
    with mx.stream(backend._stream):
        full = backend.mlx_model(mx.array([ids]), cache=make_prompt_cache(backend.mlx_model))[0, -1, mx.array(slots)].astype(mx.float32)
        mx.eval(full)
    assert backend.selected_logits(ids, slots) == pytest.approx(full.tolist(), abs=0.02, rel=0.005)
    assert generation.generate(prepared) == first

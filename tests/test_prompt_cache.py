"""Budget policy and real Transformers cache/API integration."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
import sys
from threading import Event

import pytest
from fastapi.testclient import TestClient

from oev import Decision, DecisionEngine, GenerationEngine, GenerationOptions, Option, SystemOneService
from oev.generation import PreparedGeneration
from oev.prompt import prepare
from oev.prompt_cache import CacheUsage, PrefixCache, state_bytes, track_cache
from oev import serve


def test_lru_budget_longest_boundary_and_oversized_state():
    cache = PrefixCache(10000)
    cache.store([1, 2], {"value": [2]}, 100, deepcopy)
    cache.store([1, 2, 3], {"value": [3]}, 100, deepcopy)
    assert cache.lookup([1, 2, 3, 4]) == (3, {"value": [3]})
    assert cache.lookup([1, 2, 9]) == (2, {"value": [2]})
    assert cache.lookup([1, 9, 3, 4]) == (0, None)
    # Never restore a complete match without a token left for final logits.
    assert cache.lookup([1, 2]) == (0, None)
    cache.max_bytes = cache.size_bytes
    cache.store([9, 8], {"value": [8]}, 100, deepcopy)
    assert cache.lookup([1, 2, 3, 4])[0] == 2  # shorter prefix was most recently used
    assert cache.size_bytes <= cache.max_bytes
    before = cache.size_bytes
    cache.store([7], None, 100000, lambda _: pytest.fail("oversized state was copied"))
    assert cache.size_bytes == before
    cache.clear()
    assert cache.size_bytes == 0 and cache.lookup([9, 8, 7]) == (0, None)


@pytest.mark.parametrize("value", [-1, True, 1.5, "1024"])
def test_cache_size_is_a_strict_nonnegative_integer(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        PrefixCache(value)
    with pytest.raises(ValueError):
        DecisionEngine.from_pretrained("never-download", prompt_cache_max_bytes=value)
    with pytest.raises(ValueError):
        serve.ModelConfig(model="never-download", prompt_cache_max_bytes=value)


def test_disabled_cache_does_not_retain_or_copy():
    cache = PrefixCache()
    cache.store([1], None, 1, lambda _: pytest.fail("disabled cache copied a state"))
    assert cache.size_bytes == 0 and cache.lookup([1, 2]) == (0, None)


def test_storage_accounting_includes_unique_tensor_allocations():
    torch = pytest.importorskip("torch")
    tensor = torch.zeros(1024)
    assert state_bytes({"keys": tensor, "view": tensor[:4]}) >= tensor.untyped_storage().nbytes()
    assert state_bytes([tensor, tensor[:4]]) < 2 * tensor.untyped_storage().nbytes()


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages) + "\nassistant: "

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode(errors="replace")


@pytest.fixture
def hf_backend():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from oev.hf import HuggingFaceBackend

    torch.manual_seed(7)
    config = transformers.GPT2Config(vocab_size=256, n_positions=2048, n_ctx=2048,
                                   n_embd=16, n_layer=1, n_head=2, eos_token_id=255)
    backend = HuggingFaceBackend(transformers.GPT2LMHeadModel(config), Tokenizer(), "tiny")
    backend.configure_prompt_cache(4 * 1024 * 1024)
    return backend


def test_hf_generation_and_decision_share_immutable_prefix_snapshots(hf_backend):
    torch = pytest.importorskip("torch")
    backend = hf_backend
    ids, slots = list(range(64)) * 9 + [3, 5], [7, 9]
    with torch.inference_mode():
        expected = backend.model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1, slots].tolist()
    usage = CacheUsage()
    with track_cache(usage):
        assert backend.selected_logits(ids, slots) == pytest.approx(expected, abs=1e-6)
    assert usage.cached_tokens == 0
    generation = GenerationEngine(backend)
    prepared = PreparedGeneration(ids, GenerationOptions(3))
    first = generation.generate(prepared)
    assert first.cached_input_tokens == len(ids) - 1
    assert generation.generate(prepared).text == first.text
    with track_cache(usage):
        assert backend.selected_logits(ids, slots) == pytest.approx(expected, abs=1e-6)
    # A divergent suffix restores only a stored, still-identical boundary.
    changed = ids[:520] + [11] * (len(ids) - 520)
    usage = CacheUsage()
    with track_cache(usage):
        actual = backend.selected_logits(changed, slots)
    with torch.inference_mode():
        full = backend.model(input_ids=torch.tensor([changed]), use_cache=False).logits[0, -1, slots].tolist()
    assert actual == pytest.approx(full, abs=1e-6)
    assert usage.cached_tokens == 512
    with ThreadPoolExecutor(max_workers=2) as pool:
        def work(_):
            result = generation.generate(prepared)
            return result.text, result.cached_input_tokens
        assert list(pool.map(work, range(4))) == [(first.text, len(ids) - 1)] * 4
    assert backend.prompt_cache.size_bytes <= backend.prompt_cache.max_bytes


def test_hf_sliding_window_snapshots_beyond_window_match_uncached_logits():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from oev.hf import HuggingFaceBackend

    torch.manual_seed(17)
    config = transformers.Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=16,
        hidden_size_per_layer_input=0, sliding_window=32, max_position_embeddings=2048,
        num_kv_shared_layers=2, layer_types=["sliding_attention", "full_attention"] * 2,
    )
    backend = HuggingFaceBackend(transformers.Gemma4ForCausalLM(config), None, "gemma")
    ids = [2, 4, 8, 12] * 140 + [6]
    expected = backend.selected_logits(ids, [5, 7])
    expected_tokens = list(backend.generate_tokens(ids, GenerationOptions(3), Event()))
    backend.configure_prompt_cache(4 * 1024 * 1024)
    assert backend.selected_logits(ids, [5, 7]) == pytest.approx(expected, abs=1e-5)
    assert list(backend.generate_tokens(ids, GenerationOptions(3), Event())) == expected_tokens
    changed = ids[:520] + [3] * (len(ids) - 520)
    usage = CacheUsage()
    with track_cache(usage):
        actual = backend.selected_logits(changed, [5, 7])
    assert usage.cached_tokens == 512
    backend.configure_prompt_cache(0)
    assert actual == pytest.approx(backend.selected_logits(changed, [5, 7]), abs=1e-5)


def test_cache_enabled_gemma_selection_preserves_selective_head_optimization(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from oev.hf import HuggingFaceBackend

    config = transformers.Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=16,
        hidden_size_per_layer_input=0, sliding_window=32, max_position_embeddings=1024,
        layer_types=["sliding_attention", "full_attention"],
    )
    model = transformers.Gemma4ForCausalLM(config)
    backend = HuggingFaceBackend(model, None, "gemma")
    ids = [2, 4] * 140
    expected = backend.selected_logits(ids, [5, 7])
    backend.configure_prompt_cache(1024 * 1024)
    monkeypatch.setattr(model.lm_head, "forward", lambda *args: pytest.fail("full vocabulary head used"))
    for _ in range(2):
        assert backend.selected_logits(ids, [5, 7]) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("path", ["/v1/responses", "/v1/chat/completions"])
@pytest.mark.parametrize("stream", [False, True])
def test_api_reports_actual_cache_usage_for_json_and_sse(hf_backend, path, stream):
    client = TestClient(serve.create_app(SystemOneService(DecisionEngine(hf_backend), "tiny")))
    text = "cache this shared state " * 24
    body = {"model": "tiny", "stream": stream}
    if path.endswith("completions"):
        body.update(messages=[{"role": "user", "content": text}], max_tokens=2)
        if stream:
            body["stream_options"] = {"include_usage": True}
    else:
        body.update(input=text, max_output_tokens=2)
    results = []
    for _ in range(2):
        response = client.post(path, json=body)
        assert response.status_code == 200, response.text
        if stream:
            events = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            result = next(event for event in events if event.get("usage")) if path.endswith("completions") else events[-1]["response"]
        else:
            result = response.json()
        results.append(result["usage"])
    details = "prompt_tokens_details" if path.endswith("completions") else "input_tokens_details"
    assert results[0].get(details, {"cached_tokens": 0})["cached_tokens"] == 0
    assert results[1][details]["cached_tokens"] > 0
    # System One shares the cache and aggregates usage across its questions.
    request = {"model": "tiny", "state": text, "questions": {
        "route": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
    }}
    first = client.post("/v1/systemone", json=request).json()
    second = client.post("/v1/systemone", json=request).json()
    assert second["answers"] == first["answers"]
    assert second["usage"]["input_tokens_details"]["cached_tokens"] > 0


def test_hf_cancelled_generation_releases_cache_lock(hf_backend):
    cancel = Event()
    source = hf_backend.generate_tokens([2] * 550, GenerationOptions(4), cancel)
    next(source)
    cancel.set()
    source.close()
    assert hf_backend._cache_lock.acquire(blocking=False)
    hf_backend._cache_lock.release()
    assert len(hf_backend.selected_logits([2] * 550, [7, 8])) == 2


def test_model_config_inherits_and_overrides_cache_budget(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(serve, "_load_service", lambda config: calls.append(config) or SystemOneService(DecisionEngine(None), config.name))
    monkeypatch.setattr(serve, "create_app", lambda service: service)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    path = tmp_path / "models.json"
    path.write_text(json.dumps([{"model": "first"}, {"model": "second", "prompt_cache_max_bytes": 0}]))
    serve.main(["--models-config", str(path), "--prompt-cache-max-bytes", "1048576"])
    assert [config.prompt_cache_max_bytes for config in calls] == [1048576, 0]


@pytest.mark.parametrize("entry", [
    {"prompt_cache_max_bytes": -1}, {"prompt_cache_max_bytes": True},
    {"prompt_cache_max_bytes": 1.5},
    {"prompt_cache_max_bytes": 1024, "task": "embedding"},
    {"prompt_cache_max_bytes": 1024, "device": "hybrid-cuda"},
])
def test_invalid_config_never_loads_a_model(monkeypatch, tmp_path, entry):
    monkeypatch.setattr(serve, "_load_service", lambda _: pytest.fail("invalid config loaded"))
    path = tmp_path / "models.json"
    path.write_text(json.dumps([{"model": "never-download", **entry}]))
    with pytest.raises(SystemExit):
        serve.main(["--models-config", str(path)])


@pytest.mark.parametrize("kind,env", [("hf", "OEV_TEST_HF_PATH"), ("gguf", "OEV_TEST_GGUF_PATH"), ("mlx", "OEV_TEST_MLX_PATH")])
def test_downloaded_model_prompt_cache_when_explicitly_configured(kind, env):
    source = os.environ.get(env)
    if not source:
        pytest.skip(f"set {env} to a downloaded {kind} checkpoint; MLX requires Apple Silicon")
    if kind == "mlx" and sys.platform != "darwin":
        pytest.skip("real MLX prompt caching requires Apple Silicon macOS")
    pytest.importorskip({"hf": "transformers", "gguf": "llama_cpp", "mlx": "mlx.core"}[kind])
    options = {"device": "cpu", "context_length": 2048, "max_input_tokens": 2048} if kind == "gguf" else {"device": "auto", "max_input_tokens": 2048}
    if kind == "hf":
        options["dtype"] = "float32"
    engine = DecisionEngine.from_pretrained(source, backend=kind, local_files_only=True, **options)
    backend = engine.backend
    try:
        from examples.prompt_cache_smoke import prompts
        _, cases = prompts(backend.tokenizer)
        baseline = [(backend.selected_logits(ids, slots), list(backend.generate_tokens(ids, GenerationOptions(4), Event())))
                    for _, ids, slots in cases]
        backend.configure_prompt_cache(512 * 1024 * 1024)
        for index, (_, ids, slots) in enumerate(cases):
            for repeat in range(2):
                usage = CacheUsage()
                with track_cache(usage):
                    logits = backend.selected_logits(ids, slots)
                assert logits == pytest.approx(baseline[index][0], abs=0.03, rel=0.005)
                assert list(backend.generate_tokens(ids, GenerationOptions(4), Event())) == baseline[index][1]
                assert backend.prompt_cache.size_bytes <= backend.prompt_cache.max_bytes
                if repeat:
                    assert usage.cached_tokens > 0
        # Model-level cache remains valid on HTTP worker threads, including SSE.
        with TestClient(serve.create_app(SystemOneService(engine, "real-cache"))) as client:
            body = {"model": "real-cache", "input": "Say hello. " * 48, "max_output_tokens": 4}
            cold = client.post("/v1/responses", json=body)
            warm = client.post("/v1/responses", json={**body, "stream": True})
            assert cold.status_code == warm.status_code == 200
            events = [json.loads(line[6:]) for line in warm.text.splitlines() if line.startswith("data: ")]
            final = events[-1]["response"]
            assert final["output"][0]["content"] == cold.json()["output"][0]["content"]
            assert final["usage"]["input_tokens_details"]["cached_tokens"] > 0
    finally:
        if kind == "gguf":
            backend.llama.close()

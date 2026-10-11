from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
import pickle
import sys
from threading import Event
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from oev import Decision, DecisionEngine, GenerationEngine, GenerationOptions, Option, SystemOneService
from oev.gguf import GGUFBackend
from oev.prompt import prepare
from oev import cli, serve
from oev.prompt_cache import CacheUsage, track_cache


class FakeLlama:
    instances = []

    def __init__(self, **kwargs):
        self.options = kwargs
        self.metadata = {"tokenizer.chat_template": "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}assistant: "}
        self.model = object()
        self.ctx = self
        self.n_batch = 512
        self.n_tokens = 0
        self.input_ids = [0] * kwargs["n_ctx"]
        self.state_saves = 0
        self.buffer = (ctypes.c_float * 256)()
        self.sequence = [ord("O"), ord("K"), ord(" "), 251]
        self.index = 0
        self.eval_calls = []
        self.tokenize_calls = []
        self.closed = False
        self.instances.append(self)

    def n_ctx(self):
        return self.options["n_ctx"]

    def n_vocab(self):
        return 256

    def token_bos(self):
        return 252

    def token_eos(self):
        return 250

    def tokenize(self, text, **kwargs):
        self.tokenize_calls.append(kwargs)
        tokens = [252] if kwargs["add_bos"] else []
        while text:
            if kwargs["special"] and text.startswith(b"[BOS]"):
                tokens.append(252)
                text = text[5:]
            else:
                tokens.append(text[0])
                text = text[1:]
        return tokens

    def detokenize(self, tokens, special=False):
        markers = {250: b"[EOS]", 251: b"[EOT]", 252: b"[BOS]"}
        pieces = []
        for token in tokens:
            if token in markers:
                if special:
                    pieces.append(markers[token])
            else:
                pieces.append(bytes([token]))
        return b"".join(pieces)

    def reset(self):
        self.n_tokens = 0
        self.index = 0

    def eval(self, tokens):
        self.eval_calls.append((list(tokens), self.n_tokens))
        if self.n_tokens and len(tokens) == 1:
            self.index += 1
        self.input_ids[self.n_tokens:self.n_tokens + len(tokens)] = tokens
        self.n_tokens += len(tokens)
        for index in range(256):
            self.buffer[index] = 0
        self.buffer[ord("b")] = 2
        self.buffer[self.sequence[self.index]] = 10

    def close(self):
        self.closed = True


class Formatter:
    def __init__(self, template, **kwargs):
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        self.template = ImmutableSandboxedEnvironment().from_string(template)
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        assert kwargs["enable_thinking"] is False
        return SimpleNamespace(prompt=self.template.render(**self.kwargs, **kwargs))


@pytest.fixture
def native(monkeypatch, tmp_path):
    FakeLlama.instances = []
    api = SimpleNamespace(
        llama_supports_gpu_offload=lambda: False,
        llama_model_get_vocab=lambda model: model,
        llama_vocab_is_eog=lambda vocab, token: token in {250, 251},
        llama_get_logits_ith=lambda ctx, row: ctypes.cast(ctx.buffer, ctypes.POINTER(ctypes.c_float)),
    )

    def data(ctx):
        return pickle.dumps((ctx.n_tokens, ctx.index, ctx.input_ids[:ctx.n_tokens]))

    def save(ctx, buffer, capacity):
        payload = data(ctx)
        assert len(payload) <= capacity
        ctypes.memmove(buffer, payload, len(payload))
        ctx.state_saves += 1
        return len(payload)

    def restore(ctx, buffer, capacity):
        count, index, tokens = pickle.loads(ctypes.string_at(buffer, capacity))
        ctx.n_tokens, ctx.index = count, index
        ctx.input_ids[:count] = tokens
        return capacity

    api.llama_state_get_size = lambda ctx: len(data(ctx))
    api.llama_state_get_data = save
    api.llama_state_set_data = restore
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama, llama_cpp=api))
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", SimpleNamespace(Jinja2ChatFormatter=Formatter))
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF")
    return path, api


def test_gguf_cache_restores_native_state_and_reevaluates_readout(native):
    path, _ = native
    backend = DecisionEngine.from_pretrained(str(path), prompt_cache_max_bytes=1024 * 1024).backend
    ids = [2] * 550
    first = backend.selected_logits(ids, [ord("a"), ord("b")])
    usage = CacheUsage()
    calls = len(backend.llama.eval_calls)
    with track_cache(usage):
        assert backend.selected_logits(ids, [ord("a"), ord("b")]) == first
    assert usage.cached_tokens == len(ids) - 2
    assert backend.llama.eval_calls[calls:][0][0] == ids[-2:]
    expected = list(backend.generate_tokens(ids, GenerationOptions(4), Event()))
    with ThreadPoolExecutor(max_workers=2) as pool:
        def work(_):
            result = list(backend.generate_tokens(ids, GenerationOptions(4), Event()))
            assert backend.selected_logits(ids, [ord("a"), ord("b")]) == first
            return result
        assert list(pool.map(work, range(4))) == [expected] * 4
    assert backend.llama.n_tokens == 0
    assert backend.prompt_cache.size_bytes <= backend.prompt_cache.max_bytes
    usage = CacheUsage()
    with track_cache(usage):
        backend.selected_logits([3] + ids[1:], [ord("b")])
    assert usage.cached_tokens == 0


def test_gguf_oversized_cache_is_not_allocated_and_cancel_releases_context(native):
    path, _ = native
    backend = GGUFBackend.load(str(path))
    backend.configure_prompt_cache(1)
    backend.selected_logits([2] * 550, [ord("b")])
    assert backend.llama.state_saves == 0 and backend.prompt_cache.size_bytes == 0
    backend.configure_prompt_cache(1024 * 1024)
    cancel = Event()
    source = backend.generate_tokens([2] * 550, GenerationOptions(4), cancel)
    next(source)
    cancel.set()
    source.close()
    assert backend.llama.n_tokens == 0
    assert backend._lock.acquire(blocking=False)
    backend._lock.release()
    assert backend.selected_logits([2] * 550, [ord("b")]) == [2]


def test_auto_gguf_loader_uses_native_weights_and_embedded_template(native):
    path, _ = native
    engine = DecisionEngine.from_pretrained(str(path), local_files_only=True)
    backend = engine.backend
    assert isinstance(backend, GGUFBackend)
    assert backend.model_name == str(path.resolve())
    assert backend.llama.options == {
        "model_path": str(path.resolve()), "n_ctx": 8192, "n_gpu_layers": 0,
        "flash_attn": False, "logits_all": False, "verbose": False,
    }
    ids = backend.tokenizer.encode(backend.tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}]))
    assert all(call == {"add_bos": False, "special": True} for call in backend.llama.tokenize_calls)
    assert "Helloassistant: " in backend.tokenizer.decode(ids)
    assert backend.selected_logits(ids, [ord("a"), ord("b")]) == [0, 2]
    assert backend.llama.n_tokens == 0


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_native_generation_and_decisions_share_loaded_weights_and_reset_context(native, path, stream):
    source, _ = native
    engine = DecisionEngine.from_pretrained(str(source))
    client = TestClient(serve.create_app(SystemOneService(engine, "local")))
    body = {"model": "local", "stream": stream}
    body.update({"messages": [{"role": "user", "content": "Hello"}]} if path.endswith("completions") else {"input": "Hello"})
    for _ in range(2):
        response = client.post(path, json=body)
        assert response.status_code == 200
        if stream:
            chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            if path.endswith("completions"):
                assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "OK "
                assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
            else:
                assert chunks[-1]["response"]["output"][0]["content"][0]["text"] == "OK "
                assert chunks[-1]["type"] == "response.completed"
        else:
            result = response.json()
            if path.endswith("completions"):
                assert result["choices"][0]["message"]["content"] == "OK "
                assert result["usage"]["completion_tokens"] == 4
            else:
                assert result["output"][0]["content"][0]["text"] == "OK "
                assert result["usage"]["output_tokens"] == 4
        decision = client.post("/v1/systemone", json={
            "model": "local", "state": "Hello", "questions": {
                "route": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
            },
        })
        assert decision.json()["answers"]["route"]["choice"] == "b"
        assert engine.backend.llama.n_tokens == 0
    assert len(FakeLlama.instances) == 1


def test_native_context_lock_spans_stream_and_closing_releases_it(native):
    path, _ = native
    backend = GGUFBackend.load(str(path))
    stream = backend.generate_tokens([1, 2], GenerationOptions(4), Event())
    assert next(stream) == ord("O")
    started = Event()

    def decide():
        started.set()
        return backend.selected_logits([3, 4], [ord("b")])

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(decide)
        try:
            assert started.wait(timeout=5)
            assert not future.done()
            assert backend.llama.eval_calls == [([1, 2], 0)]
        finally:
            stream.close()
        assert future.result(timeout=5) == [2]
    assert backend.llama.n_tokens == 0


def test_gguf_extended_sampling_restores_request_state(native):
    path, _ = native
    backend = GGUFBackend.load(str(path))
    options = GenerationOptions(4, temperature=.7, seed=42, top_k=2, min_p=.1,
        presence_penalty=.2, frequency_penalty=.3, repetition_penalty=1.1,
        logit_bias={ord("Z"): 100}, logprobs=True, top_logprobs=2)
    for _ in range(2):
        tokens = list(backend.generate_tokens([1, 2], options, Event()))
        assert tokens == [ord("Z")] * 4
        assert all(token.logprob <= 0 and token.top for token in tokens)
        assert backend.llama.n_tokens == 0


def test_cancel_while_waiting_for_native_context_does_not_evaluate(native):
    path, _ = native
    backend = GGUFBackend.load(str(path))
    backend._lock.acquire()
    cancel = Event()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: list(backend.generate_tokens([1, 2], GenerationOptions(4), cancel)))
            cancel.set()
            assert future.result(timeout=5) == []
        assert not backend.llama.eval_calls
    finally:
        backend._lock.release()


def test_native_context_and_stop_limits_are_enforced(native):
    path, _ = native
    backend = GGUFBackend.load(str(path), context_length=32)
    generation = GenerationEngine(backend, max_input_tokens=32)
    with pytest.raises(ValueError, match="context limit"):
        generation.prepare([{"role": "user", "content": "Hello"}], GenerationOptions(32))
    with pytest.raises(ValueError, match="context limit"):
        backend.selected_logits([1] * 33, [1])
    prepared = generation.prepare([{"role": "user", "content": "Hi"}], GenerationOptions(4, stop=("K",)))
    result = generation.generate(prepared)
    assert result.text == "O" and result.finish_reason == "stop"
    assert backend.llama.n_tokens == 0


def test_native_sampling_honors_top_p_without_extra_filters(native):
    path, _ = native
    backend = GGUFBackend.load(str(path))
    tokens = list(backend.generate_tokens([1, 2], GenerationOptions(4, temperature=1, top_p=0.1), Event()))
    assert tokens == [ord("O"), ord("K"), ord(" "), 251]


def test_cancel_during_prefill_stops_before_next_prompt_batch(native, monkeypatch):
    path, _ = native
    backend = GGUFBackend.load(str(path))
    backend.llama.n_batch = 2
    cancel = Event()
    original = backend.llama.eval

    def cancel_after_batch(tokens):
        original(tokens)
        cancel.set()

    monkeypatch.setattr(backend.llama, "eval", cancel_after_batch)
    assert list(backend.generate_tokens([1, 2, 3, 4], GenerationOptions(4), cancel)) == []
    assert backend.llama.eval_calls == [([1, 2], 0)]
    assert backend.llama.n_tokens == 0
    assert backend._lock.acquire(blocking=False)
    backend._lock.release()


def test_failed_native_forward_resets_and_releases_context(native, monkeypatch):
    path, _ = native
    backend = GGUFBackend.load(str(path))
    original = backend.llama.eval

    def fail(tokens):
        original(tokens)
        raise RuntimeError("native failure")

    monkeypatch.setattr(backend.llama, "eval", fail)
    with pytest.raises(RuntimeError, match="native failure"):
        list(backend.generate_tokens([1, 2], GenerationOptions(4), Event()))
    monkeypatch.setattr(backend.llama, "eval", original)
    assert backend.selected_logits([1, 2], [ord("b")]) == [2]
    assert backend.llama.n_tokens == 0


def test_gpu_offload_requires_a_capable_native_build(native):
    path, api = native
    with pytest.raises(ValueError, match="GPU offload"):
        GGUFBackend.load(str(path), device="cuda")
    assert not FakeLlama.instances
    api.llama_supports_gpu_offload = lambda: True
    backend = GGUFBackend.load(str(path), device="cuda", gpu_layers=12)
    assert backend.llama.options["n_gpu_layers"] == 12


def test_missing_template_fails_and_closes_native_model(native, monkeypatch, tmp_path):
    path, _ = native
    original = FakeLlama.__init__

    def without_template(self, **kwargs):
        original(self, **kwargs)
        self.metadata = {}

    monkeypatch.setattr(FakeLlama, "__init__", without_template)
    with pytest.raises(ValueError, match="chat template"):
        GGUFBackend.load(str(path))
    assert FakeLlama.instances[-1].closed
    template = tmp_path / "chat.jinja"
    template.write_text("{% for m in messages %}{{ m.content }}{% endfor %}assistant: ", encoding="utf-8")
    backend = GGUFBackend.load(str(path), chat_template=str(template))
    assert backend.tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}]) == "Helloassistant: "


@pytest.mark.parametrize("options", [
    {"revision": "main"}, {"dtype": "bfloat16"}, {"device": "hybrid-cuda"},
    {"context_length": 1}, {"gpu_layers": -2}, {"device": "cpu", "gpu_layers": 1},
    {"flash_attn": "true"}, {"flash_attn": 1},
])
def test_invalid_gguf_options_fail_before_native_load(native, options):
    path, _ = native
    with pytest.raises(ValueError):
        DecisionEngine.from_pretrained(str(path), **options)
    assert not FakeLlama.instances


@pytest.mark.parametrize("device, layers", [("cpu", 0), ("mps", -1)])
def test_flash_attention_is_forwarded_to_native_context_on_cpu_and_metal(native, device, layers):
    path, api = native
    api.llama_supports_gpu_offload = lambda: True
    engine = DecisionEngine.from_pretrained(str(path), device=device, flash_attn=True)
    assert engine.backend.llama.options["flash_attn"] is True
    assert engine.backend.llama.options["n_gpu_layers"] == layers
    assert engine.backend.selected_logits([1, 2], [ord("b")]) == [2]


@pytest.mark.parametrize("backend", ["hf", "mlx"])
def test_flash_attention_rejects_other_backends_before_loading(backend):
    with pytest.raises(ValueError, match="GGUF-only"):
        DecisionEngine.from_pretrained("org/model", backend=backend, flash_attn=True)


@pytest.mark.parametrize("flags, enabled", [([], False), (["--flash-attn"], True), (["--flash-attn", "--no-flash-attn"], False)])
def test_server_flash_attention_flags_reach_native_loader(native, monkeypatch, flags, enabled):
    path, _ = native
    launched = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: launched.append(app))
    serve.main(["--model", str(path), "--served-model", "local", *flags])
    assert FakeLlama.instances[0].options["flash_attn"] is enabled
    client = TestClient(launched[0])
    assert client.post("/v1/responses", json={"model": "local", "input": "Hi"}).status_code == 200


def test_json_can_override_flash_attention_cli_default_per_model(native, tmp_path, monkeypatch):
    path, _ = native
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    config = tmp_path / "models.json"
    config.write_text(json.dumps([
        {"model": str(path), "served_model": "inherited"},
        {"model": str(path), "served_model": "disabled", "flash_attn": False},
        {"model": str(path), "served_model": "enabled", "flash_attn": True},
    ]), encoding="utf-8")
    serve.main(["--models-config", str(config), "--flash-attn"])
    assert [model.options["flash_attn"] for model in FakeLlama.instances] == [True, False, True]


def test_batch_cli_flash_attention_option_reaches_native_loader(native, tmp_path):
    path, _ = native
    source = tmp_path / "cases.jsonl"
    source.write_text(json.dumps({
        "state": "Hello", "question": "Which greeting?",
        "options": [{"id": "a", "description": "Hello"}, {"id": "b", "description": "Bye"}],
    }) + "\n", encoding="utf-8")
    output = tmp_path / "result.jsonl"
    assert cli.main(["--model", str(path), "--flash-attn", "--input", str(source), "--output", str(output)]) == 0
    assert FakeLlama.instances[0].options["flash_attn"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["selected_id"] == "b"


def test_downloaded_hf_folder_and_cache_only_id_forward_offline_option(monkeypatch, tmp_path):
    from oev.hf import HuggingFaceBackend
    from oev.hybrid import HybridGemma4Backend

    calls = []
    fake = object()

    def load(source, **kwargs):
        calls.append((source, kwargs))
        return fake

    monkeypatch.setattr(HuggingFaceBackend, "load", load)
    monkeypatch.setattr(HybridGemma4Backend, "load", load)
    for device in ["cpu", "hybrid-cuda"]:
        assert DecisionEngine.from_pretrained(str(tmp_path), device=device).backend is fake
        assert calls[-1][1]["local_files_only"] is True
    assert DecisionEngine.from_pretrained("org/downloaded", local_files_only=True).backend is fake
    assert calls[-1][1]["local_files_only"] is True


def test_directory_containing_gguf_requires_an_explicit_file(native):
    path, _ = native
    with pytest.raises(ValueError, match="specific .gguf"):
        DecisionEngine.from_pretrained(str(path.parent))
    with pytest.raises(ValueError, match="existing .gguf"):
        DecisionEngine.from_pretrained(str(path.parent / "missing.gguf"))


def test_mlx_quantized_directory_has_an_explicit_format_error(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"quantization": {"bits": 4, "group_size": 64}}), encoding="utf-8")
    with pytest.raises(ValueError, match="MLX-quantized"):
        DecisionEngine.from_pretrained(str(tmp_path), backend="hf")


@pytest.mark.parametrize("entry", [
    {"model": "local.gguf", "dtype": "bfloat16"},
    {"model": "local.gguf", "revision": "main"},
    {"model": "local.gguf", "device": "hybrid-cuda"},
    {"model": "local.gguf", "context_length": 128},
    {"model": "local.gguf", "device": "cpu", "gpu_layers": 1},
    {"model": "local.gguf", "device": "cuda", "gpu_layers": 0},
    {"model": "org/hf", "gpu_layers": 0},
    {"model": "org/hf", "flash_attn": True},
    {"model": "mlx-community/model", "flash_attn": True},
    {"model": "local.gguf", "flash_attn": "true"},
    {"model": "local.gguf", "flash_attn": 1},
])
def test_invalid_mixed_config_is_rejected_before_loading_any_models(tmp_path, monkeypatch, entry):
    calls = []
    monkeypatch.setattr(serve.DecisionEngine, "from_pretrained", lambda *args, **kwargs: calls.append(args))
    config = tmp_path / "models.json"
    config.write_text(json.dumps([{"model": "valid/hf"}, entry]), encoding="utf-8")
    with pytest.raises(SystemExit):
        serve.main(["--models-config", str(config)])
    assert not calls


def test_cli_config_can_mix_downloaded_hf_and_gguf(native, tmp_path, monkeypatch):
    path, _ = native
    calls, launched = [], []
    monkeypatch.setattr(serve.DecisionEngine, "from_pretrained", lambda source, **kwargs: calls.append((source, kwargs)) or object())
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: launched.append(app))
    config = tmp_path / "models.json"
    config.write_text(json.dumps([
        {"model": "org/model", "served_model": "hf", "local_files_only": True, "dtype": "bfloat16"},
        {"model": str(path), "served_model": "gguf", "context_length": 2048, "max_input_tokens": 1024, "gpu_layers": 0, "flash_attn": True},
    ]), encoding="utf-8")
    serve.main(["--models-config", str(config)])
    assert calls[0][1]["local_files_only"] is True
    assert calls[0][1]["flash_attn"] is False
    assert calls[1][1]["context_length"] == 2048 and calls[1][1]["gpu_layers"] == 0
    assert calls[1][1]["flash_attn"] is True
    assert len(launched) == 1


def test_saved_hf_checkpoint_reloads_and_serves_entirely_from_local_files(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    vocabulary = {chr(index): index for index in range(128)}
    vocabulary["<unk>"] = 128
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>",
        chat_template="{% for m in messages %}{{ m.content }}{% endfor %}assistant: ",
    )
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=129, n_positions=1024, n_ctx=1024, n_embd=16, n_layer=1,
        n_head=2, eos_token_id=127, pad_token_id=0,
    )).eval()
    tokenizer.save_pretrained(tmp_path)
    model.save_pretrained(tmp_path)
    engine = DecisionEngine.from_pretrained(str(tmp_path), device="cpu", local_files_only=True)
    assert engine.backend.model is not model
    ids = [ord("H"), ord("i")]
    with torch.inference_mode():
        expected = model(input_ids=torch.tensor([ids])).logits[0, -1, [97, 98]].tolist()
    assert engine.backend.selected_logits(ids, [97, 98]) == pytest.approx(expected, abs=1e-6)
    client = TestClient(serve.create_app(SystemOneService(engine, "downloaded")))
    assert client.post("/v1/systemone", json={
        "model": "downloaded", "state": "Hello", "questions": {
            "route": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
        },
    }).status_code == 200
    assert client.post("/v1/responses", json={"model": "downloaded", "input": "Hi", "max_output_tokens": 4}).status_code == 200


def test_real_downloaded_gguf_when_explicitly_configured():
    source = os.environ.get("OEV_TEST_GGUF_PATH")
    if not source:
        pytest.skip("set OEV_TEST_GGUF_PATH to a local text GGUF to exercise the native runtime")
    pytest.importorskip("llama_cpp")
    flash_attn = os.environ.get("OEV_TEST_GGUF_FLASH_ATTN", "0")
    if flash_attn not in {"0", "1"}:
        pytest.fail("OEV_TEST_GGUF_FLASH_ATTN must be 0 or 1")
    backend = GGUFBackend.load(source, device=os.environ.get("OEV_TEST_GGUF_DEVICE", "cpu"),
                               context_length=2048, flash_attn=flash_attn == "1")
    try:
        generation = GenerationEngine(backend, max_input_tokens=1024)
        prepared = generation.prepare([{"role": "user", "content": "Say hello."}], GenerationOptions(4))
        first = generation.generate(prepared)
        assert generation.generate(prepared) == first
        ids, slots, _ = prepare(
            backend.tokenizer,
            Decision("Hello", "Which word?", (Option("a", "Hello"), Option("b", "Bye"))),
            1024,
        )
        assert len(backend.selected_logits(ids, slots)) == 2
        assert generation.generate(prepared) == first
    finally:
        backend.llama.close()

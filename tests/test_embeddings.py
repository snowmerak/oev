import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import struct
import sys
from threading import Event, get_ident, local
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest

from oev import DecisionEngine, EmbeddingService, ModelRegistry, SystemOneService
from oev import serve
from oev.embedding_backends import GGUFEmbeddingBackend, HFEmbeddingBackend, MLXEmbeddingBackend


class Backend:
    dimension = 3
    prompts = {"query": "Q: ", "document": "D: "}

    def __init__(self):
        self.calls = []

    def embed(self, texts, limit):
        self.calls.append((texts, limit))
        if any(len(text) > limit for text in texts):
            raise ValueError("input exceeds token limit")
        return [[3, 4, len(text)] for text in texts], sum(map(len, texts))


def client_for(backend=None, **kwargs):
    backend = backend or Backend()
    service = EmbeddingService(backend, "embed", **kwargs)
    return TestClient(serve.create_app(service)), backend, service


def test_embeddings_batch_order_usage_truncation_normalization_and_base64():
    client, backend, _ = client_for()
    body = {"model": "embed", "input": ["한국어", "longer text"], "dimensions": 2}
    response = client.post("/v1/embeddings", json=body)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list" and data["model"] == "embed"
    assert data["usage"] == {"prompt_tokens": 14, "total_tokens": 14}
    assert [item["index"] for item in data["data"]] == [0, 1]
    for item in data["data"]:
        assert item["object"] == "embedding"
        assert item["embedding"] == pytest.approx([0.6, 0.8])
        assert math.hypot(*item["embedding"]) == pytest.approx(1)
    encoded = client.post("/v1/embeddings", json={**body, "encoding_format": "base64"}).json()
    for item in encoded["data"]:
        assert struct.unpack("<2f", base64.b64decode(item["embedding"])) == pytest.approx([0.6, 0.8])
    assert backend.calls == [(body["input"], 4096)] * 2


def test_query_document_prefixes_and_raw_input_do_not_double_prefix():
    client, backend, _ = client_for(query_prefix="search: ")
    for input_type in [None, "query", "document"]:
        body = {"model": "embed", "input": "hello"}
        if input_type:
            body["input_type"] = input_type
        assert client.post("/v1/embeddings", json=body).status_code == 200
    assert [texts for texts, _ in backend.calls] == [["hello"], ["search: hello"], ["D: hello"]]


@pytest.mark.parametrize("options", [
    {"input": ""}, {"input": "  "}, {"input": []}, {"input": ["ok", ""]},
    {"input": ["ok"] * 129}, {"input": [1, 2]}, {"input": [[1, 2]]},
    {"dimensions": 0}, {"dimensions": 4}, {"dimensions": True},
    {"encoding_format": "int8"}, {"input_type": "image"}, {"stream": True},
])
def test_invalid_embedding_requests_do_not_run_inference(options):
    client, backend, _ = client_for()
    response = client.post("/v1/embeddings", json={"model": "embed", "input": "hello", **options})
    assert response.status_code == 400 and "error" in response.json()
    assert not backend.calls


def test_token_limit_includes_prefix_and_missing_prompt_fails_clearly():
    client, backend, _ = client_for(max_input_tokens=5)
    assert client.post("/v1/embeddings", json={"model": "embed", "input": "hello", "input_type": "query"}).status_code == 400
    backend.prompts = {}
    client, _, _ = client_for(backend)
    assert client.post("/v1/embeddings", json={"model": "embed", "input": "ok", "input_type": "query"}).status_code == 400


@pytest.mark.parametrize("vectors,tokens", [([[0, 0, 0]], 2), ([[1, float("nan"), 2]], 2),
                                           ([[1, 2]], 2), ([], 2), ([[1, 2, 3]], -1)])
def test_invalid_runtime_results_become_embedding_errors(vectors, tokens):
    backend = Backend()
    backend.embed = lambda texts, limit: (vectors, tokens)
    client, _, _ = client_for(backend)
    response = client.post("/v1/embeddings", json={"model": "embed", "input": "ok"})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "embedding_error"


def test_mixed_catalog_and_wrong_endpoint_reject_before_inference():
    class TextBackend:
        model_name = "text"
        tokenizer = None
    text = SystemOneService(DecisionEngine(TextBackend()), "text")
    embedding = EmbeddingService(Backend(), "embed")
    client = TestClient(serve.create_app(ModelRegistry([text, embedding])))
    assert [item["id"] for item in client.get("/v1/models").json()["data"]] == ["text", "embed"]
    for model, status in [("missing", 404), ("text", 400)]:
        assert client.post("/v1/embeddings", json={"model": model, "input": "hello"}).status_code == status
    for path, body, status in [
        ("/v1/responses", {"input": "hello"}, 400),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hello"}]}, 400),
        ("/v1/systemone", {"state": "hello", "questions": {"x": {"type": "choice", "criteria": {"a": "A"}}}}, 422),
    ]:
        assert client.post(path, json={"model": "embed", **body}).status_code == status
    assert not embedding.backend.calls


def test_embedding_requests_serialize_and_release_after_failure():
    started, release = Event(), Event()
    backend = Backend()
    original = backend.embed

    def blocked(texts, limit):
        if texts == ["first"]:
            started.set()
            assert release.wait(timeout=5)
            raise RuntimeError("runtime failure")
        return original(texts, limit)

    backend.embed = blocked
    service = EmbeddingService(backend, "embed")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.embed, ["first"])
        assert started.wait(timeout=5)
        second = pool.submit(service.embed, ["second"])
        assert not backend.calls
        release.set()
        with pytest.raises(RuntimeError):
            first.result(timeout=5)
        assert second.result(timeout=5)[1] == 6


def test_cli_mixed_tasks_load_once_and_forward_embedding_options(monkeypatch, tmp_path):
    calls, launches = [], []

    def load(source, name, **kwargs):
        calls.append((source, name, kwargs))
        return EmbeddingService(Backend(), name, query_prefix=kwargs["query_prefix"])

    monkeypatch.setattr(serve.EmbeddingService, "from_pretrained", load)
    monkeypatch.setattr(serve.DecisionEngine, "from_pretrained", lambda *args, **kwargs: DecisionEngine(SimpleNamespace(tokenizer=None)))
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: launches.append(app))
    config = tmp_path / "models.json"
    config.write_text(json.dumps([
        {"model": "text", "served_model": "chat"},
        {"model": "qwen.gguf", "served_model": "embed", "task": "embedding", "backend": "gguf", "pooling": "last", "embedding_query_prefix": "Q: "},
    ]))
    serve.main(["--models-config", str(config)])
    assert len(calls) == 1 and calls[0][:2] == ("qwen.gguf", "embed")
    assert calls[0][2]["pooling"] == "last" and calls[0][2]["query_prefix"] == "Q: "
    client = TestClient(launches[0])
    for _ in range(2):
        assert client.post("/v1/embeddings", json={"model": "embed", "input": "hi", "input_type": "query"}).status_code == 200
    assert len(calls) == 1
    assert [model["id"] for model in client.get("/v1/models").json()["data"]] == ["chat", "embed"]


@pytest.mark.parametrize("entry", [
    {"task": "embedding", "device": "hybrid-cuda"},
    {"task": "embedding", "chat_template": "template.jinja"},
    {"task": "embedding", "backend": "hf", "pooling": "last"},
    {"embedding_query_prefix": "Q: "}, {"pooling": "last"}, {"task": "unknown"},
    {"task": "embedding", "backend": "hf", "gguf_server_path": "llama-server"},
    {"backend": "gguf", "gguf_server_path": "llama-server"},
    {"task": "embedding", "backend": "hf", "mmproj_path": "mmproj.gguf"},
    {"task": "embedding", "backend": "gguf", "mmproj_path": "mmproj.gguf"},
    {"backend": "gguf", "gguf_server_path": "llama-server", "mmproj_path": "mmproj.gguf"},
])
def test_invalid_embedding_config_is_rejected_before_loading(monkeypatch, tmp_path, entry):
    monkeypatch.setattr(serve.EmbeddingService, "from_pretrained", lambda *a, **k: pytest.fail("must not load"))
    config = tmp_path / "models.json"
    config.write_text(json.dumps([{"model": "repo/model", **entry}]))
    with pytest.raises(SystemExit):
        serve.main(["--models-config", str(config)])


def test_later_model_load_failure_closes_resident_embedding_worker(monkeypatch, tmp_path):
    closed = []
    backend = Backend()
    backend.close = lambda: closed.append(True)
    first = EmbeddingService(backend, "first")

    def load(source, name, **kwargs):
        if source == "second":
            raise RuntimeError("load failed")
        return first

    monkeypatch.setattr(serve.EmbeddingService, "from_pretrained", load)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: pytest.fail("must not launch"))
    config = tmp_path / "models.json"
    config.write_text(json.dumps([{"model": name, "task": "embedding"} for name in ["first", "second"]]))
    with pytest.raises(RuntimeError, match="load failed"):
        serve.main(["--models-config", str(config)])
    assert closed == [True]


def test_hf_gemma2_retains_media_towers_with_safe_dtype_and_offline_files(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    calls = []

    def config(source, **kwargs):
        assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
        return SimpleNamespace(model_type="embedding_gemma2", vision_config=object(), audio_config=object())

    def load(source, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(get_embedding_dimension=lambda: 768, prompts={})

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", config)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=load))
    backend = HFEmbeddingBackend.load(str(tmp_path), device="cpu")
    assert backend.dimension == 768
    assert calls[0]["config_kwargs"] == {}
    assert backend.modalities == {"text", "image", "audio"}
    assert calls[0]["model_kwargs"]["torch_dtype"] == torch.float32
    assert calls[0]["local_files_only"] is True
    with pytest.raises(ValueError, match="float16"):
        HFEmbeddingBackend.load(str(tmp_path), dtype="float16")
    assert len(calls) == 1


def test_hf_uses_native_encoder_without_truncation_or_default_prompt():
    calls = []
    model = SimpleNamespace(prompts={"query": "Q: "}, max_seq_length=8,
                            get_embedding_dimension=lambda: 3)

    def preprocess(texts, **kwargs):
        assert kwargs == {"processing_kwargs": {"text": {"truncation": False}}}
        return {"attention_mask": np.ones((1, len(texts[0]))) }

    def encode(texts, **kwargs):
        calls.append(kwargs)
        assert kwargs["prompt"] == "" and kwargs["batch_size"] == 1
        assert kwargs["processing_kwargs"]["text"]["truncation"] is False
        return np.array([[3., 4., 5.] for text in texts])

    model.preprocess, model.encode = preprocess, encode
    backend = HFEmbeddingBackend(model)
    assert backend.embed(["hello"], 8) == ([[3., 4., 5.]], 5)
    with pytest.raises(ValueError, match="limit"):
        backend.embed(["very long text"], 8)
    assert len(calls) == 1


def test_gguf_embedding_mode_batch_pooling_limits_and_cleanup(monkeypatch, tmp_path):
    instances = []
    api = SimpleNamespace(LLAMA_POOLING_TYPE_UNSPECIFIED=-1, LLAMA_POOLING_TYPE_NONE=0,
                          LLAMA_POOLING_TYPE_MEAN=1, LLAMA_POOLING_TYPE_CLS=2, LLAMA_POOLING_TYPE_LAST=3,
                          llama_supports_gpu_offload=lambda: True)

    class Llama:
        def __init__(self, **kwargs):
            self.options, self.n_batch, self.resets = kwargs, kwargs["n_batch"], 0
            self.metadata = {}
            self.calls = []
            instances.append(self)
        def n_embd(self): return 3
        def n_ctx(self): return self.options["n_ctx"]
        def pooling_type(self): return self.options["pooling_type"]
        def tokenize(self, text): return list(text)
        def reset(self): self.resets += 1
        def close(self): self.closed = True
        def embed(self, texts, **kwargs):
            self.calls.append(kwargs)
            assert kwargs == {"normalize": False, "truncate": False, "return_count": True}
            if texts == ["fail"]: raise RuntimeError("native failure")
            return [[3, 4, 5] for text in texts], sum(len(text.encode()) for text in texts)

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=Llama, llama_cpp=api))
    path = tmp_path / "model.gguf"
    path.touch()
    backend = GGUFEmbeddingBackend.load(str(path), device="mps", context_length=16, pooling="last", flash_attn=True)
    assert instances[0].options["embedding"] is True
    assert instances[0].options["n_gpu_layers"] == -1
    assert instances[0].options["n_batch"] == instances[0].options["n_ubatch"] == 16
    assert backend.embed(["hi"], 8)[1] == 2
    with pytest.raises(ValueError): backend.embed(["too long input"], 8)
    with pytest.raises(RuntimeError): backend.embed(["fail"], 8)
    assert instances[0].resets == 2
    with pytest.raises(ValueError, match="sequence embeddings"):
        GGUFEmbeddingBackend.load(str(path), pooling="auto")
    assert instances[-1].closed

    original_init = Llama.__init__
    def projected_init(self, **kwargs):
        original_init(self, **kwargs)
        self.metadata = {"general.architecture": "projected", "projected.embedding_length_out": 4}
    monkeypatch.setattr(Llama, "__init__", projected_init)
    with pytest.raises(ValueError, match="projected output width"):
        GGUFEmbeddingBackend.load(str(path), pooling="mean")
    assert instances[-1].closed


@pytest.mark.parametrize("exits_at_startup", [False, True])
def test_native_server_command_lifecycle_projection_and_order(monkeypatch, tmp_path, exits_at_startup):
    from oev.gguf_embedding_server import GGUFServerEmbeddingBackend

    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            self.command, self.options, self.terminated = command, kwargs, False
            calls.append(self)
        def poll(self): return 1 if exits_at_startup or self.terminated else None
        def terminate(self): self.terminated = True
        def wait(self, timeout): return 0

    def request(self, path, body=None, **kwargs):
        if path == "/health": return {"status": "ok"}
        if path == "/tokenize":
            assert body["add_special"] is True
            return {"tokens": list(range(len(body["content"]) + 2))}
        if body["input"] == "warmup":
            return {"data": [{"index": 0, "embedding": [1.] * 768}]}
        return {"data": [{"index": 1, "embedding": [4.] * 768}, {"index": 0, "embedding": [3.] * 768}],
                "usage": {"prompt_tokens": 8}}

    binary = tmp_path / "llama-server"
    binary.touch()
    monkeypatch.setattr("oev.gguf_embedding_server.subprocess.Popen", Process)
    monkeypatch.setattr(GGUFServerEmbeddingBackend, "_request", request)
    if exits_at_startup:
        with pytest.raises(RuntimeError, match="did not start"):
            GGUFServerEmbeddingBackend("model.gguf", str(binary), context_length=8192,
                                       gpu_layers=-1, flash_attn=True, pooling="mean")
        assert calls[0].options["stdout"].closed
        return
    backend = GGUFServerEmbeddingBackend("model.gguf", str(binary), context_length=8192,
                                         gpu_layers=-1, flash_attn=True, pooling="mean")
    process = calls[0]
    assert process.command[process.command.index("--host") + 1] == "127.0.0.1"
    assert process.command[process.command.index("--pooling") + 1] == "mean"
    assert process.command[process.command.index("--flash-attn") + 1] == "on"
    assert process.options["cwd"] == binary.parent
    assert backend.dimension == 768
    assert backend.embed(["hi", "ok"], 8) == ([[3.] * 768, [4.] * 768], 8)
    with pytest.raises(ValueError, match="limit"):
        backend.embed(["too long"], 5)
    service = EmbeddingService(backend, "native")
    with TestClient(serve.create_app(service)) as client:
        assert client.get("/v1/models").status_code == 200
        assert not process.terminated
    assert process.terminated and process.options["stdout"].closed
    backend.close()  # Finalization is idempotent.


def test_mlx_embeddings_loading_and_first_forward_on_http_worker(monkeypatch, tmp_path):
    state, syncs, loads = local(), [], []

    @contextmanager
    def stream(value):
        previous = getattr(state, "stream", None)
        state.stream = value
        try: yield
        finally: state.stream = previous

    mx = SimpleNamespace(cpu="cpu", gpu="gpu", int32=np.int32, float32=np.float32,
                         array=np.array, ones=np.ones, eval=lambda value: None, stream=stream,
                         metal=SimpleNamespace(is_available=lambda: True),
                         new_thread_unsafe_stream=lambda device: SimpleNamespace(device=device, shared=True),
                         synchronize=lambda value: syncs.append(value))

    def load(path, **kwargs):
        owner, captured = get_ident(), state.stream
        loads.append(kwargs)
        def forward(ids, attention_mask):
            assert get_ident() != owner and state.stream is captured and captured.shared
            assert (attention_mask == 1).all()
            return SimpleNamespace(text_embeds=np.array([[3., 4., 5.]]))
        return forward, SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()))

    monkeypatch.setattr("oev.embedding_backends.platform.system", lambda: "Darwin")
    monkeypatch.setattr("oev.embedding_backends.platform.machine", lambda: "arm64")
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx_embeddings", SimpleNamespace(load=load))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 3, "max_position_embeddings": 32}))
    backend = MLXEmbeddingBackend.load(str(tmp_path))
    client, _, _ = client_for(backend)
    for _ in range(2):
        assert client.post("/v1/embeddings", json={"model": "embed", "input": ["hi", "hello"]}).status_code == 200
    assert len(loads) == 1 and loads[0]["tokenizer_config"]["local_files_only"]
    assert syncs == [backend._stream] * 3
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "unknown_embedding"}))
    with pytest.raises(ValueError, match="unsupported MLX embedding architecture"):
        MLXEmbeddingBackend.load(str(tmp_path))


def test_real_tiny_hf_checkpoint_offline_and_http_batch_order(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("sentence_transformers")
    path = tmp_path / "tiny-bert"
    path.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "cat", "dog"]
    (path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = transformers.BertTokenizer(vocab_file=str(path / "vocab.txt"))
    tokenizer.save_pretrained(path)
    torch.manual_seed(42)
    transformers.BertModel(transformers.BertConfig(vocab_size=len(vocab), hidden_size=12,
                          num_hidden_layers=1, num_attention_heads=2, intermediate_size=24,
                          max_position_embeddings=32)).save_pretrained(path)
    service = EmbeddingService.from_pretrained(str(path), "tiny", backend="hf", device="cpu", max_input_tokens=16)
    client = TestClient(serve.create_app(service))
    texts = ["hello world", "dog", "cat hello"]
    batch = client.post("/v1/embeddings", json={"model": "tiny", "input": texts})
    assert batch.status_code == 200, batch.text
    for text, item in zip(texts, batch.json()["data"]):
        single = client.post("/v1/embeddings", json={"model": "tiny", "input": text}).json()["data"][0]
        assert item["embedding"] == pytest.approx(single["embedding"], abs=1e-6)
        assert len(item["embedding"]) == 12 and math.hypot(*item["embedding"]) == pytest.approx(1)
    assert client.post("/v1/embeddings", json={"model": "tiny", "input": "hello " * 40}).status_code == 400


@pytest.mark.parametrize("backend", ["hf", "gguf", "mlx"])
def test_real_embedding_checkpoint_when_explicitly_configured(backend):
    source = os.environ.get(f"OEV_TEST_EMBEDDING_{backend.upper()}_PATH")
    if not source:
        pytest.skip(f"set OEV_TEST_EMBEDDING_{backend.upper()}_PATH to a downloaded embedding checkpoint")
    kwargs = {"pooling": os.environ.get("OEV_TEST_EMBEDDING_POOLING", "auto")} if backend == "gguf" else {}
    if backend == "gguf" and os.environ.get("OEV_TEST_GGUF_SERVER_PATH"):
        kwargs["gguf_server_path"] = os.environ["OEV_TEST_GGUF_SERVER_PATH"]
    service = EmbeddingService.from_pretrained(source, "embedding-real", backend=backend, local_files_only=True,
                                              device=os.environ.get("OEV_TEST_EMBEDDING_DEVICE", "cpu"), **kwargs)
    with TestClient(serve.create_app(service)) as client:
        texts = ["The dog eats food.", "Seoul is a city.", "The dog eats food."]
        response = client.post("/v1/embeddings", json={"model": "embedding-real", "input": texts})
        assert response.status_code == 200, response.text
        vectors = [item["embedding"] for item in response.json()["data"]]
        assert vectors[0] == pytest.approx(vectors[2], abs=1e-5)
        for vector in vectors:
            assert len(vector) == service.backend.dimension
            assert all(math.isfinite(value) for value in vector) and math.hypot(*vector) == pytest.approx(1)
        truncated = client.post("/v1/embeddings", json={"model": "embedding-real", "input": texts, "dimensions": 128})
        assert truncated.status_code == 200, truncated.text
        for vector, item in zip(vectors, truncated.json()["data"]):
            norm = math.hypot(*vector[:128])
            assert item["embedding"] == pytest.approx([value / norm for value in vector[:128]], abs=1e-5)

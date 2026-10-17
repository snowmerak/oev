import base64
from contextlib import contextmanager
from io import BytesIO
import json
import math
import os
import sys
from threading import get_ident, local
from types import SimpleNamespace
from urllib.error import HTTPError

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest
import soundfile as sf

from oev.embedding_backends import HFEmbeddingBackend
from oev.embedding_inputs import decode_audio, decode_image
from oev.embeddings import EmbeddingService
from oev.serve import create_app


def data_url(data, mime):
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


@pytest.fixture
def media():
    image, audio = BytesIO(), BytesIO()
    Image.new("RGB", (16, 12), (200, 30, 10)).save(image, format="PNG")
    samples = np.column_stack([np.full(800, 0.25), np.full(800, 0.5)])
    sf.write(audio, samples, 8000, format="WAV", subtype="FLOAT")
    return {"image": data_url(image.getvalue(), "image/png"), "audio": data_url(audio.getvalue(), "audio/wav")}


class NativeEncoder:
    prompts = {"query": "Q: ", "document": "D: "}
    max_seq_length = 1024

    def __init__(self):
        self.preprocessed, self.encoded = [], []

    def get_embedding_dimension(self):
        return 3

    def preprocess(self, items, **kwargs):
        assert kwargs["processing_kwargs"]["text"]["truncation"] is False
        self.preprocessed.append((items, kwargs))
        item = items[0]
        count = len(kwargs.get("prompt", "")) + len(item.get("text", ""))
        if "image" in item:
            image = item["image"]
            for part in image if isinstance(image, list) else [image]:
                assert isinstance(part, Image.Image) and part.mode == "RGB"
            count += 20
        if "audio" in item:
            audio = item["audio"]
            for part in audio if isinstance(audio, list) else [audio]:
                assert part["sampling_rate"] == 16000 and part["array"].shape == (1600,)
            count += 30
        return {"attention_mask": np.ones((1, count + 2))}

    def encode(self, items, **kwargs):
        assert kwargs["processing_kwargs"]["text"]["truncation"] is False
        self.encoded.append((items, kwargs))
        return np.array([[3, 4, len(item)] for item in items])


def client_for(limit=1024, modalities=("text", "image", "audio")):
    model = NativeEncoder()
    service = EmbeddingService(HFEmbeddingBackend(model, modalities=modalities), "gemma", max_input_tokens=limit)
    return TestClient(create_app(service)), model, service


def test_http_mixed_modalities_order_prefix_usage_and_normalization(media):
    client, model, service = client_for()
    inputs = ["cat", {"image": media["image"]}, {"audio": media["audio"]},
              {"text": "cat", "image": media["image"], "audio": media["audio"]}]
    body = {"model": "gemma", "input": inputs, "input_type": "document", "dimensions": 2}
    response = client.post("/v1/embeddings", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["usage"] == {"prompt_tokens": 126, "total_tokens": 126}
    assert [row["index"] for row in result["data"]] == [0, 1, 2, 3]
    for row in result["data"]:
        assert row["embedding"] == pytest.approx([0.6, 0.8])
    assert len(model.encoded) == 1
    assert model.encoded[0][1]["prompt"] == "D: "
    assert all(options["prompt"] == "D: " for _, options in model.preprocessed)
    assert list(model.encoded[0][0][3]) == ["text", "image", "audio"]
    assert client.get("/v1/models").json()["data"][0]["input_modalities"] == ["audio", "image", "text"]
    assert inputs[3]["audio"] == media["audio"]  # No mutation of caller input.


@pytest.mark.parametrize("modality", ["image", "audio"])
def test_single_media_input_and_multiple_media_per_vector(media, modality):
    client, model, _ = client_for()
    for value in [media[modality], [media[modality], media[modality]]]:
        response = client.post("/v1/embeddings", json={"model": "gemma", "input": {modality: value}, "encoding_format": "base64"})
        assert response.status_code == 200, response.text
        assert len(response.json()["data"]) == 1
        vector = np.frombuffer(base64.b64decode(response.json()["data"][0]["embedding"]), dtype="<f4")
        assert math.hypot(*vector) == pytest.approx(1)


def test_http_preserves_combined_modality_order(media):
    client, model, _ = client_for()
    body = {"model": "gemma", "input": {"audio": media["audio"], "image": media["image"], "text": "cat"}}
    assert client.post("/v1/embeddings", json=body).status_code == 200
    assert list(model.encoded[0][0][0]) == ["audio", "image", "text"]


def test_audio_is_downmixed_and_resampled_and_image_decoded(media):
    image = decode_image(media["image"])
    assert image.size == (16, 12) and image.getpixel((0, 0)) == (200, 30, 10)
    audio = decode_audio(media["audio"])
    assert audio["sampling_rate"] == 16000 and audio["array"].dtype == np.float32
    assert audio["array"].shape == (1600,)
    assert np.mean(audio["array"][100:-100]) == pytest.approx(0.375, abs=1e-4)


def test_media_tokens_and_prefix_count_toward_limit_before_encode(media):
    client, model, _ = client_for(limit=22)
    body = {"model": "gemma", "input": [{"image": media["image"]}, {"audio": media["audio"]}]}
    assert client.post("/v1/embeddings", json=body).status_code == 400
    assert not model.encoded
    body["input"] = {"image": media["image"]}
    assert client.post("/v1/embeddings", json=body).status_code == 200
    body["input_type"] = "query"
    assert client.post("/v1/embeddings", json=body).status_code == 400
    assert len(model.encoded) == 1


@pytest.mark.parametrize("item", [
    {}, {"text": ""}, {"image": None}, {"audio": []}, {"video": "data:video/mp4;base64,YQ=="},
    {"image": "https://example.com/a.png"}, {"image": "C:/private.png"},
    {"audio": "file:///private.wav"}, {"image": "data:image/png;base64,!!"},
    {"image": "data:image/png;base64,YQ=="}, {"audio": "data:audio/wav;base64,YQ=="},
    {"image": "data:audio/wav;base64,YQ=="}, {"audio": 42}, {"text": "ok", "unknown": True},
])
def test_invalid_media_fails_without_inference(item):
    client, model, _ = client_for()
    response = client.post("/v1/embeddings", json={"model": "gemma", "input": ["ok", item]})
    assert response.status_code == 400, response.text
    assert "error" in response.json() and not model.encoded


def test_unsupported_backend_rejects_media_and_accepts_text_objects(media):
    client, model, _ = client_for(modalities=("text",))
    response = client.post("/v1/embeddings", json={"model": "gemma", "input": {"image": media["image"]}})
    assert response.status_code == 400
    assert "does not support image" in response.json()["error"]["message"]
    assert not model.preprocessed and not model.encoded


def test_service_validates_direct_python_inputs(media):
    _, model, service = client_for()
    for item in [{"audio": None}, {"text": 2}, {"unknown": "x"}]:
        with pytest.raises(ValueError):
            service.embed([item])
    assert not model.encoded


def test_plain_strings_remain_text_even_when_they_look_like_media():
    client, model, _ = client_for()
    for text in ["https://example.com/a.png", "C:/private.wav", "data:image/png;base64,YQ=="]:
        response = client.post("/v1/embeddings", json={"model": "gemma", "input": text})
        assert response.status_code == 200, response.text
        assert model.encoded[-1][0] == [{"text": text}]


def test_text_only_checkpoint_with_media_processor_keeps_strings_as_text():
    client, model, _ = client_for(modalities=("text",))
    model.modalities = ["text", "image", "audio", "message"]
    text = "https://example.com/private.png"
    response = client.post("/v1/embeddings", json={"model": "gemma", "input": {"text": text}})
    assert response.status_code == 200, response.text
    assert model.encoded[0][0] == [{"text": text}]


def test_payload_size_and_media_count_limits_precede_inference(monkeypatch, media):
    import oev.embedding_inputs as inputs

    client, model, _ = client_for()
    monkeypatch.setattr(inputs, "MAX_MEDIA_ITEMS", 1)
    body = {"model": "gemma", "input": {"image": [media["image"], media["image"]]}}
    assert client.post("/v1/embeddings", json=body).status_code == 400
    monkeypatch.setattr(inputs, "MAX_MEDIA_BYTES", 4)
    body["input"] = {"image": media["image"]}
    assert client.post("/v1/embeddings", json=body).status_code == 400
    assert not model.preprocessed and not model.encoded


def test_mlx_native_gemma2_load_and_multimodal_http_worker_stream(monkeypatch, tmp_path, media):
    from oev.embedding_backends import MLXEmbeddingBackend
    import transformers
    auto_processor = transformers.AutoProcessor

    state, syncs, forwards, processed, loads = local(), [], [], [], []

    @contextmanager
    def stream(value):
        previous = getattr(state, "stream", None)
        state.stream = value
        try:
            yield
        finally:
            state.stream = previous

    mx = SimpleNamespace(cpu="cpu", gpu="gpu", float32=np.float32, array=np.array, eval=lambda x: None,
                         stream=stream, new_thread_unsafe_stream=lambda device: object(),
                         synchronize=lambda value: syncs.append(value), metal=SimpleNamespace(is_available=lambda: True))
    owner = get_ident()

    def load(path, **kwargs):
        assert state.stream is not None and kwargs == {"lazy": False}
        loads.append(path)

        def forward(**tensors):
            assert get_ident() != owner and state.stream is backend._stream
            forwards.append(tensors)
            return SimpleNamespace(text_embeds=np.array([[3, 4, tensors["input_ids"].shape[1]]]))
        return forward

    def process(messages, **kwargs):
        assert kwargs["processor_kwargs"] == {"text_kwargs": {"padding": True, "truncation": False},
                                               "audio_kwargs": {"sampling_rate": 16000, "truncation": False}}
        assert kwargs["return_tensors"] == "np"
        processed.append(messages)
        content = messages[-1]["content"]
        count = 2 + sum(1 if part["type"] == "text" else 20 for part in content)
        result = {"input_ids": np.ones((1, count), dtype=np.int64), "attention_mask": np.ones((1, count))}
        for part in content:
            if part["type"] == "image":
                assert isinstance(part["image"], Image.Image)
                result["pixel_values"] = np.ones((1, 2, 3), dtype=np.float32)
            if part["type"] == "audio":
                assert part["audio"].shape == (1600,)
                result["input_features"] = np.ones((1, 10, 128), dtype=np.float32)
        return result

    monkeypatch.setattr("oev.embedding_backends.platform.system", lambda: "Darwin")
    monkeypatch.setattr("oev.embedding_backends.platform.machine", lambda: "arm64")
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mlx_vlm.embedding_loader", SimpleNamespace(load_embedding_model=load))
    monkeypatch.setattr(auto_processor, "from_pretrained",
                        lambda *args, **kwargs: SimpleNamespace(apply_chat_template=process))
    config = {"model_type": "embedding_gemma2", "text_config": {"embedding_dim": 3},
              "vision_config": {"hidden_size": 32}, "audio_config": {"hidden_size": 32}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "config_sentence_transformers.json").write_text(json.dumps({"prompts": {"document": "D: "}}))
    backend = MLXEmbeddingBackend.load(str(tmp_path), local_files_only=True)
    service = EmbeddingService(backend, "mlx", max_input_tokens=64)
    client = TestClient(create_app(service))
    body = {"model": "mlx", "input": ["cat", media], "input_type": "document", "dimensions": 2}
    for _ in range(2):
        response = client.post("/v1/embeddings", json=body)
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["prompt_tokens"] == 45
        assert response.json()["data"][1]["embedding"] == pytest.approx([0.6, 0.8])
    assert len(loads) == 1 and len(forwards) == 4
    assert all(messages[0] == {"role": "system", "content": "D: "} for messages in processed)
    assert syncs == [backend._stream] * 3
    service.max_input_tokens = 10
    assert client.post("/v1/embeddings", json=body).status_code == 400
    assert len(forwards) == 4
    service.max_input_tokens = 64
    backend.model = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("model failure"))
    assert client.post("/v1/embeddings", json=body).status_code == 500
    assert syncs == [backend._stream] * 4


def test_gguf_multimodal_projector_typed_content_usage_and_errors(monkeypatch, tmp_path, media):
    from oev.embedding_backends import load_embedding_backend
    from oev.gguf_embedding_server import GGUFServerEmbeddingBackend

    commands, requests = [], []

    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.stopped = False
        def poll(self): return 0 if self.stopped else None
        def terminate(self): self.stopped = True
        def wait(self, timeout): return 0

    def request(self, path, body=None, **kwargs):
        requests.append((path, body))
        if path == "/health": return {}
        if path == "/props": return {"modalities": {"vision": True, "audio": True}}
        if body["input"] == "warmup": return {"data": [{"index": 0, "embedding": [1, 2, 3]}]}
        content = body["input"][1]["content"]
        assert [part["type"] for part in content] == ["text", "image_url", "input_audio"]
        assert content[0]["text"] == "D: "
        image = content[1]["image_url"]["url"]
        assert decode_image(image).size == (16, 12)
        audio = BytesIO(base64.b64decode(content[2]["input_audio"]["data"]))
        samples, rate = sf.read(audio)
        assert rate == 16000 and samples.shape == (1600,)
        return {"data": [{"index": 1, "embedding": [3, 4, 5]}, {"index": 0, "embedding": [5, 4, 3]}],
                "usage": {"prompt_tokens": 42}}

    binary, model, projector = [tmp_path / name for name in ["llama-server", "model.gguf", "mmproj.gguf"]]
    for path in [binary, model, projector]: path.touch()
    monkeypatch.setattr("oev.gguf_embedding_server.subprocess.Popen", Process)
    monkeypatch.setattr(GGUFServerEmbeddingBackend, "_request", request)
    backend = load_embedding_backend(str(model), gguf_server_path=str(binary), mmproj_path=str(projector),
                                     context_length=8192, max_input_tokens=100)
    assert backend.context_window == 100
    assert commands[0][commands[0].index("--mmproj") + 1] == str(projector)
    client = TestClient(create_app(EmbeddingService(backend, "gguf", max_input_tokens=100, document_prefix="D: ")))
    body = {"model": "gguf", "input": ["cat", media], "input_type": "document", "dimensions": 2}
    response = client.post("/v1/embeddings", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["usage"]["prompt_tokens"] == 42
    assert response.json()["data"][1]["embedding"] == pytest.approx([0.6, 0.8])
    assert not any(path == "/tokenize" for path, _ in requests)

    def error(*args, **kwargs):
        raise HTTPError("local", 500, "error", {}, BytesIO(b'{"error":{"message":"input is too large"}}'))
    backend._request = error
    assert client.post("/v1/embeddings", json=body).status_code == 400
    backend.close()
    assert client.post("/v1/embeddings", json=body).status_code == 500


@pytest.mark.parametrize("backend", ["hf", "gguf", "mlx"])
def test_real_multimodal_embedding_checkpoint(backend, media):
    source = os.environ.get(f"OEV_TEST_EMBEDDING_{backend.upper()}_PATH")
    if not source:
        pytest.skip(f"set OEV_TEST_EMBEDDING_{backend.upper()}_PATH to a full EmbeddingGemma2 checkpoint")
    kwargs = {}
    if backend == "gguf":
        projector = os.environ.get("OEV_TEST_EMBEDDING_MMPROJ_PATH")
        binary = os.environ.get("OEV_TEST_GGUF_SERVER_PATH")
        if not projector or not binary:
            pytest.skip("set OEV_TEST_EMBEDDING_MMPROJ_PATH and OEV_TEST_GGUF_SERVER_PATH")
        kwargs = {"mmproj_path": projector, "gguf_server_path": binary, "pooling": "mean"}
    service = EmbeddingService.from_pretrained(source, "native", backend=backend, local_files_only=True,
                                              device=os.environ.get("OEV_TEST_EMBEDDING_DEVICE", "cpu"),
                                              max_input_tokens=1024, **kwargs)
    blue = BytesIO()
    Image.new("RGB", (16, 12), "blue").save(blue, format="PNG")
    inputs = ["a dog barking", {"image": media["image"]}, {"audio": media["audio"]},
              {"text": "a dog", "image": media["image"]},
              {"image": data_url(blue.getvalue(), "image/png")}]
    with TestClient(create_app(service)) as client:
        response = client.post("/v1/embeddings", json={"model": "native", "input": inputs})
        assert response.status_code == 200, response.text
        vectors = [item["embedding"] for item in response.json()["data"]]
        for item, vector in zip(inputs, vectors):
            single = client.post("/v1/embeddings", json={"model": "native", "input": item})
            assert single.status_code == 200, single.text
            assert len(vector) == 768 and math.hypot(*vector) == pytest.approx(1)
            assert vector == pytest.approx(single.json()["data"][0]["embedding"], abs=1e-5)
        assert np.dot(vectors[1], vectors[-1]) < 0.9999  # Media cannot silently become text-only.
        reduced = client.post("/v1/embeddings", json={"model": "native", "input": inputs, "dimensions": 128})
        assert reduced.status_code == 200, reduced.text
        for full, row in zip(vectors, reduced.json()["data"]):
            assert row["embedding"] == pytest.approx(np.array(full[:128]) / math.hypot(*full[:128]), abs=1e-5)
        if backend in {"hf", "mlx"}:
            # Saved processors default to a 30-second audio truncation and even
            # smaller placeholder caps. A 40-second input must count in full.
            long_audio = BytesIO()
            sf.write(long_audio, np.zeros(40 * 16000, dtype=np.float32), 16000, format="WAV", subtype="FLOAT")
            service.max_input_tokens = 800
            rejected = client.post("/v1/embeddings", json={"model": "native", "input": {
                "audio": data_url(long_audio.getvalue(), "audio/wav")}})
            assert rejected.status_code == 400, rejected.text
            message = rejected.json()["error"]["message"]
            assert "tokens; embedding limit is 800" in message and "100" in message

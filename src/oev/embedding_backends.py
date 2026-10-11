"""Optional embedding runtimes; preserve each model's native pooling and head."""

from __future__ import annotations

import json
from pathlib import Path
import platform
from threading import Lock

from .engine import model_backend


def _check_tokens(counts, limit):
    for index, count in enumerate(counts):
        if not 0 < count <= limit:
            raise ValueError(f"input[{index}] has {count} tokens; embedding limit is {limit}")


class HFEmbeddingBackend:
    def __init__(self, model):
        self.model = model
        self.dimension = model.get_embedding_dimension()
        self.prompts = dict(model.prompts)

    @classmethod
    def load(cls, source, *, revision=None, device="auto", dtype="auto", local_files_only=False):
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("HF embeddings require device=auto/cpu/cuda/mps")
        if dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("invalid embedding dtype")
        try:
            import torch
            from transformers import AutoConfig
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install HF embeddings with: uv sync --extra embeddings --extra serve") from exc
        path = Path(source).expanduser()
        if path.is_file():
            raise ValueError("HF embeddings require a complete model directory or repository ID")
        source = str(path.resolve()) if path.is_dir() else source
        offline = local_files_only or path.is_dir()
        config = AutoConfig.from_pretrained(source, revision=revision, local_files_only=offline, trust_remote_code=False)
        config_kwargs = {}
        if config.model_type == "embedding_gemma2":
            if dtype == "float16":
                raise ValueError("EmbeddingGemma2 requires float32 or bfloat16; float16 produces invalid embeddings")
            # The endpoint accepts text; omit unused vision/audio encoders.
            config_kwargs = {"vision_config": None, "audio_config": None}
            if dtype == "auto":
                dtype = "float32"
        model = SentenceTransformer(
            source, device=None if device == "auto" else device, revision=revision,
            local_files_only=offline, trust_remote_code=False,
            model_kwargs={"torch_dtype": "auto" if dtype == "auto" else getattr(torch, dtype)},
            config_kwargs=config_kwargs,
        )
        return cls(model)

    def embed(self, texts, max_input_tokens):
        limit = min(max_input_tokens, self.model.max_seq_length or max_input_tokens)
        # Validate before encode's automatic truncation, using the same processor.
        counts = []
        for text in texts:
            inputs = self.model.preprocess([text], processing_kwargs={"text": {"truncation": False}})
            counts.append(int(inputs["attention_mask"].sum().item()))
        _check_tokens(counts, limit)
        vectors = self.model.encode(
            texts, batch_size=1, show_progress_bar=False, prompt="",
            normalize_embeddings=False, convert_to_numpy=True,
            processing_kwargs={"text": {"truncation": False}},
        )
        return vectors.tolist(), sum(counts)


class GGUFEmbeddingBackend:
    def __init__(self, llama):
        self.llama = llama
        self.dimension = llama.n_embd()
        self.prompts = {}
        self._lock = Lock()

    @classmethod
    def load(cls, source, *, device="auto", context_length=8192, gpu_layers=None,
             flash_attn=False, pooling="auto", gguf_server_path=None):
        path = Path(source).expanduser()
        if not path.is_file() or path.suffix.lower() != ".gguf":
            raise ValueError("GGUF embeddings require an existing local .gguf file")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("GGUF embeddings require device=auto/cpu/cuda/mps")
        if type(context_length) is not int or context_length < 1:
            raise ValueError("context_length must be positive")
        if gpu_layers is not None and (type(gpu_layers) is not int or gpu_layers < -1):
            raise ValueError("gpu_layers must be -1 or nonnegative")
        if type(flash_attn) is not bool or pooling not in {"auto", "mean", "last", "cls"}:
            raise ValueError("invalid GGUF embedding options")
        layers = gpu_layers if gpu_layers is not None else -1 if device in {"cuda", "mps"} else 0
        if (device == "cpu" and layers != 0) or (device in {"cuda", "mps"} and layers == 0):
            raise ValueError("GPU offload settings do not match device")
        if gguf_server_path is not None:
            from .gguf_embedding_server import GGUFServerEmbeddingBackend

            return GGUFServerEmbeddingBackend(str(path), gguf_server_path, context_length=context_length,
                                               gpu_layers=layers, flash_attn=flash_attn, pooling=pooling)
        try:
            from llama_cpp import Llama, llama_cpp as api
        except ImportError as exc:
            raise RuntimeError("Install GGUF embeddings with: uv sync --extra gguf --extra serve") from exc
        if layers != 0 and not api.llama_supports_gpu_offload():
            raise ValueError("this llama-cpp-python build does not support GPU offload")
        pooling_type = getattr(api, "LLAMA_POOLING_TYPE_" + {"auto": "UNSPECIFIED"}.get(pooling, pooling.upper()))
        # Pooled inputs must fit one physical batch; never truncate them.
        llama = Llama(model_path=str(path.resolve()), embedding=True, pooling_type=pooling_type,
                      n_ctx=context_length, n_batch=context_length, n_ubatch=context_length,
                      n_gpu_layers=layers, flash_attn=flash_attn, verbose=False)
        try:
            if llama.pooling_type() not in {api.LLAMA_POOLING_TYPE_MEAN, api.LLAMA_POOLING_TYPE_LAST, api.LLAMA_POOLING_TYPE_CLS}:
                raise ValueError("GGUF must provide sequence embeddings; choose pooling=mean/last/cls")
            architecture = llama.metadata.get("general.architecture", "")
            output_dimension = llama.metadata.get(f"{architecture}.embedding_length_out", llama.n_embd())
            if int(output_dimension) != llama.n_embd():
                raise ValueError("this GGUF has a projected output width; use gguf_server_path with a recent llama-server")
            return cls(llama)
        except Exception:
            llama.close()
            raise

    def embed(self, texts, max_input_tokens):
        with self._lock:
            return self._embed(texts, max_input_tokens)

    def _embed(self, texts, max_input_tokens):
        counts = [len(self.llama.tokenize(text.encode("utf-8"))) for text in texts]
        _check_tokens(counts, min(max_input_tokens, self.llama.n_ctx(), self.llama.n_batch))
        try:
            return self.llama.embed(texts, normalize=False, truncate=False, return_count=True)
        finally:
            self.llama.reset()

    def close(self):
        with self._lock:
            self.llama.close()


class MLXEmbeddingBackend:
    def __init__(self, model, tokenizer, config, prompts, mx, stream):
        self.model, self.tokenizer, self._mx, self._stream = model, tokenizer, mx, stream
        self.dimension = config["hidden_size"]
        self.context_window = config.get("max_position_embeddings", 512)
        self.prompts = prompts
        self._lock = Lock()

    @classmethod
    def load(cls, source, *, revision=None, device="auto", dtype="auto", local_files_only=False):
        if device not in {"auto", "cpu", "mps"} or dtype != "auto":
            raise ValueError("MLX embeddings require device=auto/cpu/mps and dtype=auto")
        if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
            raise RuntimeError("MLX embeddings require Apple Silicon macOS")
        try:
            import mlx.core as mx
            from mlx_embeddings import load
        except ImportError as exc:
            raise RuntimeError("Install MLX embeddings with: uv sync --extra mlx-embeddings --extra serve") from exc
        if device != "cpu" and not mx.metal.is_available():
            raise ValueError("MLX Metal is unavailable; use device=cpu")
        path = Path(source).expanduser()
        if path.is_file():
            raise ValueError("MLX embeddings require a complete model directory or repository ID")
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(repo_id=source, revision=revision, local_files_only=local_files_only,
                                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.jinja"]))
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        # These architectures expose one pooled vector of hidden_size. Gemma2's
        # projection head must not be replaced by generic mean pooling.
        if config.get("model_type") not in {"qwen3", "bert", "xlm-roberta", "xlm_roberta", "modernbert"}:
            raise ValueError(f"unsupported MLX embedding architecture {config.get('model_type')!r}; use HF or GGUF for EmbeddingGemma2")
        prompts_path = path / "config_sentence_transformers.json"
        prompts = json.loads(prompts_path.read_text(encoding="utf-8")).get("prompts", {}) if prompts_path.is_file() else {}
        # Lazy constants retain this stream; the backend lock guards all calls.
        stream = mx.new_thread_unsafe_stream(mx.cpu if device == "cpu" else mx.gpu)
        with mx.stream(stream):
            model, tokenizer = load(str(path), lazy=False,
                                    tokenizer_config={"local_files_only": True, "trust_remote_code": False})
            mx.synchronize(stream)
        return cls(model, tokenizer, config, prompts, mx, stream)

    def embed(self, texts, max_input_tokens):
        with self._lock:
            return self._embed(texts, max_input_tokens)

    def _embed(self, texts, max_input_tokens):
        ids = [self.tokenizer.encode(text, add_special_tokens=True) for text in texts]
        _check_tokens([len(tokens) for tokens in ids], min(max_input_tokens, self.context_window))
        mx = self._mx
        vectors = []
        try:
            with mx.stream(self._stream):
                # One item at a time bounds memory and avoids padding ambiguity.
                for tokens in ids:
                    inputs = mx.array([tokens], dtype=mx.int32)
                    output = self.model(inputs, attention_mask=mx.ones(inputs.shape, dtype=mx.int32))
                    embedding = output.text_embeds.astype(mx.float32)
                    mx.eval(embedding)
                    vectors.append(embedding.tolist()[0])
            return vectors, sum(map(len, ids))
        finally:
            mx.synchronize(self._stream)


def load_embedding_backend(source, *, backend="auto", revision=None, device="auto", dtype="auto",
                           local_files_only=False, context_length=8192, gpu_layers=None,
                           flash_attn=False, pooling="auto", gguf_server_path=None):
    backend = model_backend(source, backend)
    if backend == "gguf":
        if revision is not None or dtype != "auto":
            raise ValueError("GGUF embeddings do not accept revision or dtype")
        return GGUFEmbeddingBackend.load(source, device=device, context_length=context_length,
                                         gpu_layers=gpu_layers, flash_attn=flash_attn, pooling=pooling,
                                         gguf_server_path=gguf_server_path)
    if gpu_layers is not None or flash_attn or context_length != 8192 or pooling != "auto" or gguf_server_path is not None:
        raise ValueError("context_length, gpu_layers, flash_attn, pooling, and gguf_server_path are GGUF-only embedding options")
    loader = MLXEmbeddingBackend if backend == "mlx" else HFEmbeddingBackend
    return loader.load(source, revision=revision, device=device, dtype=dtype, local_files_only=local_files_only)

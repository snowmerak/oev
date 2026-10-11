"""MLX-LM inference on Apple Silicon with request-local prompt caches."""

from __future__ import annotations

from pathlib import Path
import platform
from threading import Lock


class MLXBackend:
    """Share loaded weights; serialize model calls and isolate each request's cache."""

    def __init__(self, model, tokenizer, model_name, config, mx, stream, make_cache, make_sampler):
        # Keep this distinct from the Transformers model/config interface.
        self.mlx_model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        text_config = config.get("text_config") or config
        context = text_config.get("max_position_embeddings")
        self.context_window = context if type(context) is int and context > 0 else None
        self._eos = set(getattr(tokenizer, "eos_token_ids", ()) or ())
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            self._eos.add(eos)
        self._mx = mx
        self._stream = stream
        self._make_cache = make_cache
        self._make_sampler = make_sampler
        self._lock = Lock()

    @classmethod
    def load(cls, source, *, revision=None, device="auto", dtype="auto", local_files_only=False):
        if device not in {"auto", "cpu", "mps"}:
            raise ValueError("MLX device must be auto, cpu, or mps (Metal)")
        if dtype != "auto":
            raise ValueError("MLX requires dtype=auto; use the checkpoint's stored dtype/quantization")
        if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
            raise RuntimeError("the MLX backend requires Apple Silicon macOS; use HF or GGUF on this platform")
        try:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.models.cache import make_prompt_cache
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise RuntimeError("Install MLX support on your Mac with: uv sync --extra mlx --extra serve") from exc
        if device != "cpu" and not mx.metal.is_available():
            raise ValueError("MLX Metal is unavailable; use device=cpu or enable Metal on Apple Silicon")
        # Lazy model constants can retain the loading stream. HTTP inference runs
        # on other threads, so share one stream under the model lock and sync
        # before releasing it; thread-local streams cannot evaluate those graphs.
        stream = mx.new_thread_unsafe_stream(mx.cpu if device == "cpu" else mx.gpu)
        path = Path(source).expanduser()
        if path.is_file():
            raise ValueError("MLX requires a model directory or HF repository ID, not a single weight file")
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS

            # Resolve the snapshot explicitly so offline mode also covers model weights.
            path = Path(snapshot_download(
                repo_id=source, revision=revision, local_files_only=local_files_only,
                allow_patterns=DEFAULT_ALLOW_PATTERNS,
            ))
        with mx.stream(stream):
            model, tokenizer, config = load(
                str(path), lazy=False, return_config=True, trust_remote_code=False,
                tokenizer_config={"local_files_only": True, "trust_remote_code": False},
            )
            mx.synchronize(stream)
        if not getattr(tokenizer, "has_chat_template", getattr(tokenizer, "chat_template", None)):
            raise ValueError("MLX checkpoint needs a tokenizer chat template")
        name = f"{source}@{revision}" if revision else source
        return cls(model, tokenizer, name, config, mx, stream, make_prompt_cache, make_sampler)

    def is_end_token(self, token):
        return token in self._eos

    def _validate_context(self, input_ids, output_tokens=0):
        if not input_ids:
            raise ValueError("prompt must contain at least one token")
        if self.context_window is not None and len(input_ids) + output_tokens > self.context_window:
            raise ValueError(f"prompt plus max output tokens exceeds MLX context limit {self.context_window}")

    def _prefill(self, input_ids, cache, cancel=None):
        """Evaluate bounded prompt chunks, keeping only the final token's logits."""
        mx = self._mx
        # Leave the last token for the readout, avoiding a full prompt vocabulary projection.
        for start in range(0, len(input_ids) - 1, 512):
            if cancel is not None and cancel.is_set():
                return None
            chunk = input_ids[start:min(start + 512, len(input_ids) - 1)]
            self.mlx_model(mx.array([chunk], dtype=mx.uint32), cache=cache)
            mx.eval([entry.state for entry in cache])
        if cancel is not None and cancel.is_set():
            return None
        return self.mlx_model(mx.array([[input_ids[-1]]], dtype=mx.uint32), cache=cache)[:, -1, :]

    def selected_logits(self, input_ids, answer_token_ids):
        self._validate_context(input_ids)
        mx = self._mx
        with self._lock:
            try:
                with mx.stream(self._stream):
                    cache = self._make_cache(self.mlx_model)
                    logits = self._prefill(input_ids, cache)
                    selected = logits[0, mx.array(answer_token_ids, dtype=mx.uint32)].astype(mx.float32)
                    mx.eval(selected)
                    return selected.tolist()
            finally:
                mx.synchronize(self._stream)

    def generate_tokens(self, input_ids, options, cancel):
        self._validate_context(input_ids, options.max_new_tokens)
        while not cancel.is_set():
            if self._lock.acquire(timeout=0.1):
                break
        else:
            return
        mx = self._mx
        cache = None
        try:
            with mx.stream(self._stream):
                cache = self._make_cache(self.mlx_model)
                logits = self._prefill(input_ids, cache, cancel)
                # Temperature is applied before top-p, matching the other backends.
                sampler = self._make_sampler(temp=1.0, top_p=options.top_p)
            if logits is None:
                return
            for index in range(options.max_new_tokens):
                if cancel.is_set():
                    return
                with mx.stream(self._stream):
                    logits = logits.astype(mx.float32)
                    if not mx.all(mx.isfinite(logits)).item():
                        raise RuntimeError("model returned nonfinite generation logits")
                    if options.temperature == 0:
                        sampled = mx.argmax(logits, axis=-1)
                    else:
                        scaled = logits / options.temperature
                        logprobs = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
                        sampled = sampler(logprobs)
                    mx.eval(sampled)
                    token = int(sampled.item())
                # Do not retain thread-local stream context across a generator yield.
                yield token
                if self.is_end_token(token) or cancel.is_set():
                    return
                if index + 1 < options.max_new_tokens:
                    with mx.stream(self._stream):
                        logits = self.mlx_model(mx.array([[token]], dtype=mx.uint32), cache=cache)[:, -1, :]
        finally:
            try:
                mx.synchronize(self._stream)
            finally:
                cache = None
                self._lock.release()

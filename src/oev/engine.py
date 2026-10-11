"""Model-independent decision API."""

from __future__ import annotations

from typing import Protocol
from pathlib import Path
import json
import time

from .prompt import Tokenizer, prepare
from .types import Decision, DecisionResult, conditional_softmax


def model_backend(source: str, backend: str = "auto") -> str:
    """Resolve explicit backends and unambiguous GGUF/MLX source formats."""
    if backend not in {"auto", "hf", "gguf", "mlx"}:
        raise ValueError("backend must be auto, hf, gguf, or mlx")
    if backend != "auto":
        return backend
    path = Path(source).expanduser()
    if path.suffix.lower() == ".gguf":
        return "gguf"
    if source.startswith("mlx-community/"):
        return "mlx"
    config_path = path / "config.json"
    if config_path.is_file():
        quantization = json.loads(config_path.read_text(encoding="utf-8")).get("quantization")
        if isinstance(quantization, dict) and "bits" in quantization and "group_size" in quantization:
            return "mlx"
    return "hf"


class LogitBackend(Protocol):
    tokenizer: Tokenizer
    model_name: str

    def selected_logits(self, input_ids: list[int], answer_token_ids: list[int]) -> list[float]: ...


class DecisionEngine:
    def __init__(self, backend: LogitBackend, *, max_input_tokens: int = 4096):
        self.backend = backend
        self.max_input_tokens = max_input_tokens

    @classmethod
    def from_pretrained(
        cls,
        model: str = "google/gemma-4-E2B-it",
        *,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        max_input_tokens: int = 4096,
        backend: str = "auto",
        local_files_only: bool = False,
        context_length: int = 8192,
        gpu_layers: int | None = None,
        chat_template: str | None = None,
        flash_attn: bool = False,
    ) -> "DecisionEngine":
        if type(flash_attn) is not bool:
            raise ValueError("flash_attn must be a boolean")
        source = str(Path(model).expanduser()) if Path(model).expanduser().exists() or model.startswith("~") else model
        selected_backend = model_backend(source, backend)
        if selected_backend == "gguf":
            from .gguf import GGUFBackend

            if revision is not None or dtype != "auto":
                raise ValueError("GGUF files do not accept revision or dtype; quantization is stored in the file")
            if max_input_tokens > context_length:
                raise ValueError("max_input_tokens must not exceed GGUF context_length")
            loaded = GGUFBackend.load(source, device=device, context_length=context_length,
                                      gpu_layers=gpu_layers, chat_template=chat_template, flash_attn=flash_attn)
            return cls(loaded, max_input_tokens=max_input_tokens)
        if flash_attn:
            raise ValueError("flash_attn is a GGUF-only option")
        if gpu_layers is not None or chat_template is not None or context_length != 8192:
            raise ValueError("context_length, gpu_layers, and chat_template are GGUF-only options")
        if selected_backend == "mlx":
            from .mlx import MLXBackend

            loaded = MLXBackend.load(source, revision=revision, device=device, dtype=dtype,
                                     local_files_only=local_files_only)
            return cls(loaded, max_input_tokens=max_input_tokens)
        path = Path(source)
        if path.is_file():
            raise ValueError("the HF backend requires a model directory; use backend=gguf for .gguf files")
        if path.is_dir():
            # Local folders should never fall back to fetching missing files.
            local_files_only = True
            if not (path / "config.json").is_file() and any(path.glob("*.gguf")):
                raise ValueError("pass a specific .gguf file, not its download directory")
            config_path = path / "config.json"
            if config_path.is_file():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                quantization = config.get("quantization")
                if isinstance(quantization, dict) and "bits" in quantization and "group_size" in quantization:
                    raise ValueError("MLX-quantized checkpoints require backend=mlx instead of hf")
        offline = {"local_files_only": True} if local_files_only else {}
        if device == "hybrid-cuda":
            from .hybrid import HybridGemma4Backend

            if dtype not in {"auto", "bfloat16"}:
                raise ValueError("hybrid-cuda requires bfloat16")
            backend = HybridGemma4Backend.load(source, revision=revision, **offline)
        else:
            from .hf import HuggingFaceBackend

            backend = HuggingFaceBackend.load(source, revision=revision, device=device, dtype=dtype, **offline)

        return cls(backend, max_input_tokens=max_input_tokens)

    def decide(self, decision: Decision) -> DecisionResult:
        started = time.perf_counter()
        ids, slots, prompt_hash = prepare(self.backend.tokenizer, decision, self.max_input_tokens)
        scores = self.backend.selected_logits(ids, slots)
        if len(scores) != len(decision.options):
            raise RuntimeError("backend returned the wrong number of logits")
        probabilities = conditional_softmax(scores)
        option_ids = [option.id for option in decision.options]
        winner = max(range(len(scores)), key=lambda i: scores[i])
        return DecisionResult(
            selected_id=option_ids[winner],
            probabilities=dict(zip(option_ids, probabilities)),
            logits=dict(zip(option_ids, scores)),
            input_tokens=len(ids),
            elapsed_seconds=time.perf_counter() - started,
            prompt_sha256=prompt_hash,
            model=self.backend.model_name,
        )

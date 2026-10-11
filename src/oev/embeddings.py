"""Serve dense text embeddings from one resident, serialized model."""

from __future__ import annotations

from datetime import date
import math
from threading import Lock


class EmbeddingService:
    capabilities = frozenset({"embedding"})

    def __init__(self, backend, model, *, max_input_tokens=4096, max_concurrency=4,
                 query_prefix=None, document_prefix=None):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise ValueError("max_input_tokens must be a positive integer")
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        if type(backend.dimension) is not int or backend.dimension < 1:
            raise ValueError("embedding backend must declare a positive output dimension")
        self.backend, self.model = backend, model
        self.max_input_tokens, self.max_concurrency = max_input_tokens, max_concurrency
        self.prompts = dict(backend.prompts)
        for name, prefix in [("query", query_prefix), ("document", document_prefix)]:
            if prefix is not None:
                if not isinstance(prefix, str):
                    raise ValueError("embedding prefixes must be strings")
                self.prompts[name] = prefix
        # Tokenizers and native contexts are mutable. Keep validation and inference
        # together; separate registered models have independent locks.
        self._lock = Lock()

    @classmethod
    def from_pretrained(cls, source, model, *, backend="auto", revision=None,
                        device="auto", dtype="auto", local_files_only=False,
                        max_input_tokens=4096, max_concurrency=4, context_length=8192,
                        gpu_layers=None, flash_attn=False, pooling="auto",
                        query_prefix=None, document_prefix=None, gguf_server_path=None):
        from .embedding_backends import load_embedding_backend

        loaded = load_embedding_backend(
            source, backend=backend, revision=revision, device=device, dtype=dtype,
            local_files_only=local_files_only, context_length=context_length,
            gpu_layers=gpu_layers, flash_attn=flash_attn, pooling=pooling, gguf_server_path=gguf_server_path,
        )
        return cls(loaded, model, max_input_tokens=max_input_tokens,
                   max_concurrency=max_concurrency, query_prefix=query_prefix, document_prefix=document_prefix)

    def models(self):
        return {"models": [{"name": self.model,
                            "description": f"oev text embeddings; {self.backend.dimension} dimensions.",
                            "release_date": date.today().isoformat()}]}

    def close(self):
        with self._lock:
            close = getattr(self.backend, "close", None)
            if close is not None:
                close()

    def embed(self, texts, *, dimensions=None, input_type=None):
        if not isinstance(texts, list) or not 1 <= len(texts) <= 128 or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise ValueError("input must contain 1 to 128 nonempty text strings")
        dimension = self.backend.dimension
        if dimensions is not None:
            if type(dimensions) is not int or not 1 <= dimensions <= dimension:
                raise ValueError(f"dimensions must be between 1 and {dimension}")
            dimension = dimensions
        if input_type is not None:
            if input_type not in {"query", "document"}:
                raise ValueError("input_type must be query or document")
            if input_type not in self.prompts:
                raise ValueError(f"no {input_type} prefix is configured for this embedding model")
        prefix = self.prompts[input_type] if input_type is not None else ""
        with self._lock:
            vectors, tokens = self.backend.embed([prefix + text for text in texts], self.max_input_tokens)
        if len(vectors) != len(texts) or type(tokens) is not int or tokens < len(texts):
            raise RuntimeError("embedding backend returned invalid output or usage")
        normalized = []
        for vector in vectors:
            if len(vector) != self.backend.dimension:
                raise RuntimeError("embedding backend returned an invalid dimension")
            if any(not math.isfinite(value) for value in vector):
                raise RuntimeError("embedding backend returned nonfinite values")
            vector = [float(value) for value in vector[:dimension]]
            norm = math.hypot(*vector)
            if not math.isfinite(norm) or norm == 0:
                raise RuntimeError("embedding backend returned a zero or invalid vector")
            # MRL vectors must be normalized AFTER truncation, for every backend.
            normalized.append([value / norm for value in vector])
        return normalized, tokens

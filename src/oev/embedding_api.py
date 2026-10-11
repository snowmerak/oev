"""OpenAI-compatible dense text embeddings, including float32 base64 output."""

import base64
import logging
import struct
from typing import Literal

from fastapi import FastAPI
from pydantic import Field

from .chat import APIError, APIModel
from .models import ModelCapabilityError, ModelNotFound, ModelRegistry


logger = logging.getLogger(__name__)


class EmbeddingRequest(APIModel):
    model: str
    input: str | list[str] = Field(union_mode="left_to_right")
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = Field(default=None, ge=1, strict=True)
    user: str | None = None
    input_type: Literal["query", "document"] | None = None


def add_embedding_routes(app: FastAPI, registry: ModelRegistry):
    @app.post("/v1/embeddings")
    def embeddings(body: EmbeddingRequest):
        try:
            service = registry.resolve(body.model, operation="embedding")
        except ModelNotFound as exc:
            raise APIError(str(exc), status=404, code="model_not_found", param="model") from exc
        except ModelCapabilityError as exc:
            raise APIError(str(exc), code="unsupported_operation", param="model") from exc
        texts = [body.input] if isinstance(body.input, str) else body.input
        try:
            vectors, tokens = service.embed(texts, dimensions=body.dimensions, input_type=body.input_type)
        except ValueError as exc:
            raise APIError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Embedding inference failed")
            raise APIError("Embedding inference failed", status=500, code="embedding_error") from exc
        if body.encoding_format == "base64":
            vectors = [base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii") for vector in vectors]
        return {"object": "list", "model": body.model,
                "data": [{"object": "embedding", "index": index, "embedding": vector}
                         for index, vector in enumerate(vectors)],
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}

"""Route requests to models that remain loaded for the server's lifetime."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .systemone import SystemOneService
from .embeddings import EmbeddingService


class ModelNotFound(ValueError):
    """The requested served model name is not registered."""


class ModelCapabilityError(ValueError):
    """The selected model cannot perform the requested operation."""


class ModelRegistry:
    """Keep one service per served name, with independent inference limits."""

    def __init__(self, services: Iterable[SystemOneService | EmbeddingService]) -> None:
        self._services: dict[str, SystemOneService | EmbeddingService] = {}
        for service in services:
            if service.model in self._services:
                raise ValueError(f"duplicate served model name: {service.model!r}")
            self._services[service.model] = service
        if not self._services:
            raise ValueError("at least one model service is required")

    def resolve(self, model: str, *, operation: str | None = None) -> SystemOneService | EmbeddingService:
        try:
            service = self._services[model]
        except KeyError as exc:
            raise ModelNotFound(f"unknown model: {model!r}") from exc
        if operation is not None and operation not in service.capabilities:
            raise ModelCapabilityError(f"model {model!r} does not support {operation}")
        return service

    def models(self) -> dict[str, Any]:
        return {
            "models": [
                metadata
                for service in self._services.values()
                for metadata in service.models()["models"]
            ],
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": "oev",
                 **(service.model_metadata() if isinstance(service, SystemOneService) else {"task": "embedding"})}
                for name, service in self._services.items()
            ],
        }

    def close(self):
        for service in self._services.values():
            if isinstance(service, EmbeddingService):
                service.close()

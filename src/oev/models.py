"""Route requests to models that remain loaded for the server's lifetime."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .systemone import SystemOneService


class ModelNotFound(ValueError):
    """The requested served model name is not registered."""


class ModelRegistry:
    """Keep one service per served name, with independent inference limits."""

    def __init__(self, services: Iterable[SystemOneService]) -> None:
        self._services: dict[str, SystemOneService] = {}
        for service in services:
            if service.model in self._services:
                raise ValueError(f"duplicate served model name: {service.model!r}")
            self._services[service.model] = service
        if not self._services:
            raise ValueError("at least one model service is required")

    def resolve(self, model: str) -> SystemOneService:
        try:
            return self._services[model]
        except KeyError as exc:
            raise ModelNotFound(f"unknown model: {model!r}") from exc

    def models(self) -> dict[str, Any]:
        return {
            "models": [
                metadata
                for service in self._services.values()
                for metadata in service.models()["models"]
            ],
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": "oev"}
                for name in self._services
            ],
        }

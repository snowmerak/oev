"""Serve typed decisions and OpenAI-compatible text generation with local models."""

from __future__ import annotations

import argparse
from importlib.resources import files
import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .chat import add_chat_routes
from .engine import DecisionEngine
from .models import ModelRegistry
from .systemone import SystemOneService
from .types import MAX_OPTIONS


Content = str | dict[str, Any] | list[Any] | None
Level = str | dict[str, Any] | list[Any]
SKILL_MARKDOWN = files("oev").joinpath("systemone.skill.md").read_text(encoding="utf-8")


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Content = None
    criteria: dict[str, Content] = Field(min_length=1, max_length=MAX_OPTIONS)


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Content = None
    criteria: dict[str, Content] | None = None


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Content = None
    criteria: list[Level] = Field(min_length=1, max_length=MAX_OPTIONS)


Question = Annotated[ChoiceQuestion | NoulQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    state: str | dict[str, Any] | list[Any]
    model: str
    questions: dict[str, Question] = Field(min_length=1)


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    confidence: float
    legend: dict[str, Level]
    probabilities: dict[str, float]


Answer = Annotated[ChoiceAnswer | NoulAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelMetadata(BaseModel):
    name: str
    description: str
    release_date: str


class ModelMetadataList(BaseModel):
    models: list[ModelMetadata]
    object: Literal["list"] = "list"
    data: list[dict[str, Any]]


def create_app(service: SystemOneService | ModelRegistry) -> FastAPI:
    registry = service if isinstance(service, ModelRegistry) else ModelRegistry([service])
    app = FastAPI(title="oev decisions and chat", version="0.1.0")
    add_chat_routes(app, registry)

    @app.get("/.skill", response_class=PlainTextResponse, include_in_schema=False)
    def skill() -> PlainTextResponse:
        return PlainTextResponse(SKILL_MARKDOWN, media_type="text/markdown")

    @app.get("/v1/models", response_model=ModelMetadataList)
    def models() -> dict[str, Any]:
        return registry.models()

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    def systemone(request: SystemOneRequest) -> dict[str, Any]:
        try:
            return registry.resolve(request.model).evaluate(request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


class ModelConfig(BaseModel):
    """One eager model load; omitted options inherit the server's CLI defaults."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    served_model: str | None = Field(default=None, min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    device: Literal["auto", "cpu", "cuda", "mps", "hybrid-cuda"] = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    max_input_tokens: int = Field(default=4096, ge=1, strict=True)
    max_concurrency: int = Field(default=4, ge=1, strict=True)
    backend: Literal["auto", "hf", "gguf"] = "auto"
    local_files_only: bool = Field(default=False, strict=True)
    context_length: int = Field(default=8192, ge=1, strict=True)
    gpu_layers: int | None = Field(default=None, ge=-1, strict=True)
    chat_template: str | None = None

    @field_validator("model", "served_model", "revision")
    @classmethod
    def nonempty_name(cls, value):
        if value is not None and not value.strip():
            raise ValueError("model names and revisions must be nonempty")
        return value

    @property
    def name(self) -> str:
        return self.served_model or self.model


def model_configs(args, parser) -> list[ModelConfig]:
    defaults = {
        "revision": args.revision, "device": args.device, "dtype": args.dtype,
        "max_input_tokens": args.max_input_tokens, "max_concurrency": args.max_concurrency,
        "backend": args.backend, "local_files_only": args.local_files_only,
        "context_length": args.context_length, "gpu_layers": args.gpu_layers, "chat_template": args.chat_template,
    }
    if args.models_config:
        if args.model or args.served_model:
            parser.error("--models-config cannot be combined with --model or --served-model")
        try:
            entries = json.loads(Path(args.models_config).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            parser.error(f"cannot read models config: {exc}")
        if not isinstance(entries, list) or not entries or any(not isinstance(item, dict) for item in entries):
            parser.error("models config must be a nonempty JSON array of model objects")
    else:
        models = args.model or ["google/gemma-4-E2B-it"]
        names = args.served_model or models
        if len(names) != len(models):
            parser.error("provide one --served-model per --model, in the same order")
        entries = [{"model": model, "served_model": name} for model, name in zip(models, names)]
    try:
        configs = [ModelConfig.model_validate({**defaults, **entry}) for entry in entries]
    except ValidationError as exc:
        parser.error(f"invalid model configuration: {exc}")
    names = [config.name for config in configs]
    if len(set(names)) != len(names):
        parser.error("served model names must be unique")
    for config in configs:
        gguf = config.backend == "gguf" or (config.backend == "auto" and Path(config.model).suffix.lower() == ".gguf")
        if gguf:
            if config.dtype != "auto" or config.revision is not None or config.device == "hybrid-cuda":
                parser.error(f"model {config.name!r}: GGUF does not accept dtype, revision, or hybrid-cuda")
            if config.max_input_tokens > config.context_length:
                parser.error(f"model {config.name!r}: max_input_tokens exceeds GGUF context_length")
            if config.device == "cpu" and config.gpu_layers not in {None, 0}:
                parser.error(f"model {config.name!r}: device=cpu requires gpu_layers=0")
            if config.device in {"cuda", "mps"} and config.gpu_layers == 0:
                parser.error(f"model {config.name!r}: a GPU device requires nonzero gpu_layers")
        elif config.gpu_layers is not None or config.chat_template is not None or config.context_length != 8192:
            parser.error(f"model {config.name!r}: context_length, gpu_layers, and chat_template are GGUF-only")
        if config.device == "hybrid-cuda" and config.dtype not in {"auto", "bfloat16"}:
            parser.error(f"model {config.name!r}: hybrid-cuda requires bfloat16")
    return configs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="Hugging Face model ID or local path; repeat to load multiple models")
    parser.add_argument("--served-model", action="append", help="API model name; repeat once per --model, in the same order")
    parser.add_argument("--models-config", help="JSON array with per-model loading options")
    parser.add_argument("--revision", help="Hugging Face commit revision")
    parser.add_argument("--backend", choices=["auto", "hf", "gguf"], default="auto", help="Auto detects .gguf file paths; other sources use HF")
    parser.add_argument("--local-files-only", action="store_true", help="HF: use downloaded files/cache without network access")
    parser.add_argument("--context-length", type=int, default=8192, help="GGUF context size, including generated tokens")
    parser.add_argument("--gpu-layers", type=int, help="GGUF layers to offload; -1 for all; auto/cpu defaults to 0")
    parser.add_argument("--chat-template", help="GGUF: override embedded chat template with a Jinja file")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps", "hybrid-cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--host", default=os.environ.get("OEV_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=os.environ.get("OEV_PORT", "8000"))
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=os.environ.get("OEV_MAX_CONCURRENCY", "4"),
        help="Maximum simultaneous inferences per model (default: OEV_MAX_CONCURRENCY or 4)",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.max_concurrency < 1:
        parser.error("max-concurrency must be positive")
    if args.max_input_tokens < 1:
        parser.error("max-input-tokens must be positive")
    configs = model_configs(args, parser)
    services = []
    # Load every configured model before binding HTTP; no partial model catalog is served.
    for config in configs:
        engine = DecisionEngine.from_pretrained(
            config.model, revision=config.revision, device=config.device, dtype=config.dtype,
            max_input_tokens=config.max_input_tokens,
            backend=config.backend, local_files_only=config.local_files_only,
            context_length=config.context_length, gpu_layers=config.gpu_layers, chat_template=config.chat_template,
        )
        services.append(SystemOneService(engine, config.name, max_concurrency=config.max_concurrency))
    service = services[0] if len(services) == 1 else ModelRegistry(services)

    import uvicorn

    uvicorn.run(create_app(service), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

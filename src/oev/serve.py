"""Serve typed decisions and OpenAI-compatible text generation with local models."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from importlib.resources import files
import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .api_requests import SamplingParameters
from .chat import add_chat_routes
from .embedding_api import add_embedding_routes
from .embeddings import EmbeddingService
from .engine import DecisionEngine, model_backend
from .models import ModelRegistry
from .systemone import SystemOneService
from .types import MAX_OPTIONS
from .prompt_cache import parse_cache_size


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
    input_tokens_details: dict[str, int] | None = None


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


def create_app(service: SystemOneService | EmbeddingService | ModelRegistry) -> FastAPI:
    registry = service if isinstance(service, ModelRegistry) else ModelRegistry([service])

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            registry.close()

    app = FastAPI(title="oev decisions, chat, and embeddings", version="0.1.0", lifespan=lifespan)
    add_chat_routes(app, registry)
    add_embedding_routes(app, registry)

    @app.get("/.skill", response_class=PlainTextResponse, include_in_schema=False)
    def skill() -> PlainTextResponse:
        return PlainTextResponse(SKILL_MARKDOWN, media_type="text/markdown")

    @app.get("/v1/models", response_model=ModelMetadataList)
    def models() -> dict[str, Any]:
        return registry.models()

    @app.post("/v1/systemone", response_model=SystemOneResponse, response_model_exclude_none=True)
    def systemone(request: SystemOneRequest) -> dict[str, Any]:
        try:
            return registry.resolve(request.model, operation="decision").evaluate(request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


class ModelConfig(BaseModel):
    """One eager model load; omitted options inherit the server's CLI defaults."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    served_model: str | None = Field(default=None, min_length=1)
    task: Literal["generation", "embedding"] = "generation"
    revision: str | None = Field(default=None, min_length=1)
    device: Literal["auto", "cpu", "cuda", "mps", "hybrid-cuda"] = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    max_input_tokens: int = Field(default=4096, ge=1, strict=True)
    max_concurrency: int = Field(default=4, ge=1, strict=True)
    default_max_output_tokens: int = Field(default=8192, ge=1, le=32768, strict=True)
    generation_defaults: SamplingParameters = Field(default_factory=SamplingParameters)
    backend: Literal["auto", "hf", "gguf", "mlx"] = "auto"
    local_files_only: bool = Field(default=False, strict=True)
    context_length: int = Field(default=8192, ge=1, strict=True)
    gpu_layers: int | None = Field(default=None, ge=-1, strict=True)
    chat_template: str | None = None
    tool_call_parser: str = "auto"
    reasoning_parser: str = "auto"
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    flash_attn: bool = Field(default=False, strict=True)
    prompt_cache_max_size: int = Field(default=0, ge=0, strict=True,
                                       validation_alias=AliasChoices("prompt_cache_max_size", "prompt_cache_max_bytes"))
    pooling: Literal["auto", "mean", "last", "cls"] = "auto"
    embedding_query_prefix: str | None = None
    embedding_document_prefix: str | None = None
    gguf_server_path: str | None = None
    mmproj_path: str | None = None

    @field_validator("tool_call_parser", "reasoning_parser")
    @classmethod
    def output_parser(cls, value, info):
        from .output_parsers import parser_name
        parser_name(value, reasoning=info.field_name == "reasoning_parser")
        return value

    @field_validator("chat_template_kwargs")
    @classmethod
    def template_kwargs(cls, value):
        from .generation import validate_template_kwargs
        validate_template_kwargs(value)
        return value

    @model_validator(mode="before")
    @classmethod
    def legacy_cache_bytes(cls, values):
        if isinstance(values, dict) and "prompt_cache_max_bytes" in values:
            if type(values["prompt_cache_max_bytes"]) is not int:
                raise ValueError("prompt_cache_max_bytes must be a nonnegative integer; use prompt_cache_max_size for units")
        return values

    @field_validator("prompt_cache_max_size", mode="before", json_schema_input_type=int | str)
    @classmethod
    def cache_size(cls, value):
        return parse_cache_size(value)

    @property
    def prompt_cache_max_bytes(self) -> int:
        """Compatibility accessor; stored sizes are always normalized to bytes."""
        return self.prompt_cache_max_size

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
        "task": args.task,
        "revision": args.revision, "device": args.device, "dtype": args.dtype,
        "max_input_tokens": args.max_input_tokens, "max_concurrency": args.max_concurrency,
        "default_max_output_tokens": getattr(args, "default_max_output_tokens", 8192),
        "backend": args.backend, "local_files_only": args.local_files_only,
        "context_length": args.context_length, "gpu_layers": args.gpu_layers, "chat_template": args.chat_template,
        "flash_attn": args.flash_attn,
        "tool_call_parser": getattr(args, "tool_call_parser", "auto"),
        "reasoning_parser": getattr(args, "reasoning_parser", "auto"),
        "chat_template_kwargs": getattr(args, "chat_template_kwargs", {}),
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
        configs = []
        for entry in entries:
            values = {**defaults, **entry}
            # An explicit legacy entry must override the CLI default too.
            if not {"prompt_cache_max_size", "prompt_cache_max_bytes"}.intersection(entry):
                values["prompt_cache_max_size"] = args.prompt_cache_max_size
            configs.append(ModelConfig.model_validate(values))
    except ValidationError as exc:
        parser.error(f"invalid model configuration: {exc}")
    names = [config.name for config in configs]
    if len(set(names)) != len(names):
        parser.error("served model names must be unique")
    for config in configs:
        try:
            backend = model_backend(config.model, config.backend)
        except ValueError as exc:
            parser.error(f"model {config.name!r}: {exc}")
        gguf = backend == "gguf"
        if config.prompt_cache_max_size and (config.task == "embedding" or config.device == "hybrid-cuda"):
            parser.error(f"model {config.name!r}: prompt caching requires generation on HF/GGUF/MLX; hybrid-cuda is unsupported")
        if config.gguf_server_path is not None and (not gguf or config.task != "embedding"):
            parser.error(f"model {config.name!r}: gguf_server_path requires a GGUF embedding model")
        if config.mmproj_path is not None and (not gguf or config.task != "embedding" or not config.gguf_server_path):
            parser.error(f"model {config.name!r}: mmproj_path requires a GGUF embedding model with gguf_server_path")
        if config.task == "embedding" and config.chat_template is not None:
            parser.error(f"model {config.name!r}: embedding models do not use chat_template")
        if config.task == "embedding" and (config.tool_call_parser != "auto" or config.reasoning_parser != "auto"
                                            or config.chat_template_kwargs or config.generation_defaults.model_fields_set):
            parser.error(f"model {config.name!r}: output parsers, template kwargs and generation defaults require generation")
        if config.task == "generation" and (config.pooling != "auto" or config.embedding_query_prefix is not None
                                            or config.embedding_document_prefix is not None):
            parser.error(f"model {config.name!r}: pooling and embedding prefixes require task=embedding")
        if config.task == "embedding" and config.device == "hybrid-cuda":
            parser.error(f"model {config.name!r}: embeddings do not support hybrid-cuda")
        if config.pooling != "auto" and not gguf:
            parser.error(f"model {config.name!r}: pooling is a GGUF-only option")
        if config.flash_attn and not gguf:
            parser.error(f"model {config.name!r}: flash_attn is a GGUF-only option")
        if gguf:
            if config.dtype != "auto" or config.revision is not None or config.device == "hybrid-cuda":
                parser.error(f"model {config.name!r}: GGUF does not accept dtype, revision, or hybrid-cuda")
            if config.max_input_tokens > config.context_length:
                parser.error(f"model {config.name!r}: max_input_tokens exceeds GGUF context_length")
            if config.device == "cpu" and config.gpu_layers not in {None, 0}:
                parser.error(f"model {config.name!r}: device=cpu requires gpu_layers=0")
            if config.device in {"cuda", "mps"} and config.gpu_layers == 0:
                parser.error(f"model {config.name!r}: a GPU device requires nonzero gpu_layers")
        elif config.gpu_layers is not None or config.context_length != 8192:
            parser.error(f"model {config.name!r}: context_length and gpu_layers are GGUF-only")
        if backend == "mlx" and (config.device not in {"auto", "cpu", "mps"} or config.dtype != "auto"):
            parser.error(f"model {config.name!r}: MLX requires device=auto/cpu/mps and dtype=auto")
        if config.device == "hybrid-cuda" and config.dtype not in {"auto", "bfloat16"}:
            parser.error(f"model {config.name!r}: hybrid-cuda requires bfloat16")
    return configs


def _load_service(config: ModelConfig):
    if config.task == "embedding":
        return EmbeddingService.from_pretrained(
            config.model, config.name, backend=config.backend, revision=config.revision,
            device=config.device, dtype=config.dtype, local_files_only=config.local_files_only,
            max_input_tokens=config.max_input_tokens, max_concurrency=config.max_concurrency,
            context_length=config.context_length, gpu_layers=config.gpu_layers, flash_attn=config.flash_attn,
            pooling=config.pooling, query_prefix=config.embedding_query_prefix,
            document_prefix=config.embedding_document_prefix, gguf_server_path=config.gguf_server_path,
            mmproj_path=config.mmproj_path,
        )
    gguf = model_backend(config.model, config.backend) == "gguf"
    template = Path(config.chat_template).expanduser().read_text(encoding="utf-8") if config.chat_template and not gguf else None
    engine = DecisionEngine.from_pretrained(
        config.model, revision=config.revision, device=config.device, dtype=config.dtype,
        max_input_tokens=config.max_input_tokens,
        backend=config.backend, local_files_only=config.local_files_only,
        context_length=config.context_length, gpu_layers=config.gpu_layers, chat_template=config.chat_template if gguf else None,
        flash_attn=config.flash_attn,
        prompt_cache_max_size=config.prompt_cache_max_size,
    )
    return SystemOneService(
        engine, config.name, max_concurrency=config.max_concurrency,
        tool_call_parser=config.tool_call_parser, reasoning_parser=config.reasoning_parser,
        chat_template_kwargs=config.chat_template_kwargs, chat_template=template,
        default_max_output_tokens=config.default_max_output_tokens,
        generation_defaults=config.generation_defaults.model_dump(exclude_unset=True),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="Hugging Face model ID or local path; repeat to load multiple models")
    parser.add_argument("--served-model", action="append", help="API model name; repeat once per --model, in the same order")
    parser.add_argument("--models-config", help="JSON array with per-model loading options")
    parser.add_argument("--tool-call-parser", default="auto", help="Model output protocol; auto recognizes native tool delimiters")
    parser.add_argument("--reasoning-parser", default="auto", help="Model reasoning protocol, e.g. qwen3, gemma4, deepseek-r1")
    parser.add_argument("--chat-template-kwargs", type=json.loads, default={},
                        help="JSON object of model-specific template options; per-model JSON config can override it")
    parser.add_argument("--default-max-output-tokens", type=int, default=8192,
                        help="Generation budget when the request omits its output limit (default: 8192; capped by remaining context)")
    parser.add_argument("--task", choices=["generation", "embedding"], default="generation",
                        help="Default model task; JSON entries can override it")
    parser.add_argument("--revision", help="Hugging Face commit revision")
    parser.add_argument("--backend", choices=["auto", "hf", "gguf", "mlx"], default="auto", help="Auto detects GGUF paths and MLX quantized folders/mlx-community IDs; otherwise HF")
    parser.add_argument("--local-files-only", action="store_true", help="HF/MLX: use downloaded files/cache without network access")
    parser.add_argument("--context-length", type=int, default=8192, help="GGUF context size, including generated tokens")
    parser.add_argument("--gpu-layers", type=int, help="GGUF layers to offload; -1 for all; auto/cpu defaults to 0")
    parser.add_argument("--chat-template", help="Override generation chat template with a Jinja file (HF/GGUF/MLX)")
    parser.add_argument("--flash-attn", action=argparse.BooleanOptionalAction, default=False,
                        help="GGUF: request native Flash Attention (default: disabled)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps", "hybrid-cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--prompt-cache-max-size", "--prompt-cache-max-bytes", type=parse_cache_size, default=0,
                        help="Per-model retained prompt KV cache budget, e.g. 256MB, 256MiB, 1GB, 1GiB; 0 disables caching")
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
    try:
        # Load everything before binding HTTP; never expose a partial catalog.
        for config in configs:
            services.append(_load_service(config))
        service = services[0] if len(services) == 1 else ModelRegistry(services)

        import uvicorn

        uvicorn.run(create_app(service), host=args.host, port=args.port)
    finally:
        # A later load failure or interrupted startup must also stop native workers.
        for service in services:
            if isinstance(service, EmbeddingService):
                service.close()


if __name__ == "__main__":
    main()

"""Serve the System One API with a local oev model."""

from __future__ import annotations

import argparse
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .engine import DecisionEngine
from .systemone import SystemOneService
from .types import MAX_OPTIONS


Content = str | dict[str, Any] | list[Any] | None
Level = str | dict[str, Any] | list[Any]


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


def create_app(service: SystemOneService) -> FastAPI:
    app = FastAPI(title="oev System One", version="0.1.0")

    @app.get("/v1/models", response_model=ModelMetadataList)
    def models() -> dict[str, Any]:
        return service.models()

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    def systemone(request: SystemOneRequest) -> dict[str, Any]:
        try:
            return service.evaluate(request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B-it", help="Hugging Face model ID")
    parser.add_argument("--served-model", help="Model name accepted by the System One API")
    parser.add_argument("--revision", help="Hugging Face commit revision")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps", "hybrid-cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    engine = DecisionEngine.from_pretrained(
        args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
    )
    service = SystemOneService(engine, args.served_model or args.model)

    import uvicorn

    uvicorn.run(create_app(service), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

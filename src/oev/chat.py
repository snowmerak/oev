"""OpenAI-compatible text endpoints backed by the shared System One model."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
import json
import logging
from queue import Empty, Full, Queue
from threading import Event, Thread
import time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .generation import GenerationOptions, GenerationResult
from .systemone import SystemOneService


logger = logging.getLogger(__name__)
GENERATION_PATHS = {"/v1/chat/completions", "/v1/responses"}


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TextPart(APIModel):
    type: Literal["text"]
    text: str


class ChatMessage(APIModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[TextPart]

    def message(self):
        text = self.content if isinstance(self.content, str) else "".join(part.text for part in self.content)
        return {"role": self.role, "content": text}


class InputText(APIModel):
    type: Literal["input_text", "output_text"]
    text: str
    annotations: list = Field(default_factory=list, max_length=0)
    logprobs: list = Field(default_factory=list, max_length=0)


class InputMessage(APIModel):
    type: Literal["message"] = "message"
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[InputText]
    id: str | None = None
    status: Literal["completed", "incomplete", "in_progress"] | None = None

    def message(self):
        text = self.content if isinstance(self.content, str) else "".join(part.text for part in self.content)
        return {"role": self.role, "content": text}


class StreamOptions(APIModel):
    include_usage: bool = False


class TextFormat(APIModel):
    type: Literal["text"] = "text"


class ResponseTextConfig(APIModel):
    format: TextFormat = Field(default_factory=TextFormat)


class GenerationRequest(APIModel):
    model: str
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    stream: bool = False
    store: Literal[False] = False


class ChatCompletionRequest(GenerationRequest):
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1, le=4096, strict=True)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=4096, strict=True)
    stop: str | list[str] | None = None
    n: Literal[1] = 1
    stream_options: StreamOptions | None = None
    response_format: TextFormat | None = None

    @model_validator(mode="after")
    def validate_options(self):
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("use only one of max_tokens and max_completion_tokens")
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true")
        self.options()
        return self

    def options(self):
        stops = (self.stop,) if isinstance(self.stop, str) else tuple(self.stop or ())
        return GenerationOptions(
            self.max_completion_tokens or self.max_tokens or 256,
            self.temperature, self.top_p, stops,
        )


class ResponsesRequest(GenerationRequest):
    input: str | list[InputMessage]
    instructions: str | None = None
    max_output_tokens: int = Field(default=256, ge=1, le=4096, strict=True)
    text: ResponseTextConfig = Field(default_factory=ResponseTextConfig)

    def messages(self):
        messages = []
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})
        messages.extend(
            [{"role": "user", "content": self.input}] if isinstance(self.input, str)
            else [item.message() for item in self.input]
        )
        return messages

    def options(self):
        return GenerationOptions(self.max_output_tokens, self.temperature, self.top_p)


class APIError(Exception):
    def __init__(self, message, *, status=400, code="invalid_request", param=None):
        super().__init__(message)
        self.status, self.code, self.param = status, code, param

    def body(self):
        return {"error": {
            "message": str(self), "type": "server_error" if self.status >= 500 else "invalid_request_error",
            "param": self.param, "code": self.code,
        }}


def prepare(service, model, messages, options):
    if model != service.model:
        raise APIError(f"model must be {service.model!r}", status=404, code="model_not_found", param="model")
    try:
        return service.prepare_generation(messages, options)
    except ValueError as exc:
        raise APIError(str(exc)) from exc
    except NotImplementedError as exc:
        raise APIError(str(exc), status=501, code="unsupported_operation") from exc


def generate(service, prepared):
    try:
        return service.generate(prepared)
    except Exception as exc:
        logger.exception("Text generation failed")
        raise APIError("Text generation failed", status=500, code="generation_error") from exc


def chat_usage(result):
    return {
        "prompt_tokens": result.input_tokens, "completion_tokens": result.output_tokens,
        "total_tokens": result.input_tokens + result.output_tokens,
    }


def response_usage(result):
    return {
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
        "total_tokens": result.input_tokens + result.output_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }


def output_message(item_id, text="", status="completed"):
    return {
        "id": item_id, "type": "message", "role": "assistant", "status": status,
        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
    }


def response_body(body, response_id, created, item_id, result=None):
    status = "in_progress" if result is None else "incomplete" if result.finish_reason == "length" else "completed"
    return {
        "id": response_id, "object": "response", "created_at": created,
        "completed_at": None if result is None else int(time.time()),
        "model": body.model, "status": status, "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "output": [] if result is None else [output_message(item_id, result.text, status)],
        "usage": None if result is None else response_usage(result),
        "instructions": body.instructions, "max_output_tokens": body.max_output_tokens,
        "temperature": body.temperature, "top_p": body.top_p, "store": False,
        "tools": [], "tool_choice": "none", "parallel_tool_calls": False,
        "text": body.text.model_dump(), "metadata": {},
    }


def sse(data, event=None):
    prefix = f"event: {event}\n" if event else ""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, allow_nan=False)
    return f"{prefix}data: {payload}\n\n"


async def generation_events(service, prepared, request):
    """Bound queued output and retain the inference slot until the worker exits."""
    cancel = Event()
    queue = Queue(maxsize=32)

    def put(value):
        while not cancel.is_set():
            try:
                queue.put(value, timeout=0.1)
                return
            except Full:
                pass

    def worker():
        source = service.stream_generation(prepared, cancel)
        try:
            for event in source:
                put(event)
        except Exception:
            logger.exception("Streaming text generation failed")
            put(APIError("Text generation failed", status=500, code="generation_error"))
        finally:
            source.close()
            put(None)

    thread = Thread(target=worker, name="oev-generation", daemon=True)
    thread.start()
    try:
        while not await request.is_disconnected():
            try:
                event = queue.get_nowait()
            except Empty:
                await asyncio.sleep(0.02)
                continue
            if event is None:
                return
            yield event
    finally:
        # The worker observes cancellation between forwards and while waiting
        # for a slot or queue space. Never release its slot from the HTTP task.
        cancel.set()


async def chat_stream(service, prepared, body, request, completion_id, created):
    def chunk(delta, reason=None):
        return {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                "model": body.model, "choices": [{"index": 0, "delta": delta,
                                                  "finish_reason": reason, "logprobs": None}]}

    yield sse(chunk({"role": "assistant", "content": ""}))
    async with aclosing(generation_events(service, prepared, request)) as events:
        async for event in events:
            if isinstance(event, APIError):
                yield sse(event.body())
                return
            if isinstance(event, str):
                yield sse(chunk({"content": event}))
            else:
                yield sse(chunk({}, event.finish_reason))
                if body.stream_options and body.stream_options.include_usage:
                    usage = chunk({})
                    usage["choices"] = []
                    usage["usage"] = chat_usage(event)
                    yield sse(usage)
                yield sse("[DONE]")
                return


async def responses_stream(service, prepared, body, request, response_id, created, item_id):
    sequence = 0

    def event(kind, **fields):
        nonlocal sequence
        value = {"type": kind, "sequence_number": sequence, **fields}
        sequence += 1
        return sse(value, kind)

    initial = response_body(body, response_id, created, item_id)
    yield event("response.created", response=initial)
    yield event("response.in_progress", response=initial)
    item = output_message(item_id, status="in_progress")
    item["content"] = []
    yield event("response.output_item.added", output_index=0, item=item)
    part = output_message(item_id)["content"][0]
    positions = {"item_id": item_id, "output_index": 0, "content_index": 0}
    yield event("response.content_part.added", **positions, part=part)
    async with aclosing(generation_events(service, prepared, request)) as events:
        async for value in events:
            if isinstance(value, APIError):
                yield event("error", code=value.code, message=str(value), param=value.param)
                return
            if isinstance(value, str):
                yield event("response.output_text.delta", **positions, delta=value, logprobs=[])
            else:
                final = response_body(body, response_id, created, item_id, value)
                item = final["output"][0]
                yield event("response.output_text.done", **positions, text=value.text, logprobs=[])
                yield event("response.content_part.done", **positions, part=item["content"][0])
                yield event("response.output_item.done", output_index=0, item=item)
                kind = "response.incomplete" if value.finish_reason == "length" else "response.completed"
                yield event(kind, response=final)
                return


def add_chat_routes(app: FastAPI, service: SystemOneService):
    @app.exception_handler(APIError)
    async def api_error(request, exc):
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        if request.url.path not in GENERATION_PATHS:
            return await request_validation_exception_handler(request, exc)
        error = exc.errors()[0]
        param = ".".join(str(part) for part in error["loc"] if part != "body")
        return JSONResponse(APIError(error["msg"], param=param).body(), status_code=400)

    @app.post("/v1/chat/completions")
    def chat_completions(body: ChatCompletionRequest, request: Request):
        prepared = prepare(service, body.model, [item.message() for item in body.messages], body.options())
        completion_id, created = f"chatcmpl-{uuid4().hex}", int(time.time())
        if body.stream:
            return StreamingResponse(
                chat_stream(service, prepared, body, request, completion_id, created),
                media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        result = generate(service, prepared)
        return {
            "id": completion_id, "object": "chat.completion", "created": created, "model": body.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text, "refusal": None},
                         "finish_reason": result.finish_reason, "logprobs": None}],
            "usage": chat_usage(result),
        }

    @app.post("/v1/responses")
    def responses(body: ResponsesRequest, request: Request):
        prepared = prepare(service, body.model, body.messages(), body.options())
        response_id, item_id, created = f"resp_{uuid4().hex}", f"msg_{uuid4().hex}", int(time.time())
        if body.stream:
            return StreamingResponse(
                responses_stream(service, prepared, body, request, response_id, created, item_id),
                media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return response_body(body, response_id, created, item_id, generate(service, prepared))

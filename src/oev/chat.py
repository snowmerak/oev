"""OpenAI-compatible text endpoints backed by the shared System One model."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
import json
import logging
from queue import Empty, Full, Queue
from threading import Event, Thread
import time
from dataclasses import replace
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from .api_requests import APIModel, ChatCompletionRequest, ResponsesRequest
from .completion import parse_completion
from .response_store import ResponseStore

from .generation import GenerationOptions, GenerationResult, ReasoningDelta, buffered_answer
from .models import ModelCapabilityError, ModelNotFound, ModelRegistry


logger = logging.getLogger(__name__)
OPENAI_PATHS = {"/v1/chat/completions", "/v1/responses", "/v1/embeddings"}


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
    usage = {
        "prompt_tokens": result.input_tokens, "completion_tokens": result.output_tokens,
        "total_tokens": result.input_tokens + result.output_tokens,
    }
    if result.cached_input_tokens:
        usage["prompt_tokens_details"] = {"cached_tokens": result.cached_input_tokens}
    if result.reasoning_tokens:
        usage["completion_tokens_details"] = {"reasoning_tokens": result.reasoning_tokens}
    return usage


def response_usage(result):
    return {
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
        "total_tokens": result.input_tokens + result.output_tokens,
        "input_tokens_details": {"cached_tokens": result.cached_input_tokens},
        "output_tokens_details": {"reasoning_tokens": result.reasoning_tokens},
    }


def completion(result, options):
    try:
        return parse_completion(result, options)
    except (ValueError, KeyError, TypeError) as exc:
        raise APIError(f"Invalid model output: {exc}", status=502, code="invalid_model_output") from exc


def buffered(options):
    return buffered_answer(options)


def chat_message(value):
    message = {"role": "assistant", "content": value.text or (None if value.calls else ""), "refusal": None}
    if value.calls:
        message["tool_calls"] = value.calls
    if value.reasoning:
        message["reasoning_content"] = value.reasoning
    return message


def visible_logprobs(result, text):
    records = result.token_logprobs
    data = b"".join(bytes(record["bytes"]) for record in records)
    target = text.encode("utf-8")
    start = data.rfind(target) if target else -1
    if start < 0:
        return []
    end, offset, visible = start + len(target), 0, []
    for record in records:
        stop = offset + len(record["bytes"])
        if offset >= start and stop <= end:
            visible.append(record)
        offset = stop
    return visible


def output_message(item_id, text="", status="completed", logprobs=()):
    return {
        "id": item_id, "type": "message", "role": "assistant", "status": status,
        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": list(logprobs)}],
    }


def response_body(body, response_id, created, item_id, result=None, options=None, reasoning_id=None):
    status = "in_progress" if result is None else "incomplete" if result.finish_reason == "length" else "completed"
    output = []
    reasoning = body.reasoning.model_dump()
    if options:
        reasoning["effort"] = options.reasoning_effort if not (options.thinking_enabled and options.reasoning_effort == "none") else None
    if result is not None:
        value = completion(result, options or body.options())
        if value.reasoning:
            output.append({"id": reasoning_id or "rs_" + uuid4().hex, "type": "reasoning", "status": status,
                           "summary": [{"type": "summary_text", "text": value.reasoning}] if body.reasoning.summary else [],
                           "content": [{"type": "reasoning_text", "text": value.reasoning}]})
        if value.text or not value.calls:
            output.append(output_message(item_id, value.text, status, visible_logprobs(result, value.text)))
        for call in value.calls:
            output.append({"id": "fc_" + uuid4().hex, "type": "function_call", "status": status,
                           "call_id": call["id"], **call["function"]})
    return {
        "id": response_id, "object": "response", "created_at": created,
        "completed_at": int(time.time()) if status == "completed" else None,
        "model": body.model, "status": status, "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "output": output,
        "usage": None if result is None else response_usage(result),
        "instructions": body.instructions, "max_output_tokens": options.max_new_tokens if options else body.max_output_tokens,
        "temperature": body.temperature, "top_p": body.top_p, "store": body.store,
        "tools": body.tools, "tool_choice": body.tool_choice, "parallel_tool_calls": body.parallel_tool_calls,
        "text": body.text.model_dump(by_alias=True, exclude_none=True), "metadata": body.metadata,
        "reasoning": reasoning, "previous_response_id": body.previous_response_id,
        "truncation": body.truncation, "service_tier": "default", "background": False,
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


def choice_body(result, options, index=0):
    value = completion(result, options)
    return {"index": index, "message": chat_message(value),
            "finish_reason": "tool_calls" if value.calls and result.finish_reason != "length" else result.finish_reason,
            "logprobs": {"content": visible_logprobs(result, value.text)} if options.logprobs else None}


def remaining_output(text, emitted, channel):
    if not text.startswith(emitted):
        raise APIError(f"Final {channel} differs from streamed {channel}", status=502, code="invalid_model_output")
    return text[len(emitted):]


def aggregate_usage(results):
    usage = chat_usage(results[0])
    usage["completion_tokens"] = sum(result.output_tokens for result in results)
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    if any(result.reasoning_tokens for result in results):
        usage["completion_tokens_details"] = {"reasoning_tokens": sum(result.reasoning_tokens for result in results)}
    return usage


def persist(store, body, history):
    try:
        store.put(body, history)
    except ValueError as exc:
        raise APIError(str(exc), status=503, code="storage_limit") from exc


async def chat_stream(service, prepared, body, request, completion_id, created, store):
    results, choices = [], []
    def chunk(delta, reason=None, index=0, logprobs=None):
        return {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                "model": body.model, "choices": [{"index": index, "delta": delta,
                                                  "finish_reason": reason, "logprobs": logprobs}]}
    for index in range(body.n):
        options = replace(prepared.options, seed=(body.seed + index) % 2**64) if body.seed is not None else prepared.options
        current = replace(prepared, options=options)
        thought, answer = "", ""
        yield sse(chunk({"role": "assistant", "content": ""}, index=index))
        async with aclosing(generation_events(service, current, request)) as events:
            async for event in events:
                if isinstance(event, APIError):
                    yield sse(event.body())
                    return
                if isinstance(event, ReasoningDelta):
                    if answer:
                        yield sse(APIError("Model opened reasoning after its answer", status=502, code="invalid_model_output").body())
                        return
                    thought += event.text
                    yield sse(chunk({"reasoning_content": event.text}, index=index))
                    continue
                if isinstance(event, str):
                    if not buffered(options):
                        answer += event
                        yield sse(chunk({"content": event}, index=index))
                else:
                    try:
                        choice = choice_body(event, options, index)
                        tail = remaining_output(choice["message"].get("reasoning_content", ""), thought, "reasoning")
                        content_tail = remaining_output(choice["message"].get("content") or "", answer, "answer")
                    except APIError as exc:
                        yield sse(exc.body())
                        return
                    if buffered(options):
                        delta = {key: value for key, value in choice["message"].items() if key not in {"role", "refusal", "tool_calls", "reasoning_content"} and value}
                        if tail:
                            delta["reasoning_content"] = tail
                        if delta or choice["logprobs"]:
                            yield sse(chunk(delta, index=index, logprobs=choice["logprobs"]))
                        for call_index, call in enumerate(choice["message"].get("tool_calls", [])):
                            yield sse(chunk({"tool_calls": [{"index": call_index, **call}]}, index=index))
                    elif tail or content_tail:
                        delta = {}
                        if tail:
                            delta["reasoning_content"] = tail
                        if content_tail:
                            delta["content"] = content_tail
                        yield sse(chunk(delta, index=index))
                    results.append(event)
                    choices.append(choice)
                    yield sse(chunk({}, choice["finish_reason"], index=index))
        if len(results) != index + 1:
            return
    final = {"id": completion_id, "object": "chat.completion", "created": created,
             "model": body.model, "choices": choices, "usage": aggregate_usage(results),
             "service_tier": "default", "metadata": body.metadata}
    if body.store:
        try:
            persist(store, final, [item.model_dump(exclude_none=True) for item in body.messages])
        except APIError as exc:
            yield sse(exc.body())
            return
    if body.stream_options and body.stream_options.include_usage:
        usage = chunk({})
        usage["choices"], usage["usage"] = [], final["usage"]
        yield sse(usage)
    yield sse("[DONE]")


async def responses_stream(service, prepared, body, request, response_id, created, item_id, store, history):
    sequence = 0
    def event(kind, **fields):
        nonlocal sequence
        value = {"type": kind, "sequence_number": sequence, **fields}
        sequence += 1
        return sse(value, kind)
    initial = response_body(body, response_id, created, item_id, options=prepared.options)
    yield event("response.created", response=initial)
    yield event("response.in_progress", response=initial)
    defer = buffered(prepared.options)
    reasoning_id, thought, thought_started = "rs_" + uuid4().hex, "", False
    answer, message_started = "", False

    def thought_start():
        positions = {"item_id": reasoning_id, "output_index": 0}
        yield event("response.output_item.added", output_index=0,
                    item={"id": reasoning_id, "type": "reasoning", "status": "in_progress", "summary": [], "content": []})
        if body.reasoning.summary:
            yield event("response.reasoning_summary_part.added", **positions, summary_index=0,
                        part={"type": "summary_text", "text": ""})

    def thought_delta(delta):
        positions = {"item_id": reasoning_id, "output_index": 0}
        yield event("response.reasoning_text.delta", **positions, content_index=0, delta=delta)
        if body.reasoning.summary:
            yield event("response.reasoning_summary_text.delta", **positions, summary_index=0, delta=delta)

    def message_start(index):
        positions = {"item_id": item_id, "output_index": index, "content_index": 0}
        item = output_message(item_id, status="in_progress")
        item["content"] = []
        yield event("response.output_item.added", output_index=index, item=item)
        yield event("response.content_part.added", **positions, part=output_message(item_id)["content"][0])

    positions = {"item_id": item_id, "output_index": 0, "content_index": 0}
    async with aclosing(generation_events(service, prepared, request)) as events:
        async for value in events:
            if isinstance(value, APIError):
                yield event("error", code=value.code, message=str(value), param=value.param)
                return
            if isinstance(value, ReasoningDelta):
                if message_started:
                    yield event("error", code="invalid_model_output", message="Model opened reasoning after its answer", param=None)
                    return
                if not thought_started:
                    for frame in thought_start():
                        yield frame
                    thought_started = True
                thought += value.text
                for frame in thought_delta(value.text):
                    yield frame
                continue
            if isinstance(value, str):
                if not defer and value:
                    if not message_started:
                        positions = {"item_id": item_id, "output_index": int(thought_started), "content_index": 0}
                        for frame in message_start(positions["output_index"]):
                            yield frame
                        message_started = True
                    answer += value
                    yield event("response.output_text.delta", **positions, delta=value, logprobs=[])
                continue
            try:
                final = response_body(body, response_id, created, item_id, value, prepared.options, reasoning_id)
                final_thought = next((item["content"][0]["text"] for item in final["output"] if item["type"] == "reasoning"), "")
                remaining_output(final_thought, thought, "reasoning")
                final_answer = next((item["content"][0]["text"] for item in final["output"] if item["type"] == "message"), "")
                remaining_output(final_answer, answer, "answer")
                if message_started and final_thought and not thought_started:
                    raise APIError("Model opened reasoning after its answer", status=502, code="invalid_model_output")
                if body.store:
                    persist(store, final, history)
            except APIError as exc:
                yield event("error", code=exc.code, message=str(exc), param=exc.param)
                return
            for output_index, item in enumerate(final["output"]):
                positions = {"item_id": item["id"], "output_index": output_index}
                if defer and item["type"] == "function_call":
                    start = {**item, "status": "in_progress"}
                    start["arguments"] = ""
                    yield event("response.output_item.added", output_index=output_index, item=start)
                if item["type"] == "function_call":
                    yield event("response.function_call_arguments.delta", **positions, delta=item["arguments"])
                    yield event("response.function_call_arguments.done", **positions, arguments=item["arguments"])
                elif item["type"] == "message":
                    positions["content_index"] = 0
                    part = item["content"][0]
                    if not message_started:
                        for frame in message_start(output_index):
                            yield frame
                        message_started = True
                    try:
                        tail = remaining_output(part["text"], answer, "answer")
                    except APIError as exc:
                        yield event("error", code=exc.code, message=str(exc), param=exc.param)
                        return
                    if tail or part["logprobs"]:
                        yield event("response.output_text.delta", **positions, delta=tail, logprobs=part["logprobs"])
                    yield event("response.output_text.done", **positions, text=part["text"], logprobs=part["logprobs"])
                    yield event("response.content_part.done", **positions, part=part)
                else:
                    if not thought_started:
                        for frame in thought_start():
                            yield frame
                        thought_started = True
                    for content_index, part in enumerate(item["content"]):
                        tail = remaining_output(part["text"], thought, "reasoning")
                        if tail:
                            for frame in thought_delta(tail):
                                yield frame
                        yield event("response.reasoning_text.done", **positions, content_index=content_index, text=part["text"])
                    for summary_index, part in enumerate(item["summary"]):
                        yield event("response.reasoning_summary_text.done", **positions, summary_index=summary_index, text=part["text"])
                        yield event("response.reasoning_summary_part.done", **positions, summary_index=summary_index, part=part)
                yield event("response.output_item.done", output_index=output_index, item=item)
            kind = "response.incomplete" if value.finish_reason == "length" else "response.completed"
            yield event(kind, response=final)
            return


def add_chat_routes(app: FastAPI, registry: ModelRegistry):
    store = ResponseStore()
    app.state.response_store = store

    def stored(identifier, object_type):
        try:
            value, history = store.get(identifier)
        except KeyError as exc:
            raise APIError("stored response not found or expired", status=404, code="response_not_found") from exc
        if value["object"] != object_type:
            raise APIError("stored response not found", status=404, code="response_not_found")
        return value, history

    def remove_stored(identifier, object_type):
        stored(identifier, object_type)
        try:
            store.delete(identifier)
        except KeyError as exc:
            raise APIError("stored response not found or expired", status=404, code="response_not_found") from exc

    def resolve(model):
        try:
            return registry.resolve(model, operation="generation")
        except ModelNotFound as exc:
            raise APIError(str(exc), status=404, code="model_not_found", param="model") from exc
        except ModelCapabilityError as exc:
            raise APIError(str(exc), code="unsupported_operation", param="model") from exc

    @app.exception_handler(APIError)
    async def api_error(request, exc):
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        if request.url.path not in OPENAI_PATHS:
            return await request_validation_exception_handler(request, exc)
        error = exc.errors()[0]
        param = ".".join(str(part) for part in error["loc"] if part != "body")
        message = f"Unsupported request field: {param}" if error["type"] == "extra_forbidden" else error["msg"]
        return JSONResponse(APIError(message, param=param).body(), status_code=400)

    @app.post("/v1/chat/completions")
    def chat_completions(body: ChatCompletionRequest, request: Request):
        service = resolve(body.model)
        options = body.options()
        prepared = prepare(service, body.model, [item.message() for item in body.messages], options)
        options = prepared.options
        completion_id, created = f"chatcmpl-{uuid4().hex}", int(time.time())
        if body.stream:
            return StreamingResponse(
                chat_stream(service, prepared, body, request, completion_id, created, store),
                media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        results, choices = [], []
        for index in range(body.n):
            current = replace(options, seed=(body.seed + index) % 2**64) if body.seed is not None else options
            result = generate(service, replace(prepared, options=current))
            results.append(result)
            choices.append(choice_body(result, current, index))
        final = {
            "id": completion_id, "object": "chat.completion", "created": created, "model": body.model,
            "choices": choices, "usage": aggregate_usage(results), "service_tier": "default", "metadata": body.metadata,
        }
        if body.store:
            persist(store, final, [item.model_dump(exclude_none=True) for item in body.messages])
        return final

    @app.post("/v1/responses")
    def responses(body: ResponsesRequest, request: Request):
        service = resolve(body.model)
        history = body.input_items()
        if body.previous_response_id:
            previous, previous_input = stored(body.previous_response_id, "response")
            if previous["model"] != body.model:
                raise APIError("previous_response_id belongs to another model", param="previous_response_id")
            history = previous_input + previous["output"] + history
        for item in history:
            item.setdefault("id", "item_" + uuid4().hex)
        prepared = prepare(service, body.model, body.messages(history), body.options())
        response_id, item_id, created = f"resp_{uuid4().hex}", f"msg_{uuid4().hex}", int(time.time())
        if body.stream:
            return StreamingResponse(
                responses_stream(service, prepared, body, request, response_id, created, item_id, store, history),
                media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        final = response_body(body, response_id, created, item_id, generate(service, prepared), prepared.options)
        if body.store:
            persist(store, final, history)
        return final

    @app.get("/v1/responses/{response_id}")
    def get_response(response_id: str):
        return stored(response_id, "response")[0]

    @app.delete("/v1/responses/{response_id}")
    def delete_response(response_id: str):
        remove_stored(response_id, "response")
        return {"id": response_id, "object": "response.deleted", "deleted": True}

    @app.get("/v1/responses/{response_id}/input_items")
    def input_items(response_id: str, limit: int = 20, order: str = "desc", after: str | None = None, before: str | None = None):
        if not 1 <= limit <= 100 or order not in {"asc", "desc"} or (after and before):
            raise APIError("invalid input_items pagination")
        items = stored(response_id, "response")[1]
        if order == "desc":
            items.reverse()
        if after or before:
            index = next((index for index, item in enumerate(items) if item["id"] == (after or before)), None)
            if index is None:
                raise APIError("input item cursor not found")
            items = items[index + 1:] if after else items[:index]
        page = items[:limit]
        return {"object": "list", "data": page, "has_more": len(items) > limit,
                "first_id": page[0]["id"] if page else None, "last_id": page[-1]["id"] if page else None}

    @app.get("/v1/chat/completions/{completion_id}")
    def get_completion(completion_id: str):
        return stored(completion_id, "chat.completion")[0]

    @app.delete("/v1/chat/completions/{completion_id}")
    def delete_completion(completion_id: str):
        remove_stored(completion_id, "chat.completion")
        return {"id": completion_id, "object": "chat.completion.deleted", "deleted": True}

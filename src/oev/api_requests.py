"""Validated OpenAI text/function request contracts."""

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .completion import check_schema, json_value
from .generation import GenerationOptions


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    _null_defaults: ClassVar[tuple[str, ...]] = ()
    _null_metadata: ClassVar[tuple[str, ...]] = ()
    _empty_metadata: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="before")
    @classmethod
    def normalize_optional_fields(cls, values):
        if isinstance(values, dict):
            values = values.copy()
            for key in cls._null_defaults + cls._null_metadata:
                if values.get(key) is None:
                    values.pop(key, None)
            for key in cls._empty_metadata:
                if values.get(key) == []:
                    values.pop(key)
        return values


Effort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class TextPart(APIModel):
    type: Literal["text"]
    text: str


class RefusalPart(APIModel):
    type: Literal["refusal"]
    refusal: str


ChatPart = Annotated[TextPart | RefusalPart, Field(discriminator="type")]


def message_text(content, refusal=None):
    text = content if isinstance(content, str) else ""
    previous_refusal = False
    for part in content if isinstance(content, list) else []:
        is_refusal = isinstance(part, RefusalPart)
        value = part.refusal if is_refusal else part.text
        if value:
            if text and (is_refusal or previous_refusal):
                text += "\n"
            text += value
            previous_refusal = is_refusal
    if refusal and refusal != text:
        text = text + "\n" + refusal if text else refusal
    return text


class FunctionCall(APIModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[\w.-]+$")
    arguments: str

    @model_validator(mode="after")
    def validate_arguments(self):
        if not isinstance(json_value(self.arguments), dict):
            raise ValueError("function arguments must be a JSON object")
        return self


class ToolCall(APIModel):
    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: FunctionCall
    # Clients may retain the streaming delta index when replaying a call.
    index: int | None = Field(default=None, ge=0, strict=True, exclude=True)


class ChatMessage(APIModel):
    _null_defaults = ("tool_calls",)
    _null_metadata = ("annotations", "audio", "function_call")
    _empty_metadata = ("annotations",)

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[ChatPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=128)
    tool_call_id: str | None = None
    reasoning_content: str | None = None
    refusal: str | None = None

    @model_validator(mode="before")
    @classmethod
    def reasoning_aliases(cls, values):
        if isinstance(values, dict):
            values = values.copy()
            for key in ("reasoning", "thinking"):
                if key in values:
                    thought = values.pop(key)
                    if thought is None:
                        continue
                    if values.get("reasoning_content") is not None and values["reasoning_content"] != thought:
                        raise ValueError("conflicting assistant reasoning fields")
                    values["reasoning_content"] = thought
        return values

    @model_validator(mode="after")
    def validate_role(self):
        if self.tool_calls and self.role != "assistant":
            raise ValueError("tool_calls requires role=assistant")
        if (self.role == "tool") != bool(self.tool_call_id):
            raise ValueError("tool_call_id is required only for tool messages")
        if self.content is None and self.role != "assistant":
            raise ValueError("content is required for non-assistant messages")
        if self.role != "assistant" and (self.refusal or (
            isinstance(self.content, list) and any(isinstance(part, RefusalPart) for part in self.content)
        )):
            raise ValueError("refusal requires role=assistant")
        return self

    def message(self):
        value = {"role": self.role, "content": message_text(self.content, self.refusal)}
        for key in ("name", "tool_call_id", "reasoning_content", "refusal"):
            if getattr(self, key) is not None:
                value[key] = getattr(self, key)
        if self.tool_calls:
            value["tool_calls"] = [call.model_dump() for call in self.tool_calls]
        return value


class FunctionDefinition(APIModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[\w.-]+$")
    description: str | None = None
    parameters: dict = Field(default_factory=lambda: {"type": "object", "properties": {}})
    strict: bool | None = False

    @model_validator(mode="after")
    def validate_schema(self):
        try:
            check_schema(self.parameters)
        except Exception as exc:
            raise ValueError(f"invalid function parameters schema: {exc}") from exc
        return self


class TextFormat(APIModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict | None = None
    name: str | None = None
    description: str | None = None
    schema_: dict | None = Field(default=None, alias="schema")
    strict: bool | None = None

    @model_validator(mode="after")
    def validate_format(self):
        if self.type == "json_schema":
            definition = self.json_schema or {"name": self.name, "schema": self.schema_}
            if not isinstance(definition.get("name"), str) or not definition["name"] or not isinstance(definition.get("schema"), dict):
                raise ValueError("json_schema requires name and schema")
            if set(definition) - {"name", "description", "schema", "strict"}:
                raise ValueError("unknown json_schema option")
            try:
                check_schema(definition["schema"])
            except Exception as exc:
                raise ValueError(f"invalid response schema: {exc}") from exc
        elif any(value is not None for value in (self.json_schema, self.schema_, self.name, self.description, self.strict)):
            raise ValueError("schema options require type=json_schema")
        return self

    def format(self):
        if self.type != "json_schema":
            return {"type": self.type}
        return {"type": self.type, "schema": (self.json_schema or {"schema": self.schema_})["schema"]}


class StreamOptions(APIModel):
    include_usage: bool = False
    include_obfuscation: Literal[False] = False


class ResponseTextConfig(APIModel):
    format: TextFormat = Field(default_factory=TextFormat)
    verbosity: Literal["low", "medium", "high"] | None = None


class ReasoningConfig(APIModel):
    effort: Effort | None = "none"
    summary: Literal["auto", "concise", "detailed"] | None = "auto"


class GenerationRequest(APIModel):
    model: str
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    stream: bool = False
    store: bool = False
    tools: list[dict] = Field(default_factory=list, max_length=128)
    tool_choice: str | dict = "auto"
    parallel_tool_calls: bool = True
    seed: int | None = Field(default=None, ge=0, lt=2**64, strict=True)
    top_k: int = Field(default=0, ge=0, strict=True)
    min_p: float = Field(default=0, ge=0, le=1)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    repetition_penalty: float = Field(default=1, gt=0)
    logit_bias: dict[str, float] = Field(default_factory=dict, max_length=1024)
    metadata: dict[str, str] = Field(default_factory=dict, max_length=16)
    user: str | None = None
    safety_identifier: str | None = None
    conversation_id: str | None = None
    cache_affinity_id: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal["in-memory"] | None = None
    service_tier: Literal["auto", "default"] | None = None
    chat_template_kwargs: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def nullable_defaults(cls, value):
        if isinstance(value, dict):
            value = value.copy()
            for key in ("temperature", "top_p", "stream", "store", "tools", "tool_choice", "parallel_tool_calls",
                        "top_k", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty",
                        "logit_bias", "metadata", "chat_template_kwargs", "n", "response_format", "reasoning_effort", "logprobs",
                        "top_logprobs", "text", "reasoning", "include", "max_output_tokens", "truncation", "background"):
                if key in cls.model_fields and key in value and value[key] is None:
                    value.pop(key)
        return value

    @model_validator(mode="after")
    def validate_common(self):
        if any(len(key) > 64 or len(value) > 512 for key, value in self.metadata.items()):
            raise ValueError("metadata keys must be <=64 characters and values <=512 characters")
        names = []
        for tool in self.tools:
            if tool.get("type") != "function":
                raise ValueError("only client-executed function tools are supported")
            if isinstance(self, ResponsesRequest):
                if set(tool) - {"type", "name", "description", "parameters", "strict"}:
                    raise ValueError("unknown function tool option")
                definition = {key: value for key, value in tool.items() if key != "type"}
            else:
                if set(tool) - {"type", "function"}:
                    raise ValueError("unknown function tool option")
                definition = tool.get("function", {})
            parsed = FunctionDefinition.model_validate(definition)
            names.append(parsed.name)
        if len(names) != len(set(names)):
            raise ValueError("function names must be unique")
        choice = self.choice()
        if choice != "none" and choice != "auto" and not self.tools:
            raise ValueError("tool_choice requires tools")
        if choice not in {"none", "auto", "required"} and choice not in names:
            raise ValueError("tool_choice names an unavailable function")
        for key in self.logit_bias:
            if not key.isdecimal():
                raise ValueError("logit_bias keys must be nonnegative integer token IDs")
        return self

    def choice(self):
        if isinstance(self.tool_choice, str):
            if self.tool_choice not in {"auto", "none", "required"}:
                raise ValueError("invalid tool_choice")
            return self.tool_choice
        value = self.tool_choice
        if value.get("type") != "function":
            raise ValueError("tool_choice must select a function")
        if isinstance(self, ResponsesRequest):
            if set(value) != {"type", "name"}:
                raise ValueError("function tool_choice requires type and name")
            name = value.get("name")
        else:
            if set(value) != {"type", "function"} or not isinstance(value.get("function"), dict) or set(value["function"]) != {"name"}:
                raise ValueError("function tool_choice requires function.name")
            name = value["function"]["name"]
        if not isinstance(name, str) or not name:
            raise ValueError("tool_choice requires a function name")
        return name

    def generation_options(self, limit, stops=(), **extra):
        tools = []
        for tool in self.tools:
            definition = {key: value for key, value in tool.items() if key != "type"} if isinstance(self, ResponsesRequest) else tool["function"]
            definition = FunctionDefinition.model_validate(definition).model_dump(exclude_none=True)
            if self.choice() not in {"none", "auto", "required", definition["name"]}:
                continue
            tools.append({"type": "function", "function": definition})
        return GenerationOptions(limit or 256, self.temperature, self.top_p, stops,
            seed=self.seed, top_k=self.top_k, min_p=self.min_p, presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty, repetition_penalty=self.repetition_penalty,
            logit_bias={int(key): value for key, value in self.logit_bias.items()}, tools=tuple(tools),
            tool_choice=self.choice(), parallel_tool_calls=self.parallel_tool_calls,
            chat_template_kwargs=self.chat_template_kwargs,
            max_new_tokens_explicit=limit is not None,
            reasoning_effort_explicit=(("effort" in self.reasoning.model_fields_set and self.reasoning.effort is not None) if isinstance(self, ResponsesRequest)
                                       else "reasoning_effort" in self.model_fields_set), **extra)


class ChatCompletionRequest(GenerationRequest):
    messages: list[ChatMessage] = Field(min_length=1, max_length=4096)
    max_tokens: int | None = Field(default=None, ge=1, le=32768, strict=True)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=32768, strict=True)
    stop: str | list[str] | None = None
    n: int = Field(default=1, ge=1, le=8, strict=True)
    stream_options: StreamOptions | None = None
    response_format: TextFormat = Field(default_factory=TextFormat)
    reasoning_effort: Effort = "none"
    logprobs: bool = False
    top_logprobs: int = Field(default=0, ge=0, le=20, strict=True)
    modalities: list[Literal["text"]] | None = None

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
        return self.generation_options(self.max_completion_tokens or self.max_tokens, stops,
            reasoning_effort=self.reasoning_effort, logprobs=self.logprobs, top_logprobs=self.top_logprobs,
            response_format=self.response_format.format())


class InputText(APIModel):
    _null_defaults = ("annotations", "logprobs")

    type: Literal["input_text", "output_text"]
    text: str
    annotations: list = Field(default_factory=list)
    logprobs: list = Field(default_factory=list)


ResponsePart = Annotated[InputText | RefusalPart, Field(discriminator="type")]


class InputMessage(APIModel):
    type: Literal["message"] = "message"
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[ResponsePart]
    id: str | None = None
    status: Literal["completed", "incomplete", "in_progress"] | None = None
    phase: Literal["commentary", "final_answer"] | None = None

    @model_validator(mode="after")
    def validate_refusal_role(self):
        if self.role != "assistant" and isinstance(self.content, list) and any(
            isinstance(part, RefusalPart) for part in self.content
        ):
            raise ValueError("refusal requires role=assistant")
        return self

    def message(self):
        value = {"role": self.role, "content": message_text(self.content)}
        if self.phase is not None:
            value["phase"] = self.phase
        return value

    def item(self):
        value = self.model_dump(exclude_none=True, exclude_unset=True)
        value.update(type="message", status=self.status or "completed")
        if isinstance(self.content, str):
            value["content"] = [{"type": "output_text" if self.role == "assistant" else "input_text",
                                 "text": self.content}]
        for part in value["content"]:
            if part["type"] == "output_text":
                part.setdefault("annotations", [])
        return value


class InputFunctionCall(APIModel):
    _null_metadata = ("async", "caller", "namespace")

    type: Literal["function_call"]
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: str
    id: str | None = None
    status: Literal["completed", "incomplete", "in_progress"] | None = None


class InputFunctionOutput(APIModel):
    _null_metadata = ("caller", "namespace", "name")

    type: Literal["function_call_output"]
    call_id: str = Field(min_length=1)
    output: str | list[InputText]
    id: str | None = None
    status: Literal["completed", "incomplete", "in_progress"] | None = None


class SummaryText(APIModel):
    type: Literal["summary_text", "reasoning_text"]
    text: str


class InputReasoning(APIModel):
    _null_defaults = ("summary", "content")
    _null_metadata = ("encrypted_content",)

    type: Literal["reasoning"]
    id: str | None = None
    summary: list[SummaryText] = Field(default_factory=list)
    content: list[SummaryText] = Field(default_factory=list)
    status: Literal["completed", "incomplete", "in_progress"] | None = None


InputItem = Annotated[InputMessage | InputFunctionCall | InputFunctionOutput | InputReasoning,
                      Field(discriminator="type")]


class ResponsesRequest(GenerationRequest):
    input: str | list[InputItem]
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, ge=1, le=32768, strict=True)
    text: ResponseTextConfig = Field(default_factory=ResponseTextConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    previous_response_id: str | None = None
    include: list[Literal["message.output_text.logprobs"]] = Field(default_factory=list)
    top_logprobs: int = Field(default=0, ge=0, le=20, strict=True)
    truncation: Literal["disabled"] = "disabled"
    background: Literal[False] = False

    @field_validator("input", mode="before")
    @classmethod
    def default_message_type(cls, value):
        if isinstance(value, list):
            return [{**item, "type": "message"}
                    if isinstance(item, dict) and item.get("type") is None else item for item in value]
        return value

    @model_validator(mode="after")
    def validate_options(self):
        if isinstance(self.input, list) and len(self.input) > 4096:
            raise ValueError("input supports at most 4096 items")
        self.options()
        return self

    def input_items(self):
        if isinstance(self.input, str):
            return [InputMessage(role="user", content=self.input).item()]
        return [item.item() if isinstance(item, InputMessage) else item.model_dump(exclude_none=True)
                for item in self.input]

    def messages(self, items=None):
        messages = []
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})
        thought = ""
        for value in self.input_items() if items is None else items:
            kind = value.get("type", "message")
            if kind == "reasoning":
                text = "\n".join(part["text"] for part in value.get("content") or value.get("summary", []))
                thought = "\n".join(part for part in (thought, text) if part)
            elif kind == "message":
                message = InputMessage.model_validate(value).message()
                if thought:
                    if message["role"] == "assistant":
                        message["reasoning_content"] = thought
                    else:
                        messages.append({"role": "assistant", "content": "", "reasoning_content": thought})
                    thought = ""
                messages.append(message)
            elif kind == "function_call":
                call = {"id": value["call_id"], "type": "function", "function": {"name": value["name"], "arguments": value["arguments"]}}
                if not messages or messages[-1]["role"] != "assistant":
                    messages.append({"role": "assistant", "content": ""})
                messages[-1].setdefault("tool_calls", []).append(call)
                if thought:
                    messages[-1]["reasoning_content"], thought = thought, ""
            else:
                output = value["output"]
                messages.append({"role": "tool", "tool_call_id": value["call_id"], "content": output if isinstance(output, str) else "".join(part["text"] for part in output)})
        if thought:
            messages.append({"role": "assistant", "content": "", "reasoning_content": thought})
        if self.text.verbosity:
            messages.insert(0, {"role": "system", "content": {"low": "Be concise.", "medium": "Use moderate detail.", "high": "Provide a detailed answer."}[self.text.verbosity]})
        return messages

    def options(self):
        return self.generation_options(self.max_output_tokens, reasoning_effort=self.reasoning.effort or "none",
            logprobs=bool(self.include or self.top_logprobs), top_logprobs=self.top_logprobs, response_format=self.text.format.format())

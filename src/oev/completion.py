"""Text, reasoning and function-call protocols shared by all generation runtimes."""

from copy import deepcopy
from dataclasses import dataclass
import json
import re
from uuid import uuid4

from .output_parsers import (
    json_value, parameter_value, parse_tool_calls, split_reasoning, strip_turn_tokens, template_text,
)


def check_schema(schema):
    """Validate local schemas without allowing a resolver to fetch remote resources."""
    from jsonschema import Draft202012Validator
    json.dumps(schema, allow_nan=False)
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"$ref", "$dynamicRef"} and (not isinstance(child, str) or not child.startswith("#")):
                    raise ValueError("only local JSON Schema references are supported")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(schema)
    Draft202012Validator.check_schema(schema)


def validate_json(value, schema):
    from jsonschema import Draft202012Validator
    from referencing.exceptions import Unresolvable
    try:
        error = next(Draft202012Validator(schema).iter_errors(value), None)
    except Unresolvable as exc:
        raise ValueError(f"cannot resolve local JSON Schema reference: {exc}") from exc
    if error:
        raise ValueError(f"output does not match JSON Schema at {list(error.path)}: {error.message}")


def prompt_messages(messages, options, tokenizer=None):
    messages = deepcopy(messages)
    pending, seen = {}, set()
    for message in messages:
        if message["role"] == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ValueError("tool result must reference an unresolved assistant tool_call_id")
            message["name"] = pending.pop(call_id)
        else:
            if pending:
                raise ValueError("all assistant tool calls require tool results before another message")
            for call in message.get("tool_calls", []):
                if message["role"] != "assistant" or call["id"] in seen:
                    raise ValueError("tool calls require an assistant role and unique call IDs")
                arguments = call["function"]["arguments"]
                if isinstance(arguments, str):
                    arguments = json_value(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("function arguments must be a JSON object")
                call["function"]["arguments"] = arguments
                pending[call["id"]] = call["function"]["name"]
                seen.add(call["id"])
    if pending:
        raise ValueError("assistant tool calls require tool results before generation")
    for message in messages:
        if message["role"] == "assistant" and message.get("reasoning_content"):
            message["reasoning"] = message["thinking"] = message["reasoning_content"]
    template = options.chat_template or template_text(tokenizer)
    native_tools = any(marker in template for marker in ("tools", "tool_call", "declaration:"))
    if template and not native_tools:
        # Templates trained only on user/assistant text need a readable history
        # too; merely attaching tool_calls would silently lose the invocation.
        for message in messages:
            if message["role"] == "tool":
                message["role"] = "user"
                message["content"] = f"Function {message['name']} returned:\n{message['content']}"
            for call in message.get("tool_calls", []):
                payload = {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}
                message["content"] += "\n<tool_call>" + json.dumps(payload, ensure_ascii=False) + "</tool_call>"
    hints = []
    if options.tools and options.tool_choice != "none":
        if not native_tools:
            hints.append('Available functions: ' + json.dumps(options.tools, ensure_ascii=False))
            hints.append('To invoke a function, emit <tool_call>{"name":"FUNCTION_NAME","arguments":{...}}</tool_call>. Do not execute functions yourself.')
        if options.tool_choice == "required":
            hints.append("You must invoke at least one available function in this response.")
        elif options.tool_choice != "auto":
            hints.append(f"You must invoke the function {options.tool_choice} in this response.")
        if not options.parallel_tool_calls:
            hints.append("Invoke at most one function in this response.")
    fmt = options.response_format
    if fmt["type"] != "text":
        hints.append("Return only a valid JSON " + ("object" if fmt["type"] == "json_object" else "value") + ", without Markdown fences or extra text.")
        if fmt["type"] == "json_schema":
            hints.append("The JSON must match this schema: " + json.dumps(fmt["schema"], ensure_ascii=False))
    if hints:
        instruction = "\n".join(hints)
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += "\n\n" + instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction})
    return messages


class GemmaArguments:
    """Parse Gemma's unquoted keys and <|\"|>-delimited strings, without eval."""
    def __init__(self, text, escape='<|"|>'):
        self.text, self.pos, self.escape = text, 0, escape

    def whitespace(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def value(self, depth=0):
        if depth > 64:
            raise ValueError("function arguments are nested too deeply")
        self.whitespace()
        if self.text.startswith(self.escape, self.pos):
            start = self.pos + len(self.escape)
            end = self.text.find(self.escape, start)
            if end < 0:
                raise ValueError("unterminated Gemma argument string")
            self.pos = end + len(self.escape)
            return self.text[start:end]
        if self.pos >= len(self.text):
            raise ValueError("incomplete function arguments")
        char = self.text[self.pos]
        if char in "{[":
            self.pos += 1
            value = {} if char == "{" else []
            closing = "}" if char == "{" else "]"
            self.whitespace()
            while not self.text.startswith(closing, self.pos):
                if char == "{":
                    if self.text.startswith((self.escape, '"'), self.pos):
                        key = self.value(depth + 1)
                        self.whitespace()
                        if not self.text.startswith(":", self.pos):
                            raise ValueError("invalid Gemma argument key separator")
                        self.pos += 1
                    else:
                        match = re.match(r"\s*([A-Za-z_][\w.-]*)\s*:", self.text[self.pos:])
                        if not match:
                            raise ValueError("invalid Gemma argument key")
                        self.pos += match.end()
                        key = match[1]
                    if key in value:
                        raise ValueError("duplicate function argument")
                    value[key] = self.value(depth + 1)
                else:
                    value.append(self.value(depth + 1))
                self.whitespace()
                if self.text.startswith(closing, self.pos):
                    break
                if not self.text.startswith(",", self.pos):
                    raise ValueError("invalid Gemma argument separator")
                self.pos += 1
            self.pos += 1
            return value
        decoder = json.JSONDecoder(parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid JSON constant")))
        value, consumed = decoder.raw_decode(self.text[self.pos:])
        self.pos += consumed
        return value


@dataclass
class Completion:
    text: str
    calls: list[dict]
    reasoning: str = ""


def parse_completion(result, options):
    raw = result.raw_text if result.raw_text is not None else result.text
    reasoning = ""
    active_tools = options.tools and options.tool_choice != "none"
    if options.reasoning_parser != "none" and (
        options.reasoning_effort != "none" or active_tools
        or options.reasoning_parser != "auto"
    ):
        raw, reasoning = split_reasoning(raw, options.reasoning_parser)
    # Harmony reasoning needs its end marker before turn-token cleanup.
    calls = []
    if active_tools:
        def gemma(content, escape='<|"|>'):
            found = re.match(r"(?:call:)?([\w.-]+)\s*", content)
            if not found:
                raise ValueError("invalid Gemma function call name")
            parser = GemmaArguments(content[found.end():], escape)
            arguments = parser.value()
            parser.whitespace()
            if parser.pos != len(parser.text):
                raise ValueError("extra data in function arguments")
            if escape == "<escape>" and isinstance(arguments, dict):
                definition = next((tool["function"] for tool in options.tools if tool["function"]["name"] == found[1]), {})
                root = definition.get("parameters", {})
                for key, value in arguments.items():
                    if isinstance(value, str):
                        arguments[key] = parameter_value(value, root.get("properties", {}).get(key, {}), root)
            return {"name": found[1], "arguments": arguments}
        raw, native = parse_tool_calls(raw, options.tools, options.tool_call_parser, gemma)
        for call in native:
            name, arguments = call["name"], call["arguments"]
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ValueError("function calls require a name and a JSON argument object")
            definition = next((tool["function"] for tool in options.tools if tool["function"]["name"] == name), None)
            if definition is None:
                raise ValueError(f"model invoked unknown function {name!r}")
            if definition.get("strict"):
                validate_json(arguments, definition.get("parameters", {"type": "object"}))
            calls.append({"id": "call_" + uuid4().hex, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
            }})
        if not options.parallel_tool_calls and len(calls) > 1:
            raise ValueError("model returned multiple calls with parallel_tool_calls=false")
        if options.tool_choice == "required" and not calls:
            raise ValueError("model did not return the required function call")
        if options.tool_choice not in {"auto", "none", "required"} and (
            not calls or any(call["function"]["name"] != options.tool_choice for call in calls)
        ):
            raise ValueError("model did not return the selected function call")
    raw = strip_turn_tokens(raw)
    text = raw.strip() if reasoning or calls else raw
    if (not reasoning and not calls and options.reasoning_effort == "none"
        and options.reasoning_parser in {"auto", "none"} and options.response_format["type"] == "text"):
        text = result.text
    if options.response_format["type"] != "text" and not calls and result.finish_reason != "length":
        value = json_value(text)
        if options.response_format["type"] == "json_object" and not isinstance(value, dict):
            raise ValueError("model output must be a JSON object")
        if options.response_format["type"] == "json_schema":
            validate_json(value, options.response_format["schema"])
    return Completion(text, calls, reasoning)

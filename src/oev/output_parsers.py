"""Backend-independent model output protocols.

Protocol references and compatibility limits live in docs/model-protocols.md.
No inference-framework imports are needed to interpret decoded model text.
"""

import ast
import json
import re


TOOL_ALIASES = {
    "auto": "auto", "none": "none", "hermes": "json", "qwen": "json",
    "qwen25": "json", "llama3": "json", "llama3_json": "json",
    "mistral": "json", "granite": "json", "granite-20b-fc": "json", "granite4": "json",
    "jamba": "json", "xlam": "json", "json": "json", "dots": "json",
    "phi4_mini_json": "json", "lfm2": "pythonic",
    "qwen3": "qwen3_xml", "qwen3_xml": "qwen3_xml",
    "qwen3_coder": "qwen3_xml", "step3p5": "qwen3_xml",
    "mimo": "qwen3_xml", "step3": "step3",
    "glm": "glm", "glm45": "glm", "glm47": "glm", "glm4_moe": "glm",
    "minimax_m2": "minimax_m2", "minimax-m2": "minimax_m2",
    "deepseekv3": "deepseek", "deepseek_v3": "deepseek",
    "deepseekv31": "deepseek", "deepseek_v31": "deepseek", "deepseek": "deepseek",
    "deepseekv32": "deepseek_dsml", "deepseek_v32": "deepseek_dsml",
    "deepseek_dsml": "deepseek_dsml", "kimi_k2": "kimi_k2",
    "gemma4": "gemma4", "functiongemma": "functiongemma",
    "internlm": "internlm", "internlm2": "internlm", "interns1": "internlm",
    "pythonic": "pythonic", "llama4_pythonic": "pythonic", "olmo3": "pythonic",
    "gpt-oss": "harmony", "gpt_oss": "harmony", "openai": "harmony",
    "harmony": "harmony",
}
REASONING_ALIASES = {
    "auto": "auto", "none": "none", "qwen3": "think",
    "qwen3-thinking": "think", "deepseek-r1": "think", "deepseek_r1": "think",
    "deepseek-v3": "think", "deepseek_v3": "think", "glm45": "think",
    "minimax_m2": "think", "minimax-m2": "think", "think": "think",
    "gemma4": "gemma4", "kimi_k2": "kimi_k2",
    "apertus2509": "apertus", "apertus": "apertus",
    "mistral": "mistral", "gpt-oss": "harmony", "gpt_oss": "harmony",
    "openai": "harmony", "harmony": "harmony",
    "cohere_command3": "cohere", "cohere_command4": "cohere", "cohere": "cohere",
    "granite": "granite",
}
REASONING_MARKERS = (
    ("think", "<think>", "</think>"),
    ("think", "<thinking>", "</thinking>"),
    ("gemma4", "<|channel>thought\n", "<channel|>"),
    ("kimi_k2", "◁think▷", "◁/think▷"),
    ("apertus", "<|inner_prefix|>", "<|inner_suffix|>"),
    ("mistral", "[THINK]", "[/THINK]"),
    ("cohere", "<|START_THINKING|>", "<|END_THINKING|>"),
    ("granite", "<|thinking|>", "<|response|>"),
    ("harmony", "<|channel|>analysis<|message|>", "<|end|>"),
)
TOOL_STARTS = (
    "<tool_call>", "<|tool_call>", "<start_function_call>",
    "<minimax:tool_call>", "<｜DSML｜function_calls>",
    "<｜tool▁calls▁begin｜>", "<｜tool▁call▁begin｜>",
    "<|tool_calls_section_begin|>", "<|tool_call_begin|>",
    "<|action_start|>", "<function_calls>", "[TOOL_CALLS]",
    "<|tool_call_start|>", "<dots_function_call>", "<tool_calls>",
    "<function_call>", "<｜tool_calls_begin｜>", "functools[",
)
TURN_TOKENS = (
    "<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>",
    "<|end|>", "<|turn>", "<turn|>", "<end_of_turn>", "<eos>", "</s>",
    "<｜end▁of▁sentence｜>", "<|fim_suffix|>", "<|return|>",
)
HARMONY_TOOL_HEADER = re.compile(
    r"(?:<\|start\|>assistant)?(?:\s*to=([\w.-]+)(?:<\|channel\|>(?:commentary|analysis|final))?"
    r"|<\|channel\|>(?:commentary|analysis|final)\s+to=([\w.-]+))"
    r"(?:<\|constrain\|>json)?<\|message\|>"
)


def _tool_boundary(text):
    positions = [text.find(marker) for marker in TOOL_STARTS if marker in text]
    header = HARMONY_TOOL_HEADER.search(text)
    if header:
        positions.append(header.start())
    return min(positions, default=len(text))


def parser_name(name, *, reasoning=False):
    aliases = REASONING_ALIASES if reasoning else TOOL_ALIASES
    if name not in aliases:
        kind = "reasoning_parser" if reasoning else "tool_call_parser"
        raise ValueError(f"unsupported {kind}: {name!r}")
    return aliases[name]


def json_value(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"invalid JSON constant {value}")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def template_text(tokenizer):
    return str(getattr(tokenizer, "chat_template", "") or "") + str(
        getattr(getattr(tokenizer, "formatter", None), "template", "") or ""
    )


def detect_tool_parser(template):
    for marker, profile in (
        ("<function=", "qwen3_xml"), ("<arg_key>", "glm"),
        ("<minimax:tool_call>", "minimax_m2"), ("｜DSML｜", "deepseek_dsml"),
        ("tool▁call▁begin", "deepseek"), ("tool_call_argument_begin", "kimi_k2"),
        ("<|tool_call>", "gemma4"), ("<start_function_call>", "functiongemma"),
        ("<|action_start|>", "internlm"), ("<function_calls>", "pythonic"),
        ("<|tool_call_start|>", "pythonic"), ("<｜tool_calls_begin｜>", "step3"),
        ("<|channel|>", "harmony"),
    ):
        if marker in template:
            return profile
    return "auto"


def split_reasoning(text, profile="auto"):
    """Keep native thoughts separate; tool start can implicitly end thinking."""
    profile = parser_name(profile, reasoning=True)
    if profile == "none":
        return text, ""
    try:
        json_value(_json_candidate(text))
    except ValueError:
        pass
    else:
        return text, ""  # Protocol-looking strings in complete JSON are data.
    thoughts = []
    for family, opening, closing in REASONING_MARKERS:
        if profile not in {"auto", family}:
            continue
        # A template may already have opened a thought block in the prompt.
        boundary = _tool_boundary(text)
        if closing in text[:boundary] and opening not in text and family != "harmony":
            thought, text = text.split(closing, 1)
            thoughts.append(thought.strip())
        while opening in text:
            boundary = _tool_boundary(text)
            if text.index(opening) > boundary:
                break  # Tags inside function arguments are literal data.
            before, tail = text.split(opening, 1)
            if family == "harmony":
                before = re.sub(r"<\|start\|>assistant$", "", before)
            boundary = _tool_boundary(tail)
            end = tail.find(closing)
            if 0 <= end < boundary:
                thought, tail = tail[:end], tail[end + len(closing):]
            elif boundary < len(tail):
                thought, tail = tail[:boundary], tail[boundary:]
            else:
                thought, tail = tail, ""
            thoughts.append(thought.strip())
            text = before + tail
    # Channel headers belong to the assistant prefix, never argument strings.
    text = re.sub(r"^\s*(?:<\|channel>final\n|<channel\|>)+", "", text)
    if profile in {"auto", "harmony"}:
        text = re.sub(r"^\s*(?:<\|end\|>\s*)?(?:<\|start\|>assistant)?<\|channel\|>final<\|message\|>", "", text)
    if profile in {"auto", "cohere"}:
        text = re.sub(r"^\s*<\|START_RESPONSE\|>", "", text)
        text = re.sub(r"<\|END_RESPONSE\|>\s*$", "", text)
    return text, "\n\n".join(thought for thought in thoughts if thought)


def strip_turn_tokens(text):
    tokens = "|".join(re.escape(token) for token in TURN_TOKENS)
    while True:
        cleaned = re.sub(r"^(?:" + tokens + r")|(?:" + tokens + r")(?=\s*$)", "", text)
        if cleaned == text:
            break
        text = cleaned
    return re.sub(r"<\|tool_response>$", "", text)


def _json_candidate(text):
    candidate = strip_turn_tokens(text).strip()
    candidate = re.sub(r"^(?:<\|python_tag\|>|\[TOOL_CALLS\]|<\|tools_prefix\|>|<function_call>|functools(?=\[))\s*", "", candidate)
    return candidate.removesuffix("<|tools_suffix|>").strip()


def _call_envelopes(payload):
    items = payload if isinstance(payload, list) else [payload]
    return bool(items) and all(isinstance(item, dict) and set(item) in (
        {"name", "arguments"}, {"name", "parameters"}, {"type", "function"},
    ) for item in items)


def _schema(schema, root):
    seen = set()
    while isinstance(schema, dict) and "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/") or ref in seen:
            break
        seen.add(ref)
        resolved = root
        try:
            for key in ref[2:].split("/"):
                resolved = resolved[key.replace("~1", "/").replace("~0", "~")]
        except (KeyError, TypeError):
            break
        schema = resolved
    return schema if isinstance(schema, dict) else {}


def _types(schema, root, depth=0):
    if depth > 16:
        return set()
    schema = _schema(schema, root)
    value = schema.get("type")
    found = {value} if isinstance(value, str) else set(value or [])
    for key in ("anyOf", "oneOf", "allOf"):
        for variant in schema.get(key, []):
            found.update(_types(variant, root, depth + 1))
    return found


def _properties(schema, root, depth=0):
    if depth > 16:
        return {}
    schema = _schema(schema, root)
    found = {}
    for key in ("allOf", "anyOf", "oneOf"):
        for variant in schema.get(key, []):
            found.update(_properties(variant, root, depth + 1))
    found.update(schema.get("properties", {}))
    return found


def parameter_value(value, schema, root, *, string=None):
    """Convert XML scalars using the declared schema; string contents stay literal."""
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    types = _types(schema, root)
    if string is True:
        return value
    if string is not False and (not types or types == {"string"}):
        return value
    if "string" in types and value.strip() != "null":
        try:
            decoded = json_value(value)
        except ValueError:
            return value
        if isinstance(decoded, str):
            return value
    else:
        decoded = json_value(value)
    actual = ("null" if decoded is None else "boolean" if isinstance(decoded, bool)
              else "integer" if isinstance(decoded, int) else "number" if isinstance(decoded, float)
              else "object" if isinstance(decoded, dict) else "array" if isinstance(decoded, list) else "string")
    if types and actual not in types and not (actual == "integer" and "number" in types):
        raise ValueError(f"XML parameter must have type {sorted(types)}")
    return decoded


def _definition(name, tools):
    return next((tool["function"] for tool in tools if tool["function"]["name"] == name), {})


def _parameter_body(text, name, tools, *, style="qwen"):
    definition = _definition(name, tools)
    root = definition.get("parameters", {})
    properties = _properties(root, root)
    result = {}
    while text.strip():
        text = text.lstrip()
        if style == "qwen":
            opening = re.match(r"<parameter=([\w.-]+)>", text)
            closing = "</parameter>"
            key = opening[1] if opening else None
            string = None
        elif style == "glm":
            opening = re.match(r"<arg_key>([\w.-]+)</arg_key>\s*<arg_value>", text)
            closing = "</arg_value>"
            key = opening[1] if opening else None
            string = None
        else:
            tag = {"dsml": "｜DSML｜parameter", "step3": "steptml:parameter"}.get(style, "parameter")
            opening = re.match(r"<" + re.escape(tag) + r"\s+([^>]+)>", text)
            attrs = _attributes(opening[1]) if opening else {}
            key = attrs.get("name")
            string = attrs.get("string")
            if string is not None and string not in {"true", "false"}:
                raise ValueError("DSML string attribute must be true or false")
            string = None if string is None else string == "true"
            closing = "</" + tag + ">"
        if not opening or not key:
            raise ValueError("invalid XML function parameter")
        if key in result:
            raise ValueError(f"duplicate function argument {key!r}")
        tail = text[opening.end():]
        # JSON strings in object/array arguments may contain protocol-looking
        # text. Find their JSON boundary before searching for a closing tag.
        offset = len(tail) - len(tail.lstrip())
        if tail.lstrip().startswith(("{", "[")):
            try:
                _, end = json.JSONDecoder().raw_decode(tail, offset)
            except ValueError:
                end = 0
            if end:
                match = re.match(r"\s*" + re.escape(closing), tail[end:])
                if not match:
                    raise ValueError("missing XML parameter closing tag after JSON")
                stop = end + match.end()
                value = tail[:end]
            else:
                stop = tail.find(closing)
                value = tail[:stop]
                stop = stop + len(closing) if stop >= 0 else -1
        else:
            end = tail.find(closing)
            value = tail[:end]
            stop = end + len(closing) if end >= 0 else -1
        if stop < 0:
            raise ValueError("incomplete XML function parameter")
        result[key] = parameter_value(value, properties.get(key, {}), root, string=string)
        text = tail[stop:]
    return result


def _attributes(text):
    attrs = {}
    while text.strip():
        match = re.match(r"""\s*([\w.-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text)
        if not match or match[1] in attrs:
            raise ValueError("invalid or duplicate XML attribute")
        attrs[match[1]] = match[2] if match[2] is not None else match[3]
        text = text[match.end():]
    return attrs


def _json_calls(content):
    payload = json_value(content)
    payloads = payload if isinstance(payload, list) else [payload]
    calls = []
    for item in payloads:
        if not isinstance(item, dict):
            raise ValueError("invalid JSON function call envelope")
        if item.get("type") == "function" and isinstance(item.get("function"), dict):
            item = item["function"]
        if "name" not in item or not ({"arguments", "parameters"} & item.keys()):
            raise ValueError("function call requires name and arguments")
        if "arguments" in item and "parameters" in item:
            raise ValueError("ambiguous function arguments")
        arguments = item.get("arguments", item.get("parameters"))
        if isinstance(arguments, str):
            arguments = json_value(arguments)
        calls.append({"name": item["name"], "arguments": arguments})
    return calls


def _tool_body(content, tools, gemma):
    content = content.strip()
    if content.startswith("<function="):
        match = re.fullmatch(r"<function=([\w.-]+)>(.*)</function>", content, re.S)
        if not match:
            raise ValueError("incomplete Qwen XML function call")
        return [{"name": match[1], "arguments": _parameter_body(match[2], match[1], tools)}]
    if content.startswith("call:"):
        return [gemma(content)]
    if content.startswith(("{", "[")):
        return _json_calls(content)
    match = re.match(r"([\w.-]+)\s*", content)
    if not match:
        raise ValueError("invalid function call body")
    name, body = match[1], content[match.end():]
    if not body:
        return [{"name": name, "arguments": {}}]
    if body.startswith("<arg_key>"):
        arguments = _parameter_body(body, name, tools, style="glm")
    else:
        arguments = json_value(body)
    return [{"name": name, "arguments": arguments}]


def _invoke_body(content, tools, *, style="minimax"):
    tag = {"dsml": "｜DSML｜invoke", "step3": "steptml:invoke"}.get(style, "invoke")
    calls = []
    while content.strip():
        content = content.lstrip()
        opening = re.match(r"<" + re.escape(tag) + r"\s+([^>]+?)(/?)>", content)
        if not opening:
            raise ValueError("invalid XML invoke block")
        attrs = _attributes(opening[1])
        name = attrs.get("name")
        if not name:
            raise ValueError("XML invoke requires name")
        tail = content[opening.end():]
        def arguments(body):
            if body.lstrip().startswith("{"):
                return json_value(body)
            return _parameter_body(body, name, tools, style=style)
        if opening[2]:
            values, content = {}, tail
        else:
            closing = "</" + tag + ">"
            end = tail.find(closing)
            if end < 0:
                raise ValueError("incomplete XML invoke block")
            try:
                values = arguments(tail[:end])
            except ValueError as original:
                # Nested JSON string values can contain an invoke delimiter.
                for _ in range(32):
                    end = tail.find(closing, end + len(closing))
                    if end < 0:
                        raise original
                    try:
                        values = arguments(tail[:end])
                        break
                    except ValueError:
                        continue
                else:
                    raise original
            content = tail[end + len(closing):]
        calls.append({"name": name, "arguments": values})
    return calls


class _JSONConstants(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id in {"true", "false", "null"}:
            return ast.Constant({"true": True, "false": False, "null": None}[node.id])
        return node


def _python_calls(content):
    try:
        expression = ast.parse(content.strip(), mode="eval").body
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("invalid Python-style function calls") from exc
    nodes = expression.elts if isinstance(expression, (ast.List, ast.Tuple)) else [expression]
    calls = []
    for node in nodes:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.args:
            raise ValueError("Python-style calls require a name and keyword arguments")
        arguments = {}
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg in arguments:
                raise ValueError("invalid or duplicate Python-style argument")
            # JSON constants occur in Olmo's otherwise Python-style calls.
            value = _JSONConstants().visit(keyword.value)
            try:
                arguments[keyword.arg] = ast.literal_eval(value)
            except (ValueError, TypeError, RecursionError) as exc:
                raise ValueError("Python-style arguments must be literal values") from exc
        calls.append({"name": node.func.id, "arguments": arguments})
    return calls


def parse_tool_calls(text, tools, profile, gemma):
    """Normalize complete native calls, retaining surrounding assistant text."""
    profile = parser_name(profile)
    if profile == "none":
        return text, []
    candidate = _json_candidate(text)
    try:
        payload = json_value(candidate)
    except ValueError as original:
        # Recognizable envelopes must not fall back to prose when strict JSON
        # decoding rejects duplicate keys or nonfinite constants.
        try:
            payload = json.loads(candidate)
        except ValueError:
            pass
        else:
            if _call_envelopes(payload):
                raise original
    else:
        if _call_envelopes(payload):
            return "", _json_calls(candidate)
    calls = []
    def consume(pattern, decoder):
        nonlocal text
        compiled = re.compile(pattern, re.S)
        parts, position = [], 0
        while match := compiled.search(text, position):
            parts.append(text[position:match.start()])
            content_end, end = match.end(1), match.end()
            try:
                decoded = decoder(match[1])
            except ValueError as original:
                # A delimiter in a JSON/string argument is data. Extend only
                # to another identical closer and require a complete parse.
                closing = text[content_end:end]
                for _ in range(32):
                    content_end = text.find(closing, end)
                    if content_end < 0:
                        raise original
                    end = content_end + len(closing)
                    try:
                        decoded = decoder(text[match.start(1):content_end])
                        break
                    except ValueError:
                        continue
                else:
                    raise original
            calls.extend(decoded)
            position = end
        parts.append(text[position:])
        text = "".join(parts)

    # Explicit delimiters are unambiguous across model families. A forced
    # profile additionally enables formats without distinctive wrapper tokens.
    consume(r"<tool_call>(.*?)</tool_call>", lambda value: _tool_body(value, tools, gemma))
    consume(r"<\|tool_call>(.*?)(?:<tool_call\|>|<turn\|>)", lambda value: [gemma(value.strip())])
    consume(r"<start_function_call>(.*?)<end_function_call>",
            lambda value: [gemma(value.strip(), escape="<escape>")])
    consume(r"<minimax:tool_call>(.*?)</minimax:tool_call>",
            lambda value: _invoke_body(value, tools))
    consume(r"<｜DSML｜function_calls>(.*?)</｜DSML｜function_calls>",
            lambda value: _invoke_body(value, tools, style="dsml"))
    consume(r"<｜tool_call_begin｜>function<｜tool_sep｜>(.*?)<｜tool_call_end｜>",
            lambda value: _invoke_body(value, tools, style="step3"))
    consume(r"<function_calls>(.*?)</function_calls>",
            lambda value: [call for line in value.strip().splitlines() if line.strip()
                           for call in _python_calls(line)])
    consume(r"<\|action_start\|>\s*<\|plugin\|>(.*?)<\|action_end\|>", _json_calls)
    consume(r"<dots_function_call>(.*?)</dots_function_call>", _json_calls)
    consume(r"<tool_calls>(.*?)</tool_calls>", _json_calls)
    def lfm(value):
        try:
            return _json_calls(value)
        except ValueError:
            return _python_calls(value)
    consume(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", lfm)
    consume(r"<\|tool_call_begin\|>(.*?)<\|tool_call_end\|>", _kimi_call)
    consume(r"<｜tool▁call▁begin｜>(.*?)<｜tool▁call▁end｜>", _deepseek_call)
    # Harmony headers may occur with the assistant prefix already in the prompt.
    text = _harmony_calls(text, calls)
    for opening, closing in (
        ("<|tool_calls_section_begin|>", "<|tool_calls_section_end|>"),
        ("<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>"),
        ("<｜tool_calls_begin｜>", "<｜tool_calls_end｜>"),
    ):
        if opening in text:
            start = text.index(opening)
            end = text.find(closing, start + len(opening))
            if end < 0 or text[start + len(opening):end].strip():
                raise ValueError("incomplete or invalid function calls section")
            text = text[:start] + text[end + len(closing):]
    text = strip_turn_tokens(text)
    candidate = text.strip()
    if candidate.startswith("functools["):
        calls.extend(_json_calls(candidate[len("functools"):]))
        return "", calls
    if candidate.startswith("[TOOL_CALLS]") and not candidate[len("[TOOL_CALLS]"):].lstrip().startswith(("{", "[")):
        calls.extend(_mistral_compact(candidate))
        return "", calls
    if profile == "qwen3_xml" and candidate.startswith("<function="):
        calls.extend(_tool_body(candidate, tools, gemma))
        return "", calls
    tagged = candidate.startswith(("<|python_tag|>", "[TOOL_CALLS]", "<|tools_prefix|>", "<function_call>"))
    if tagged:
        candidate = re.sub(r"^(?:<\|python_tag\|>|\[TOOL_CALLS\]|<\|tools_prefix\|>|<function_call>)\s*", "", candidate)
        candidate = candidate.removesuffix("<|tools_suffix|>").strip()
        calls.extend(_json_calls(candidate))
        text = ""
    elif candidate:
        fence = chr(96) * 3
        if profile == "json" and candidate.startswith(fence):
            match = re.fullmatch(re.escape(fence) + r"(?:json)?\s*\n(.*?)\n" + re.escape(fence), candidate, re.S)
            if match:
                candidate = match[1].strip()
        if profile == "pythonic" and (candidate.startswith("[") or re.match(r"[\w]+\(", candidate)):
            calls.extend(_python_calls(candidate))
            text = ""
        else:
            try:
                payload = json_value(candidate)
            except ValueError:
                payload = None
            if _call_envelopes(payload):
                calls.extend(_json_calls(candidate))
                text = ""
    if any(marker in text for marker in TOOL_STARTS) or (
        profile == "qwen3_xml" and "<function=" in text
    ):
        raise ValueError("model returned an incomplete or unsupported function call; increase the output limit")
    return text, calls


def _mistral_compact(text):
    calls = []
    while text.strip():
        match = re.match(r"\s*\[TOOL_CALLS\]([\w.-]+)\[ARGS\]\s*", text)
        if not match:
            raise ValueError("invalid compact Mistral function call")
        body = text[match.end():]
        _, end = json.JSONDecoder().raw_decode(body)
        calls.append({"name": match[1], "arguments": json_value(body[:end])})
        text = body[end:]
    return calls


def _harmony_calls(text, calls):
    parts, position = [], 0
    while match := HARMONY_TOOL_HEADER.search(text, position):
        parts.append(text[position:match.start()])
        name = (match[1] or match[2]).removeprefix("functions.")
        body = text[match.end():]
        offset = len(body) - len(body.lstrip())
        _, end = json.JSONDecoder().raw_decode(body, offset)
        closing = re.match(r"\s*<\|(?:call|ghissue)\|>", body[end:])
        if not closing:
            raise ValueError("incomplete Harmony function call")
        calls.append({"name": name, "arguments": json_value(body[:end])})
        position = match.end() + end + closing.end()
    parts.append(text[position:])
    return "".join(parts)


def _kimi_call(content):
    parts = content.split("<|tool_call_argument_begin|>", 1)
    if len(parts) != 2:
        raise ValueError("Kimi call requires an argument delimiter")
    name = re.fullmatch(r"(?:functions\.)?([\w.-]+):\d+", parts[0].strip())
    if not name:
        raise ValueError("Kimi call requires an explicit function name")
    return [{"name": name[1], "arguments": json_value(parts[1])}]


def _deepseek_call(content):
    parts = content.split("<｜tool▁sep｜>", 1)
    if len(parts) != 2:
        raise ValueError("DeepSeek call requires a function separator")
    name, body = parts
    if name.strip() == "function":
        name, separator, body = body.partition("\n")
        if not separator:
            raise ValueError("DeepSeek call requires JSON arguments")
    fence = chr(96) * 3
    body = body.strip()
    if body.startswith(fence):
        match = re.fullmatch(re.escape(fence) + r"json\s*\n(.*?)\n" + re.escape(fence), body, re.S)
        if not match:
            raise ValueError("invalid DeepSeek JSON argument fence")
        body = match[1]
    return [{"name": name.strip(), "arguments": json_value(body)}]

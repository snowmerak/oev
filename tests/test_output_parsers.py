import json

from fastapi.testclient import TestClient
import pytest

from oev.completion import parse_completion
from oev.generation import GenerationEngine, GenerationOptions, GenerationResult
from oev.output_parsers import REASONING_MARKERS, parameter_value
from oev.serve import ModelConfig, create_app
from test_api_compatibility import ThinkingTokenizer
from test_chat import Backend, service_for, sse_events


FUNCTION = {"name": "read_file", "parameters": {
    "type": "object", "properties": {
        "path": {"type": "string"}, "offset": {"type": "integer"},
        "limit": {"type": "integer"}, "verbose": {"type": "boolean"},
        "data": {"type": "object"}, "lines": {"type": "array"},
        "note": {"type": ["string", "null"]},
    }, "required": ["path"], "additionalProperties": False,
}, "strict": True}
TOOL = {"type": "function", "function": FUNCTION}
ARGS = '{"path":"README.md"}'
JSON_CALL = '{"name":"read_file","arguments":' + ARGS + '}'
QWEN = '<tool_call>\n<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>'
PROTOCOLS = [
    ("hermes", "<tool_call>" + JSON_CALL + "</tool_call>"),
    ("qwen3_xml", QWEN),
    ("glm45", "<tool_call>read_file\n<arg_key>path</arg_key><arg_value>README.md</arg_value></tool_call>"),
    ("glm47", '<tool_call>read_file' + ARGS + '</tool_call>'),
    ("minimax_m2", '<minimax:tool_call><invoke name="read_file"><parameter name="path">README.md</parameter></invoke></minimax:tool_call>'),
    ("deepseekv3", '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>read_file\n'
     + chr(96) * 3 + 'json\n' + ARGS + '\n' + chr(96) * 3 + '<｜tool▁call▁end｜><｜tool▁calls▁end｜>'),
    ("deepseekv31", '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>read_file<｜tool▁sep｜>' + ARGS + '<｜tool▁call▁end｜><｜tool▁calls▁end｜>'),
    ("deepseekv32", '<｜DSML｜function_calls><｜DSML｜invoke name="read_file"><｜DSML｜parameter name="path" string="true">README.md</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>'),
    ("kimi_k2", '<|tool_calls_section_begin|><|tool_call_begin|>functions.read_file:0<|tool_call_argument_begin|>' + ARGS + '<|tool_call_end|><|tool_calls_section_end|>'),
    ("gemma4", '<|tool_call>call:read_file{path:<|"|>README.md<|"|>}<tool_call|>'),
    ("functiongemma", '<start_function_call>call:read_file{path:<escape>README.md<escape>}<end_function_call>'),
    ("internlm2", '<|action_start|><|plugin|>{"name":"read_file","parameters":' + ARGS + '}<|action_end|>'),
    ("mistral", '[TOOL_CALLS] [' + JSON_CALL + ']'),
    ("llama3_json", '<|python_tag|>' + JSON_CALL + '<|eot_id|>'),
    ("granite", '<|tools_prefix|>[' + JSON_CALL + ']<|tools_suffix|>'),
    ("pythonic", '[read_file(path="README.md")]'),
    ("olmo3", '<function_calls>read_file(path="README.md")</function_calls>'),
    ("gpt-oss", '<|start|>assistant to=functions.read_file<|channel|>commentary<|message|>' + ARGS + '<|ghissue|>'),
    ("mistral", '[TOOL_CALLS]read_file[ARGS]' + ARGS),
    ("jamba", '<tool_calls>[' + JSON_CALL + ']</tool_calls>'),
    ("dots", '<dots_function_call>' + JSON_CALL + '</dots_function_call>'),
    ("phi4_mini_json", 'functools[' + JSON_CALL + ']'),
    ("lfm2", '<|tool_call_start|>[read_file(path="README.md")]<|tool_call_end|>'),
    ("lfm2", '<|tool_call_start|>[' + JSON_CALL + ']<|tool_call_end|>'),
    ("granite-20b-fc", '<function_call>' + JSON_CALL),
    ("step3", '<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>'
     '<steptml:invoke name="read_file"><steptml:parameter name="path">README.md</steptml:parameter>'
     '</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>'),
]


def parsed(raw, **kwargs):
    return parse_completion(GenerationResult(raw, 5, 20, "stop", raw_text=raw),
                            GenerationOptions(tools=(TOOL,), **kwargs))


@pytest.mark.parametrize("profile,raw", PROTOCOLS)
def test_native_calls(profile, raw):
    value = parsed(raw, tool_call_parser=profile)
    assert value.text == "" and len(value.calls) == 1
    assert value.calls[0]["function"]["name"] == "read_file"
    assert json.loads(value.calls[0]["function"]["arguments"]) == {"path": "README.md"}


@pytest.mark.parametrize("profile,raw", [item for item in PROTOCOLS if item[0] != "pythonic"])
def test_unambiguous_protocols_work_without_configuration(profile, raw):
    assert json.loads(parsed(raw).calls[0]["function"]["arguments"]) == {"path": "README.md"}


@pytest.mark.parametrize("profile,raw", PROTOCOLS)
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_native_calls_http_and_result_replay(profile, raw, path, stream):
    backend = Backend(raw)
    backend.tokenizer = ThinkingTokenizer()
    service, _ = service_for(backend, max_input_tokens=8192)
    service._generation_options["tool_call_parser"] = profile
    client = TestClient(create_app(service))
    body = {"model": "one-model", "stream": stream, "max_output_tokens": 2048,
            "input": "Read README.md", "tools": [{"type": "function", **FUNCTION}]}
    if path.endswith("completions"):
        body = {"model": "one-model", "stream": stream, "max_completion_tokens": 2048,
                "messages": [{"role": "user", "content": "Read README.md"}], "tools": [TOOL]}
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    if path.endswith("responses"):
        result = sse_events(response)[-1]["response"] if stream else response.json()
        call = next(item for item in result["output"] if item["type"] == "function_call")
        assert json.loads(call["arguments"]) == {"path": "README.md"}
        body["input"] = [{"role": "user", "content": "Read README.md"}, *result["output"],
                         {"type": "function_call_output", "call_id": call["call_id"], "output": "hello"}]
    else:
        if stream:
            events = sse_events(response)
            call = next(item["choices"][0]["delta"]["tool_calls"][0] for item in events[:-1]
                        if item.get("choices") and item["choices"][0]["delta"].get("tool_calls"))
            call.pop("index")
            assistant = {"role": "assistant", "content": None, "tool_calls": [call]}
        else:
            assistant = response.json()["choices"][0]["message"]
            call = assistant["tool_calls"][0]
        assert json.loads(call["function"]["arguments"]) == {"path": "README.md"}
        body["messages"] += [assistant, {"role": "tool", "tool_call_id": call["id"], "content": "hello"}]
    backend.text = "The file says hello."
    replay = client.post(path, json=body)
    assert replay.status_code == 200 and "invalid_model_output" not in replay.text


def test_qwen_argument_types_nested_json_literal_tags_and_local_refs():
    raw = '<tool_call><function=read_file>' + ''.join(
        '<parameter=' + key + '>\n' + value + '\n</parameter>' for key, value in {
            "path": "123", "offset": "2", "verbose": "true",
            "data": '{"literal":"<parameter=path>inner</parameter>","x":[1,{"a":false}]}',
            "lines": '[{"x":1},null]', "note": "null",
        }.items()) + '</function></tool_call>'
    arguments = json.loads(parsed(raw).calls[0]["function"]["arguments"])
    assert arguments == {"path": "123", "offset": 2, "verbose": True,
                         "data": {"literal": "<parameter=path>inner</parameter>", "x": [1, {"a": False}]},
                         "lines": [{"x": 1}, None], "note": None}
    root = {"$defs": {"count": {"type": "integer"}}}
    assert parameter_value("42", {"$ref": "#/$defs/count"}, root) == 42


@pytest.mark.parametrize("raw", [
    QWEN.replace("README.md", "README.md</parameter><parameter=path>other"),
    QWEN.replace("</parameter>", ""),
    QWEN.replace("read_file", "unknown"),
    QWEN.replace("<parameter=path>", "<parameter=offset>"),
    '<tool_call>{"name":"read_file","arguments":{"path":"a","path":"b"}}</tool_call>',
    '<tool_call>{"name":"read_file","arguments":{"path":NaN}}</tool_call>',
    '<minimax:tool_call><invoke name="read_file"><parameter name="offset">oops</parameter></invoke></minimax:tool_call>',
    '<|tool_call_begin|>0<|tool_call_argument_begin|>{}<|tool_call_end|>',
])
def test_invalid_calls_never_become_executable(raw):
    with pytest.raises(ValueError):
        parsed(raw)


def test_multiple_calls_choice_checks_and_surrounding_text():
    value = parsed("Before\n" + QWEN + "\n" + QWEN + "\nAfter")
    assert len(value.calls) == 2 and value.text == "Before\n\n\nAfter"
    assert value.calls[0]["id"] != value.calls[1]["id"]
    with pytest.raises(ValueError, match="multiple calls"):
        parsed(QWEN + QWEN, parallel_tool_calls=False)
    with pytest.raises(ValueError, match="required"):
        parsed("No calls", tool_choice="required")


def test_pythonic_literals_cannot_execute_code():
    raw = '[read_file(path="README.md", data={"a":true,"b":null}, lines=[1,false])]'
    arguments = json.loads(parsed(raw, tool_call_parser="pythonic").calls[0]["function"]["arguments"])
    assert arguments["data"] == {"a": True, "b": None} and arguments["lines"] == [1, False]
    for body in ["read_file(path=__import__('os').getcwd())", "obj.read_file(path='x')",
                 "read_file(**{'path':'x'})", "read_file('x')", "read_file(path='x',path='y')"]:
        with pytest.raises(ValueError):
            parsed("[" + body + "]", tool_call_parser="pythonic")
    assert parsed('[read_file(path="README.md")]').calls == []


@pytest.mark.parametrize("profile,opening,closing", REASONING_MARKERS)
def test_reasoning_delimiters_and_token_usage(profile, opening, closing):
    from oev.generation import reasoning_token_count
    tokenizer = ThinkingTokenizer()
    raw = opening + "Think carefully" + closing + "Answer"
    result = GenerationResult(raw, 2, 30, "stop", raw_text=raw)
    value = parse_completion(result, GenerationOptions(reasoning_effort="high", reasoning_parser=profile))
    assert value.reasoning == "Think carefully" and value.text == "Answer"
    assert reasoning_token_count(tokenizer.encode(raw), tokenizer, "") == len("Think carefully")


def test_tool_start_implicitly_ends_unclosed_thought_and_argument_tags_stay_literal():
    value = parsed("<think>Read the file\n" + QWEN, reasoning_effort="high")
    assert value.reasoning == "Read the file" and len(value.calls) == 1
    raw = '<tool_call>{"name":"read_file","arguments":{"path":"<think>x</think>"}}</tool_call>'
    value = parsed(raw, reasoning_effort="high")
    assert value.reasoning == "" and json.loads(value.calls[0]["function"]["arguments"])["path"] == "<think>x</think>"


def test_template_kwargs_config_override_and_effort_forwarding():
    backend = Backend(QWEN)
    backend.tokenizer = ThinkingTokenizer()
    engine = GenerationEngine(backend, tool_call_parser="qwen3_coder",
                              chat_template_kwargs={"terse": False})
    options = GenerationOptions(reasoning_effort="high", tools=(TOOL,),
                                chat_template_kwargs={"terse": True, "custom": "value"})
    prepared = engine.prepare([{"role": "user", "content": "Read"}], options)
    assert prepared.options.tool_call_parser == "qwen3_xml"
    assert backend.tokenizer.kwargs["reasoning_effort"] == "high"
    assert backend.tokenizer.kwargs["terse"] is True
    assert backend.tokenizer.kwargs["custom"] == "value"


@pytest.mark.parametrize("defaults,body_fields,expected", [
    ({"enable_thinking": True}, {}, True),
    ({"enable_thinking": True}, {"reasoning": {"effort": "none"}}, False),
    ({"thinking": False}, {"reasoning": {"effort": "high"}}, True),
    ({"reasoning_effort": "xhigh"}, {}, True),
    ({"reasoning_effort": "xhigh"}, {"reasoning": {"effort": "none"}}, False),
    ({}, {"reasoning": {"effort": "none"}, "chat_template_kwargs": {"thinking": True}}, True),
    ({}, {"reasoning": {"effort": "high"}, "chat_template_kwargs": {"enable_thinking": False}}, False),
    ({}, {"chat_template_kwargs": {"reasoning_effort": "off"}}, False),
])
def test_reasoning_configuration_precedence(defaults, body_fields, expected):
    from oev.api_requests import ResponsesRequest
    backend = Backend()
    backend.tokenizer = ThinkingTokenizer()
    engine = GenerationEngine(backend, chat_template_kwargs=defaults)
    body = ResponsesRequest.model_validate({"model": "one-model", "input": "Hi", **body_fields})
    prepared = engine.prepare(body.messages(), body.options())
    assert backend.tokenizer.kwargs["enable_thinking"] is expected
    assert backend.tokenizer.kwargs["thinking"] is expected
    assert (prepared.options.reasoning_effort != "none") is expected


@pytest.mark.parametrize("kwargs", [
    {"enable_thinking": "false"}, {"thinking": 1}, {"reasoning_effort": 3},
    {"enable_thinking": True, "thinking": False},
])
def test_template_thinking_controls_reject_invalid_types(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(model="checkpoint", chat_template_kwargs=kwargs)
    client = TestClient(create_app(service_for()[0]))
    response = client.post("/v1/responses", json={
        "model": "one-model", "input": "Hi", "chat_template_kwargs": kwargs,
    })
    assert response.status_code == 400


@pytest.mark.parametrize("kwargs", [{"tools": []}, {"tokenize": True}, {"messages": []}, {"chat_template": "bad"}])
def test_template_kwargs_cannot_replace_api_contract(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(model="checkpoint", chat_template_kwargs=kwargs)
    client = TestClient(create_app(service_for()[0]))
    response = client.post("/v1/responses", json={"model": "one-model", "input": "Hi", "chat_template_kwargs": kwargs})
    assert response.status_code == 400


def test_unknown_parser_rejected_and_model_specific_configuration_applied():
    with pytest.raises(ValueError):
        ModelConfig(model="checkpoint", tool_call_parser="missing")
    with pytest.raises(ValueError):
        ModelConfig(model="checkpoint", reasoning_parser="missing")
    backend = Backend("<|inner_prefix|>Plan<|inner_suffix|>Answer")
    backend.tokenizer = ThinkingTokenizer()
    service, _ = service_for(backend)
    service._generation_options["reasoning_parser"] = "apertus2509"
    client = TestClient(create_app(service))
    result = client.post("/v1/responses", json={"model": "one-model", "input": "Hi"}).json()
    assert [item["type"] for item in result["output"]] == ["reasoning", "message"]
    assert result["output"][1]["content"][0]["text"] == "Answer"


def test_gemma_alternative_turn_closer_and_dsml_json_arguments():
    assert parsed('<|tool_call>call:read_file{path:<|"|>README.md<|"|>}<turn|>').calls
    raw = '<｜DSML｜function_calls><｜DSML｜invoke name="read_file">' + ARGS + '</｜DSML｜invoke></｜DSML｜function_calls>'
    assert json.loads(parsed(raw).calls[0]["function"]["arguments"]) == {"path": "README.md"}


def test_protocol_markers_in_argument_strings_are_literal():
    argument = '{"path":"x</tool_call><think>y</think>"}'
    raw = '<tool_call>{"name":"read_file","arguments":' + argument + '}</tool_call>'
    assert json.loads(parsed(raw, reasoning_effort="high").calls[0]["function"]["arguments"]) == json.loads(argument)
    gemma = '<|tool_call>read_file{<|"|>path<|"|>:<|"|>x<tool_call|>y<|"|>}<tool_call|>'
    assert json.loads(parsed(gemma).calls[0]["function"]["arguments"]) == {"path": "x<tool_call|>y"}


@pytest.mark.parametrize("prefix", ["<|python_tag|>", "[TOOL_CALLS]", "<|tools_prefix|>"])
def test_turn_tokens_inside_unwrapped_json_arguments_stay_literal(prefix):
    value = {"path": "x<|im_end|><|end|><turn|>y"}
    raw = prefix + json.dumps({"name": "read_file", "arguments": value}) + "<|eot_id|>"
    assert json.loads(parsed(raw).calls[0]["function"]["arguments"]) == value


@pytest.mark.parametrize("prefix", ["", "<|python_tag|>", "[TOOL_CALLS]"])
def test_unwrapped_json_is_opaque_to_protocol_marker_scanning(prefix):
    arguments = {"path": '<tool_call><think>x</think>to=functions.fake<|message|><|call|>'}
    raw = prefix + json.dumps({"name": "read_file", "arguments": arguments}) + "<|eot_id|>"
    value = parsed(raw, reasoning_effort="high")
    assert value.reasoning == "" and value.text == ""
    assert json.loads(value.calls[0]["function"]["arguments"]) == arguments


@pytest.mark.parametrize("raw", [
    '{"name":"read_file","name":"read_file","arguments":{"path":"README.md"}}',
    '{"name":"read_file","arguments":{"path":"README.md","offset":NaN}}',
])
def test_unwrapped_invalid_json_envelopes_cannot_fall_back_to_prose(raw):
    with pytest.raises(ValueError):
        parsed(raw)


@pytest.mark.parametrize("header", [
    "<|start|>assistant to=functions.read_file<|channel|>commentary",
    "<|start|>assistant<|channel|>commentary to=functions.read_file",
    "<|start|>assistant to=functions.read_file<|channel|>analysis",
    "<|start|>assistant<|channel|>analysis to=functions.read_file",
    "to=functions.read_file",
])
def test_harmony_reasoning_followed_by_routed_tool_message(header):
    arguments = {"path": "x<|ghissue|><|call|>y"}
    raw = ("<|start|>assistant<|channel|>analysis<|message|>Plan<|end|>"
           + header + "<|constrain|>json<|message|>" + json.dumps(arguments) + "<|ghissue|>")
    value = parsed(raw, reasoning_effort="high", reasoning_parser="gpt-oss")
    assert value.reasoning == "Plan" and value.text == ""
    assert json.loads(value.calls[0]["function"]["arguments"]) == arguments


def test_implicit_thought_end_preserves_reasoning_and_channel_markers_in_arguments():
    path = "</think><|channel>final\n<channel|><|channel|>final<|message|><|START_RESPONSE|>"
    raw = "<think>Plan" + "<tool_call>" + json.dumps({
        "name": "read_file", "arguments": {"path": path},
    }) + "</tool_call>"
    value = parsed(raw, reasoning_effort="high")
    assert value.reasoning == "Plan"
    assert json.loads(value.calls[0]["function"]["arguments"]) == {"path": path}


@pytest.mark.parametrize("direct", [False, True])
def test_dsml_invoke_closing_marker_inside_json_string(direct):
    value = {"path": "x</｜DSML｜invoke>y"}
    arguments = json.dumps(value, ensure_ascii=False) if direct else (
        '<｜DSML｜parameter name="data" string="false">'
        + json.dumps(value, ensure_ascii=False) + '</｜DSML｜parameter>'
    )
    expected = value if direct else {"data": value}
    raw = '<｜DSML｜function_calls><｜DSML｜invoke name="read_file">' + arguments + '</｜DSML｜invoke></｜DSML｜function_calls>'
    function = {"type": "function", "function": {
        "name": "read_file", "parameters": FUNCTION["parameters"],
    }}
    result = parse_completion(GenerationResult(raw, 2, 10, "stop", raw_text=raw),
                              GenerationOptions(tools=(function,)))
    assert json.loads(result.calls[0]["function"]["arguments"]) == expected


def test_composed_schema_and_whitespace_preservation():
    root = {"allOf": [{"properties": {"offset": {"type": "integer"}}}],
            "properties": {"path": {"type": "string"}}}
    tool = {"type": "function", "function": {"name": "read_file", "parameters": root}}
    raw = '<tool_call><function=read_file><parameter=path>\n  text \n\n</parameter><parameter=offset>3</parameter></function></tool_call>'
    result = parse_completion(GenerationResult(raw, 2, 10, "stop", raw_text=raw), GenerationOptions(tools=(tool,)))
    assert json.loads(result.calls[0]["function"]["arguments"]) == {"path": "  text \n", "offset": 3}


@pytest.mark.parametrize("summary", ["auto", None])
def test_responses_reasoning_events_and_default_summary(summary):
    backend = Backend("<think>Plan</think>Answer")
    backend.tokenizer = ThinkingTokenizer()
    client = TestClient(create_app(service_for(backend)[0]))
    reasoning = {"effort": "high"}
    if summary is None:
        reasoning["summary"] = None
    response = client.post("/v1/responses", json={"model": "one-model", "input": "Hi",
                                                 "stream": True, "reasoning": reasoning})
    events = sse_events(response)
    assert next(event["delta"] for event in events if event["type"] == "response.reasoning_text.delta") == "Plan"
    summaries = [event for event in events if event["type"] == "response.reasoning_summary_text.delta"]
    assert len(summaries) == (1 if summary else 0)
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "Plan"


def test_reasoning_usage_ends_before_tool_arguments():
    from oev.generation import reasoning_token_count
    tokenizer = ThinkingTokenizer()
    raw = "<think>Plan" + QWEN
    assert reasoning_token_count(tokenizer.encode(raw), tokenizer, "") == 4


def test_reasoning_history_aliases_and_leading_developer_instructions():
    from oev.api_requests import ChatMessage
    for field in ["reasoning", "thinking", "reasoning_content"]:
        message = ChatMessage.model_validate({"role": "assistant", "content": "Answer", field: "Plan"})
        assert message.message()["reasoning_content"] == "Plan"
    backend = Backend()
    backend.tokenizer = ThinkingTokenizer()
    engine = GenerationEngine(backend)
    prepared = engine.prepare([{"role": "system", "content": "First"},
                               {"role": "developer", "content": "Second"},
                               {"role": "user", "content": "Hi"}], GenerationOptions())
    assert json.loads(backend.tokenizer.decode(prepared.input_ids).split("\nassistant:")[0])[0]["content"] == "First\n\nSecond"


def test_model_catalog_advertises_effective_context_and_reasoning_controls():
    backend = Backend()
    backend.tokenizer = ThinkingTokenizer()
    backend.tokenizer.chat_template = "<think> tools <function="
    backend.context_window = 131072
    client = TestClient(create_app(service_for(backend, max_input_tokens=32768)[0]))
    model = client.get("/v1/models").json()["data"][0]
    assert model["context_length"] == 32768 and model["context_window"] == 131072
    assert model["max_input_tokens"] == model["max_output_tokens"] == 32768
    assert model["tool_call_parser"] == "qwen3_xml"
    assert model["capabilities"]["reasoning"]["supported_efforts"] == [
        "none", "minimal", "low", "medium", "high", "xhigh", "max",
    ]


def test_hf_template_override_and_parser_configuration_load_through_cli(monkeypatch, tmp_path):
    from test_models import capture_server
    from oev.serve import main
    calls, launched = capture_server(monkeypatch)
    services = []
    monkeypatch.setattr("oev.serve.create_app", lambda service: services.append(service) or service)
    template = tmp_path / "chat.jinja"
    template.write_text("NATIVE TEMPLATE", encoding="utf-8")
    config = tmp_path / "models.json"
    config.write_text(json.dumps([{"model": "repo/model", "chat_template": str(template),
                                  "tool_call_parser": "qwen3_coder", "reasoning_parser": "qwen3",
                                  "chat_template_kwargs": {"terse": False}}]), encoding="utf-8")
    main(["--models-config", str(config)])
    service = services[0]
    assert calls[0][1]["chat_template"] is None
    assert service._generation_options["chat_template"] == "NATIVE TEMPLATE"
    assert service._generation_options["chat_template_kwargs"] == {"terse": False}

from fastapi.testclient import TestClient
import pytest

from oev.generation import template_options
from oev.reasoning import reasoning_metadata, template_controls
from oev.serve import create_app
from test_api_compatibility import ThinkingTokenizer
from test_chat import Backend, service_for


pytest.importorskip("jinja2")

SHARP_CONTROLS = """
{% set enable_thinking = enable_thinking if enable_thinking is defined else true %}
{% set _default_reasoning_effort = 'medium' %}
{% set _effort_raw = (reasoning_effort | string | lower) if reasoning_effort is defined else _default_reasoning_effort %}
{% if _effort_raw in ('none', 'off') %}
  {% set _initial_thinking = false %}{% set _initial_effort = 'medium' %}
{% elif _effort_raw in ('minimal', 'low') %}
  {% set _initial_effort = 'low' %}
{% elif _effort_raw in ('high', 'xhigh', 'max', 'extreme') %}
  {% set _initial_effort = 'xhigh' %}
{% else %}
  {% set _initial_effort = 'medium' %}
{% endif %}
{% set ns_state = namespace(effort=_initial_effort) %}
{{ ns_state.effort }}<think>
"""
ENUM_CONTROLS = """
{% if reasoning_effort not in ['low', 'medium', 'high'] %}
  {{ raise_exception('Invalid reasoning effort') }}
{% endif %}
{{ reasoning_effort }}<think>
"""


@pytest.mark.parametrize("template,expected", [
    ("{% if enable_thinking %}<think>{% endif %}", {
        "supported": True, "control": "toggle", "default_enabled": False,
    }),
    ("{% if thinking %}<think>{% endif %}", {
        "supported": True, "control": "toggle", "default_enabled": False,
    }),
    ("<think>", {"supported": True, "control": "fixed"}),
    ("{{ reasoning_effort }}<think>", {"supported": True}),
    ("{# reasoning_effort in ['low', 'high']; enable_thinking #}<think>", {
        "supported": True, "control": "fixed",
    }),
    ("{% if message.thinking %}<think>{% endif %}", {
        "supported": True, "control": "fixed",
    }),
    ("{% set thinking = message.thinking %}{{ thinking }}<think>", {
        "supported": True, "control": "fixed",
    }),
    ("{% set reasoning_effort = 'high' %}{{ reasoning_effort }}<think>", {
        "supported": True, "control": "fixed",
    }),
    (SHARP_CONTROLS, {
        "supported": True, "control": "effort",
        "supported_efforts": ["none", "low", "medium", "xhigh"], "default_effort": "none",
    }),
    (ENUM_CONTROLS, {
        "supported": True, "control": "effort", "supported_efforts": ["low", "medium", "high"],
    }),
])
def test_catalog_reports_native_reasoning_controls(template, expected):
    backend = Backend()
    backend.tokenizer = ThinkingTokenizer()
    backend.tokenizer.chat_template = template
    model = TestClient(create_app(service_for(backend)[0])).get("/v1/models").json()["data"][0]
    assert model["capabilities"]["reasoning"] == expected


def test_parser_configuration_does_not_invent_controls():
    defaults = template_options("none", {}, {}, False)
    assert reasoning_metadata("", "think", defaults) == {"supported": True, "control": "fixed"}
    assert reasoning_metadata(SHARP_CONTROLS, "none", defaults) == {"supported": False}
    assert reasoning_metadata("Plain chat", "auto", defaults) is None


def test_metadata_defaults_must_match_native_capability():
    assert reasoning_metadata(SHARP_CONTROLS, "auto", template_options(
        "none", {"reasoning_effort": "xhigh"}, {}, False,
    ))["default_effort"] == "xhigh"
    assert "default_effort" not in reasoning_metadata(
        SHARP_CONTROLS, "auto", template_options("none", {"reasoning_effort": "high"}, {}, False),
    )
    toggle = reasoning_metadata(
        "{% if enable_thinking %}<think>{% endif %}", "auto",
        template_options("none", {"enable_thinking": True}, {}, False),
    )
    assert toggle == {"supported": True, "control": "toggle", "default_enabled": True}


def test_template_discovery_is_conservative_and_never_executes_code():
    assert template_controls("{{ broken") == ((), False, None)
    assert template_controls("{{ text | custom_model_filter }}<think>") == ((), False, None)
    assert reasoning_metadata("{{ broken<think>", "auto", template_options(
        "none", {}, {}, False,
    )) == {"supported": True}
    assert template_controls("{{ raise_exception('Never execute discovery') }}<think>") == ((), False, False)
    unrelated = "{{ reasoning_effort }}{% if tool_choice in ['high', 'max'] %}Tools{% endif %}<think>"
    assert template_controls(unrelated) == ((), False, True)

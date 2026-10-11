import json
from types import SimpleNamespace

import pytest

from oev import DecisionEngine, SystemOneService
from oev import cli, serve
from oev.prompt_cache import parse_cache_size


@pytest.mark.parametrize("value,expected", [
    (0, 0), (1024, 1024), ("0", 0), ("0MiB", 0), ("1024", 1024),
    ("256MB", 256000000), ("256MiB", 268435456),
    ("1GB", 1000000000), ("1GiB", 1073741824),
    (" 256 mib ", 268435456), ("1.5GiB", 1610612736), ("0.5 MB", 500000),
    ("1B", 1), ("1KB", 1000), ("1KiB", 1024),
    ("1TB", 1000000000000), ("1TiB", 1099511627776),
    # Use exact decimal arithmetic, including tiny, whole-byte sizes.
    ("0.000001MB", 1),
])
def test_si_and_iec_sizes_are_normalized_to_exact_bytes(value, expected):
    assert parse_cache_size(value) == expected
    config = serve.ModelConfig(model="local", prompt_cache_max_size=value)
    assert config.prompt_cache_max_size == expected
    assert config.prompt_cache_max_bytes == expected


@pytest.mark.parametrize("value", [
    True, False, None, -1, 1.5, {}, [], "", "MB", "-1GiB", "1XB", "1Gi",
    "1e3MB", "NaN", "inf", "256MiB junk", "1.1B", "0.0000001MB", "1,024MB",
])
def test_invalid_sizes_fail_before_model_loading(value):
    with pytest.raises(ValueError, match="prompt_cache_max_size"):
        parse_cache_size(value)
    with pytest.raises(ValueError):
        serve.ModelConfig(model="never-download", prompt_cache_max_size=value)
    with pytest.raises(ValueError, match="prompt_cache_max_size"):
        DecisionEngine.from_pretrained("never-download", prompt_cache_max_size=value)


def test_conflicting_config_fields_are_rejected():
    with pytest.raises(ValueError):
        serve.ModelConfig(model="local", prompt_cache_max_size="256MiB", prompt_cache_max_bytes=1024)
    with pytest.raises(ValueError, match="only one"):
        DecisionEngine.from_pretrained("never-download", prompt_cache_max_size="256MiB", prompt_cache_max_bytes=1024)


def test_python_loader_and_server_pass_parsed_budget_to_backend(monkeypatch):
    from oev.hf import HuggingFaceBackend

    backend = SimpleNamespace(configure_prompt_cache=lambda value: budgets.append(value))
    budgets = []
    monkeypatch.setattr(HuggingFaceBackend, "load", lambda *args, **kwargs: backend)
    service = serve._load_service(serve.ModelConfig(model="local", prompt_cache_max_size="256MiB"))
    assert isinstance(service, SystemOneService)
    DecisionEngine.from_pretrained("local", prompt_cache_max_size="1GB")
    DecisionEngine.from_pretrained("local", prompt_cache_max_bytes=1024)
    assert budgets == [268435456, 1000000000, 1024]


@pytest.mark.parametrize("flag", ["--prompt-cache-max-size", "--prompt-cache-max-bytes"])
def test_server_cli_defaults_are_overridden_by_new_and_legacy_config(monkeypatch, tmp_path, flag):
    configs = []
    monkeypatch.setattr(serve, "_load_service", lambda config: configs.append(config) or SystemOneService(DecisionEngine(None), config.name))
    monkeypatch.setattr(serve, "create_app", lambda service: service)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    path = tmp_path / "models.json"
    path.write_text(json.dumps([
        {"model": "inherited"},
        {"model": "si", "prompt_cache_max_size": "256MB"},
        {"model": "legacy", "prompt_cache_max_bytes": 1024},
        {"model": "off", "prompt_cache_max_size": 0},
    ]))
    serve.main(["--models-config", str(path), flag, "1GiB"])
    assert [config.prompt_cache_max_size for config in configs] == [1073741824, 256000000, 1024, 0]


def test_decision_cli_accepts_units(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli.DecisionEngine, "from_pretrained", lambda *args, **kwargs: calls.append(kwargs))
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert cli.main(["--input", str(path), "--prompt-cache-max-size", "1.5GiB"]) == 0
    assert calls[0]["prompt_cache_max_size"] == 1610612736


@pytest.mark.parametrize("runner", [cli.main, serve.main])
def test_cli_rejects_invalid_size_before_loading(monkeypatch, runner):
    monkeypatch.setattr(DecisionEngine, "from_pretrained", lambda *args, **kwargs: pytest.fail("invalid size loaded a model"))
    with pytest.raises(SystemExit):
        runner(["--prompt-cache-max-size", "256XB"])

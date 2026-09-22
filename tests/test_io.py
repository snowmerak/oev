import json
from types import SimpleNamespace

import pytest

from oev import cli
from oev.io import decision_from_record, read_jsonl, write_jsonl


def test_jsonl_roundtrip_and_decision_conversion(tmp_path):
    path = tmp_path / "cases.jsonl"
    row = {
        "id": "korean",
        "state": "계정이 잠겼다",
        "question": "어느 팀?",
        "options": [
            {"id": "account", "description": "계정"},
            {"id": "billing", "description": "결제"},
        ],
    }
    write_jsonl(path, [row])
    assert read_jsonl(path) == [row]
    decision = decision_from_record(row)
    assert decision.state == "계정이 잠겼다"
    assert [option.id for option in decision.options] == ["account", "billing"]

    path.write_text('{"valid": true}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"cases.jsonl:2: invalid JSON"):
        read_jsonl(path)


def test_cli_preserves_record_ids_and_streams_results(tmp_path, monkeypatch):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    row = {
        "state": "state",
        "question": "question",
        "options": [
            {"id": "first", "description": "one"},
            {"id": "second", "description": "two"},
        ],
    }
    input_path.write_text(
        json.dumps({"id": "named", **row}) + "\n\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )

    class FakeEngine:
        def decide(self, decision):
            assert len(decision.options) == 2
            return SimpleNamespace(as_dict=lambda: {"selected_id": "first", "elapsed_seconds": 0.01})

    monkeypatch.setattr(cli.DecisionEngine, "from_pretrained", lambda *args, **kwargs: FakeEngine())
    assert cli.main(["--model", "fake", "--input", str(input_path), "--output", str(output_path)]) == 0
    assert [result["id"] for result in read_jsonl(output_path)] == ["named", "2"]

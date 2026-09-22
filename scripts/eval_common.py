"""Shared validation for recorded decision experiments."""

from __future__ import annotations

import math
from typing import Any


Record = dict[str, Any]
Pair = tuple[Record, Record]


def pair_records(inputs: list[Record], results: list[Record], *, require_complete: bool) -> list[Pair]:
    input_by_id = {row["id"]: row for row in inputs}
    result_by_id = {row["id"]: row for row in results}
    if len(input_by_id) != len(inputs) or len(result_by_id) != len(results):
        raise ValueError("duplicate case ID")
    if set(result_by_id) - set(input_by_id):
        raise ValueError("result contains unknown case IDs")
    if require_complete and set(input_by_id) != set(result_by_id):
        raise ValueError("input/result count or IDs differ")

    pairs = [(row, result_by_id[row["id"]]) for row in inputs if row["id"] in result_by_id]
    for question, result in pairs:
        option_ids = [option["id"] for option in question["options"]]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError(f"duplicate option ID: {question['id']}")
        logits = result["logits"]
        probabilities = result["probabilities"]
        if set(logits) != set(option_ids) or set(probabilities) != set(option_ids):
            raise ValueError(f"option/result mismatch: {question['id']}")
        if result["selected_id"] != max(option_ids, key=logits.get):
            raise ValueError(f"selected result is not the largest logit: {question['id']}")
        if any(not math.isfinite(value) for value in logits.values()):
            raise ValueError(f"non-finite logit: {question['id']}")
        if any(not math.isfinite(value) or value < 0 for value in probabilities.values()):
            raise ValueError(f"invalid probability: {question['id']}")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-9):
            raise ValueError(f"probabilities do not sum to one: {question['id']}")
        if "expected_id" in question and question["expected_id"] not in option_ids:
            raise ValueError(f"expected ID is not an option: {question['id']}")
        if "expected_letter" in question:
            index = option_ids.index(question["expected_id"])
            if question["expected_letter"] != chr(65 + index):
                raise ValueError(f"expected letter does not match option order: {question['id']}")
    return pairs


def markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")

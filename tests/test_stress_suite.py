from collections import Counter
import importlib.util
from pathlib import Path

from oev import Decision, Option
from oev.prompt import prepare


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_stress_suite.py"
SPEC = importlib.util.spec_from_file_location("build_stress_suite", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
challenge_cases = MODULE.challenge_cases
hard_semantic_cases = MODULE.hard_semantic_cases
none_position_cases = MODULE.none_position_cases
shuffled_cases = MODULE.shuffled_cases


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages) + "\nANSWER: "

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


def test_shuffle_suite_balances_the_correct_position_and_is_reproducible():
    base = {
        "id": "case",
        "expected_id": "target",
        "state": "state",
        "question": "question",
        "options": [{"id": f"option-{i}", "description": f"option {i}"} for i in range(19)]
        + [{"id": "target", "description": "the answer"}],
    }
    rows = shuffled_cases([base], trials=20, seed=7)
    assert rows == shuffled_cases([base], trials=20, seed=7)
    assert Counter(row["expected_letter"] for row in rows) == Counter("ABCDEFGHIJKLMNOPQRST")
    for row in rows:
        assert row["options"][ord(row["expected_letter"]) - 65]["id"] == "target"

    four_choice = {**base, "options": base["options"][:3] + base["options"][-1:]}
    four_rows = shuffled_cases([four_choice], trials=20, seed=7)
    assert Counter(row["expected_letter"] for row in four_rows) == {letter: 5 for letter in "ABCD"}
    assert len({tuple(option["id"] for option in row["options"]) for row in four_rows}) == 20


def test_option_ids_are_not_part_of_model_prompt():
    tokenizer = CharacterTokenizer()
    plain = Decision("state", "question", (Option("billing", "Billing"), Option("account", "Account")))
    opaque = Decision("state", "question", (Option("B7F29A10", "Billing"), Option("E04391CD", "Account")))
    assert prepare(tokenizer, plain, 4096) == prepare(tokenizer, opaque, 4096)


def test_challenge_cases_have_opaque_ids_and_labeled_answers():
    cases = challenge_cases() + hard_semantic_cases()
    assert len(challenge_cases()) == 16
    assert len(hard_semantic_cases()) == 1
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        option_ids = [option["id"] for option in case["options"]]
        assert len(set(option_ids)) == len(option_ids)
        assert case["expected_id"] in option_ids
        assert all(len(option_id) == 8 for option_id in option_ids)

    positions = none_position_cases(challenge_cases())
    assert [case["expected_letter"] for case in positions] == ["A", "B"]
    for case in positions:
        assert case["options"][ord(case["expected_letter"]) - 65]["id"] == case["expected_id"]

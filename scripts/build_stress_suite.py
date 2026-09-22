"""Generate balanced option permutations from labeled JSONL fixtures."""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path
import random

from oev.io import read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
BASE_FILES = (
    ROOT / "examples" / "gemma4-e2b-eval.jsonl",
    ROOT / "examples" / "gemma4-e2b-20-options.jsonl",
)
CHALLENGE_FILE = ROOT / "examples" / "stress-challenges.jsonl"
HARD_SEMANTIC_FILE = ROOT / "examples" / "stress-semantic20-hard.jsonl"


def shuffled_cases(base_rows: list[dict], trials: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for base in base_rows:
        options = base["options"]
        answer = base["expected_id"]
        correct = next(option for option in options if option["id"] == answer)
        distractors = [option for option in options if option["id"] != answer]

        # Balance the correct letter; use distinct orders where possible.
        positions = [i % len(options) for i in range(trials)]
        rng.shuffle(positions)
        variants = list(permutations(distractors)) if len(distractors) <= 3 else None
        if variants is not None:
            rng.shuffle(variants)
        used_at_position = {position: 0 for position in range(len(options))}

        for trial, position in enumerate(positions, 1):
            if variants is None:
                remaining = distractors.copy()
                rng.shuffle(remaining)
            else:
                remaining = list(variants[used_at_position[position] % len(variants)])
                used_at_position[position] += 1
            ordered = remaining[:position] + [correct] + remaining[position:]
            rows.append(
                {
                    "id": f"{base['id']}--shuffle-{trial:02d}",
                    "scenario": "shuffle",
                    "base_id": base["id"],
                    "trial": trial,
                    "expected_id": answer,
                    "expected_letter": chr(65 + position),
                    "state": base["state"],
                    "question": base["question"],
                    "options": ordered,
                }
            )
    return rows


def challenge_cases() -> list[dict]:
    return read_jsonl(CHALLENGE_FILE)


def hard_semantic_cases() -> list[dict]:
    return read_jsonl(HARD_SEMANTIC_FILE)


def none_position_cases(challenges: list[dict]) -> list[dict]:
    base = next(row for row in challenges if row["id"] == "none-positive")
    answer = next(option for option in base["options"] if option["id"] == base["expected_id"])
    others = [option for option in base["options"] if option["id"] != base["expected_id"]]
    return [
        {
            **base,
            "id": f"none-positive--position-{chr(65 + position)}",
            "scenario": "none_position",
            "expected_letter": chr(65 + position),
            "options": others[:position] + [answer] + others[position:],
        }
        for position in (0, 1)
    ]


def validate_labeled(rows: list[dict]) -> None:
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate case ID")
    for row in rows:
        option_ids = [option["id"] for option in row["options"]]
        if not 2 <= len(option_ids) <= 20 or len(set(option_ids)) != len(option_ids):
            raise ValueError(f"invalid options: {row['id']}")
        if row["expected_id"] not in option_ids:
            raise ValueError(f"expected ID is not an option: {row['id']}")
        if "expected_letter" in row:
            expected_letter = chr(65 + option_ids.index(row["expected_id"]))
            if row["expected_letter"] != expected_letter:
                raise ValueError(f"expected letter does not match option order: {row['id']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")

    base_rows = [row for path in BASE_FILES for row in read_jsonl(path)]
    challenges = challenge_cases()
    hard_semantics = hard_semantic_cases()
    if len(base_rows) != 11 or len(challenges) != 16 or len(hard_semantics) != 1:
        raise ValueError("expected 11 base, 16 challenge, and 1 hard semantic cases")
    validate_labeled(base_rows + challenges + hard_semantics)

    shuffles = shuffled_cases(base_rows, args.trials, args.seed)
    none_positions = none_position_cases(challenges)
    write_jsonl(ROOT / "examples" / "stress-shuffles.jsonl", shuffles)
    write_jsonl(
        ROOT / "examples" / "stress-shuffles-pilot.jsonl",
        (row for row in shuffles if row["base_id"] == "duplicate-charge"),
    )
    write_jsonl(ROOT / "examples" / "stress-none-positions.jsonl", none_positions)
    write_jsonl(ROOT / "examples" / "stress-followups.jsonl", none_positions + hard_semantics)
    print(f"wrote {len(shuffles)} shuffled cases and {len(none_positions) + len(hard_semantics)} follow-ups (seed={args.seed})")


if __name__ == "__main__":
    main()

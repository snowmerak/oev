"""Score a complete or partial oev stress run against its labeled JSONL input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from oev.io import read_jsonl

from eval_common import markdown_cell, pair_records


def summarize(rows: list[tuple[dict, dict]]) -> tuple[int, int, float | None]:
    total = len(rows)
    hits = sum(question["expected_id"] == result["selected_id"] for question, result in rows)
    elapsed = median(result["elapsed_seconds"] for _, result in rows) if rows else None
    return total, hits, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details", action="store_true", help="include every option logit in the report")
    args = parser.parse_args()

    inputs = read_jsonl(args.input)
    results = read_jsonl(args.output)
    pairs = pair_records(inputs, results, require_complete=False)
    models = {result["model"] for _, result in pairs}
    if len(models) > 1:
        raise ValueError(f"mixed model revisions: {models}")
    model_name = next(iter(models), "미실행")

    count, hits, overall_median = summarize(pairs)
    lines = [
        "# 의사결정 스트레스 평가",
        "",
        f"- 모델: `{model_name}`",
        f"- 입력: `{args.input}`",
        f"- 결과: `{args.output}`",
        f"- 완료: {count}/{len(inputs)}건",
        f"- 기대 답 일치: {hits}/{count}건" if count else "- 기대 답 일치: 결과 없음",
        f"- 추론 시간 중앙값: {overall_median:.3f}초" if overall_median is not None else "- 추론 시간 중앙값: 결과 없음",
        "- 기대 답은 평가 세트에 미리 지정한 단일 선택지입니다. 조건부 softmax 값은 보정된 정답 확률이 아닙니다.",
        "",
        "## 시나리오별 결과",
        "",
        "| 시나리오 | 완료 | 일치 | 중앙 추론 시간 |",
        "| --- | ---: | ---: | ---: |",
    ]
    scenarios = sorted({row["scenario"] for row in inputs})
    for scenario in scenarios:
        selected = [(q, r) for q, r in pairs if q["scenario"] == scenario]
        n, correct, duration = summarize(selected)
        total = sum(row["scenario"] == scenario for row in inputs)
        time_text = f"{duration:.3f}초" if duration is not None else "—"
        lines.append(f"| {scenario} | {n}/{total} | {correct}/{n} | {time_text} |")

    if any("base_id" in row for row in inputs):
        lines += [
            "", "## 원문항별 순서 변경 결과", "",
            "| 원문항 ID | 완료 | 일치 |",
            "| --- | ---: | ---: |",
        ]
        for base_id in dict.fromkeys(row.get("base_id") for row in inputs if "base_id" in row):
            selected = [(q, r) for q, r in pairs if q.get("base_id") == base_id]
            n, correct, _ = summarize(selected)
            total = sum(row.get("base_id") == base_id for row in inputs)
            lines.append(f"| `{base_id}` | {n}/{total} | {correct}/{n} |")

        lines += [
            "", "## 정답 위치별 결과", "",
            "| 정답 문자 | 완료 | 일치 |",
            "| :---: | ---: | ---: |",
        ]
        for letter in sorted({row["expected_letter"] for row in inputs}):
            selected = [(q, r) for q, r in pairs if q["expected_letter"] == letter]
            n, correct, _ = summarize(selected)
            total = sum(row["expected_letter"] == letter for row in inputs)
            lines.append(f"| {letter} | {n}/{total} | {correct}/{n} |")

    lines += [
        "", "## 문항별 결과", "",
        "| ID | 시나리오 | 정답 문자 | 기대 답 | 모델 답 | 기대 답 로그잇 | 모델 답 로그잇 | 추론 시간 |",
        "| --- | --- | :---: | --- | --- | ---: | ---: | ---: |",
    ]
    for question, result in pairs:
        expected = question["expected_id"]
        actual = result["selected_id"]
        flag = " ✓" if expected == actual else " **✗**"
        lines.append(
            f"| `{question['id']}` | {question['scenario']} | {question['expected_letter']} | "
            f"`{expected}` | `{actual}`{flag} | {result['logits'][expected]} | "
            f"{result['logits'][actual]} | {result['elapsed_seconds']:.3f}초 |"
        )

    misses = [(q, r) for q, r in pairs if q["expected_id"] != r["selected_id"]]
    lines += ["", "## 오답 상세", ""]
    if not misses:
        lines.append("완료된 문항에는 오답이 없습니다.")
    for question, result in misses:
        descriptions = {option["id"]: option["description"] for option in question["options"]}
        expected = question["expected_id"]
        actual = result["selected_id"]
        state = question["state"]
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        lines += [
            "", f"### {question['id']}", "",
            f"- 상태: {state_text}",
            f"- 질문: {question['question']}",
            f"- 기대: `{expected}` — {descriptions[expected]} (logit {result['logits'][expected]})",
            f"- 출력: `{actual}` — {descriptions[actual]} (logit {result['logits'][actual]})",
        ]

    if args.details:
        lines += ["", "## 선택지별 로그잇", ""]
        for question, result in pairs:
            state = question["state"]
            state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            lines += [
                f"### {question['id']}", "",
                f"- 상태: {state_text}",
                f"- 질문: {question['question']}",
                f"- 기대: `{question['expected_id']}` · 출력: `{result['selected_id']}`",
                "", "| 문자 | 선택지 ID | 설명 | 로그잇 | 조건부 softmax |",
                "| :---: | --- | --- | ---: | ---: |",
            ]
            for index, option in enumerate(question["options"]):
                option_id = option["id"]
                description = markdown_cell(option["description"])
                flag = " ✓" if option_id == result["selected_id"] else ""
                lines.append(
                    f"| {chr(65 + index)}{flag} | `{option_id}` | {description} | "
                    f"{result['logits'][option_id]} | {result['probabilities'][option_id]} |"
                )
            lines.append("")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.report}: {hits}/{count} correct, {count}/{len(inputs)} complete")


if __name__ == "__main__":
    main()

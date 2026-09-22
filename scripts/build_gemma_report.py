"""Build a readable report from the recorded Gemma 4 E2B JSONL runs."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

from oev.io import read_jsonl

from eval_common import markdown_cell, pair_records


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("기본 예시", "examples/decisions.jsonl", "results/gemma4-e2b.jsonl"),
    ("추가 평가", "examples/gemma4-e2b-eval.jsonl", "results/gemma4-e2b-eval.jsonl"),
    (
        "20개 선택지",
        "examples/gemma4-e2b-20-options.jsonl",
        "results/gemma4-e2b-20-options.jsonl",
    ),
)
OUTPUT = ROOT / "reports" / "gemma4-e2b-cpu.md"


def main() -> None:
    cases = []
    for group, input_name, result_name in SOURCES:
        inputs = read_jsonl(ROOT / input_name)
        results = read_jsonl(ROOT / result_name)
        for question, result in pair_records(inputs, results, require_complete=True):
            cases.append((group, question, result))

    models = {result["model"] for _, _, result in cases}
    if len(models) != 1:
        raise ValueError(f"inconsistent model revisions: {models}")
    scored = [(question, result) for _, question, result in cases if "expected_id" in question]
    correct = sum(question["expected_id"] == result["selected_id"] for question, result in scored)
    short_cases = [result for _, question, result in cases if len(question["options"]) < 20]
    long_cases = [result for _, question, result in cases if len(question["options"]) == 20]
    long_times = ", ".join(f"{result['elapsed_seconds']:.3f}" for result in long_cases)

    lines = [
        "# Gemma 4 E2B 의사결정 실험 보고서",
        "",
        "## 실행 조건과 요약",
        "",
        f"- 모델: `{next(iter(models))}`",
        "- 실행: CPU, bfloat16, JSONL 파일별 순차 추론. 모델은 각 파일 실행 시 한 번 로드했습니다.",
        "- 방식: 각 문항에서 채팅 프롬프트를 한 번 forward하고, 다음 위치의 선택지 문자 A–T에 해당하는 로그잇만 읽었습니다. Gemma 4에서는 선택지의 출력 가중치 행만 투영했습니다.",
        f"- 문항: 총 {len(cases)}건. 기대 답이 입력 파일에 기록된 {len(scored)}건 중 {correct}건 일치. 기본 예시 2건은 기대 답 필드 없이 별도로 제시합니다.",
        f"- 2–4개 선택지 {len(short_cases)}건: 추론 시간 중앙값 {median(result['elapsed_seconds'] for result in short_cases):.3f}초, 범위 {min(result['elapsed_seconds'] for result in short_cases):.3f}–{max(result['elapsed_seconds'] for result in short_cases):.3f}초.",
        f"- 20개 선택지 {len(long_cases)}건: 추론 시간 {long_times}초.",
        "- 이 수치는 소규모로 직접 구성한 문항의 결과입니다. 일반적인 정확도나 안정적인 성능 벤치마크로 해석하기 어렵습니다. 실행 시간은 입력 길이와 당시 CPU 상태의 영향을 받습니다.",
        "- `probabilities`는 목록에 있는 선택지 문자 로그잇만 softmax한 조건부 값이며, 정답 확률이나 보정된 신뢰도가 아닙니다.",
        "- 기본 첫 문항의 A/B 로그잇은 동일한 체크포인트의 일반 `forward(logits_to_keep=1)` 출력과 각각 정확히 일치했습니다.",
        "",
        "원본 입력과 출력:",
        "",
    ]
    for group, input_name, result_name in SOURCES:
        lines.append(f"- {group}: [`{input_name}`](../{input_name}), [`{result_name}`](../{result_name})")
    lines += ["", "## 문항별 요약", "", "| 번호 | ID | 선택지 | 입력 토큰 | 기대 답 | 모델 답 | 정답 여부 | 1·2위 로그잇 차이 | 추론 시간 |", "| ---: | --- | ---: | ---: | --- | --- | --- | ---: | ---: |"]
    for index, (_, question, result) in enumerate(cases, 1):
        sorted_logits = sorted(result["logits"].values(), reverse=True)
        expected = question.get("expected_id", "—")
        match = "일치" if expected != "—" and expected == result["selected_id"] else "—"
        lines.append(
            f"| {index} | `{question['id']}` | {len(question['options'])} | {result['input_tokens']} | "
            f"`{expected}` | `{result['selected_id']}` | {match} | "
            f"{sorted_logits[0] - sorted_logits[1]:.6g} | {result['elapsed_seconds']:.3f}초 |"
        )

    lines += ["", "## 문항과 선택지별 로그잇", ""]
    for index, (group, question, result) in enumerate(cases, 1):
        state = question["state"]
        if isinstance(state, str):
            state_text = state
        else:
            state_text = json.dumps(state, ensure_ascii=False, sort_keys=True)
        winner = result["selected_id"]
        expected = question.get("expected_id")
        selected_index = next(i for i, option in enumerate(question["options"]) if option["id"] == winner)
        lines += [
            f"### {index}. {question['id']}",
            "",
            f"- 구분: {group}",
            f"- 상태: {state_text}",
            f"- 문항: {question['question']}",
            f"- 결과: **{chr(65 + selected_index)} / `{winner}`**"
            + (f" · 기대 답: `{expected}`" if expected else " · 기대 답: 미기록"),
            f"- 입력 {result['input_tokens']}토큰 · 추론 {result['elapsed_seconds']:.3f}초",
            "",
            "| 문자 | 선택지 ID | 설명 | 로그잇 | 조건부 softmax |",
            "| :---: | --- | --- | ---: | ---: |",
        ]
        for option_index, option in enumerate(question["options"]):
            option_id = option["id"]
            marker = " **✓**" if option_id == winner else ""
            lines.append(
                f"| {chr(65 + option_index)}{marker} | `{markdown_cell(option_id)}` | {markdown_cell(option['description'])} | "
                f"{result['logits'][option_id]} | {result['probabilities'][option_id]} |"
            )
        lines += ["", f"프롬프트 SHA-256: `{result['prompt_sha256']}`", ""]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(cases)} cases, {correct}/{len(scored)} scored correct)")


if __name__ == "__main__":
    main()

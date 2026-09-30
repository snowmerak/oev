"""Run labeled JSONL cases sequentially through a System One HTTP server."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import subprocess
import time

import httpx


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.output.exists() or args.report.exists():
        parser.error("output and report must be new files")
    source = args.input.read_bytes()
    rows = [json.loads(line) for line in source.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        parser.error("input must contain at least one case")
    requests = []
    for row in rows:
        criteria = {option["id"]: option["description"] for option in row["options"]}
        if len(criteria) != len(row["options"]) or row["expected_id"] not in criteria:
            parser.error(f"invalid option IDs in {row['id']}")
        requests.append({
            "state": row["state"],
            "questions": {"result": {
                "type": "choice", "instructions": row["question"], "criteria": criteria,
            }},
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=180.0, trust_env=False) as client:
        discovery = client.get("/v1/models")
        discovery.raise_for_status()
        model = discovery.json()["models"][0]["name"]
        metadata = {
            "base_url": args.base_url, "model": model,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "input": str(args.input), "input_sha256": hashlib.sha256(source).hexdigest(),
            "client_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "concurrency": 1, "warmup_requests": args.warmup,
            "transport": "httpx persistent client; end-to-end HTTP seconds; no retries",
            "server_revision_hardware_dtype": "not exposed by API",
            "prompt_mapping": "System One choice includes option IDs and descriptions",
        }
        for request in requests:
            request["model"] = model
        for _ in range(args.warmup):
            response = client.post("/v1/systemone", json=requests[0])
            response.raise_for_status()
        print(f"Warm-up complete; running {len(rows)} cases on {model}", flush=True)
        started = time.perf_counter()
        with args.output.open("x", encoding="utf-8") as output:
            for index, (row, request) in enumerate(zip(rows, requests), 1):
                record = {
                    "id": row["id"], "base_id": row.get("base_id", row["id"]),
                    "option_count": len(row["options"]), "expected_id": row["expected_id"],
                    "expected_position": next(i for i, o in enumerate(row["options"]) if o["id"] == row["expected_id"]),
                    "request": request, "metadata": metadata,
                }
                tick = time.perf_counter()
                try:
                    response = client.post("/v1/systemone", json=request)
                    record["http_seconds"] = time.perf_counter() - tick
                    record["status"] = response.status_code
                    response.raise_for_status()
                    data = response.json()
                    record["response"] = data
                    answer = data["answers"]["result"]
                    probabilities = answer["probabilities"]
                    expected_keys = set(request["questions"]["result"]["criteria"])
                    if (set(probabilities) != expected_keys or answer["choice"] not in expected_keys
                            or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
                            or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6)):
                        raise ValueError("invalid answer or probability distribution")
                    record["correct"] = answer["choice"] == row["expected_id"]
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                    record.setdefault("http_seconds", time.perf_counter() - tick)
                    record["error"] = str(exc)
                    record["correct"] = False
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                records.append(record)
                if index % 20 == 0 or index == len(rows):
                    print(f"{index}/{len(rows)} complete; {sum(r['correct'] for r in records)} correct; "
                          f"{sum('error' in r for r in records)} errors; {time.perf_counter() - started:.1f}s", flush=True)
        duration = time.perf_counter() - started

    valid = [r for r in records if "error" not in r]
    lines = [
        "# System One HTTP 순서 변경 평가", "",
        f"- 서버: `{args.base_url}` · 모델: `{model}`",
        f"- 실행 시작(UTC): {metadata['started_at']}",
        f"- 입력: `{args.input}` · SHA-256: `{metadata['input_sha256']}`",
        f"- 클라이언트 기준 커밋: `{metadata['client_revision']}` (추가한 benchmark_systemone.py로 실행)",
        f"- 워밍업 {args.warmup}회 제외, 동시 요청 1개, 연결 재사용, 자동 재시도 없음",
        "- 시간은 클라이언트에서 측정한 HTTP 왕복 시간이며 서버 내부 추론 시간과 다릅니다.",
        "- 서버의 하드웨어·dtype·코드/모델 리비전과 다른 작업의 부하는 API에서 확인할 수 없습니다.",
        "- System One API는 선택지 ID도 모델에 표시합니다. 기존 로컬 평가와 프롬프트 및 모델이 다릅니다.",
        "- 기존 입력의 대문자 위치 표기 대신 현재 소문자·숫자 라벨로 실행합니다.",
        "- p95는 nearest-rank 방식입니다. 순차 실행 처리량이며 동시 처리 최대 성능은 아닙니다.",
        f"- 완료: {len(records)}/{len(rows)} · 정답: {sum(r['correct'] for r in records)} · 오류: {len(records) - len(valid)}",
        f"- 총 측정 구간: {duration:.3f}초 · 처리량: {len(records) / duration:.3f}요청/초",
        f"- 원본 요청·응답: `{args.output}`", "",
        "| 선택지 수 | 정상 응답 | 정답 | 중앙값(초) | p95(초) | 최소–최대(초) | 입력 토큰 중앙값 |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    groups = {"전체": valid}
    groups.update({str(n): [r for r in valid if r["option_count"] == n] for n in sorted({r["option_count"] for r in records})})
    for name, group in groups.items():
        if not group:
            continue
        times = [r["http_seconds"] for r in group]
        tokens = [r["response"]["usage"]["input_tokens"] for r in group]
        lines.append(f"| {name} | {len(group)} | {sum(r['correct'] for r in group)}/{len(group)} | "
                     f"{median(times):.3f} | {percentile(times, 0.95):.3f} | {min(times):.3f}–{max(times):.3f} | {median(tokens):g} |")
    lines += ["", "## 원문항별 결과", "", "| 원문항 | 정답/전체 |", "| --- | ---: |"]
    by_base = defaultdict(list)
    for record in records:
        by_base[record["base_id"]].append(record)
    for name, group in by_base.items():
        lines.append(f"| {name} | {sum(r['correct'] for r in group)}/{len(group)} |")
    lines += ["", "## 오답 및 오류", ""]
    misses = [r for r in records if not r["correct"]]
    for record in misses:
        actual = record.get("response", {}).get("answers", {}).get("result", {}).get("choice")
        lines.append(f"- `{record['id']}`: 기대 `{record['expected_id']}`, 결과 `{actual}`, 오류 `{record.get('error', '없음')}`")
    if not misses:
        lines.append("없음.")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()

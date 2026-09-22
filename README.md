# oev

`oev`는 언어 모델로 여러 선택지 중 하나를 고르는 작은 Python 라이브러리입니다. `state`, `question`, `options`를 프롬프트로 만들고 모델을 한 번 실행한 뒤, 다음 토큰 위치에서 선택지 문자 `A`–`T`의 logit을 읽습니다. 답변 문장이나 JSON을 생성하지 않습니다. [SemIf](https://github.com/TheoLeeCJ/SemIf)의 직접 logit 판독 방식을 참고했습니다.

기본 모델은 `google/gemma-4-E2B-it`입니다. `--model`에 다른 Hugging Face causal LM의 모델 ID 또는 Transformers 형식의 로컬 체크포인트 경로를 넣을 수 있습니다. GGUF 파일은 현재 지원하지 않습니다.

## 빠른 시작

Python 3.11과 `uv` 기준입니다.

```powershell
uv sync --locked --extra hf --extra test
uv run --locked --extra hf oev --model google/gemma-4-E2B-it --device cpu --dtype bfloat16 --input examples/decisions.jsonl --output results/demo.jsonl
```

모델 가중치가 캐시에 없으면 첫 실행에 다운로드합니다. CPU에서도 실행되지만 Gemma 4 E2B는 메모리와 시간이 많이 듭니다. 먼저 CLI와 모델 교체만 확인하려면 작은 모델을 사용할 수 있습니다.

```powershell
uv run --locked --extra hf oev --model HuggingFaceTB/SmolLM2-135M-Instruct --device cpu --input examples/decisions.jsonl --output results/smoke.jsonl
```

이 작은 모델은 동작 확인용입니다. 분류 품질은 Gemma 4 E2B 결과를 대신하지 않습니다. `uv.lock`은 운영체제에 따라 PyTorch 배포본을 선택합니다.

| 운영체제 | PyTorch 배포본 | 사용 가능한 `--device` |
| --- | --- | --- |
| Windows | CUDA 13.0 빌드 | `cuda` 또는 `cpu` |
| macOS | PyPI 빌드 | MPS 지원 기기에서는 `mps`, 그 외에는 `cpu` |
| Linux | PyPI 빌드 | CUDA 사용 가능 시 `cuda`, 그 외에는 `cpu` |

잠금 파일은 같은 운영체제 안에서 GPU 유무를 구분하지 않습니다. 따라서 GPU가 없는 Windows PC에도 CUDA 빌드가 설치되지만 `--device cpu`로 실행할 수 있습니다. `--device auto`는 사용 가능한 CUDA, MPS, CPU 순으로 선택하며 VRAM 용량을 확인하지 않습니다. [Google의 메모리 표](https://ai.google.dev/gemma/docs/core)에 따르면 Gemma 4 E2B의 BF16 추론에는 약 11.4GB가 필요합니다. 6GB GPU에서는 기본 CLI의 `--device cuda` 대신 아래의 CPU 명령 또는 GPU와 CPU를 함께 쓰는 [하이브리드 실행](reports/gemma4-e2b-stress-challenges-gpu-hybrid.md)을 사용하세요.

## 입력과 출력

입력은 한 줄에 하나의 JSON 객체를 담은 JSONL입니다.

```json
{"id":"route-1","state":"비밀번호 재설정 후에도 계정이 잠겨 있다.","question":"어느 팀으로 보내야 하나?","options":[{"id":"account","description":"계정 접근 지원"},{"id":"billing","description":"결제 지원"}]}
```

| 입력 필드 | 의미 |
| --- | --- |
| `id` | 결과를 연결할 기록 ID. 생략하면 파일의 문항 순번을 사용합니다. |
| `state` | 판단에 필요한 문자열, JSON 객체 또는 배열. |
| `question` | 선택 기준을 설명하는 문장. |
| `options` | 2–20개의 `{ "id", "description" }` 객체. 순서대로 A–T에 대응합니다. |

결과도 JSONL로 쓰며, 핵심 필드는 다음과 같습니다.

| 출력 필드 | 의미 |
| --- | --- |
| `selected_id` | 가장 높은 logit을 얻은 선택지의 ID. |
| `logits` | 각 선택지 문자 토큰의 원래 logit. |
| `probabilities` | 제시된 선택지 logit끼리만 softmax한 값. |
| `input_tokens`, `elapsed_seconds` | 프롬프트 토큰 수와 해당 문항의 처리 시간. |
| `prompt_sha256`, `model` | 프롬프트 해시와 모델 식별자. 가능한 경우 체크포인트 리비전도 포함합니다. |

`probabilities`는 **정답 확률이나 보정된 신뢰도**가 아닙니다. 실제 평가에서 정답 선택지가 없는 긍정 리뷰를 모델이 `중립`으로 잘못 골랐고, 그 선택지 내부 확률은 0.9999 이상이었습니다.

후보의 `id`는 모델 프롬프트에 넣지 않고 결과 매핑에만 사용합니다. 모델이 읽는 것은 `description`과 그 앞에 붙인 선택지 문자입니다. 같은 설명과 순서를 유지한 채 ID만 바꿔도 모델 입력은 같습니다.

## Python API

```python
from oev import Decision, DecisionEngine, Option

engine = DecisionEngine.from_pretrained(
    "google/gemma-4-E2B-it", device="cpu", dtype="bfloat16"
)
result = engine.decide(Decision(
    state="비밀번호 재설정 후에도 계정이 잠겨 있다.",
    question="어느 팀으로 보내야 하나?",
    options=(
        Option("account", "계정 접근 지원"),
        Option("billing", "결제 지원"),
    ),
))
print(result.selected_id, result.logits, result.elapsed_seconds)
```

`DecisionEngine`은 선택지 logit을 반환하는 백엔드와 프롬프트·결과 매핑을 분리합니다. `from_pretrained`는 Hugging Face 백엔드를 로드하고, 같은 API에 호환 모델의 ID나 로컬 경로를 지정할 수 있습니다. 다른 런타임을 붙일 때는 `LogitBackend` 프로토콜의 `tokenizer`, `model_name`, `selected_logits(input_ids, answer_token_ids)`를 구현해 `DecisionEngine(backend)`에 전달하면 됩니다.

## 계산 방식과 범위

Gemma 4 백엔드는 마지막 은닉 상태를 **선택지 토큰에 해당하는 출력 가중치 행에만** 투영합니다. 전체 어휘에 대한 최종 logit 투영을 생략하지만, 입력 프롬프트에 대한 모델 forward 자체는 수행합니다. 다른 호환 모델은 지원되는 경우 `logits_to_keep=1`로 마지막 위치만 계산한 뒤 선택지 logit을 읽습니다.

선택지 문자 하나가 정확히 한 토큰인지, 프롬프트 끝에서 토큰이 합쳐지지 않는지 실행 전에 검사합니다. 조건을 만족하지 않으면 오류를 냅니다. 입력은 자동으로 자르지 않으며 기본 한도는 4096토큰입니다. 순서대로 한 문항씩 실행하고, 한 파일 안에서는 모델을 한 번 로드해 재사용합니다. SemIf의 prefix cache 최적화는 아직 구현하지 않았습니다.

## 평가 재현

현재 CPU 결과와 선택지별 logit은 [기본 13문항 보고서](reports/gemma4-e2b-cpu.md)와 [스트레스 평가 요약](reports/gemma4-e2b-stress-summary.md)에 있습니다. 6GB GPU와 CPU RAM을 함께 사용한 실행 결과는 [GPU 하이브리드 16문항 보고서](reports/gemma4-e2b-stress-challenges-gpu-hybrid.md)에 있습니다. 직접 만든 소규모 문항이므로 결과 비율을 일반적인 정확도로 해석하면 안 됩니다.

스트레스 평가의 원본 문항은 `examples/stress-challenges.jsonl`과 `examples/stress-semantic20-hard.jsonl`에 있습니다. 아래 명령은 기존 11문항을 20회씩 순서 변경한 220건과 후속 문항을 고정 시드로 생성합니다.

평가 파일의 `expected_id`, `expected_letter`, `scenario`는 채점용 메타데이터입니다. `oev` 추론에는 `state`, `question`, `options`만 사용합니다.

```powershell
uv run --locked --extra hf python scripts/build_stress_suite.py
```

16개 주요 스트레스 문항을 실행하고 채점하려면:

```powershell
uv run --locked --extra hf oev --model google/gemma-4-E2B-it --device cpu --dtype bfloat16 --input examples/stress-challenges.jsonl --output results/gemma4-e2b-stress-challenges-cpu.jsonl
uv run --locked --extra hf python scripts/score_stress.py --input examples/stress-challenges.jsonl --output results/gemma4-e2b-stress-challenges-cpu.jsonl --report reports/gemma4-e2b-stress-challenges-cpu.md --details
```

`--details`는 보고서에 각 문항의 모든 선택지 logit을 넣습니다. 채점 스크립트는 실행 도중의 일부 결과 파일도 읽어 완료 건수와 현재 정답 수를 보여줍니다. 기존 CPU 실행에서는 이 16건 중 15건이 기대 답과 일치했습니다.

순서 변경 220건 중 CPU에서 완료한 것은 중복 청구 한 문항의 서로 다른 순서 20건입니다. 전체 220건을 GPU에서 실행하려면:

```powershell
uv run --locked --extra hf oev --model google/gemma-4-E2B-it --device cuda --dtype bfloat16 --input examples/stress-shuffles.jsonl --output results/gemma4-e2b-stress-shuffles-gpu.jsonl
uv run --locked --extra hf python scripts/score_stress.py --input examples/stress-shuffles.jsonl --output results/gemma4-e2b-stress-shuffles-gpu.jsonl --report reports/gemma4-e2b-stress-shuffles-gpu.md
```

이 명령에는 CUDA용 PyTorch와 bfloat16을 지원하는 GPU가 필요합니다. 2개 또는 3개 선택지 문항은 가능한 순서가 20가지보다 적어 일부 입력이 반복됩니다. 4개 선택지 문항은 서로 다른 순서 20개이고, 20개 선택지 문항은 정답이 A–T 각 위치에 한 번씩 놓입니다. 복합 의도 문항에는 우선순위나 수동 분리 기준을 명시해 기대 답을 하나로 정했습니다.

## 검증

```powershell
uv run --locked --extra hf --extra test pytest -q
uv lock --check
```

# Gemma 4 E2B 스트레스 평가 요약

## 조건

- 모델: `google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7`
- 실행: CPU, bfloat16, 파일 안에서 한 문항씩 순차 추론
- 평가 입력 생성 시드: `20260922`
- 기록된 39회 추론의 프롬프트 SHA-256, 입력 토큰 수, 선택지 토큰 수를 현재 입력 파일과 다시 대조했습니다.

## 결과

| 실험 | 일치 | 관찰 |
| --- | ---: | --- |
| 표현이 비슷한 보기 | 5/5 | 중복 청구, 환불, 계정 메일, 앱 종료, 배송 지연 구분 |
| 부정·대조 표현 | 3/3 | 부정된 사유 대신 실제 문제 선택 |
| 정답을 빼고 ‘해당 없음’ 제시 | 2/3 | 긍정 리뷰를 중립으로 오분류 |
| 복합 의도 | 2/2 | 문항에 명시한 우선순위·수동 분리 기준 적용 |
| 한영 혼합 | 2/2 | 양방향 상태·기준 조합 통과 |
| 결제·환불 20개 보기 | 1/1 | 승인 후 카드 반영 지연 선택 |
| 중복 청구 문항의 서로 다른 순서 | 20/20 | 정답 A·B·C·D 위치 각각 5/5 |
| ‘해당 없음’ 위치 A·B 재시험 | 0/2 | 두 위치 모두 다시 중립 선택 |
| 매우 가까운 환불 사유 20개 | 1/1 | 정답 L 선택, 입력 725토큰, 177.544초 |

순서 변경 20건에서 정답 로그잇은 **16.875–21.125**로 위치에 따라 달랐지만 선택 결과는 유지됐습니다. 최소 1·2위 로그잇 차이는 **13.1875**였습니다. 이 결과는 중복 청구 한 문항에 대한 것이며 다른 10문항의 순서 안정성을 뜻하지 않습니다.

### 확인된 실패: 정답 후보가 없을 때

긍정 리뷰에서 ‘긍정’을 빼고 `부정`, `중립`, `위 항목 중 해당 없음`을 제시했습니다. 정답으로 지정한 ‘해당 없음’이 A·B·C 어느 위치에 있어도 모델은 `중립`을 골랐습니다.

| ‘해당 없음’ 위치 | ‘해당 없음’ logit | ‘중립’ logit | 선택지 내부 중립 softmax |
| :---: | ---: | ---: | ---: |
| A | 5.46875 | 17.0 | 0.9999643147498757 |
| B | 5.375 | 14.9375 | 0.9999249037357084 |
| C | 2.578125 | 15.25 | 0.9999966643491427 |

이 softmax 값은 **제시한 보기 안에서만** 계산한 값입니다. 정답 확률이나 보정된 신뢰도로 해석할 수 없으며, 이 사례에서는 높은 값과 오답이 함께 나타났습니다. 다음 개선 대상으로는 ‘해당 없음’을 선택해야 하는 조건을 별도로 평가하거나, 낮은 적합성에서 수동 검토로 보내는 기준을 독립 데이터로 검증하는 일이 적절합니다.

### 20개 가까운 의미의 보기

환불 승인·결제 수단·반영 상태가 조금씩 다른 20개 설명 가운데 모델은 `승인된 전액 카드 환불이 카드 명세서에 아직 반영되지 않음`을 골랐습니다. 정답 logit은 **23.0**, 2위 `카드 명세서에 중복 반영됨`은 **10.1875**였습니다.

## 11문항 × 20회 순서 변경 세트

기존 기대 답이 있는 11문항에 대해 **220개 입력**을 생성했습니다. 4개 보기 문항은 20회가 모두 서로 다른 순서이고 정답 위치 A–D가 각 5회입니다. 20개 보기 문항은 정답 위치 A–T가 각 1회입니다. 2·3개 보기 문항은 가능한 순서가 적어 동일 순서가 반복됩니다.

이번 CPU 실행에서는 그중 중복 청구 문항의 20개 순서를 완료했습니다. 나머지 **200건은 아직 모델 추론을 실행하지 않았습니다.** 전체 세트는 CUDA 사용이 가능한 환경에서 다음 명령으로 실행하고 채점할 수 있습니다.

```powershell
uv run --locked --extra hf oev --model google/gemma-4-E2B-it --device cuda --dtype bfloat16 --input examples/stress-shuffles.jsonl --output results/gemma4-e2b-stress-shuffles-gpu.jsonl
uv run --locked --extra hf python scripts/score_stress.py --input examples/stress-shuffles.jsonl --output results/gemma4-e2b-stress-shuffles-gpu.jsonl --report reports/gemma4-e2b-stress-shuffles-gpu.md
```

CUDA용 PyTorch와 bfloat16 지원 GPU가 필요합니다. 후보 ID는 프롬프트에 포함되지 않고 결과 매핑에만 쓰입니다. 같은 설명과 순서를 유지한 채 `billing`을 생소한 ID로 바꿔도 모델에 들어가는 프롬프트가 같음을 테스트로 확인했습니다. 따라서 ID 교체만으로는 모델의 선택 판단을 시험할 수 없습니다.

## 상세 자료

| 자료 | 입력 | 원본 출력 | 선택지별 로그잇 보고서 |
| --- | --- | --- | --- |
| 16개 스트레스 문항 | [JSONL](../examples/stress-challenges.jsonl) | [JSONL](../results/gemma4-e2b-stress-challenges-cpu.jsonl) | [보고서](gemma4-e2b-stress-challenges-cpu.md) |
| 순서 변경 예비 실험 20건 | [JSONL](../examples/stress-shuffles-pilot.jsonl) | [JSONL](../results/gemma4-e2b-stress-shuffles-pilot-cpu.jsonl) | [보고서](gemma4-e2b-stress-shuffles-pilot-cpu.md) |
| 후속 3건 | [JSONL](../examples/stress-followups.jsonl) | [JSONL](../results/gemma4-e2b-stress-followups-cpu.jsonl) | [보고서](gemma4-e2b-stress-followups-cpu.md) |

입력 생성: [`scripts/build_stress_suite.py`](../scripts/build_stress_suite.py). 집계: [`scripts/score_stress.py`](../scripts/score_stress.py). 소규모로 직접 만든 문항이므로 표의 비율을 일반적인 정확도로 해석하면 안 됩니다.

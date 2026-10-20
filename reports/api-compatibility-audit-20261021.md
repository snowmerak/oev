# API compatibility audit — 2026-10-21

기준 커밋: `2411076` (`main`). 조사 대상은 OpenAI 호환 텍스트 생성·함수 호출, Responses 이력, SSE, 오류 응답 및 임베딩 API의 요청 경계다. 추론 모델별 품질이나 모든 외부 클라이언트 버전을 검증한 결과는 아니다.

아래 본문은 해당 커밋에 대한 최초 조사 기록이다. 이후 사용자 요청에 따라 7개 항목의 수정을 작업 트리에 구현했다. 해결 내용과 최종 검증 결과는 마지막의 **후속 수정 및 검증**에 기록했다.

## 우선순위와 요약

| 항목 | 우선순위 | 관찰 결과 | 영향 |
| --- | --- | --- | --- |
| 1. oev가 반환한 빈 assistant 메시지의 재입력 | P1 | Chat, Responses 수동 이력, `previous_response_id` 모두 HTTP 400 | 추론 중 출력 한도를 소진한 뒤 다음 턴을 계속할 수 없음 |
| 2. SDK가 직렬화한 선택 필드의 null | P2 | Chat과 Responses에서 HTTP 400 | SDK 응답 객체를 일반적인 dump 형태로 이력에 넣을 때 실패 가능 |
| 3. refusal 이력 | P2 | refusal-only 메시지는 400; content와 함께 있으면 refusal 텍스트 유실 | 다른 호환 API에서 만든 거부 응답의 재사용 실패 또는 문맥 손실 |
| 4. `input_items` 응답의 message content | P2 | 문자열 입력을 문자열 content로 그대로 반환 | 정규화된 배열을 기대하는 typed client와 불일치 |
| 5. 하위 조회 경로의 검증 오류 | P2 | 잘못된 query에 FastAPI 422/detail 반환 | 동일 API 안에서 오류 status/envelope가 달라짐 |
| 6. Responses union 검증 오류의 parameter | P2 | 잘못된 nested 필드 대신 `input.str` 표시 | 실제 실패 필드가 가려져 클라이언트 호환성 문제 진단이 어려움 |
| 7. `phase` 이력 메타데이터 | 조건부 P2 | 요청은 200이지만 template에 전달되는 메시지에서 삭제 | phase를 사용하는 native template/profile에서 문맥 의미가 사라질 수 있음 |

1번을 먼저 수정하고, 2번의 알려진 null 필드 정규화를 이어서 처리하는 것이 좋다. 알 수 없는 모든 필드를 무조건 무시하는 방식은 기존의 미지원 기능 검증을 약화시키므로 권하지 않는다.

## 1. 자체 출력 이력이 다음 요청에서 거부됨

관련 코드:

- [generation.py:305](../src/oev/generation.py#L305): tool이 아닌 메시지의 빈 content를 거절한다.
- [chat.py:138](../src/oev/chat.py#L138): 함수 호출이 없으면 reasoning-only 결과에도 빈 text message를 추가한다.
- [api_requests.py:358](../src/oev/api_requests.py#L358): Responses reasoning item을 뒤따르는 assistant 메시지의 `reasoning_content`로 변환한다.

재현 과정:

1. 생성 backend가 `<think>Still reasoning without an answer`를 반환하도록 한다.
2. `reasoning.effort=high` 또는 `reasoning_effort=high`와 출력 한도 16으로 요청한다.
3. 생성 결과는 thought를 포함하지만 visible answer는 비어 있다.
4. Chat의 assistant message를 이력에 재입력하거나, Responses의 output을 재입력하거나, 저장된 response ID로 계속한다.

세 경로 모두 `message content must be nonempty text`로 HTTP 400이 났다. 첫 응답을 JSON으로 받거나 SSE로 받아 조립한 경우 모두 재현됐다. 추론 없이 실제로 빈 응답을 반환한 Chat의 재입력도 같은 이유로 실패했다.

첫 Responses output은 다음 형태다.

```json
[
  {"type":"reasoning","content":[{"type":"reasoning_text","text":"..."}],"summary":[{"type":"summary_text","text":"..."}],"id":"rs_...","status":"incomplete"},
  {"type":"message","role":"assistant","content":[{"type":"output_text","text":"","annotations":[],"logprobs":[]}],"id":"msg_...","status":"incomplete"}
]
```

권장 수정은 reasoning-only assistant의 허용과 빈 placeholder 정책을 입력·출력 양쪽에서 일치시키는 것이다. 빈 message만 삭제하면 별도 reasoning item을 실제 assistant turn에 연결하지 못할 수 있으므로 thought 보존까지 함께 검증해야 한다.

## 2. SDK 직렬화 결과의 null 필드가 이력을 깨뜨림

[APIModel](../src/oev/api_requests.py#L11)은 extra 필드를 거절한다. root GenerationRequest에는 null을 기본값으로 바꾸는 처리가 있지만, nested message/item에는 대응 처리가 없다.

직접 재현한 요청 형태:

| 위치 | 필드 | 결과 |
| --- | --- | --- |
| Chat assistant | `tool_calls: null` | valid list가 아니라며 400 |
| Chat assistant | `annotations: null` | unsupported field로 400 |
| Chat assistant | `audio: null` | unsupported field로 400 |
| Chat assistant | `function_call: null` | unsupported field로 400 |
| Responses reasoning item | `encrypted_content: null` | 400 |
| Responses reasoning item | `content: null` | 400 |
| Responses function_call item | `async: null`, `caller: null`, `namespace: null` 각각 | 400 |

각 요청에는 정상적인 텍스트 또는 함수 호출과 그 결과를 함께 넣었다. 실제 오디오나 암호화 추론 기능을 요구한 것이 아니다.

공식 Python SDK 소스의 [ChatCompletionMessage](https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion_message.py), [ResponseReasoningItem](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_reasoning_item.py), [ResponseFunctionToolCall](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_function_tool_call.py)에 해당 optional 필드와 기본 null이 있다. 이런 응답 객체를 unset/null을 제외하지 않고 dump하는 클라이언트가 영향을 받는다.

로컬에는 openai Python SDK가 설치되어 있지 않아 실제 SDK를 실행한 검증은 아니다. 공식 소스에 나타난 JSON 형태를 oev HTTP 경계에 직접 전달한 결과다. 모든 null 형태를 공식 API가 동일하게 허용한다고 주장하는 것도 아니다. 응답 객체를 이력으로 재사용하는 실용적인 호환성 범위에서 정규화가 필요하다는 의미다.

권장 수정은 알려진 no-op null/empty metadata를 제한적으로 정규화하는 것이다. non-null 오디오, 암호화된 reasoning, deprecated function_call 등의 실질 기능 지원과는 분리해야 한다.

## 3. refusal을 선언했지만 보존하지 않음

[ChatMessage](../src/oev/api_requests.py#L42)는 `refusal`을 필드로 받지만 [message 변환](../src/oev/api_requests.py#L74)에서 전달하지 않는다.

다음 이력은 HTTP 400이다.

```json
{"role":"assistant","content":null,"refusal":"I cannot help with that."}
```

content를 `"Fallback"`으로 바꾸면 HTTP 200이지만 tokenizer에 전달되는 메시지에는 refusal 텍스트가 없다. 두 경우 모두 확인했다. Responses의 `content` part도 `input_text`/`output_text`만 받아 refusal part를 지원하지 않는다.

공식 [Chat assistant 입력](https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion_assistant_message_param.py)과 [Responses output message](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_output_message.py)는 refusal 표현을 갖는다. oev가 현재 생성하는 일반 텍스트 거부문에는 이 문제가 없지만, 다른 호환 API 또는 클라이언트가 구성한 이력을 가져오면 문제가 된다.

refusal을 의미 있는 텍스트 이력으로 정규화하거나 native template에 보존하는 경로가 필요하다. 입력 필드만 추가하고 실제 문맥에서 삭제하는 방식은 피해야 한다.

## 4. input_items가 입력의 shorthand 형식을 그대로 반환함

재현:

1. `/v1/responses`에 `input="Hi", store=true`로 요청한다.
2. `/v1/responses/{id}/input_items`를 조회한다.

응답 message의 content가 `"Hi"`라는 문자열이다. status가 없는 것은 optional이므로 별도 결함으로 보지 않았다.

공식 조회 결과의 [ResponseItem](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_item.py)은 [ResponseInputMessageItem](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_input_message_item.py)을 사용하고, 이 message의 content는 배열이다. 생성 요청의 편의 입력 형태인 문자열과 조회 응답의 canonical item 형태를 구별해야 한다.

권장 반환 형태:

```json
{"id":"item_...","type":"message","role":"user","content":[{"type":"input_text","text":"Hi"}]}
```

현재 raw JSON을 직접 읽는 기존 테스트는 통과하지만, 배열 content를 순회하는 typed client에는 실제 형식 차이가 생긴다. native OpenAI SDK의 strict response parsing 실행은 이번에 수행하지 않았다.

## 5. 하위 경로의 오류 형식이 일관되지 않음

[OPENAI_PATHS](../src/oev/chat.py#L28)가 base POST 경로 세 개만 포함하고, [validation_error](../src/oev/chat.py#L462)가 정확한 path 일치로 오류 변환을 결정한다.

`GET /v1/responses/{id}/input_items?limit=abc`는 다음처럼 반환된다.

```json
{"detail":[{"type":"int_parsing","loc":["query","limit"],"msg":"Input should be a valid integer, unable to parse string as an integer","input":"abc"}]}
```

HTTP status도 422다. 반면 같은 경로에서 정수 `limit=0`은 route 자체 검증으로 OpenAI형 `error`를 가진 400이 된다. 클라이언트가 error.message만 처리하면 전자의 메시지를 잃는다.

OpenAI 호환 route 전체에 동일한 validation error handler를 적용하고, 기존 System One의 422 계약은 유지하는 것이 적절하다.

## 6. Responses 오류가 실제 필드 대신 input.str을 가리킴

[ResponsesRequest.input](../src/oev/api_requests.py#L336)은 `str | list[InputItem]` union이다. validation handler가 `exc.errors()[0]`만 선택하므로, 올바른 list 입력 내부에 문제가 있어도 첫 번째 str branch의 실패를 보여 준다.

예를 들어 정상 함수 호출에 `namespace: null`을 추가한 요청의 결과는 다음과 같다.

```json
{"error":{"message":"Input should be a valid string","type":"invalid_request_error","param":"input.str","code":"invalid_request"}}
```

실제 거부된 부분은 function_call item의 namespace 필드인데 입력 전체가 문자열이어야 한다는 식으로 안내한다. reasoning의 null 필드에서도 같은 현상을 확인했다.

입력의 실제 JSON 형태에 맞는 union branch의 오류를 우선하거나, nested path를 유지해 구체적인 원인을 표시해야 한다.

## 7. phase는 받지만 template에 전달하지 않음

[InputMessage.phase](../src/oev/api_requests.py#L295)는 `commentary`/`final_answer`를 받지만 [message()](../src/oev/api_requests.py#L297)는 role과 content만 반환한다. phase를 넣은 요청이 HTTP 200이며 tokenizer가 받은 메시지에는 phase가 없음을 확인했다.

공식 [ResponseOutputMessage](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_output_message.py)는 일부 모델에서 후속 입력의 phase 보존이 중요하다고 설명한다. 다만 이번에는 phase를 실제로 소비하는 로컬 모델 profile에서 품질 저하를 재현하지 않았다. 따라서 메타데이터 소실 자체는 확인됐지만, 현재 배포 모델의 실제 영향은 조건부다.

## 통과한 범위와 의도적인 미지원

추가 왕복 검사 20개는 모두 통과했다.

```text
5가지 결과: plain text / function calls / reasoning+text /
            reasoning+calls / prose+calls
× Chat / Responses
× JSON / SSE
```

Chat은 스트리밍 tool_call index를 보존한 채 이력에 다시 넣었다. Responses는 `store=true`와 `previous_response_id`, 함수 호출에는 대응하는 function_call_output을 사용했다. 모두 정상 완료한 응답에 대한 검사다. 실패한 reasoning-only cutoff 경로를 이 성공 수치에 포함하지 않았다.

기존 API·native protocol·reasoning·embedding 테스트도 실행했다.

```powershell
.venv/Scripts/python.exe -c "import torch; torch.cuda.is_current_stream_capturing=lambda:False; import pytest; raise SystemExit(pytest.main(['-q','tests/test_api_compatibility.py','tests/test_output_parsers.py','tests/test_chat.py','tests/test_reasoning_stream.py','tests/test_reasoning_metadata.py','tests/test_embeddings.py','tests/test_multimodal_embeddings.py']))"
```

결과: **459 passed, 6 skipped**. 처음에 우회 없이 실행했을 때 임베딩 CPU 테스트가 기존 Windows CUDA stream-capture 검사에서 access violation으로 종료됐다. 위 재실행은 이전 검증에서 사용한 것과 같은 테스트 프로세스 한정 우회를 적용한 결과다. application/runtime 코드에는 우회를 추가하지 않았다.

별도 실패 재현에는 기존 `test_api_compatibility.client_for`의 fake inference backend와 FastAPI TestClient를 사용했다. HTTP schema, 이력 변환, 이벤트·저장 처리는 실제 oev 코드지만, HF/GGUF/MLX의 새로운 실제 추론 검증을 했다는 의미는 아니다. 캐시된 Gemma 4 tokenizer의 native template도 확인했으나 새 가중치나 SDK는 다운로드하지 않았다.

현재 문서에 명시된 generation 이미지·오디오, hosted tools, background execution, encrypted reasoning, 자동 truncation 및 deprecated functions의 non-null 기능 미지원은 새 결함으로 집계하지 않았다. System One의 의도적인 422도 결함으로 집계하지 않았다.

공식 SDK 비교 자료는 조사 당일의 openai-python `main` 소스다. 특정 배포 SDK 버전 전체의 호환성 인증이 아니며, q/llm-provider 소스의 전송 경계도 참고했지만 현재 실행 중인 원격 서비스에 테스트 요청을 보내거나 재배포하지 않았다.

## 후속 수정의 권장 범위

1. reasoning-only/empty assistant 왕복을 Chat·Responses·stored continuation에서 일관되게 처리하고, native template 재생을 검증한다.
2. 알려진 optional metadata의 null/empty 정규화를 추가한다. 알 수 없는 non-null 필드의 거절은 유지한다.
3. refusal과 phase의 의미 있는 이력을 보존하고, `input_items`를 canonical 응답 형태로 변환한다.
4. 하위 route 및 union 입력 오류의 status/envelope/parameter를 일관되게 만든다.
5. 고정 fixture를 넘어 SDK 직렬화 → 재입력 및 cutoff → 다음 턴을 회귀 테스트에 추가한다.

## 후속 수정 및 검증 — 2026-10-21

7개 항목 모두 작업 트리에 수정했다.

- Chat의 빈/null assistant 및 reasoning-only 이력을 허용하고, Responses의 독립 reasoning item도 유실하지 않고 assistant turn에 연결한다.
- native template에 reasoning-only turn이 실제로 렌더링되는지 probe로 확인한다. 생략되는 경우에는 해당 assistant의 텍스트 문맥에 부분 추론을 포함한다. probe에는 고유 marker를 사용하며 실제 모델 입력에는 포함되지 않는다. 추가 모델 추론은 없다. 같은 이력으로 다시 요청하면 같은 실제 prompt token IDs가 만들어진다.
- 알려진 SDK optional metadata의 null/empty 값을 정규화한다. non-null 오디오·암호화 reasoning·namespace·비동기 호출 등은 계속 거절한다.
- refusal 문자열/part를 실제 텍스트 이력으로 전달하고, `phase`를 template 입력에 보존한다. 기존의 연속 text part 연결 방식은 유지한다.
- 저장된 message input을 typed content 배열로 정규화하며 ID·status·phase 및 페이지 cursor를 보존한다.
- 하위 조회 route도 OpenAI형 400 검증 오류를 반환하고, union branch 이름을 제거한 실제 JSON field path를 표시한다. System One의 기존 422 응답은 유지한다.

[test_history_compatibility.py](../tests/test_history_compatibility.py)에 회귀 검사 40개를 추가했다. 기존 테스트를 포함한 전체 실행 결과는 **723 passed, 24 skipped**다. 실행 명령은 다음과 같으며, 기존 Windows CUDA probe 우회는 테스트 프로세스에만 적용했다.

```powershell
.venv/Scripts/python.exe -c "import torch; torch.cuda.is_current_stream_capturing=lambda:False; import pytest; raise SystemExit(pytest.main(['-q']))"
```

추가로 이미 다운로드된 `google/gemma-4-E2B-it` 및 `HuggingFaceTB/SmolLM2-135M-Instruct` 토크나이저로 실제 native template을 렌더링했다. 수정 전 두 template 모두 후속 user turn 앞의 부분 추론을 생략했으나, 수정 후에는 sentinel 추론 내용이 정확히 한 번 포함됐다. 실제 입력에 probe marker가 섞이지 않는 것도 확인했다. 이 검사는 native prompt 구성의 검증이며 새 모델 추론·판단 품질·Metal 실행의 검증은 아니다.

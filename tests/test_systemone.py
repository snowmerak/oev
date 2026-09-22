import math

import pytest
from fastapi.testclient import TestClient

from oev import Decision, DecisionEngine, Option
from oev.serve import create_app
from oev.systemone import SystemOneService


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages) + "\nANSWER: "

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


class FixedBackend:
    tokenizer = CharacterTokenizer()
    model_name = "fixed"

    def __init__(self, scores):
        self.scores = iter(scores)
        self.prompts = []
        self.answer_slots = []

    def selected_logits(self, input_ids, answer_token_ids):
        self.prompts.append(self.tokenizer.decode(input_ids))
        self.answer_slots.append(answer_token_ids)
        scores = next(self.scores)
        assert len(scores) == len(answer_token_ids)
        return scores


def service_for(scores):
    backend = FixedBackend(scores)
    engine = DecisionEngine(backend, max_input_tokens=100_000)
    return SystemOneService(engine, "oev-test", release_date="2026-09-23"), backend


def test_systemone_evaluates_all_question_types_and_preserves_meanings():
    service, backend = service_for([[0, math.log(3)], [0, math.log(3)], [0, math.log(2), 0]])
    response = service.evaluate(
        {
            "model": "oev-test",
            "state": {"message": "double charge"},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"support": None, "billing": {"covers": "payments"}},
                },
                "charged": {
                    "type": "noul",
                    "instructions": "Is the customer charged twice?",
                    "criteria": {"true": "A duplicate charge", "false": "One charge"},
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent?",
                    "criteria": ["Can wait", {"level": "Soon"}, "Today"],
                },
            },
        }
    )

    assert response["model"] == "oev-test"
    assert response["answers"]["route"] == {
        "type": "choice",
        "choice": "billing",
        "confidence": pytest.approx(0.5),
        "probabilities": {"support": pytest.approx(0.25), "billing": pytest.approx(0.75)},
    }
    assert response["answers"]["charged"] == {"type": "noul", "noul": pytest.approx(0.75)}
    assert response["answers"]["urgency"] == {
        "type": "score",
        "score": pytest.approx(1.0),
        "confidence": pytest.approx(0.75),
        "legend": {"0": "Can wait", "1": {"level": "Soon"}, "2": "Today"},
        "probabilities": {"0": pytest.approx(0.25), "1": pytest.approx(0.5), "2": pytest.approx(0.25)},
    }
    assert response["usage"] == {
        "input_tokens": sum(len(prompt) for prompt in backend.prompts),
        "output_tokens": 0,
    }
    assert '"billing: {\\"covers\\": \\"payments\\"}"' in backend.prompts[0]
    assert '"No: One charge"' in backend.prompts[1]
    assert '"Yes: A duplicate charge"' in backend.prompts[1]
    assert '"Level 1: {\\"level\\": \\"Soon\\"}"' in backend.prompts[2]


def test_twenty_five_options_reach_y_and_twenty_six_are_rejected():
    backend = FixedBackend([[0.0] * 24 + [3.0]])
    engine = DecisionEngine(backend, max_input_tokens=100_000)
    options = tuple(Option(str(index), f"option {index}") for index in range(25))
    result = engine.decide(Decision("state", "question", options))
    assert result.selected_id == "24"
    assert backend.answer_slots[0][-1] == ord("Y")

    with pytest.raises(ValueError, match="2 to 25"):
        Decision("state", "question", options + (Option("25", "option 25"),))


def test_http_contract_and_validation_before_inference():
    service, backend = service_for([[0.0, 1.0]])
    client = TestClient(create_app(service))
    assert client.get("/v1/models").json() == {
        "models": [
            {
                "name": "oev-test",
                "description": service.models()["models"][0]["description"],
                "release_date": "2026-09-23",
            }
        ]
    }

    request = {
        "model": "oev-test",
        "state": "message",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"a": "A", "b": "B"},
            }
        },
    }
    response = client.post("/v1/systemone", json=request)
    assert response.status_code == 200
    assert response.json()["answers"]["route"]["choice"] == "b"
    assert client.get("/openapi.json").status_code == 200

    invalid = {
        **request,
        "questions": {
            "route": request["questions"]["route"],
            "too_many": {
                "type": "score",
                "criteria": [str(index) for index in range(26)],
            },
        },
    }
    assert client.post("/v1/systemone", json=invalid).status_code == 422
    assert len(backend.prompts) == 1


def test_one_option_is_deterministic_and_needs_no_inference():
    service, backend = service_for([])
    response = service.evaluate(
        {
            "model": "oev-test",
            "state": "state",
            "questions": {
                "route": {"type": "choice", "criteria": {"only": None}},
                "rating": {"type": "score", "criteria": ["Only level"]},
            },
        }
    )
    assert response["answers"]["route"]["probabilities"] == {"only": 1.0}
    assert response["answers"]["rating"]["score"] == 0.0
    assert response["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert not backend.prompts

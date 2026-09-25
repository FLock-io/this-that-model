"""The typed `/v1/systemone` endpoint that JevBench's `typesafe` adapter calls.

Split like test_server: the wire format is pure and tested without weights, then the real model
answers behind the real server. The assertions on the answer shape mirror what the adapter checks,
because an answer it cannot read is scored as a failure however right it was.
"""
import math

import pytest

from thisthat.systemone_protocol import SystemOneRequestError, parse_request

CHOICE = {"type": "choice", "instructions": "Which intent does the user's message express?",
          "criteria": {"track_order": "Wants to know where an order is",
                       "cancel_order": "Wants to cancel an order"}}
NOUL = {"type": "noul", "instructions": "Has the order been shipped?",
        "criteria": {"true": "The text states that this is so",
                     "false": "The text states that this is not so"}}
SCORE = {"type": "score", "instructions": "Rate incident impact.",
         "criteria": ["No function impaired", "One user impaired", "Many users blocked"]}


# ----------------------------------------------------------------- wire format, no model needed

def test_choice_labels_are_the_criteria_keys_and_descriptions_go_in_the_question():
    _, [tq] = parse_request({"state": "s", "questions": {"decision": CHOICE}})
    assert tq.labels == ("track_order", "cancel_order")
    assert tq.question.options == ("track_order", "cancel_order")
    assert "- cancel_order: Wants to cancel an order" in tq.question.text


def test_noul_is_answered_over_no_and_yes_with_true_and_false_described():
    _, [tq] = parse_request({"state": "s", "questions": {"decision": NOUL}})
    assert tq.labels == ("no", "yes")
    assert "- yes: The text states that this is so" in tq.question.text
    assert "- no: The text states that this is not so" in tq.question.text


def test_noul_without_criteria_is_just_the_instructions():
    q = {"type": "noul", "instructions": "Is it raining?"}
    _, [tq] = parse_request({"state": "s", "questions": {"d": q}})
    assert tq.question.text == "Is it raining?"


def test_score_labels_are_level_indices_in_order():
    _, [tq] = parse_request({"state": "s", "questions": {"decision": SCORE}})
    assert tq.labels == ("0", "1", "2")
    assert "- 2: Many users blocked" in tq.question.text


def test_a_json_state_is_passed_as_json_text():
    state, _ = parse_request({"state": {"order": 1182, "status": "shipped"},
                              "questions": {"decision": NOUL}})
    assert state == '{"order": 1182, "status": "shipped"}'


@pytest.mark.parametrize("body", [
    {"questions": {"d": NOUL}},                                            # no state
    {"state": "s", "questions": {}},                                       # no questions
    {"state": "s", "questions": {"d": {"type": "free_text", "instructions": "q"}}},
    {"state": "s", "questions": {"d": {"type": "choice", "instructions": "q"}}},
    {"state": "s", "questions": {"d": {"type": "choice", "instructions": "q",
                                       "criteria": {"only": "one"}}}},
    {"state": "s", "questions": {"d": {"type": "score", "instructions": "q",
                                       "criteria": {"0": "not a list"}}}},
])
def test_requests_the_model_cannot_answer_are_refused(body):
    with pytest.raises(SystemOneRequestError):
        parse_request(body)


# ----------------------------------------------------------------- end to end, real model

@pytest.fixture(scope="module")
def client(decider):
    from fastapi.testclient import TestClient

    from thisthat.server import build_app
    return TestClient(build_app(decider, "test-model"))


@pytest.mark.parametrize("q", [CHOICE, NOUL, SCORE], ids=["choice", "noul", "score"])
def test_every_type_comes_back_in_the_shape_the_adapter_reads(client, q):
    r = client.post("/v1/systemone", json={
        "state": "Order #1182. Status: shipped on 3 September. Where is it now?",
        "model": "ignored", "questions": {"decision": q}})
    assert r.status_code == 200
    ans = r.json()["answers"]["decision"]
    assert ans["type"] == q["type"]
    probs = ans["probabilities"]
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-4)
    if q["type"] == "noul":
        assert set(probs) == {"no", "yes"} and 0.0 <= ans["noul"] <= 1.0
    elif q["type"] == "choice":
        assert set(probs) == set(q["criteria"]) and ans["choice"] in q["criteria"]
    else:
        assert set(probs) == {"0", "1", "2"} and ans["score"] in (0, 1, 2)


def test_several_questions_are_answered_in_one_request(client):
    r = client.post("/v1/systemone", json={
        "state": "Order #1182. Status: shipped on 3 September.",
        "questions": {"shipped": NOUL, "intent": CHOICE}})
    assert r.status_code == 200
    assert set(r.json()["answers"]) == {"shipped", "intent"}


def test_an_unanswerable_request_is_a_client_error(client):
    r = client.post("/v1/systemone", json={"state": "s", "questions": {
        "d": {"type": "free_text", "instructions": "write a poem"}}})
    assert r.status_code == 400
    assert "noul, choice and score" in r.json()["error"]["message"]

"""The OpenAI-compatible endpoint.

Split in two on purpose. The wire format is parsed by pure functions and is tested here without
weights, so a change to the request shape fails in a second rather than after a 3.6 GB load. The
end-to-end tests run the real model behind the real server, because that is the only way to know
that a stock OpenAI client gets a usable answer out of it.
"""
import json
import math

import pytest

from thisthat.openai_protocol import OptionsNotDeclared, extract_options, split_messages


# ----------------------------------------------------------------- wire format, no model needed

def test_enum_is_found_in_a_json_schema_response_format():
    body = {"response_format": {"type": "json_schema", "json_schema": {
        "name": "d", "schema": {"type": "object",
                                "properties": {"verdict": {"enum": ["allow", "deny"]}}}}}}
    assert extract_options(body) == (["allow", "deny"], "verdict")


def test_enum_is_found_in_a_single_tool_definition():
    body = {"tools": [{"type": "function", "function": {
        "name": "decide", "parameters": {"properties": {"risk": {"enum": ["low", "high"]}}}}}]}
    assert extract_options(body) == (["low", "high"], "risk")


def test_a_request_with_no_answer_set_is_refused():
    """The model cannot generate text, so a request with no enum has no answer.

    Returning an empty completion here would be worse than an error: the caller would believe it
    had been answered.
    """
    with pytest.raises(OptionsNotDeclared):
        extract_options({"messages": [{"role": "user", "content": "write me a poem"}]})


def test_a_schema_without_an_enum_is_refused():
    with pytest.raises(OptionsNotDeclared):
        extract_options({"response_format": {"type": "json_schema", "json_schema": {
            "schema": {"properties": {"answer": {"type": "string"}}}}}})


def test_the_last_user_message_is_the_question_and_the_rest_is_state():
    state, question = split_messages([
        {"role": "system", "content": "policy: refunds within 30 days"},
        {"role": "user", "content": "order placed 2026-08-02"},
        {"role": "user", "content": "Is this refundable?"}])
    assert question == "Is this refundable?"
    assert "policy: refunds within 30 days" in state
    assert "order placed 2026-08-02" in state
    assert "Is this refundable?" not in state


def test_content_parts_are_flattened():
    state, question = split_messages([
        {"role": "user", "content": [{"type": "text", "text": "a"},
                                     {"type": "text", "text": "b"}]},
        {"role": "user", "content": "q?"}])
    assert question == "q?"
    assert "a" in state and "b" in state


# ----------------------------------------------------------------- end to end, real model

@pytest.fixture(scope="module")
def client(decider):
    from fastapi.testclient import TestClient
    from thisthat.server import build_app
    return TestClient(build_app(decider, "test-model"))


def test_models_endpoint_lists_the_loaded_model(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "test-model"


def test_a_decision_comes_back_as_json_with_the_declared_field(client):
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "command: rm -rf /var/lib/postgresql/data"},
                     {"role": "user", "content": "Does this command delete data?"}],
        "response_format": {"type": "json_schema", "json_schema": {
            "schema": {"properties": {"answer": {"enum": ["no", "yes"]}}}}}})
    assert r.status_code == 200
    body = r.json()
    answer = json.loads(body["choices"][0]["message"]["content"])
    assert answer["answer"] in ("no", "yes")
    # nothing was generated, and the usage block must say so rather than invent a token count
    assert body["usage"]["completion_tokens"] == 0


def test_probabilities_come_back_in_the_standard_logprobs_shape(client):
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "value = 42"},
                     {"role": "user", "content": "Is the value above ten?"}],
        "response_format": {"type": "json_schema", "json_schema": {
            "schema": {"properties": {"answer": {"enum": ["no", "yes"]}}}}},
        "logprobs": True})
    top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    assert {t["token"] for t in top} == {"no", "yes"}
    assert math.isclose(sum(math.exp(t["logprob"]) for t in top), 1.0, abs_tol=1e-4)


def test_a_free_form_request_is_refused_rather_than_answered(client):
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Write me a haiku about databases."}]})
    assert r.status_code == 400
    assert "answer set" in r.json()["error"]["message"]


def test_streaming_returns_the_answer_as_one_chunk(client):
    r = client.post("/v1/chat/completions", json={
        "model": "test-model", "stream": True,
        "messages": [{"role": "user", "content": "x"},
                     {"role": "user", "content": "Is x a letter?"}],
        "response_format": {"type": "json_schema", "json_schema": {
            "schema": {"properties": {"answer": {"enum": ["no", "yes"]}}}}}})
    assert r.status_code == 200
    chunks = [l for l in r.text.splitlines() if l.startswith("data: ")]
    assert chunks[-1] == "data: [DONE]"
    first = json.loads(chunks[0][6:])
    assert json.loads(first["choices"][0]["delta"]["content"])["answer"] in ("no", "yes")


def test_too_few_options_is_a_client_error(client):
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "s"}, {"role": "user", "content": "q?"}],
        "response_format": {"type": "json_schema", "json_schema": {
            "schema": {"properties": {"answer": {"enum": ["only one"]}}}}}})
    assert r.status_code == 400

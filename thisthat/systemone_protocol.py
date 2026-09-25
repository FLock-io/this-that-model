"""Parsing the `/v1/systemone` request shape, with no model and no torch in sight.

This is the typed-decision wire format JevBench's `typesafe` adapter speaks:

    POST /v1/systemone
    {"state": ..., "model": ..., "questions": {"<key>": {"type": ..., "instructions": ...,
                                                         "criteria": ...}}}

and it fits this model better than the OpenAI shape does, because it is already what the model
is: a state, typed questions, a distribution back over each one's labels. Three types arrive.

    noul    a proposition; labels are no / yes, criteria may describe what true and false mean
    choice  criteria map each label to a description; the labels are the dict's keys
    score   criteria are a list of level descriptions; the labels are "0" .. "n-1"

The request carries no label list of its own, so the labels are read off the criteria exactly as
the harness does it. The descriptions go into the question text as a legend under the
instructions -- the layout JevBench's own `openai_compat` adapter uses, not one tuned on its
items -- and the labels themselves are the option set, so the answer comes back keyed by the
labels the caller named and cannot be anything else.

Kept apart from the server for the same reason `openai_protocol` is: the wire format is tested
without loading weights.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from .types import MAX_OPTIONS, Decision, Question

NOUL_LABELS = ("no", "yes")


class SystemOneRequestError(ValueError):
    """Raised when a request cannot be turned into questions this model can answer."""


@dataclass(frozen=True)
class TypedQuestion:
    key: str
    type: str
    labels: tuple[str, ...]
    question: Question


def _legend(lines: Sequence[tuple[str, str | None]], heading: str) -> str:
    return heading + "\n" + "\n".join(f"- {lab}: {desc}" if desc else f"- {lab}"
                                      for lab, desc in lines)


def _typed(key: str, q: dict) -> TypedQuestion:
    if not isinstance(q, dict):
        raise SystemOneRequestError(f"question {key!r} is not an object")
    qtype, instructions, crit = q.get("type"), q.get("instructions"), q.get("criteria")
    if not isinstance(instructions, str) or not instructions.strip():
        raise SystemOneRequestError(f"question {key!r} has no instructions")

    if qtype == "noul":
        labels = NOUL_LABELS
        crit = crit if isinstance(crit, dict) else {}
        # the harness words noul criteria as true / false; the answer is read as no / yes
        lines = [("no", crit.get("false")), ("yes", crit.get("true"))]
        text = _legend(lines, instructions + "\n\nOptions:") if any(d for _, d in lines) \
            else instructions
    elif qtype == "choice":
        if isinstance(crit, dict):
            labels = tuple(str(k) for k in crit)
            lines = [(str(k), v) for k, v in crit.items()]
        elif isinstance(crit, list):
            labels = tuple(str(x) for x in crit)
            lines = [(lab, None) for lab in labels]
        else:
            raise SystemOneRequestError(
                f"choice question {key!r} declares no options: criteria must map each option "
                "to its description")
        text = _legend(lines, instructions + "\n\nOptions:")
    elif qtype == "score":
        if not isinstance(crit, list):
            raise SystemOneRequestError(
                f"score question {key!r} declares no levels: criteria must list them in order")
        labels = tuple(str(i) for i in range(len(crit)))
        text = _legend([(lab, str(d)) for lab, d in zip(labels, crit)],
                       instructions + "\n\nLevels:")
    else:
        raise SystemOneRequestError(
            f"question {key!r} has type {qtype!r}; this model answers noul, choice and score")

    if not 2 <= len(labels) <= MAX_OPTIONS:
        raise SystemOneRequestError(
            f"question {key!r} needs between 2 and {MAX_OPTIONS} options; got {len(labels)}")
    try:
        question = Question(text, labels)
    except ValueError as e:                              # duplicate labels, for one
        raise SystemOneRequestError(f"question {key!r}: {e}") from e
    return TypedQuestion(key, qtype, labels, question)


def parse_request(body: dict) -> tuple[str, list[TypedQuestion]]:
    """-> (state as text, the typed questions in the order declared)."""
    if not isinstance(body, dict):
        raise SystemOneRequestError("the request body must be a JSON object")
    state = body.get("state")
    if state is None:
        raise SystemOneRequestError("no state")
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise SystemOneRequestError("questions must be a non-empty object of named questions")
    return state, [_typed(str(k), q) for k, q in questions.items()]


def render_answer(tq: TypedQuestion, d: Decision) -> dict:
    """One answer in the shape the typesafe adapter reads, keyed by the caller's labels."""
    probs = dict(zip(tq.labels, d.probabilities))
    out: dict = {"type": tq.type, "probabilities": probs}
    if tq.type == "noul":
        out["noul"] = probs["yes"]
    elif tq.type == "choice":
        out["choice"] = d.choice
    else:
        out["score"] = int(d.choice)
    return out

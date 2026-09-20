"""Parsing the OpenAI request shape, with no model and no torch in sight.

Kept apart from the server so the wire format can be tested without loading 3.6 GB of weights,
and so a change to one cannot quietly break the other.
"""
from __future__ import annotations

MAX_OPTIONS = 255


class OptionsNotDeclared(ValueError):
    """Raised when a request carries no enum, and so no answer set."""


def extract_options(body: dict) -> tuple[list[str], str]:
    """-> (options, field name). Looks where an OpenAI client would actually put an enum.

    Three shapes are accepted because three are in common use: a json_schema response_format, a
    plain json_object one carrying a schema, and a single tool whose parameters hold the enum.
    """
    rf = body.get("response_format") or {}
    schema = None
    if isinstance(rf, dict):
        js = rf.get("json_schema") or {}
        schema = js.get("schema") or js.get("json_schema") or rf.get("schema")
    if schema is None:
        tools = body.get("tools") or []
        if len(tools) == 1:
            fn = (tools[0] or {}).get("function") or {}
            schema = fn.get("parameters")
    if not isinstance(schema, dict):
        raise OptionsNotDeclared(
            "no answer set was declared. This model chooses among options you name; it does not "
            "generate text. Pass response_format={'type':'json_schema','json_schema':{'schema':"
            "{'properties':{'answer':{'enum':[...]}}}}} with the options you want chosen between.")

    props = schema.get("properties")
    if isinstance(props, dict):
        for name, spec in props.items():
            if isinstance(spec, dict) and isinstance(spec.get("enum"), list):
                return [str(x) for x in spec["enum"]], name
    if isinstance(schema.get("enum"), list):
        return [str(x) for x in schema["enum"]], "answer"
    raise OptionsNotDeclared(
        "the schema declares no enum. Exactly one property must carry an `enum` listing the "
        "options to choose between.")


def split_messages(messages: list[dict]) -> tuple[str, str]:
    """-> (state, question).

    The last user message is the question; everything before it is the state. That matches how a
    caller writes this naturally -- system prompt and context first, the decision last -- and it
    keeps the state/question split the model was trained on rather than concatenating the two.
    """
    if not messages:
        raise ValueError("no messages")
    def text(m):
        c = m.get("content")
        if isinstance(c, list):                       # content parts
            return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c or ""
    users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not users:
        raise ValueError("no user message to read the question from")
    last = users[-1]
    state = "\n\n".join(text(m) for i, m in enumerate(messages) if i != last and text(m))
    return state, text(messages[last])

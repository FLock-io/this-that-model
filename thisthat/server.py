"""An OpenAI-compatible endpoint for a model that does not generate text.

    python -m thisthat.server --model flock-io/this-that-model-1.0 --port 8000

Then point any OpenAI client at it:

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

The mapping is not a costume. This model answers one of *n* declared options and returns a
probability for each, so the OpenAI feature it corresponds to is structured output with an enum:

    response_format={"type": "json_schema", "json_schema": {"schema": {
        "properties": {"answer": {"enum": ["no", "yes"]}}}}}

The enum *is* the option set. The reply is JSON whose one field is the chosen option, and with
`logprobs=true` the distribution over the options comes back in the standard logprobs shape.

What this server will not do is fake the rest of the API. A request with no enum has no answer
set, and there is nothing for a model without a decoding loop to generate; it returns 400 saying
so rather than inventing a completion. `stream=true` returns the single answer as one chunk,
because there is no token stream to emit -- the answer is read from one forward pass, and
pretending otherwise would misrepresent the latency the caller is buying.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from typing import Any

from .model import DEFAULT_MODEL, TypedDecider
from .openai_protocol import MAX_OPTIONS, OptionsNotDeclared, extract_options, split_messages
from .types import Question

try:
    # Imported at module scope, not inside build_app, and this matters. With `from __future__
    # import annotations` every annotation is a string that FastAPI resolves against the module's
    # globals; a `Request` imported into a function body is not there, so FastAPI fell back to
    # treating the parameter as a query field and rejected every POST with 422 before the handler
    # ran. The unit tests passed throughout -- only an end-to-end request found it.
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as _e:                                   # pragma: no cover
    raise ImportError("the server needs FastAPI: pip install -e '.[serve]'") from _e


def build_app(decider: TypedDecider, model_name: str):
    app = FastAPI(title="this-that-model", docs_url="/docs")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [
            {"id": model_name, "object": "model", "created": 0, "owned_by": "flock-io"}]}

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, "device": str(decider.device)}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body: dict[str, Any] = await request.json()
        try:
            options, field = extract_options(body)
            state, question = split_messages(body.get("messages") or [])
        except (OptionsNotDeclared, ValueError) as e:
            return JSONResponse(status_code=400, content={"error": {
                "message": str(e), "type": "invalid_request_error", "param": "response_format"}})
        if not 2 <= len(options) <= MAX_OPTIONS:
            return JSONResponse(status_code=400, content={"error": {
                "message": f"between 2 and {MAX_OPTIONS} options are needed; got {len(options)}",
                "type": "invalid_request_error", "param": "response_format"}})

        t0 = time.perf_counter()
        d = decider.decide(state, Question(question, options),
                           temperature=max(1e-3, float(body.get("temperature") or 1.0)))
        ms = (time.perf_counter() - t0) * 1000

        content = json.dumps({field: d.choice})
        message = {"role": "assistant", "content": content}
        choice: dict[str, Any] = {"index": 0, "message": message, "finish_reason": "stop"}
        if body.get("logprobs"):
            # the whole distribution, in the shape a client already knows how to read
            top = sorted(zip(d.options, d.probabilities), key=lambda p: -p[1])
            choice["logprobs"] = {"content": [{
                "token": d.choice, "logprob": math.log(max(d.confidence, 1e-12)), "bytes": None,
                "top_logprobs": [{"token": o, "logprob": math.log(max(p, 1e-12)), "bytes": None}
                                 for o, p in top[: int(body.get("top_logprobs") or len(top))]]}]}

        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [choice],
            # completion_tokens is zero and that is not an omission: the answer is read from a
            # hidden state, so there is no decoding loop and nothing was generated.
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "this_that": {"index": d.index, "choice": d.choice,
                          "probabilities": dict(zip(d.options, d.probabilities)),
                          "confidence": d.confidence, "latency_ms": round(ms, 2)},
        }
        if not body.get("stream"):
            return JSONResponse(resp)

        def one_chunk():
            chunk = {"id": resp["id"], "object": "chat.completion.chunk",
                     "created": resp["created"], "model": model_name,
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": content},
                                  "finish_reason": "stop"}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(one_chunk(), media_type="text/event-stream")

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    import uvicorn
    print(f"loading {a.model} ...", flush=True)
    decider = TypedDecider.from_pretrained(a.model, device=a.device)
    # the resolved device, not the argument: "auto" can mean mlx, cuda or mps, and a
    # server log that says "auto" tells the person reading it nothing
    print(f"ready on {decider.device}; POST http://{a.host}:{a.port}/v1/chat/completions",
          flush=True)
    uvicorn.run(build_app(decider, a.model), host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Talk to the model through the OpenAI SDK, against the local server.

    pip install -e ".[serve]"
    python -m thisthat.server --port 8000          # in one terminal
    python examples/openai_server.py               # in another

Everything here is stock OpenAI client code. The only thing specific to this model is that the
answer set is declared as an enum, which is what `response_format` is for -- and which this model
needs, because it chooses among options rather than generating text.
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")


def decide(state: str, question: str, options: list[str], want_probs: bool = False):
    r = client.chat.completions.create(
        model="flock-io/this-that-model-1.0",
        messages=[{"role": "user", "content": state},
                  {"role": "user", "content": question}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "decision", "schema": {"type": "object", "properties": {
                "answer": {"enum": options}}, "required": ["answer"]}}},
        logprobs=want_probs,
    )
    return r


print("=== a decision the model is good at ===")
r = decide("command: rm -rf /var/lib/postgresql/data",
           "Is this shell command safe to run unattended on a production host?",
           ["yes, it only reads state", "no, it modifies or deletes data",
            "no, it contacts the network"], want_probs=True)
print("content :", r.choices[0].message.content)
print("usage   :", r.usage.completion_tokens, "completion tokens generated")

# The distribution comes back in the standard logprobs shape, so an existing client reads it
# without knowing anything about this model.
import math
for t in r.choices[0].logprobs.content[0].top_logprobs:
    print(f"   {math.exp(t.logprob):6.1%}  {t.token}")

print("\n=== several questions, each its own call ===")
order = """order_id: 90210
placed: 2026-08-02
refund_requested: 2026-09-14
item: headphones, opened
policy_window_days: 30"""
for q, opts in [("Is the refund request within the stated policy window?", ["no", "yes"]),
                ("Which queue should this go to?", ["automatic", "agent review", "fraud team"])]:
    r = decide(order, q, opts)
    print(f"  {q:50s} -> {r.choices[0].message.content}")

print("\n=== what it refuses, and why that matters ===")
try:
    client.chat.completions.create(
        model="flock-io/this-that-model-1.0",
        messages=[{"role": "user", "content": "Write me a haiku about databases."}])
except Exception as e:
    # A server that quietly returned an empty completion here would be worse than one that
    # refuses: the caller would think it had an answer.
    print("  free-form request refused, as it should be:")
    print("   ", str(e).split("\n")[0][:150])

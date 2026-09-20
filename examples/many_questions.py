"""Several questions about one state, answered in a single forward pass.

No answer is written back into the prompt, so the decisions are conditionally independent given the
input and can share the pass. Three questions cost one forward pass, not three.
"""
from thisthat import Question, TypedDecider

decider = TypedDecider.from_pretrained()

order = """order_id: 90210
placed: 2026-08-02
delivered: 2026-08-09
refund_requested: 2026-09-14
item: headphones, opened
customer_tier: gold
policy_window_days: 30"""

answers = decider.decide(order, [
    Question("Is the refund request within the stated policy window?", ["no", "yes"]),
    Question("Which queue should this go to?", ["automatic", "agent review", "fraud team"]),
    Question("Risk band", ["low", "medium", "high"]),
])

for a in answers:
    print(f"{a.question:55s} -> {a}")

# schema_first keeps the question block byte-identical across states, so its cache is computed
# once and reused: the layout to use when one decision point sees a stream of orders.
same_questions = [a.question for a in answers]
print("\nprefix reuse:", decider.decide(order, [
    Question(answers[0].question, ["no", "yes"])], layout="schema_first"))

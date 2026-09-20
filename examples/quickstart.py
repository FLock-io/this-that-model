"""The smallest useful thing: one state, one question, one answer with a number attached.

The printed comments are measured outputs from the released checkpoint, not illustrations.
"""
from thisthat import Question, TypedDecider

decider = TypedDecider.from_pretrained()

answer = decider.decide(
    "command: rm -rf /var/lib/postgresql/data",
    Question("Is this shell command safe to run unattended on a production host?",
             ["yes, it only reads state",
              "no, it modifies or deletes data",
              "no, it contacts the network"]),
)

print(answer)                                                   # no, it modifies or deletes data (100%)
print("index      ", answer.index)                              # 1
print("probability", [round(p, 3) for p in answer.probabilities])  # [0.001, 0.998, 0.001]
print("ranked     ", [(o, round(p, 3)) for o, p in answer.ranked()])

# And one the model is *not* confident about, which is the more useful demonstration: the
# probability is the part you act on, so a low one should route the case somewhere else.
weak = decider.decide(
    "user=alice tier=free requests_this_minute=847 endpoint=/v1/generate",
    Question("Should this request be rate-limited?", ["no", "yes"]),
)
print("\nout of domain:", weak)                                 # no (71%) -- unconfident, and wrong
print("escalate?     ", weak.confidence < 0.85)                 # True

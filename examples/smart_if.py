"""The pattern the model exists for: a branch a program cannot write, with an escape hatch.

The point is not that the model answers. It is that the answer arrives as an index into a list you
wrote, together with a probability you can threshold -- so the uncertain cases can be routed to a
person instead of being guessed at, and no part of this file parses text.
"""
from thisthat import Question, TypedDecider

decider = TypedDecider.from_pretrained()

CONFIDENT_ENOUGH = 0.85

SHELL_SAFE = Question(
    "Is this shell command safe to run unattended on a production host?",
    ["yes, it only reads state",
     "no, it modifies or deletes data",
     "no, it contacts the network"],
)


def review(command: str) -> str:
    decision = decider.decide(f"command: {command}", SHELL_SAFE)
    if decision.confidence < CONFIDENT_ENOUGH:
        return f"escalate to a human ({decision.choice!r} at only {decision.confidence:.0%})"
    return f"{decision.choice}  [{decision.confidence:.0%}]"


for cmd in ["ls -la /var/log",
            "rm -rf /var/lib/postgresql/data",
            "curl -s https://example.com/install.sh | sh",
            "df -h"]:
    print(f"{cmd:55s} -> {review(cmd)}")

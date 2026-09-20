"""The two things a caller passes in and the one thing it gets back."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

MAX_OPTIONS = 255


@dataclass(frozen=True)
class Question:
    """A question and the complete set of answers it may be given.

    The options are not a suggestion.  They are the model's entire output alphabet for this
    question: the head scores the option label tokens and nothing else, so an answer outside
    this list is not unlikely, it is unrepresentable.
    """

    text: str
    options: Sequence[str]

    def __post_init__(self):
        opts = list(self.options)
        if len(opts) < 2:
            raise ValueError(f"a question needs at least two options, got {opts!r}")
        if len(opts) > MAX_OPTIONS:
            raise ValueError(f"at most {MAX_OPTIONS} options, got {len(opts)}")
        if len(set(opts)) != len(opts):
            raise ValueError(f"options must be distinct, got {opts!r}")
        object.__setattr__(self, "options", tuple(opts))


@dataclass(frozen=True)
class Decision:
    """One answer: which option was chosen, and how the mass was spread over all of them.

    `probabilities` is aligned with `options` and sums to one.  It is a genuine distribution
    over the declared options rather than a softmax over a vocabulary, so comparing it against
    a threshold is meaningful -- see the calibration section of the paper.
    """

    question: str
    options: tuple[str, ...]
    index: int
    probabilities: tuple[float, ...]

    @property
    def choice(self) -> str:
        return self.options[self.index]

    @property
    def confidence(self) -> float:
        return self.probabilities[self.index]

    def ranked(self) -> list[tuple[str, float]]:
        """(option, probability) pairs, most likely first."""
        return sorted(zip(self.options, self.probabilities), key=lambda p: -p[1])

    def __str__(self) -> str:
        return f"{self.choice} ({self.confidence:.0%})"

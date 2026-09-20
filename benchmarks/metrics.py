"""Scoring, with the denominators written down.

Two conventions here are deliberate and both cost us points, so they are stated rather than buried:

  * a question that got no answer is scored as wrong, not dropped.  Dropping it conditions the
    score on a subset the system selected, and systems do not fail at random -- on the context
    ladder the unanswered questions sit entirely on the long-context arm.
  * accuracy against a sampled outcome has a ceiling below 1 whenever the truth is a probability
    rather than a label, so `ceiling` is reported next to it.  A model cannot beat the coin.
"""
from __future__ import annotations

import math
from typing import Sequence


def accuracy(picks: Sequence[int | None], gold: Sequence[int]) -> float:
    """Right answers over questions asked.  `None` counts as wrong; see the module docstring."""
    return sum(1 for p, g in zip(picks, gold) if p == g) / len(gold)


def no_answer_rate(picks: Sequence[int | None]) -> float:
    return sum(1 for p in picks if p is None) / len(picks)


def brier(p_yes: Sequence[float], truth: Sequence[float]) -> float:
    """Scalar Bernoulli Brier score, the form the recorded cohort reports."""
    return sum((p - t) ** 2 for p, t in zip(p_yes, truth)) / len(truth)


def nll(p_yes: Sequence[float], truth: Sequence[float], eps: float = 1e-9) -> float:
    return -sum(t * math.log(max(p, eps)) + (1 - t) * math.log(max(1 - p, eps))
                for p, t in zip(p_yes, truth)) / len(truth)


def squared_error_to_distribution(probs: Sequence[Sequence[float]],
                                  true_dist: Sequence[Sequence[float]]) -> float:
    """Mean squared distance to the *true distribution*, not to a sampled label.

    Where the environment defines a real probability, this is the quantity worth reporting:
    accuracy against one draw is noisy and capped, while distance to the distribution is neither.
    """
    return sum(sum((a - b) ** 2 for a, b in zip(p, q))
               for p, q in zip(probs, true_dist)) / len(true_dist)


def accuracy_ceiling(true_dist: Sequence[Sequence[float]]) -> float:
    """The best accuracy any predictor can reach against outcomes sampled from `true_dist`."""
    return sum(max(q) for q in true_dist) / len(true_dist)


def constant_predictor_error(true_dist: Sequence[Sequence[float]]) -> float:
    """The squared error of always predicting the mean distribution -- the bar to clear."""
    n, k = len(true_dist), len(true_dist[0])
    mean = [sum(q[j] for q in true_dist) / n for j in range(k)]
    return squared_error_to_distribution([mean] * n, true_dist)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Binomial interval. Worth printing whenever n is in the dozens, as it usually is here."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)

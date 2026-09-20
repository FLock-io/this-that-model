#!/usr/bin/env python3
"""Reproduce the calibration result: a question whose true answer is a probability.

    python scripts/reproduce_simulator.py

A noisy actuator executes the intended move with probability rho and otherwise picks uniformly
among the alternatives, so the probability that the executed move is safe follows from the
transition rules rather than from anyone's opinion.  That makes two measurements available which
a labelled dataset cannot support:

    accuracy   against one sampled outcome. Capped: no predictor can beat E[max(q, 1-q)].
    qL2        squared distance to the true distribution. Not capped, and the honest target.

The bar to clear is not chance, it is the constant predictor -- always answering the mean
probability, having read nothing.  Most hosted endpoints do not expose a probability at all; of
those that do, the paper found every one worse than that constant.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.metrics import (accuracy, accuracy_ceiling, constant_predictor_error,
                                squared_error_to_distribution)
from benchmarks.simulator.tasks import stochastic_set
from thisthat import DEFAULT_MODEL, TypedDecider


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("-n", type=int, default=200, help="questions")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    items = stochastic_set(n=a.n, seed=a.seed)
    truth = [it.truth[0] for it in items]
    gold = [it.answer_index[0] for it in items]
    print(f"{len(items)} noisy-actuator questions over unseen maps")

    decider = TypedDecider.from_pretrained(a.model, device=a.device)
    answers = decider.decide_batch([(it.state, it.questions) for it in items], batch_size=8)
    probs = [list(d[0].probabilities) for d in answers]
    picks = [d[0].index for d in answers]

    ceiling = accuracy_ceiling(truth)
    const = constant_predictor_error(truth)
    acc = accuracy(picks, gold)
    q_l2 = squared_error_to_distribution(probs, truth)

    print(f"\n{'':36s} {'accuracy':>9} {'qL2':>9}")
    print(f"{'accuracy ceiling (computed)':36s} {ceiling:9.3f} {'--':>9}")
    print(f"{'constant predictor':36s} {'--':>9} {const:9.4f}")
    print(f"{'this model':36s} {acc:9.3f} {q_l2:9.4f}")
    # The ceiling is an expectation over the true probabilities; accuracy is measured against one
    # draw from them.  Landing a little above it is sampling noise, not a broken bound, and
    # printing "101% of the ceiling" invites the opposite reading.
    gap = ceiling - acc
    verdict = "at the ceiling (within sampling noise)" if gap <= 0.02 else \
        f"{gap:.3f} below the ceiling"
    print(f"\n  accuracy is {verdict};")
    print(f"  the distribution is {const / q_l2:.1f}x closer to the truth than a constant's.")
    print("\n  This regenerates the evaluation set from the simulator rather than replaying a\n"
          "  fixed file, so the third decimal moves between runs of different sizes. The result\n"
          "  being reproduced is the shape: accuracy at the computed ceiling, and a squared error\n"
          "  several times smaller than the constant predictor a calibration claim has to beat.")
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"model": a.model, "n": len(items), "accuracy": acc, "q_l2": q_l2,
             "ceiling": ceiling, "constant_q_l2": const}, indent=1))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

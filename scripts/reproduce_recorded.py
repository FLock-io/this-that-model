#!/usr/bin/env python3
"""Reproduce the recorded-cohort result: 68 decision questions over 17 states.

    python scripts/reproduce_recorded.py                          # the released checkpoint
    python scripts/reproduce_recorded.py --model /path/to/local    # any typed-decision checkpoint

A third party recorded a hosted commercial decision service answering these questions, keeping the
state it was shown, its returned probability and the computed truth.  That makes an unusual thing
available: a same-input comparison against a commercial system rather than against its published
summary statistics.  The question wording is theirs, not ours -- our model was trained on a
different phrasing of the same question, so reusing theirs measures transfer rather than recall,
which is the conservative direction and the only one that makes the comparison mean anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.metrics import accuracy, brier, nll, wilson
from thisthat import DEFAULT_MODEL, Question, TypedDecider

DATA = Path(__file__).resolve().parent.parent / "data" / "recorded_68.jsonl"
EXPECTED = {"accuracy": 0.926, "brier": 0.046, "nll": 0.139}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--temperature", type=float, default=1.3,
                    help="the temperature these published numbers were produced at")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--json", default="", help="also write the results here")
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(a.data)]
    states = len({r["state"] for r in rows})
    print(f"{len(rows)} recorded questions over {states} states, from {Path(a.data).name}")

    decider = TypedDecider.from_pretrained(a.model, device=a.device)
    items = [(r["state"], [Question(r["question"], r["options"])]) for r in rows]
    answers = decider.decide_batch(items, batch_size=8, temperature=a.temperature,
                                   max_state_tokens=2048)

    picks = [d[0].index for d in answers]
    p_yes = [d[0].probabilities[1] for d in answers]
    gold = [r["answer_index"] for r in rows]
    truth = [float(g) for g in gold]

    acc = accuracy(picks, gold)
    lo, hi = wilson(sum(1 for p, g in zip(picks, gold) if p == g), len(gold))
    got = {"accuracy": acc, "brier": brier(p_yes, truth), "nll": nll(p_yes, truth)}

    ref = [r["recorded_service_p_yes"] for r in rows]
    if all(x is not None for x in ref):
        svc = {"accuracy": accuracy([1 if x >= 0.5 else 0 for x in ref], gold),
               "brier": brier(ref, truth), "nll": nll(ref, truth)}
    else:
        svc = None

    print(f"\n{'':34s} {'accuracy':>9} {'Brier':>8} {'NLL':>8}")
    if svc:
        print(f"{'hosted service (recorded)':34s} {svc['accuracy']:9.3f} "
              f"{svc['brier']:8.3f} {svc['nll']:8.3f}")
    maj = max(sum(gold) / len(gold), 1 - sum(gold) / len(gold))
    print(f"{'majority-class baseline':34s} {maj:9.3f} {'--':>8} {'--':>8}")
    print(f"{'this model':34s} {got['accuracy']:9.3f} {got['brier']:8.3f} {got['nll']:8.3f}")
    print(f"{'  95% interval on accuracy':34s} [{lo:.3f}, {hi:.3f}]   (n={len(gold)})")

    # A reproduction script that cannot fail is a demo.  Compare against what we published, and
    # say plainly when the machine in front of you does not agree with the paper.
    print()
    ok = True
    for k, want in EXPECTED.items():
        delta = abs(got[k] - want)
        tol = 0.02 if k == "accuracy" else 0.01
        flag = "ok" if delta <= tol else "DIFFERS"
        ok &= delta <= tol
        print(f"  {k:10s} got {got[k]:.3f}  published {want:.3f}  ({flag})")
    if a.json:
        Path(a.json).write_text(json.dumps({"model": a.model, "n": len(rows),
                                            "ours": got, "service": svc}, indent=1))
        print(f"\nwrote {a.json}")
    if not ok:
        print("\nNumbers differ from the published ones. Kernel and dtype differences move the\n"
              "third decimal; anything larger is worth reporting as an issue.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

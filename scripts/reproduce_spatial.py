#!/usr/bin/env python3
"""Score the model on the released spatial benchmark, and on the four held-out families.

    python scripts/reproduce_spatial.py                    # pulls the dataset from the Hub
    python scripts/reproduce_spatial.py --limit 150        # the subset the hosted models used

Two numbers come out of this and they answer different questions.

The benchmark number is in-distribution. This checkpoint was trained on all fifteen of these
question shapes -- on different items, over different maze windows, both excluded by fingerprint
and by rendered state -- so it is not comparable with a hosted model's zero-shot score on the same
set, and the script prints the caveat rather than leaving it to a footnote.

The per-surface split is the one worth watching if you are deciding whether this model fits your
data. The same maze window rendered three ways used to produce 0.457, 0.373 and 0.420; the JSON
column was chance. It is now roughly uniform. If your state looks like none of the three, measure
before you trust it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.metrics import accuracy, wilson
from thisthat import DEFAULT_MODEL, Question, TypedDecider

BENCH = "limberc/this-that-spatial-bench"


def renderer_of(state: str) -> str:
    if state.startswith("{"):
        return "json"
    if state.startswith("The agent stands at row"):
        return "rows"
    if state.startswith("Agent coordinate:"):
        return "ascii"
    if state.startswith("Snake game"):
        return "snake"
    return "whole map"


def load(path: str | None):
    if path and Path(path).exists():
        return [json.loads(l) for l in open(path)]
    from huggingface_hub import hf_hub_download
    f = hf_hub_download(BENCH, "test.jsonl", repo_type="dataset")
    return [json.loads(l) for l in open(f)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data", default="", help="local test.jsonl; downloaded from the Hub if unset")
    ap.add_argument("--limit", type=int, default=0, help="questions per family (0 = all 7,305)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    recs = load(a.data)
    if a.limit:
        per, keep = defaultdict(int), []
        for r in recs:
            if per[r["family"]] < a.limit:
                keep.append(r)
                per[r["family"]] += 1
        recs = keep
    print(f"{len(recs)} questions over {len({r['family'] for r in recs})} families")

    decider = TypedDecider.from_pretrained(a.model, device=a.device)
    out = decider.decide_batch([(r["state"], [Question(r["question"], r["options"])])
                                for r in recs], batch_size=a.batch_size, progress=True)
    picks = [d[0].index for d in out]
    gold = [r["answer_index"] for r in recs]

    acc = accuracy(picks, gold)
    lo, hi = wilson(sum(1 for p, g in zip(picks, gold) if p == g), len(gold))
    chance = sum(1 / len(r["options"]) for r in recs) / len(recs)
    print(f"\noverall {acc:.3f}  [{lo:.3f}, {hi:.3f}]   chance {chance:.3f}")
    print("\n  This is an in-distribution number: the checkpoint was trained on all fifteen of")
    print("  these question shapes, on different items and different maze windows. It is not")
    print("  comparable with a hosted model's zero-shot score on the same set.")

    by = defaultdict(lambda: [0, 0, 0.0])
    fam = defaultdict(lambda: [0, 0, 0.0])
    for r, p in zip(recs, picks):
        for d, k in ((by, renderer_of(r["state"])), (fam, r["family"])):
            d[k][0] += (p == r["answer_index"])
            d[k][1] += 1
            d[k][2] += 1 / len(r["options"])
    print(f"\n{'rendering':14s} {'n':>6} {'acc':>7} {'chance':>7}")
    for k in ("ascii", "json", "rows", "snake", "whole map"):
        if k in by:
            ok, n, ch = by[k]
            print(f"{k:14s} {n:6d} {ok/n:7.3f} {ch/n:7.3f}")
    print(f"\n{'family':18s} {'n':>6} {'acc':>7} {'chance':>7}")
    for k, (ok, n, ch) in sorted(fam.items(), key=lambda x: x[1][0] / x[1][1]):
        print(f"{k:18s} {n:6d} {ok/n:7.3f} {ch/n:7.3f}")
    print("\n  The families at the top of that list are the ones that need a search over the map.")
    print("  They are where this model stops, and no amount of data of the same kind moved them.")
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"model": a.model, "n": len(recs), "acc": acc, "chance": chance,
             "surfaces": {k: {"n": v[1], "acc": v[0] / v[1]} for k, v in by.items()},
             "families": {k: {"n": v[1], "acc": v[0] / v[1]} for k, v in fam.items()}}, indent=1))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

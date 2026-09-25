#!/usr/bin/env python3
"""Do the backends agree? Measured against float32 on the CPU, which is the one without an excuse.

    python scripts/compare_backends.py --model flock-io/this-that-model-1.1

Every backend here computes the same restricted softmax; they differ only in the precision they
carry through the backbone.  So the question is not "does MLX match PyTorch" -- both are
approximations -- but "is the MLX approximation at least as good as the one this repository has
already been shipping on Apple Silicon", which is float16 on MPS.

float32 on the CPU is the reference for that comparison.  It is slow and nobody would serve from
it, which is exactly why it is the right yardstick: it is the only configuration whose error is
not the thing under test.

What comes out is, per backend, the largest and mean absolute difference in per-option
probability across the recorded items, and the number of items whose argmax moves -- the last of
which is the one that changes an answer rather than a digit.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thisthat import DEFAULT_MODEL, Question, TypedDecider

REFERENCE = ("cpu", {"dtype": "float32"})
CANDIDATES = [("mlx", {"dtype": "float16"}), ("mlx", {"dtype": "bfloat16"}),
              ("mlx", {"dtype": "float32"}), ("mps", {"dtype": "float16"})]

SPATIAL = "limberc/this-that-spatial-bench"


def spec(text: str):
    """"mlx:float16" -> ("mlx", {"dtype": "float16"}); "cuda" -> ("cuda", {})."""
    device, _, dtype = text.partition(":")
    return device, ({"dtype": dtype} if dtype else {})


def load_items(data: str, limit: int):
    """The recorded set by default; the released benchmark when asked, for a decisive n."""
    root = Path(__file__).resolve().parent.parent
    if data == "spatial":
        from collections import defaultdict

        from huggingface_hub import hf_hub_download
        recs = [json.loads(l) for l in
                open(hf_hub_download(SPATIAL, "test.jsonl", repo_type="dataset"))]
        if limit:
            per, keep = defaultdict(int), []
            for r in recs:
                if per[r["family"]] < limit:
                    keep.append(r); per[r["family"]] += 1
            recs = keep
        return recs
    recs = [json.loads(l) for l in open(root / data)]
    return recs[:limit] if limit else recs


def probabilities(model: str, device: str, kw: dict, items, batch_size: int):
    """-> (list of per-item probability tuples, the string the decider reports for itself)."""
    decider = TypedDecider.from_pretrained(model, device=device, **kw)
    where = decider.device
    out = decider.decide_batch(items, batch_size=batch_size, progress=True)
    probs = [d[0].probabilities for d in out]
    del decider
    gc.collect()
    return probs, where


def compare(ref, got):
    """Absolute probability error and argmax movement, over items with the same option count."""
    diffs, moved = [], []
    for i, (a, b) in enumerate(zip(ref, got)):
        d = np.abs(np.asarray(a) - np.asarray(b))
        diffs.append(d.max())
        if int(np.argmax(a)) != int(np.argmax(b)):
            moved.append((i, float(max(a)), float(max(b))))
    return np.array(diffs), moved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default="data/recorded_68.jsonl",
                    help="a local jsonl, or 'spatial' for the released benchmark")
    ap.add_argument("--limit", type=int, default=0, help="items (per family, for 'spatial')")
    ap.add_argument("--reference", default="cpu:float32",
                    help="the yardstick, as device[:dtype]; float32 on the CPU unless the item "
                         "count makes that impractical, in which case say so in the write-up")
    ap.add_argument("--candidates", default="",
                    help="comma-separated device[:dtype]; defaults to every backend worth trying")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    recs = load_items(a.data, a.limit)
    items = [(r["state"], [Question(r["question"], r["options"])]) for r in recs]
    gold = [r.get("answer_index") for r in recs]
    print(f"{len(items)} items from {a.data}, {a.model}\n")

    device, kw = spec(a.reference)
    print(f"reference: {device} {kw}")
    ref, ref_where = probabilities(a.model, device, kw, items, a.batch_size)

    candidates = [spec(c) for c in a.candidates.split(",")] if a.candidates else CANDIDATES
    report = {"model": a.model, "data": a.data, "n_items": len(items), "reference": ref_where,
              "backends": {}}
    if gold[0] is not None:
        report["reference_accuracy"] = float(
            np.mean([int(np.argmax(p)) == g for p, g in zip(ref, gold)]))
        print(f"  reference accuracy {report['reference_accuracy']:.4f}")
    for device, kw in candidates:
        print(f"\ncandidate: {device} {kw}")
        try:
            got, where = probabilities(a.model, device, kw, items, a.batch_size)
        except Exception as e:                      # a machine need not have every backend
            print(f"  skipped: {type(e).__name__}: {e}")
            continue
        diffs, moved = compare(ref, got)
        acc = (float(np.mean([int(np.argmax(p)) == g for p, g in zip(got, gold)]))
               if gold[0] is not None else None)
        report["backends"][f"{where}"] = {
            "accuracy": acc,
            "max_abs_prob_diff": float(diffs.max()),
            "mean_abs_prob_diff": float(diffs.mean()),
            "p99_abs_prob_diff": float(np.percentile(diffs, 99)),
            "argmax_disagreements": len(moved),
            "disagreements": [{"item": i, "ref_confidence": c0, "got_confidence": c1}
                              for i, c0, c1 in moved],
        }
        print(f"  vs {ref_where}: max |dp| {diffs.max():.2e}, mean {diffs.mean():.2e}, "
              f"argmax moved on {len(moved)}/{len(items)}"
              + (f", accuracy {acc:.4f}" if acc is not None else ""))

    print()
    for where, r in report["backends"].items():
        n = report["n_items"]
        agree = 1 - r["argmax_disagreements"] / n
        print(f"  {where:<18} answers agree on {agree:.4%} of {n}  "
              f"max |dp| {r['max_abs_prob_diff']:.2e}"
              + (f"  accuracy {r['accuracy']:.4f}" if r["accuracy"] is not None else ""))

    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

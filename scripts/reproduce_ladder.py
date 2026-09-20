#!/usr/bin/env python3
"""Reproduce the decomposition result: does the whole state have to be read?

    python scripts/reproduce_ladder.py --sizes 32,50,80

The same questions, about the same worlds, asked twice: once from the whole rendered maze and once
from the agent-centred window the answer provably depends on. The truth is identical, so every
difference between the two curves is the cost of handing the model the whole thing.

Expect the windowed curve to stay flat and near-perfect as the world grows while the whole-state
curve decays, and expect the whole-state arm to run out of memory before 200x200 on a 16 GB card.
That failure is a result, not a crash: it is reported as a rung rather than swallowed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.metrics import accuracy
from benchmarks.simulator.tasks import context_ladder
from thisthat import DEFAULT_MODEL, TypedDecider


def score(decider, items, batch_size, max_state_tokens):
    t0 = time.perf_counter()
    answers = decider.decide_batch([(it.state, it.questions) for it in items],
                                   batch_size=batch_size, max_state_tokens=max_state_tokens)
    ms = (time.perf_counter() - t0) * 1000
    picks = [d.index for row in answers for d in row]
    gold = [g for it in items for g in it.answer_index]
    return accuracy(picks, gold), ms / max(1, len(picks))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sizes", default="32,50,80,128,200")
    ap.add_argument("--maps", type=int, default=4, help="mazes per size")
    ap.add_argument("--starts", type=int, default=3, help="agent positions per maze")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_state_tokens", type=int, default=32768)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    sizes = [int(x) for x in a.sizes.split(",") if x.strip()]
    sets = context_ladder(sizes=sizes, maps_per_size=a.maps, starts_per_map=a.starts)
    decider = TypedDecider.from_pretrained(a.model, device=a.device)

    print(f"{'size':>6}  {'whole state':>22}  {'the part it depends on':>24}")
    print(f"{'':>6}  {'accuracy':>10}{'ms/q':>12}  {'accuracy':>12}{'ms/q':>12}")
    out = {}
    for size in sizes:
        loc_acc, loc_ms = score(decider, sets[f"sim_local_{size}"], a.batch_size,
                                a.max_state_tokens)
        try:
            glob_acc, glob_ms = score(decider, sets[f"sim_global_{size}"], a.batch_size,
                                      a.max_state_tokens)
            cells = f"{glob_acc:10.3f}{glob_ms:12.1f}"
        except RuntimeError as e:
            # an allocation failure at this rung is the measurement, so record it and continue
            if "out of memory" not in str(e).lower():
                raise
            import torch
            torch.cuda.empty_cache()
            glob_acc, glob_ms = None, None
            cells = f"{'out of memory':>22}"
        print(f"{size:>6}  {cells}  {loc_acc:12.3f}{loc_ms:12.1f}")
        out[size] = {"whole_acc": glob_acc, "whole_ms": glob_ms,
                     "window_acc": loc_acc, "window_ms": loc_ms,
                     "n_questions": sum(len(it.questions) for it in sets[f"sim_local_{size}"])}

    win = [v["window_acc"] for v in out.values()]
    print(f"\n  windowed accuracy spans {min(win):.3f} to {max(win):.3f} across "
          f"{min(sizes)}x{min(sizes)} to {max(sizes)}x{max(sizes)};")
    print("  the state handed to the model is the same size at every rung, which is the point.")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

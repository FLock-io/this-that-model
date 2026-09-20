#!/usr/bin/env python3
"""Measure what one decision costs in wall-clock time, honestly.

    python scripts/measure_latency.py

Three things make the difference between a number you can quote and one you cannot:

  * batch one.  A throughput figure divided by a batch size is not the latency a caller sees.
  * the device is synchronised around each timed call.  CUDA is asynchronous; without a
    synchronise you are timing the queue submission, which is fast and meaningless.
  * warm-up iterations are discarded.  The first calls pay for kernel autotuning and allocation.

We report the spread as well as the middle. The published figure is 30.9 +/- 16.6 ms with a median
of 27.3 ms, and the gap between mean and median is the point: the distribution has a tail, and a
caller sizing a timeout needs the tail rather than the average.

Two reasons your number will differ, both benign. The published spread came from a suite whose
prompts vary much more in length than this cohort's fixed-size windows do, so expect a tighter
distribution here. And a laptop GPU that has been busy for a while is a throttled laptop GPU: a
few milliseconds above the published mean on a warm machine is thermal, not regression. Run it on
a cold card before concluding anything from a small difference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thisthat import DEFAULT_MODEL, Question, TypedDecider

DATA = Path(__file__).resolve().parent.parent / "data" / "recorded_68.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    # Across the benchmark's real items, not one prompt repeated: prompt length drives the cost,
    # and repeating a single item reports the machine's jitter (sd ~2 ms) instead of the spread a
    # caller will actually meet (sd ~17 ms). The published figure is measured this way.
    rows = [json.loads(line) for line in open(DATA)]
    items = [(r["state"], [Question(r["question"], r["options"])]) for r in rows]

    decider = TypedDecider.from_pretrained(a.model, device=a.device)
    print(f"device {decider.device}, batch 1, {len(items)} distinct items, "
          f"{a.warmup} warm-up iterations discarded")
    r = decider.time_per_decision(items, repeats=a.repeats, warmup=a.warmup)

    print(f"\n  mean   {r['mean_ms']:7.2f} ms  (sd {r['sd_ms']:.2f})")
    for k in ("p50_ms", "p90_ms", "p99_ms"):
        print(f"  {k[:3]}    {r[k]:7.2f} ms")
    print(f"\n  {1000 / r['mean_ms']:.0f} decisions per second, one question per pass")
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

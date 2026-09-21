#!/usr/bin/env python3
"""Score Laya on our spatial benchmark, under the contract our own models answer under.

Comparable means the same items, the same option sets, and the same rule for a missing answer: a
system that returns nothing is wrong, not excused. Laya's `criteria` field wants a label-to-
description mapping and our options are already the labels, so each maps to itself -- adding
descriptions we wrote would be giving one side a prompt the other never had.

The 150-per-family subset is used because that is the subset the hosted models in the paper were
scored on, which is the only cohort this number can join.
"""
from __future__ import annotations

import argparse, json, time
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/spatial_bench/test.jsonl")
    ap.add_argument("--subfolder", default="typed-decisions")
    ap.add_argument("--per_family", type=int, default=150)
    ap.add_argument("--out", default="runs/laya_spatial.json")
    a = ap.parse_args()

    import laya
    rows = [json.loads(l) for l in open(a.bench)]
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    items = [r for fam in sorted(by_fam) for r in by_fam[fam][:a.per_family]]
    print(f"{len(items)} items over {len(by_fam)} families, {a.per_family} per family")

    agent = laya.load("convaiinnovations/laya", subfolder=a.subfolder)
    print(f"loaded convaiinnovations/laya:{a.subfolder}")

    res = defaultdict(lambda: dict(n=0, ok=0, none=0, chance=0.0, ms=[]))
    t0 = time.time()
    for i, r in enumerate(items):
        q = {"q": {"type": "choice", "instructions": r["question"],
                   "criteria": {o: o for o in r["options"]}}}
        s = time.perf_counter()
        try:
            out = agent.predict({"body": r["state"]}, q)
            pick = out["answers"]["q"].get("choice")
        except Exception:
            pick = None
        ms = (time.perf_counter() - s) * 1000
        d = res[r["family"]]
        d["n"] += 1
        d["chance"] += 1 / len(r["options"])
        d["ms"].append(ms)
        if pick is None:
            d["none"] += 1            # unanswered counts as wrong, as it does for every system here
        elif pick == r["answer"]:
            d["ok"] += 1
        if i % 200 == 0:
            print(f"  {i+1}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

    out = {}
    N = OK = NONE = 0
    CH = 0.0
    allms = []
    print(f"\n{'family':20s} {'n':>5} {'acc':>7} {'chance':>7} {'no-ans':>7} {'ms/q':>7}")
    for fam in sorted(res):
        d = res[fam]
        acc = d["ok"] / d["n"]
        ch = d["chance"] / d["n"]
        med = sorted(d["ms"])[len(d["ms"]) // 2]
        out[fam] = dict(n=d["n"], acc=acc, chance=ch, no_answer=d["none"], median_ms=med)
        N += d["n"]; OK += d["ok"]; NONE += d["none"]; CH += d["chance"]; allms += d["ms"]
        print(f"{fam:20s} {d['n']:5d} {acc:7.3f} {ch:7.3f} {d['none']:7d} {med:7.0f}")
    allms.sort()
    out["__all"] = dict(n=N, acc=OK / N, chance=CH / N, no_answer=NONE,
                        median_ms=allms[len(allms) // 2], p90_ms=allms[int(len(allms) * 0.9)])
    print(f"\n{'ALL':20s} {N:5d} {OK/N:7.3f} {CH/N:7.3f} {NONE:7d} "
          f"{allms[len(allms)//2]:7.0f}   p90 {allms[int(len(allms)*0.9)]:.0f}ms")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

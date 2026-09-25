#!/usr/bin/env python3
"""What quantisation costs a model whose output is a probability.

    python scripts/measure_mlx_quantization.py --bits 8 4

For a generative model, quantisation is judged by whether the text still reads well. There is no
equivalent here. The output is a calibrated distribution over the declared options, and the
number a caller thresholds is the thing quantisation perturbs, so "the answers still look right"
is not a measurement -- accuracy can survive a quantisation that has quietly ruined the
confidence it comes with.

So this reports accuracy and the Brier score side by side, against the published values and their
tolerances, plus the per-option probability drift from float32 on the CPU and the latency bought.
The Brier column is the one that decides it.

Quantising needs a local directory, because `mlx_lm.convert` resolves the architecture from the
config's `model_type` and this checkpoint's is `qwen3_5_text`, which is not a module name. The
script writes a source tree that symlinks the weights and rewrites that one field.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.metrics import accuracy, brier
from thisthat import DEFAULT_MODEL, Question, TypedDecider
from thisthat.backends.mlx_backend import MLX_MODEL_TYPE

# the conditions the published-results tests use, so the numbers are comparable to the paper's
TEMPERATURE = 1.3
MAX_STATE_TOKENS = 2048
PUBLISHED = {"accuracy": (0.926, 0.03), "brier": (0.046, 0.02)}


def prepared_source(model: str, out: Path) -> Path:
    """A directory `mlx_lm.convert` can read: the weights symlinked, `model_type` rewritten."""
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.jinja"]))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in src.iterdir():
        if f.name != "config.json":
            (out / f.name).symlink_to(f.resolve())
    config = json.loads((src / "config.json").read_text())
    config["model_type"] = MLX_MODEL_TYPE
    (out / "config.json").write_text(json.dumps(config, indent=2))
    return out


def quantize(source: Path, bits: int, out: Path, group_size: int) -> Path:
    from mlx_lm.convert import convert
    if out.exists():
        shutil.rmtree(out)
    convert(str(source), str(out), quantize=True, q_bits=bits, q_group_size=group_size)
    return out


def score(path: str, device: str, items, gold, truth, **kw) -> dict:
    decider = TypedDecider.from_pretrained(path, device=device, **kw)
    out = decider.decide_batch(items, batch_size=8, temperature=TEMPERATURE,
                               max_state_tokens=MAX_STATE_TOKENS)
    probs = [d[0].probabilities for d in out]
    row = {"where": decider.device,
           "accuracy": accuracy([d[0].index for d in out], gold),
           "brier": brier([p[1] for p in probs], truth)}
    row["latency_ms"] = decider.time_per_decision(items[:20], repeats=40, warmup=10)["mean_ms"]
    del decider
    gc.collect()
    return row, probs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--bits", type=int, nargs="*", default=[8, 4])
    ap.add_argument("--group_size", type=int, default=64)
    ap.add_argument("--work", default="/tmp/thisthat-mlx-quant")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    recs = [json.loads(l) for l in open(root / "data" / "recorded_68.jsonl")]
    items = [(r["state"], [Question(r["question"], r["options"])]) for r in recs]
    gold = [r["answer_index"] for r in recs]
    truth = [float(r["answer_index"]) for r in recs]
    work = Path(a.work)

    print(f"{a.model}, {len(items)} recorded items, temperature {TEMPERATURE}\n")
    print("float32 on the CPU, for the drift column")
    ref_row, ref = score(a.model, "cpu", items, gold, truth, dtype="float32")

    rows = {}
    row, probs = score(a.model, "mlx", items, gold, truth)
    rows["float16"] = row | {"drift": float(max(np.abs(np.array(p) - np.array(q)).max()
                                                for p, q in zip(ref, probs)))}

    source = prepared_source(a.model, work / "source")
    for bits in a.bits:
        print(f"\nquantising to {bits} bits")
        path = quantize(source, bits, work / f"q{bits}", a.group_size)
        row, probs = score(str(path), "mlx", items, gold, truth)
        rows[f"{bits}bit"] = row | {"drift": float(max(np.abs(np.array(p) - np.array(q)).max()
                                                       for p, q in zip(ref, probs)))}

    acc0, acc_tol = PUBLISHED["accuracy"]
    br0, br_tol = PUBLISHED["brier"]
    print(f"\n{'weights':<10} {'accuracy':>9} {'brier':>8} {'max |dp|':>10} {'ms':>7}   verdict")
    print(f"{'cpu fp32':<10} {ref_row['accuracy']:>9.4f} {ref_row['brier']:>8.4f} "
          f"{'--':>10} {ref_row['latency_ms']:>7.1f}   reference")
    for name, r in rows.items():
        ok = abs(r["accuracy"] - acc0) <= acc_tol and abs(r["brier"] - br0) <= br_tol
        print(f"{name:<10} {r['accuracy']:>9.4f} {r['brier']:>8.4f} {r['drift']:>10.2e} "
              f"{r['latency_ms']:>7.1f}   {'within published tolerance' if ok else 'OUTSIDE'}")
    print(f"\npublished: accuracy {acc0} (+-{acc_tol}), brier {br0} (+-{br_tol})")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"model": a.model, "temperature": TEMPERATURE, "reference": ref_row,
             "published": PUBLISHED, "weights": rows}, indent=1) + "\n")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

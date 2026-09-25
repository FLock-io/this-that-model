"""The typed decision head: a backbone, one hidden state per answer slot, a restricted softmax.

There is no decoding loop here, and that is the whole point.  A generative model answers a
multiple-choice question by sampling tokens and hoping they spell one of the options; this one
reads the hidden state at each answer slot, scores it against the embedding of each option's
label token, and normalises over exactly those options:

    p_k(j | x) = softmax_j( <w_l(k,j), h_k> / tau )

The support of that distribution is the option list.  A malformed answer is therefore not
improbable, it is outside the sample space -- there is no token budget to exhaust, no letter to
mis-parse, and no retry path to write.

Which framework computes the inner product is not part of that claim, and lives in `backends/`.
This file owns the parts that decide what the published numbers are: how a prompt is batched,
what temperature does, and how logits become a distribution.  Those exist once and both backends
share them, so the two cannot drift apart in the arithmetic that callers actually read.
"""
from __future__ import annotations

import time
from typing import Iterable, Sequence

import numpy as np

from .backends import best_device, load_backend
from .prompt import DEFAULT_MAX_STATE_TOKENS, build
from .types import Decision, Question

DEFAULT_MODEL = "flock-io/this-that-model-1.0"

__all__ = ["DEFAULT_MODEL", "TypedDecider", "best_device"]


def _pad_to(n: int, multiple: int = 64) -> int:
    """Round the batch width up: a handful of distinct shapes means far fewer kernel recompiles."""
    return ((n + multiple - 1) // multiple) * multiple


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Row-wise softmax in float32, with -inf rows entries meaning "not an option here".

    float32 rather than the model's dtype: in half precision a masked row can underflow to all
    zeros, and a row of zeros renormalises into nonsense instead of failing.  Every row has at
    least two finite entries, so the row max is always finite and the shift is always safe.
    """
    z = logits.astype(np.float32) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class TypedDecider:
    """Answer typed questions about a state.

        decider = TypedDecider.from_pretrained()
        decider.decide("temp=91C, fan=off", Question("Throttle?", ["no", "yes"]))
    """

    def __init__(self, backend):
        self.backend = backend
        self.tokenizer = backend.tokenizer

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, name_or_path: str = DEFAULT_MODEL, *, device: str = "auto",
                        **kw) -> TypedDecider:
        """Load the model.

        `device` may be "auto" (the default), "mlx", "cuda", "mps" or "cpu".  On Apple Silicon
        with mlx-lm installed, "auto" means "mlx"; the MLX backend reads the published bf16
        checkpoint directly, so there is no conversion step and no separate repository.

        Remaining keyword arguments go to the backend -- `dtype=` to the torch one, for instance.
        """
        return cls(load_backend(name_or_path, device=device, **kw))

    @property
    def device(self):
        """Where the forward pass runs, as a string: "cuda", "mps", "mlx (bfloat16)", ..."""
        return self.backend.device

    @property
    def model(self):
        return self.backend.model

    # ------------------------------------------------------------------ public API
    def decide(self, state: str, questions: Question | Sequence[Question], *,
               temperature: float = 1.0, layout: str = "state_first",
               max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS):
        """Answer every question about `state` in a single forward pass.

        Returns one `Decision` if given one `Question`, else a list in the order asked.
        """
        single = isinstance(questions, Question)
        qs = [questions] if single else list(questions)
        out = self.decide_batch([(state, qs)], temperature=temperature, layout=layout,
                                max_state_tokens=max_state_tokens)[0]
        return out[0] if single else out

    def decide_batch(self, items: Iterable[tuple[str, Sequence[Question]]], *,
                     batch_size: int = 8, temperature: float = 1.0,
                     layout: str = "state_first",
                     max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS,
                     progress: bool = False) -> list[list[Decision]]:
        """Answer many (state, questions) pairs. Returns a list of answer-lists, input order."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        items = [(s, list(q)) for s, q in items]
        built = [build(self.tokenizer, s, q, layout=layout, max_state_tokens=max_state_tokens)
                 for s, q in items]
        # sort by length so a batch is not mostly padding, then restore the caller's order
        order = sorted(range(len(built)), key=lambda i: len(built[i]["ids"]))
        results: list[list[Decision] | None] = [None] * len(built)
        pad_id = self.backend.pad_id

        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            width = _pad_to(max(len(built[i]["ids"]) for i in chunk))
            # right padding: a causal model cannot see past the end of its own row, so the pads
            # change nothing about the real tokens and every real position keeps its index
            ids = np.full((len(chunk), width), pad_id, dtype=np.int64)
            attn = np.zeros((len(chunk), width), dtype=np.int64)
            slot_idx, slot_batch, n_opt, owner = [], [], [], []
            for b, i in enumerate(chunk):
                row = built[i]["ids"]
                ids[b, :len(row)] = row
                attn[b, :len(row)] = 1
                for k, s in enumerate(built[i]["slots"]):
                    slot_idx.append(s); slot_batch.append(b)
                    n_opt.append(built[i]["n_options"][k]); owner.append((i, k))
            logits = self.backend.slot_logits(ids, attn, np.array(slot_idx), np.array(slot_batch),
                                              np.array(n_opt))
            probs = _softmax(logits, temperature)
            for row_i, (i, k) in enumerate(owner):
                q = items[i][1][k]
                p = [float(x) for x in probs[row_i][:len(q.options)]]
                total = sum(p) or 1.0
                p = tuple(x / total for x in p)
                d = Decision(question=q.text, options=tuple(q.options),
                             index=max(range(len(p)), key=p.__getitem__), probabilities=p)
                if results[i] is None:
                    results[i] = [None] * len(items[i][1])
                results[i][k] = d
            if progress:
                print(f"\r  {min(start + batch_size, len(order))}/{len(order)}", end="", flush=True)
        if progress:
            print()
        return results

    # ------------------------------------------------------------------ measurement
    def time_per_decision(self, items, questions: Sequence[Question] | None = None, *,
                          repeats: int = 100, warmup: int = 25) -> dict:
        """Milliseconds per decision at batch one, measured the way a caller experiences it.

        `items` is a list of (state, questions) pairs, cycled through; passing a single state and
        `questions` times that one item repeatedly instead.

        Prefer the list. Timing one fixed prompt over and over reports the spread of the *machine*
        (sd ~2 ms) rather than the spread of the *workload*: prompt length drives the cost, so a
        caller sizing a timeout needs the variation across the states it will actually see. The
        published figure is measured across a benchmark's items for exactly this reason.

        Warm-up iterations are discarded and the device is synchronised around each timed call;
        without both, this measures kernel autotuning and queue depth rather than the model.
        """
        if isinstance(items, str):
            if questions is None:
                raise ValueError("pass questions alongside a single state")
            items = [(items, list(questions))]
        else:
            items = [(s, list(q)) for s, q in items]
        for i in range(warmup):
            s, q = items[i % len(items)]
            self.decide(s, q)
        samples = []
        for i in range(repeats):
            s, q = items[i % len(items)]
            self.backend.synchronize()
            t0 = time.perf_counter()
            self.decide(s, q)
            self.backend.synchronize()
            samples.append((time.perf_counter() - t0) * 1000)
        samples.sort()
        n = len(samples)
        mean = sum(samples) / n
        var = sum((x - mean) ** 2 for x in samples) / max(1, n - 1)
        return {"mean_ms": mean, "sd_ms": var ** 0.5, "p50_ms": samples[n // 2],
                "p90_ms": samples[int(0.90 * n)], "p99_ms": samples[min(int(0.99 * n), n - 1)],
                "n": n, "distinct_items": len(items),
                "backend": self.backend.name, "device": self.device,
                "questions_per_pass": len(items[0][1])}

"""The typed decision head: a backbone, one hidden state per answer slot, a restricted softmax.

There is no decoding loop here, and that is the whole point.  A generative model answers a
multiple-choice question by sampling tokens and hoping they spell one of the options; this one
reads the hidden state at each answer slot, scores it against the embedding of each option's
label token, and normalises over exactly those options:

    p_k(j | x) = softmax_j( <w_l(k,j), h_k> / tau )

The support of that distribution is the option list.  A malformed answer is therefore not
improbable, it is outside the sample space -- there is no token budget to exhaust, no letter to
mis-parse, and no retry path to write.
"""
from __future__ import annotations

import time
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from .prompt import DEFAULT_MAX_STATE_TOKENS, build, option_label_ids
from .types import MAX_OPTIONS, Decision, Question

DEFAULT_MODEL = "flock-io/this-that-model-1.0"


def _pad_to(n: int, multiple: int = 64) -> int:
    """Round the batch width up: a handful of distinct shapes means far fewer kernel recompiles."""
    return ((n + multiple - 1) // multiple) * multiple


def _mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def best_device() -> str:
    """CUDA, else Apple Silicon, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def _default_dtype(device: str) -> torch.dtype:
    """bfloat16 where it is fast, float16 on Apple Silicon, float32 on CPU.

    bfloat16 matmul on CPU is slow and on some builds unsupported. On MPS, float16 is the type
    Metal is built around; bfloat16 works on recent PyTorch but silently falls back for several
    ops, which costs more than the extra exponent range is worth for a 1.9B model doing one pass.
    """
    return {"cuda": torch.bfloat16, "mps": torch.float16}.get(device, torch.float32)


def _synchronize(device: torch.device) -> None:
    """Make an asynchronous backend finish, so a timer measures work rather than submission."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class TypedDecider:
    """Answer typed questions about a state.

        decider = TypedDecider.from_pretrained()
        decider.decide("temp=91C, fan=off", Question("Throttle?", ["no", "yes"]))
    """

    def __init__(self, model, tokenizer, device: str | torch.device | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if device is not None \
            else next(model.parameters()).device
        self._labels = torch.tensor(option_label_ids(tokenizer), device=self.device)

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, name_or_path: str = DEFAULT_MODEL, *, device: str = "auto",
                        dtype: str | torch.dtype | None = None, **kw) -> "TypedDecider":
        """Load the model. `device` may be "auto" (the default), "cuda", "mps" or "cpu".

        `dtype` defaults to whatever suits the device; pass one explicitly to override.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = best_device()
        if dtype is None:
            dtype = _default_dtype(device)
        elif isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if device == "cpu" and dtype is torch.bfloat16:
            dtype = torch.float32          # bf16 matmul on CPU is slow and often unsupported
        tok = AutoTokenizer.from_pretrained(name_or_path)
        model = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=dtype, **kw)
        model.to(device).eval()
        return cls(model, tok, device=device)

    # ------------------------------------------------------------------ the head
    @torch.no_grad()
    def _slot_logits(self, input_ids, attention_mask, slot_idx, slot_batch, n_options):
        """-> [N, MAX_OPTIONS] with options past each question's count masked to -inf."""
        hidden = self.model.model(input_ids=input_ids,
                                  attention_mask=attention_mask).last_hidden_state
        h = hidden[slot_batch, slot_idx]                       # [N, H] one vector per answer
        w = self.model.lm_head.weight[self._labels]            # [MAX_OPTIONS, H]
        logits = F.linear(h, w).float()
        past = torch.arange(MAX_OPTIONS, device=logits.device)[None, :] >= n_options[:, None]
        return logits.masked_fill(past, float("-inf"))

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
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            width = _pad_to(max(len(built[i]["ids"]) for i in chunk))
            input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
            attn = torch.zeros((len(chunk), width), dtype=torch.long)
            slot_idx, slot_batch, n_opt, owner = [], [], [], []
            for b, i in enumerate(chunk):
                ids = built[i]["ids"]
                input_ids[b, :len(ids)] = torch.tensor(ids)
                attn[b, :len(ids)] = 1
                for k, s in enumerate(built[i]["slots"]):
                    slot_idx.append(s); slot_batch.append(b)
                    n_opt.append(built[i]["n_options"][k]); owner.append((i, k))
            dev = self.device
            logits = self._slot_logits(input_ids.to(dev), attn.to(dev),
                                       torch.tensor(slot_idx, device=dev),
                                       torch.tensor(slot_batch, device=dev),
                                       torch.tensor(n_opt, device=dev))
            # float32 for the softmax: on MPS a float16 softmax over a masked row can underflow
            # to zeros, and a distribution of zeros renormalises into nonsense rather than failing
            probs = torch.softmax(logits.float() / temperature, dim=-1).cpu().numpy()
            for row, (i, k) in enumerate(owner):
                q = items[i][1][k]
                p = [float(x) for x in probs[row][:len(q.options)]]
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
            _synchronize(self.device)
            t0 = time.perf_counter()
            self.decide(s, q)
            _synchronize(self.device)
            samples.append((time.perf_counter() - t0) * 1000)
        samples.sort()
        n = len(samples)
        mean = sum(samples) / n
        var = sum((x - mean) ** 2 for x in samples) / max(1, n - 1)
        return {"mean_ms": mean, "sd_ms": var ** 0.5, "p50_ms": samples[n // 2],
                "p90_ms": samples[int(0.90 * n)], "p99_ms": samples[min(int(0.99 * n), n - 1)],
                "n": n, "distinct_items": len(items),
                "questions_per_pass": len(items[0][1])}

"""What a backend owes the decider: a padded batch of token ids in, option logits out.

The seam is deliberately narrow.  Everything above it -- prompt construction, batching, the
temperature, the softmax, the `Decision` objects -- is arithmetic that does not know what a GPU
is, and stays in `model.py` in exactly one copy.  Everything below it is one framework's idea of
a matrix multiply.  A backend that returns the wrong numbers can therefore only be wrong in one
place, and the parity test compares the two backends at precisely this line.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Backend(Protocol):
    """One framework's implementation of the typed head.

    Implementations own the model, the tokenizer and the device policy.  They own nothing else:
    the caller decides what to ask and how to read the answer.
    """

    name: str          # "torch" or "mlx"; appears in `decider.device` and in recorded results
    device: str        # human-readable placement, e.g. "cuda", "mps", "mlx (bf16)"
    tokenizer: object  # a transformers tokenizer, identical across backends by construction
    pad_id: int

    def slot_logits(self, ids: np.ndarray, attn: np.ndarray, slot_idx: np.ndarray,
                    slot_batch: np.ndarray, n_options: np.ndarray) -> np.ndarray:
        """-> float32 [N, MAX_OPTIONS], options past each question's count set to -inf.

        `ids` is [B, T] right-padded, `attn` [B, T] marks the real tokens, and slot k of row b is
        read from `ids[slot_batch[k], slot_idx[k]]`.  The return is float32 regardless of the
        dtype the model ran in: the softmax that follows is float32 in both backends, because on
        a half-precision softmax a masked row can underflow to all zeros and renormalise into
        nonsense rather than failing.
        """
        ...

    def synchronize(self) -> None:
        """Make an asynchronous backend finish, so a timer measures work rather than submission."""
        ...

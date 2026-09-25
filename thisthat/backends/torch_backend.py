"""The head on PyTorch: CUDA where there is one, Apple's MPS otherwise, CPU as the floor.

This is the path every published number in the repository was measured on, and it is unchanged
from when it lived in `model.py` -- the forward, the gather and the restricted matmul are the
same three lines.  Only the softmax moved out, up into the backend-agnostic caller.
"""
from __future__ import annotations

import numpy as np

from ..types import MAX_OPTIONS


def _mps_available() -> bool:
    import torch
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def best_torch_device() -> str:
    """CUDA, else Apple Silicon, else CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def _default_dtype(device: str):
    """bfloat16 where it is fast, float16 on Apple Silicon, float32 on CPU.

    bfloat16 matmul on CPU is slow and on some builds unsupported. On MPS, float16 is the type
    Metal is built around; bfloat16 works on recent PyTorch but silently falls back for several
    ops, which costs more than the extra exponent range is worth for a 1.9B model doing one pass.
    """
    import torch
    return {"cuda": torch.bfloat16, "mps": torch.float16}.get(device, torch.float32)


class TorchBackend:
    name = "torch"

    def __init__(self, model, tokenizer, device=None):
        import torch

        from ..prompt import option_label_ids
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self._device = torch.device(device) if device is not None \
            else next(model.parameters()).device
        self.device = str(self._device)
        self._labels = torch.tensor(option_label_ids(tokenizer), device=self._device)
        pad = tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else (tokenizer.eos_token_id or 0)

    @classmethod
    def load(cls, name_or_path: str, *, device: str = "auto", dtype=None, **kw) -> TorchBackend:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device in ("auto", "torch"):
            device = best_torch_device()
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

    def slot_logits(self, ids, attn, slot_idx, slot_batch, n_options) -> np.ndarray:
        torch = self.torch
        dev = self._device
        with torch.no_grad():
            hidden = self.model.model(
                input_ids=torch.as_tensor(ids, dtype=torch.long, device=dev),
                attention_mask=torch.as_tensor(attn, dtype=torch.long, device=dev),
            ).last_hidden_state
            h = hidden[torch.as_tensor(slot_batch, device=dev),
                       torch.as_tensor(slot_idx, device=dev)]      # [N, H] one vector per answer
            w = self.model.lm_head.weight[self._labels]            # [MAX_OPTIONS, H]
            logits = torch.nn.functional.linear(h, w).float().cpu().numpy()
        past = np.arange(MAX_OPTIONS)[None, :] >= np.asarray(n_options)[:, None]
        return np.where(past, -np.inf, logits).astype(np.float32)

    def synchronize(self) -> None:
        if self._device.type == "cuda":
            self.torch.cuda.synchronize()
        elif self._device.type == "mps":
            self.torch.mps.synchronize()

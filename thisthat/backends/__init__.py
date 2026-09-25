"""Choosing where the forward pass runs.

`device` stays the single knob it has always been.  It gained one legal value, `"mlx"`, rather
than a second parallel selector: a caller that could write `device="mps", backend="mlx"` would be
asking two questions with one answer, and every script in this repository already threads
`--device` through to here.
"""
from __future__ import annotations

import platform

from .base import Backend
from .mlx_backend import MLXBackend, mlx_available
from .torch_backend import TorchBackend, best_torch_device

TORCH_DEVICES = ("cuda", "mps", "cpu", "torch")

__all__ = [
    "TORCH_DEVICES",
    "Backend",
    "MLXBackend",
    "TorchBackend",
    "best_device",
    "load_backend",
    "mlx_available",
]


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def best_device() -> str:
    """What `device="auto"` resolves to here: CUDA, else MLX, else Apple's MPS, else CPU.

    MLX is preferred over MPS on Apple Silicon because it is faster at the same answers -- see
    `tests/test_backend_parity.py`, which is what earns it the default.  It is checked before
    torch is imported at all, so a Mac that will not use torch never pays the two seconds of
    importing it.
    """
    if _is_apple_silicon() and mlx_available():
        return "mlx"
    try:
        return best_torch_device()
    except ImportError:
        if mlx_available():
            return "mlx"
        raise ImportError(
            "no inference backend is installed: pip install 'thisthat[mlx]' on Apple Silicon, "
            "or pip install 'thisthat[torch]' anywhere else") from None


def load_backend(name_or_path: str, *, device: str = "auto", **kw) -> Backend:
    """Load the checkpoint onto whichever backend `device` names."""
    if device == "auto":
        device = best_device()
    if device == "mlx":
        if not mlx_available():
            raise ImportError(
                "device='mlx' needs mlx-lm: pip install 'thisthat[mlx]'. It runs on Apple "
                "Silicon only; elsewhere use device='cuda', 'mps' or 'cpu'.")
        return MLXBackend.load(name_or_path, **kw)
    if device in TORCH_DEVICES:
        return TorchBackend.load(name_or_path, device=device, **kw)
    raise ValueError(
        f"unknown device {device!r}; expected 'auto', 'mlx', or one of {TORCH_DEVICES}")

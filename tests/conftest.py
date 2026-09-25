"""Shared fixtures. These tests run the real model on a real accelerator.

There is no mock here on purpose. The claims this repository makes are about what a specific
checkpoint does on specific inputs, and a test that stubs the model out tests the stub. The model
is loaded once per session because loading it is the slow part, not running it.

    pytest                                  # uses the released checkpoint from the Hub
    THISTHAT_MODEL=/path/to/local pytest    # or a local one
    THISTHAT_DEVICE=mps pytest              # or a backend other than the one auto picks

CPU is excluded rather than merely slow: these tests answer dozens of questions against a 1.9B
model, and a run that takes twenty minutes is a run nobody waits for.
"""
import os

import pytest

MODEL = os.environ.get("THISTHAT_MODEL", "flock-io/this-that-model-1.0")
DEVICE = os.environ.get("THISTHAT_DEVICE", "auto")


def _accelerated_device() -> str | None:
    """The device to test on, or None if this machine has only a CPU."""
    from thisthat.backends import best_device
    try:
        device = best_device() if DEVICE == "auto" else DEVICE
    except ImportError:
        return None
    return None if device == "cpu" else device


@pytest.fixture(scope="session")
def model_name():
    return MODEL


@pytest.fixture(scope="session")
def device():
    d = _accelerated_device()
    if d is None:
        pytest.skip("these tests measure the model on an accelerator; this machine has only a CPU")
    return d


@pytest.fixture(scope="session")
def decider(device):
    from thisthat import TypedDecider
    return TypedDecider.from_pretrained(MODEL, device=device)


@pytest.fixture(scope="session")
def recorded():
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data" / "recorded_68.jsonl"
    return [json.loads(line) for line in open(path)]

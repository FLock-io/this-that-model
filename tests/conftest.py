"""Shared fixtures. These tests run the real model on a real GPU.

There is no mock here on purpose. The claims this repository makes are about what a specific
checkpoint does on specific inputs, and a test that stubs the model out tests the stub. The model
is loaded once per session because loading it is the slow part, not running it.

    pytest                                  # uses the released checkpoint from the Hub
    THISTHAT_MODEL=/path/to/local pytest    # or a local one
"""
import os

import pytest
import torch

MODEL = os.environ.get("THISTHAT_MODEL", "flock-io/this-that-model-1.0")


@pytest.fixture(scope="session")
def decider():
    if not torch.cuda.is_available():
        pytest.skip("these tests measure a GPU model; no CUDA device is visible")
    from thisthat import TypedDecider
    return TypedDecider.from_pretrained(MODEL, device="cuda")


@pytest.fixture(scope="session")
def recorded():
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data" / "recorded_68.jsonl"
    return [json.loads(line) for line in open(path)]

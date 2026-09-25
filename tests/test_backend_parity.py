"""Two backends, one set of answers.

`scripts/compare_backends.py` is where the backends are measured against each other properly, on
the recorded set and on the released benchmark. This file pins the two properties that a change
to either backend must not break, on few enough items to belong in a test run.
"""
import numpy as np
import pytest

from thisthat import Question, TypedDecider
from thisthat.backends import mlx_available

# Measured on the 68 recorded items against float32 on the CPU: float16 on MLX moves a
# probability by at most 7.1e-3, bfloat16 by 2.4e-2. The bound sits between them on purpose --
# it passes the dtype the backend ships with, and fails the one it must not silently fall back to.
MAX_PROB_DRIFT = 0.015


# Padding a prompt out by 900 tokens moves a probability by at most 3.4e-6, and -- the part that
# matters -- by the same 3.4e-6 whether the padding is 60 tokens or 900. Contamination would grow
# with the number of pads; a reduction order that changes at some tile width does not. The bound
# is set well above what was measured and still three orders of magnitude below any claim.
MAX_PADDING_NOISE = 1e-4


@pytest.mark.parametrize("filler", [60, 400])
def test_padding_does_not_change_the_answer(decider, filler):
    """A short prompt batched beside a long one must answer as it does alone, at any width.

    The MLX backend passes no attention mask, on the argument that right padding is invisible to a
    causal model: full attention masks position i to j <= i, the gated-delta recurrence at i has
    only accumulated j <= i, and the depthwise conv1d is left-padded. That argument is worth
    exactly as much as this test, which is why the test exists -- and why it runs at two widths,
    since the discriminating evidence is that the residual does not grow with the padding.
    """
    q = Question("Is the value above ten?", ["no", "yes"])
    short = ("value = 42", [q])
    long = ("value = 3. " + "Irrelevant filler sentence to lengthen the prompt. " * filler, [q])

    alone = decider.decide_batch([short])[0][0]
    padded = decider.decide_batch([short, long], batch_size=8)[0][0]

    assert alone.index == padded.index
    assert np.allclose(alone.probabilities, padded.probabilities, atol=MAX_PADDING_NOISE), (
        f"padding moved the distribution: {alone.probabilities} -> {padded.probabilities}")


@pytest.mark.skipif(not mlx_available(), reason="MLX backend not installed")
def test_mlx_agrees_with_float32_on_the_cpu(recorded, model_name):
    """The MLX backend answers the recorded items as float32 does, to within a measured bound.

    float32 on the CPU is the yardstick because it is the one configuration whose error is not the
    thing under test. Note that MLX at float32 still differs from it by ~5.5e-3: that residual is
    an implementation difference in the gated-delta kernels, not a precision one, and no dtype
    choice removes it. What the bound below protects is the part that dtype does control.
    """
    pytest.importorskip("torch")
    items = [(r["state"], [Question(r["question"], r["options"])]) for r in recorded[:8]]

    reference = TypedDecider.from_pretrained(model_name, device="cpu", dtype="float32")
    ref = [d[0].probabilities for d in reference.decide_batch(items)]
    del reference

    mlx = TypedDecider.from_pretrained(model_name, device="mlx")
    got = [d[0].probabilities for d in mlx.decide_batch(items)]

    drift = max(np.abs(np.array(a) - np.array(b)).max() for a, b in zip(ref, got))
    assert drift < MAX_PROB_DRIFT, f"probabilities drifted by {drift:.2e}"
    assert [int(np.argmax(p)) for p in ref] == [int(np.argmax(p)) for p in got], \
        "the backends chose different options"

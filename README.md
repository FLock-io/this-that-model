# this-that-model

**Typed decisions from a 1.9B model. One forward pass, no decoding loop, no parser, no retry.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2609.23886-b31b1b.svg)](https://arxiv.org/abs/2609.23886)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97-model%201.2-yellow)](https://huggingface.co/flock-io/this-that-model-1.2)
[![Spatial bench](https://img.shields.io/badge/%F0%9F%A4%97-spatial%20bench-yellow)](https://huggingface.co/datasets/limberc/this-that-spatial-bench)
[![Complex decisions](https://img.shields.io/badge/%F0%9F%A4%97-complex%20decisions-yellow)](https://huggingface.co/datasets/limberc/this-that-complex-decisions)
[![FLock API](https://img.shields.io/badge/FLock%20API-free-brightgreen.svg)](https://platform.flock.io/models)

Your program reaches a branch it cannot express in code. *Is this refund within policy? Is this
shell command safe to run unattended? Does this output satisfy the instruction it was given?*

The usual answer is to call a frontier model and parse what comes back. That works, and it costs
you a decoding loop, a per-token bill, a network hop, and a parser that has to handle the reply
arriving as a sentence, a markdown fence, a refusal, or an empty string because a reasoning budget
ran out. This model returns an index and a probability, in **31.4 ms**, on a card you already own
— or, if you would rather not own one, [free on the FLock API](https://platform.flock.io/models).

```python
from thisthat import TypedDecider, Question

decider = TypedDecider.from_pretrained("flock-io/this-that-model-1.2")

answer = decider.decide(
    "command: rm -rf /var/lib/postgresql/data",
    Question("Is this shell command safe to run unattended on a production host?",
             ["yes, it only reads state",
              "no, it modifies or deletes data",
              "no, it contacts the network"]),
)

print(answer)                # no, it modifies or deletes data (100%)
print(answer.index)          # 1
print(answer.probabilities)  # (0.003, 0.996, 0.001)

if answer.confidence < 0.8:
    escalate()               # the number is calibrated, so this threshold means something
```

Those outputs are measured, not illustrative — `1.2`, MLX at float16. The argmax is stable
across backends and the third decimal is not, which the [parity numbers](#install) below quantify.
The escalation branch is not decoration either -- ask this model something outside what it was
trained on and it will tell you it does not know:
`"user=alice tier=free requests_this_minute=847"` / *"Should this request be rate-limited?"*
returns **no (76%)**, which is both unconfident and, for 847 requests a minute, wrong. Threshold
the number and route those cases to something else.

## Three checkpoints

The architecture has not moved since the paper: 1.88 B parameters, 18 of 24 layers DeltaNet linear
attention and 6 full attention, one forward pass, zero generated tokens. What moved is the training
mixture, and each release is named after what the one before it got wrong.

| | `1.0` | `1.1` | `1.2` |
|---|---:|---:|---:|
| composed decisions, 1,710 q (chance 0.258) | 0.406 | 0.775 | **0.878** |
| composed decisions by depth, 242 q | — | 0.785 | **0.909** |
| spatial benchmark, 7,305 q | 0.839 | **0.871** | — |
| the frozen 68-question cohort | 0.941 | **1.000** | 0.985 |
| calibration qL2 ↓, `sim_event_ood` | **0.0250** | 0.0253 | 0.0274 |
| calibration qL2 ↓, `sim_local_ood` | **0.0096** | 0.0203 | 0.0161 |

A dash is a number that checkpoint's model card does not publish, not a poor one. The `1.0` column
is what the scripts in this repository reproduce; the other two are the figures
[1.1](https://huggingface.co/flock-io/this-that-model-1.1) and
[1.2](https://huggingface.co/flock-io/this-that-model-1.2) publish, and every script below takes
`--model` so you can check them on your own card.

**1.1** added the decisions 1.0 could not make: several rules applying at once, an exception that
has an exception of its own, an answer not readable from any single column of the state. 64,028
composed-decision questions over 112 rule structures and 40 domains, plus two loss terms over pairs
only a generator can make — the same decision rendered four ways, which must be answered the same,
and two policies over one state, which must not. Same weights shape, different data: 0.406 to 0.775.

**1.2** fixed *how the rule is written*. 1.1 read `bays without chilled handling are ineligible` as
though it named the eligible set — not failing to apply the rule but applying its opposite, at 0.88
mean confidence. Over eleven phrasings of one rule, against a chance rate of 0.19, 1.1 ranges from
0.00 to 0.89 and 1.2 from 0.98 to 1.00; four of 1.1's eleven rows sit *below* chance, which is not
guessing. [The table is below.](#composed-decisions)

**Read the calibration rows before you upgrade.** 1.1 gave up half the sharpness 1.0 had on
`sim_local_ood`, and 1.2 wins some of it back without reaching 1.0. If you threshold `confidence`
rather than read the argmax — which this README spends a section recommending — that is the one
axis where the newest checkpoint is not the best one, and it is worth measuring on your own
decisions before you move.

**Which to name.** `1.2` unless you are reproducing the paper, which measured `1.0`.
`from_pretrained()` with no argument still loads `1.0` for exactly that reason, so name the
checkpoint you want rather than taking the default.

## Why it cannot return garbage

The answer is read from the hidden state at a designated position and scored against the label
tokens of the options you declared, then normalised over exactly those:

$$p_k(j \mid x) = \operatorname{softmax}_j \left( \langle w_{\ell(k,j)}, h_k \rangle / \tau \right)$$

The support of that distribution **is** your option list. A malformed answer is not unlikely, it is
outside the sample space. There is no token budget to exhaust and no letter to mis-parse, so the
`try/except` around the call and the decision about what to do when parsing fails both disappear.

Because no answer is ever written back into the prompt, several questions about one state are
conditionally independent given the input and are answered in the **same** forward pass:

```python
answers = decider.decide(order_json, [
    Question("Within the refund window?", ["no", "yes"]),
    Question("Which queue?", ["standard", "priority", "manual review"]),
    Question("Risk band", ["low", "medium", "high"]),
])
for a in answers:
    print(a.question, "->", a)
```

Three decisions, one pass, not three times the price.

## Install

```bash
pip install -e .                 # transformers, plus the backend your platform runs on
pip install -e ".[bench]"        # also numpy, to regenerate the benchmarks
```

The backend comes from a marker rather than from you: PyTorch everywhere, MLX on Apple Silicon.
`.[torch]` and `.[mlx]` force the other one in alongside, which is what a dev install does.

No API key is required, and none of this repository's code will ask for one: every benchmark below
runs the local model against data vendored here or pulled from the public Hub.

**Apple Silicon.** `device="auto"` picks MLX on a Mac, and the PyTorch MPS path is still there
under `device="mps"`. There is no conversion step and no second checkpoint: MLX reads the
published bfloat16 safetensors directly.

```python
decider = TypedDecider.from_pretrained()                    # mlx on a Mac
decider = TypedDecider.from_pretrained(device="mps")        # the PyTorch path
decider = TypedDecider.from_pretrained(device="cpu", dtype="float32")
```

On an M5 Max, one question per pass across the 68 recorded states, with the device synchronised
around each call so the number is work done rather than work queued:

| backend | mean | p50 | p90 | decisions/s |
|---|---|---|---|---|
| `mlx` | **25.2 ms** (sd 3.3) | 24.5 | 30.5 | 40 |
| `mps` | 113.3 ms (sd 0.5) | 113.4 | 113.9 | 9 |

Read the gap as kernels rather than frameworks. `flash-linear-attention` and `causal_conv1d` have
no Metal build, so PyTorch runs the gated-delta recurrence and the depthwise convolution through
their reference implementations and says so on load; MLX ships its own. The MLX figure also sits
under the 30.9 ms measured on a laptop RTX 5080, which is a statement about two machines rather
than about two frameworks.

Both Apple Silicon paths load in float16 rather than the checkpoint's bfloat16. The two types cost
the same two bytes, and bfloat16 spends them on an exponent range nothing here needs while giving
up three bits of mantissa. That is not free on a model whose output is a probability: measured
against float32 on the CPU, bfloat16 on MLX moves a per-option probability by up to 2.4e-2 where
float16 moves it by 7.1e-3 (`1.1`, the 68 recorded states).

**The backends do not agree to the last digit, and the claims survive it.** Over 450 questions of
the spatial benchmark the two float16 paths pick the same option on 449, for 0.889 against 0.891
(`1.1`). Over the 68 recorded states they score the same accuracy, 0.941, and Brier scores of
0.0416 and 0.0417 (`1.0`) -- so the calibration the paper reports is the calibration you get
either way. Some of the
residual is not precision at all: MLX at float32 still differs from PyTorch at float32 by 5.5e-3,
which is the gated-delta kernels disagreeing rather than rounding, and no dtype removes it.
`scripts/compare_backends.py` is what produced every number in this paragraph, against whichever
pair of backends you point it at.

**Do not quantise this one.** The usual bargain does not apply to a model whose output is a
probability, and here it is not even a bargain. Recorded items, 1.0, `measure_mlx_quantization.py`:

| weights | accuracy | Brier | max \|dp\| vs fp32 | ms |
|---|---|---|---|---|
| float16 | 0.9412 | 0.0416 | 9.7e-03 | 27.6 |
| 8-bit | 0.9412 | 0.0415 | 5.7e-02 | 25.3 |
| 4-bit | 0.6912 | 0.1906 | 7.6e-01 | 23.9 |

Four bits costs a quarter of the accuracy and multiplies the Brier score by four and a half — the
calibration is gone, not degraded — and buys 13% of the latency, because at batch one a 1.9B
prefill is not waiting on memory. Eight bits keeps accuracy and Brier intact but moves a
probability by up to 5.7e-2, six times what float16 does, which matters if you threshold
confidence rather than just read the argmax. float16 is the default for both reasons.

## Reproducing the paper's numbers

The paper is [arXiv:2609.23886](https://arxiv.org/abs/2609.23886), and the same PDF is vendored
here as [`paper/this-that-model.pdf`](paper/this-that-model.pdf). Each script below prints what it
measured next to what we published, and says plainly when they differ.

The paper measured `1.0`, so that is what these scripts default to. Every one of them takes
`--model`, and pointing them at `flock-io/this-that-model-1.1` or `-1.2` is how you see a later
checkpoint move against the same fixed evaluation sets.

| Script | What it measures | Published result |
|---|---|---|
| `scripts/reproduce_recorded.py` | 68 recorded decision questions over 17 states, against `Jev` on the same inputs | **0.941** accuracy, Brier **0.042** (Jev: 0.765 / 0.133) |
| `scripts/reproduce_simulator.py` | Calibration where the true answer is a computed probability | accuracy **at** the computed ceiling; qL2 ~**4×** smaller than a constant predictor's |
| `scripts/reproduce_ladder.py` | Whole state vs. only the part the answer depends on, 32×32 → 200×200 | windowed flat at **~1.0**; whole state 0.48–0.65 |
| `scripts/reproduce_spatial.py` | The released 7,305-question benchmark, by family and by rendering | **0.839** against a chance rate of 0.343 |
| `scripts/measure_latency.py` | Batch-1 latency across the 68 recorded items, CUDA-synchronised, warm-up discarded | **~33 ms** on an RTX 5080 laptop GPU |
| `scripts/bench_laya.py` | The same 2,250-question subset through [Laya](https://github.com/NandhaKishorM/laya), zero-shot | **0.345** against a chance rate of 0.343 |

```bash
python scripts/reproduce_recorded.py
python scripts/reproduce_simulator.py
python scripts/reproduce_ladder.py --sizes 32,50,80
python scripts/measure_latency.py
```

Two of these regenerate their evaluation set from the simulator rather than replaying a fixed
file, so the third decimal moves between runs; the scripts say so, and compare shapes rather than
digits. The paper's headline latency of 30.9 ± 16.6 ms was measured over an internal suite whose
prompts vary far more in length than this cohort's do — expect a tighter spread here, around the
same middle.

### Against `Jev`, on inputs neither party chose

![accuracy against the cost of one pass](docs/accuracy-vs-cost.png)

| System | Accuracy | Brier ↓ | NLL ↓ | ms/question | Cost of one pass |
|---|---:|---:|---:|---:|---:|
| majority-class baseline | 0.647 | — | — | — | — |
| `claude-fable-5-1` | 0.676 | — | — | 2395 | $0.471 |
| `glm-5.3` | 0.721 | 0.204 | 0.601 | 819 | $0.008 |
| `qwen3.8-max` | 0.735 | 0.263 | 2.681 | 1014 | $0.021 |
| `NanoJev-0.6B` | 0.750 | 0.166 | 0.479 | — | — |
| **Jev** | 0.765 | 0.133 | 0.403 | — | — |
| `kimi-k3` | 0.779 | 0.143 | 0.430 | 989 | $0.009 |
| `deepseek-v4.1-flash` | 0.794 | — | — | 808 | $0.002 |
| `gpt-5.6` | 0.926 | — | — | 1180 | $0.018 |
| **this-that-model-1.0** | **0.941** | **0.042** | **0.126** | **30.9** | **$0.000014** |

A dash in the Brier and NLL columns means the endpoint returns no token probabilities, so those
quantities are not observable there — not that they are poor. Time is the median *per-question*
latency at the client; we do not report the wall clock of the whole pass, because that depends on
how many requests our harness kept in flight and is a property of the harness. Our cost is
electricity at 80 W and $0.30/kWh, which is a different kind of number from a price that has to
cover serving and margin: read our distance from the hollow square in the figure as **one order of
magnitude, not five**. Jev's x position is its *published* rate of $0.042 per million input tokens
applied to the token count we measured — its accuracy is ours to measure, its price is theirs to
state. Five further frontier models answered all 68 correctly and are omitted: at this size they
are saturated and rank nothing.

That sentence now applies to our own rows. `1.1` answers all 68 correctly (Brier 0.003, NLL 0.023)
and `1.2` misses one (0.985, Brier 0.009, NLL 0.047) — a single question is the whole difference
between them, so the cohort has stopped ranking these checkpoints against each other and only still
ranks them against the hosted service. The row worth reading here is the one the paper published.

The recorded cohort is the one comparison neither party chose: a third party put those 68 questions
to a hosted commercial decision service and published the states, the returned probabilities and
the computed truth. The wording is theirs, not ours — this model was trained on a different
phrasing of the same question, so reusing theirs measures transfer rather than recall.

### On the released spatial benchmark

7,305 questions over 15 families, [public on the Hub](https://huggingface.co/datasets/limberc/this-that-spatial-bench).
The 2,250-question subset below is the one every system answered — the first 150 questions of each
family.

| System | tokens/question | Accuracy | seen | unseen | Cost |
|---|---:|---:|---:|---:|---:|
| `gpt-5.6` | 286 | 0.897 | 0.947 | 0.879 | $7.38 |
| `claude-opus-5` | 510 | 0.892 | 0.908 | 0.886 | $109.26 |
| **this-that-model-1.0** | **0** | **0.844** | 0.873 | 0.834 | **free** |
| `claude-sonnet-5` | 722 | 0.836 | 0.792 | 0.853 | $29.12 |
| **`Jev`** † | **0** | 0.803 | 0.717 | 0.833 | — |
| `glm-5.3` | 219 | 0.789 | 0.687 | 0.827 | $1.44 |
| `kimi-k3` | 7 | 0.659 | 0.750 | 0.626 | $0.52 |
| `deepseek-v4.1-flash` | 89 | 0.512 | 0.605 | 0.478 | $0.21 |
| `deepseek-v4-pro` | 1 | 0.425 | 0.448 | 0.417 | $0.41 |
| `Laya` ‡ | **0** | 0.345 | 0.430 |  0.314 | free |
| answering constantly | — | 0.343 | 0.425 |  0.314 | — |

**Read our row with its caveat.** Every hosted system above met these fifteen question shapes for
the first time at test. This checkpoint was trained on all fifteen — on different items, over
different maze windows, both excluded from the evaluation set by fingerprint *and* by rendered
state — so ours is an in-distribution number placed beside zero-shot ones. The claim it supports is
not that a 1.88B model overtook `claude-sonnet-5`; it is that a decision shape you can generate
training data for costs $0 and 30.9 ms per decision afterwards, and one you cannot stays where it
was. Before that training the same model scored 0.409 here.

`1.1` scores **0.870** on this 2,250-question subset and **0.871** over all 7,305 — it kept 1.0's
spatial ability while adding composed decisions, which is the claim that mattered for that release.
Its card does not split those by seen and unseen, and `1.2`'s card does not report this benchmark
at all, so neither gets a row above rather than a fabricated one.

‡ [Laya](https://github.com/NandhaKishorM/laya) is the closest published work to this one by
construction: a non-autoregressive typed-decision engine over a 421M ModernBERT encoder, one
forward pass, no generated text. We ran `convaiinnovations/laya:typed-decisions` ourselves on this
exact subset, same items, same option sets, unanswered counted as wrong. It scores 0.345 against a
chance rate of 0.343, and one family of fifteen clears chance by more than a standard error.

| the state rendered as | before this round | after | chance |
|---|---:|---:|---:|
| an ASCII block | 0.457 | 0.779 | 0.388 |
| the same cells as JSON | **0.373 (chance)** | **0.936** | 0.394 |
| the same cells as prose | 0.420 | 0.942 | 0.394 |

The JSON row is why that training round happened: the model was at chance on the identical cells
in a format it had not been trained on.

## Composed decisions

Everything above asks one rule at a time. Policies written by people do not: a filter, then a
second filter, then a ranking — and the rule phrased however the person writing the policy phrased
it. That is what `1.1` and `1.2` were trained for, and it is measured on a second public benchmark
of 1,710 questions,
[`limberc/this-that-complex-decisions`](https://huggingface.co/datasets/limberc/this-that-complex-decisions).

| System | Accuracy | Generated tokens | Latency † |
|---|---:|---:|---:|
| **this-that-model-1.2** | **0.878** | **0** | **31.4 ms** |
| `claude-opus-5` | 0.834 | ~200 | ~1000 ms |
| `gpt-5.6` | 0.816 | ~200 | ~1200 ms |
| `this-that-model-1.1` | 0.775 | 0 | 50.3 ms |
| `glm-5.3` | 0.652 | ~200 | ~800 ms |
| `kimi-k3` | 0.522 | ~200 | ~1000 ms |
| `deepseek-v4.1-flash` | 0.511 | ~4 | ~800 ms |
| `deepseek-v4-pro` | 0.470 | ~4 | ~900 ms |
| `this-that-model-1.0` | 0.406 | 0 | 50.3 ms |
| `laya-typed-decisions` | 0.310 | 0 | 30.4 ms |
| answering constantly | 0.258 | — | — |

`1.0`'s row is zero-shot: this benchmark did not exist when it was trained and none of its decision
types were in its mixture. The distance from 0.406 to 0.878 is the same architecture with different
training data, which is the whole claim these two releases make.

† Those latencies are the ones `1.2`'s model card publishes, on an RTX 5080 laptop GPU. Read the
50.3 ms rows sceptically: all three checkpoints are the same architecture emitting the same zero
tokens, and this repository measures `1.0` at 30.9 ms on that same class of card, so the gap is a
property of that run rather than of those weights. `scripts/measure_latency.py --model ...` settles
it on yours.

**That table measures decision difficulty and nothing else.** On the phrasing test below,
`claude-opus-5` and `gpt-5.6` both score 1.00 on every row, including the rows where `1.1` scores
0.00. The frontier models are not worse at reading rules; they are worse at these decisions. Two
axes, and both belong on the page.

### By depth

A filter, then a second filter, then a ranking. 242 questions, held out at the question level and
at the world-state level.

| | depth 1 | depth 2 | depth 3 | overall |
|---|---:|---:|---:|---:|
| **this-that-model-1.2** | **0.94** | **0.89** | **0.87** | **0.909** |
| `this-that-model-1.1` | 0.80 | 0.80 | 0.74 | 0.785 |
| `gpt-5.6` | 0.81 | 0.56 | 0.58 | 0.678 |
| `claude-opus-5` | 0.80 | 0.53 | 0.61 | 0.669 |

Depth 2 is the hardest rung for the hosted models and the widest gap: two narrowings with many
candidates still live is where one pass over the whole state pays off most.

### The same rule, written eleven ways

100 decisions, the same gold answers, eleven ways of writing one eligibility rule. A model that
reads rules should score the same on all eleven. Chance is 0.19.

| the rule, written as | `1.1` | **`1.2`** |
|---|---:|---:|
| `Only bays with handling of chilled are eligible.` | 0.86 | **1.00** |
| `A bay is eligible only if its handling is chilled.` | 0.88 | **0.99** |
| `A bay must have handling of chilled to be used.` | 0.79 | **0.99** |
| `No bay may be used unless its handling is chilled.` | 0.89 | **0.99** |
| `Every bay is excluded except those whose handling is chilled.` | 0.87 | **1.00** |
| `A bay must not be used if its handling is not chilled.` | 0.72 | **0.99** |
| `A bay whose handling is not chilled is not eligible.` | 0.00 | **0.98** |
| `A bay whose handling is not chilled is ineligible.` | 0.00 | **0.98** |
| `Bays without chilled handling are ineligible.` | 0.03 | **0.99** |
| `Any bay that lacks chilled handling is disqualified.` | 0.05 | **1.00** |
| `A bay that fails to provide chilled handling is ruled out.` | 0.16 | **0.98** |

Four of `1.1`'s rows are below chance. A model scoring 0.00 where chance is 0.19 is not guessing —
it is reliably choosing the option the rule excludes. Asked the yes/no form (`Is bay-b eligible?`)
under such a rule, `1.1` was right 6% of the time against a chance rate of 50%.

One item of that shape, put to each checkpoint in turn:

```python
answer = decider.decide(
    '{"bays": {"bay-a": {"free": 7, "distance_m": 40, "handling": "chilled"},'
    '          "bay-b": {"free": 3, "distance_m": 12, "handling": "ambient"}},'
    ' "policy": ["Bays without chilled handling are ineligible.",'
    '            "Among those eligible, take the most free positions."]}',
    Question("Which bay is assigned?", ["bay-a", "bay-b", "hold in transit"]),
)
```

| checkpoint | answer | confidence |
|---|---|---:|
| `1.0` | `bay-b` — wrong | 0.610 |
| `1.1` | `bay-b` — wrong | **0.9998** |
| `1.2` | `bay-a` — correct | 0.9988 |

Measured here on MLX at float16, and the middle row is the one to take seriously: `1.1` is not
uncertain about the wrong answer, it is certain about it, so a confidence threshold would not have
caught it. That is the failure `1.2` exists to fix, and over 40 freshly generated decisions of that
shape `1.1` scores 0.20 against `1.2`'s 0.975. Phrased positively, both score 1.000.

**What neither card claims to have fixed.** `1.1` publishes its own weak families, and `1.2`
reports composed decisions and phrasing rather than re-measuring them, so read these as open until
someone does. Weighing several numeric criteria against one another, finding the cheapest change
that flips a decision, and judging whether the state contains enough information to decide at all
each sit near 0.50. In distribution the confidence still separates them — 0.45 when wrong against
0.77 when right on weighted criteria — so thresholding works there. Off distribution it does not:
on a rule structure absent from training `1.1` scores 0.447 against a chance rate of 0.31, with a
calibration error of 0.20. That is the one regime where this model is confident and wrong, and the
`confidence < 0.8` branch at the top of this file will not catch it. If your decisions have a shape
it has not seen, measure before you rely on it.

## Calling it over an OpenAI-compatible API

The model is hosted on the [FLock API platform](https://platform.flock.io/models), where it is **free to call** — $0 per million tokens in, $0 out, which is the listed price and not an introductory one. Get a key there, and the stock OpenAI SDK reaches it with nothing but a base URL:

```python
import os
from openai import OpenAI

client = OpenAI(base_url="https://api.flock.io/v1", api_key=os.environ["LITELLM_API_KEY"])

response = client.chat.completions.create(
    model="this-that-model-1.0",          # the id as the platform lists it
    messages=[
        {"role": "user", "content": "value=42"},
        {"role": "user", "content": "Is the value above ten?"},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "decision",
            "schema": {
                "type": "object",
                "properties": {"answer": {"enum": ["no", "yes"]}},
                "required": ["answer"],
            },
        },
    },
)
print(response.choices[0].message.content)   # {"answer":"yes"}
```

The last user message is the question; everything before it is the state. That is how a caller
writes this anyway — context first, decision last — and it preserves the state/question split the
model was trained on.

Everything below about the enum, the logprobs and the refusals holds for the hosted endpoint and
for a server you run yourself: same protocol, same weights. The two differ in the base URL and in
the model id: a bare name hosted, the Hub path when you serve it locally. Which checkpoints the
platform serves is the platform's to state rather than this file's — read the ids off
[the listing](https://platform.flock.io/models) and put the one you want in `model=`. Locally there
is no such question: any of the three Hub paths loads.
The platform listing states the same constraint this README does — typed decisions only, an enum
in `response_format` required, the last user message the question, zero completion tokens returned
— because the hosted deployment is this server, on SGLang.

### Serving it yourself

```bash
pip install -e ".[serve]"
python -m thisthat.server --model flock-io/this-that-model-1.2 --port 8000
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

r = client.chat.completions.create(
    model="flock-io/this-that-model-1.2",
    messages=[{"role": "user", "content": "command: rm -rf /var/lib/postgresql/data"},
              {"role": "user", "content": "Is this safe to run unattended?"}],
    response_format={"type": "json_schema", "json_schema": {"name": "decision", "schema": {
        "type": "object", "properties": {"answer": {"enum": [
            "yes, it only reads state",
            "no, it modifies or deletes data",
            "no, it contacts the network"]}}}}},
    logprobs=True)

r.choices[0].message.content      # '{"answer": "no, it modifies or deletes data"}'
r.usage.completion_tokens         # 0
```

**The enum is the option set**, which is why `response_format` is the right place for it: this
model chooses among options you name rather than generating text, and structured output with an
enum is the OpenAI feature that means the same thing. With `logprobs=True` the full distribution
comes back in the standard shape, so a client that already reads logprobs needs to know nothing
about this model:

```
99.7%  no, it modifies or deletes data
 0.2%  no, it contacts the network
 0.1%  yes, it only reads state
```

Two things the server deliberately does not do. A request with **no enum is refused with a 400**,
not answered — there is no decoding loop to generate a completion with, and returning an empty one
would let a caller believe it had an answer. And `usage.completion_tokens` is **0**, which is not
an omission: the answer is read from a hidden state, and reporting invented token counts would
misrepresent what the call cost. `stream=true` works and returns the answer as a single chunk,
for the same reason.

See [`examples/openai_server.py`](examples/openai_server.py) for a runnable version against the
local server; the same file reaches the hosted endpoint once the base URL, key and model id
are swapped for the three at the top of this section.

The same server also speaks the typed format decision benchmarks use, `POST /v1/systemone` —
a state and named questions of type `noul`, `choice` or `score`, answered in one forward pass
with a distribution over each question's labels:

```bash
curl -s localhost:8000/v1/systemone -d '{"state": "Order #1182. Status: shipped.",
  "questions": {"decision": {"type": "noul", "instructions": "Has the order shipped?"}}}'
# {"answers": {"decision": {"type": "noul", "probabilities": {"no": ..., "yes": ...}, "noul": ...}}, ...}
```

This is the wire format JevBench's `typesafe` adapter calls; see
[`thisthat/systemone_protocol.py`](thisthat/systemone_protocol.py) for how each type's rubric
becomes the question the model reads.

## Where the state goes

Two layouts, differing only in where the state sits:

```python
decider.decide(state, questions, layout="state_first")    # default, as trained
decider.decide(state, questions, layout="schema_first")   # questions first
```

Asking the same *m* questions about *m* different states costs `|P| + m(|S| + N)` tokens of prefill
under `schema_first` against `m(|P| + |S| + N)` under `state_first`, because the question block is
byte-identical for every state and its attention cache is computed once. For a fixed decision point
in a program — the same schema, a stream of states — that saves `(m−1)|P|`.

## Repository layout

```
thisthat/            the inference package: types, prompt layout, typed head
thisthat/backends/   where the forward pass runs: PyTorch anywhere, MLX on Apple Silicon
benchmarks/          metrics, and the simulator the evaluation sets are computed from
data/                the 68 recorded questions over 17 states
scripts/             one reproduction script per published result
examples/            short, runnable, doing one thing each
tests/               29 tests: the wire format without weights, the rest against an accelerator
```

## Citing

The paper describes `1.0`; `1.1` and `1.2` are later checkpoints of the same architecture, and
their model cards carry their own numbers.

```bibtex
@misc{cheng2026thisthat,
  title         = {this-that-model-1.0: A typed decision model that decides in 30 ms,
                   for a millionth of a cent},
  author        = {Cheng, Zehua and Dai, Wei and Sun, Jiahao},
  year          = {2026},
  eprint        = {2609.23886},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.23886}
}
```

## Licence and attribution

MIT, see [LICENSE](LICENSE). The simulator and the recorded cohort are vendored from [NanoJev](https://github.com/TianyuCodings/NanoJev) under the MIT licence and are used as published; see [NOTICE](NOTICE). We thank its authors.

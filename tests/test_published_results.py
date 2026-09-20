"""The published numbers, asserted against the released checkpoint.

These are the claims in the paper. If one of them stops holding, either the checkpoint changed or
the paper is wrong, and both are worth failing a build over. Tolerances are wide enough to absorb
kernel and dtype differences between machines and nothing else.
"""
import pytest

from benchmarks.metrics import (accuracy, accuracy_ceiling, brier, constant_predictor_error,
                                squared_error_to_distribution)
from thisthat import Question

# published in the paper, measured on the released checkpoint
RECORDED_ACCURACY = 0.926
RECORDED_BRIER = 0.046
SERVICE_ACCURACY = 0.765


def test_recorded_cohort_matches_the_published_accuracy(decider, recorded):
    items = [(r["state"], [Question(r["question"], r["options"])]) for r in recorded]
    out = decider.decide_batch(items, batch_size=8, temperature=1.3, max_state_tokens=2048)
    picks = [d[0].index for d in out]
    gold = [r["answer_index"] for r in recorded]
    acc = accuracy(picks, gold)
    assert acc == pytest.approx(RECORDED_ACCURACY, abs=0.03), (
        f"published {RECORDED_ACCURACY}, measured {acc:.3f}")


def test_recorded_cohort_matches_the_published_brier(decider, recorded):
    items = [(r["state"], [Question(r["question"], r["options"])]) for r in recorded]
    out = decider.decide_batch(items, batch_size=8, temperature=1.3, max_state_tokens=2048)
    p_yes = [d[0].probabilities[1] for d in out]
    truth = [float(r["answer_index"]) for r in recorded]
    assert brier(p_yes, truth) == pytest.approx(RECORDED_BRIER, abs=0.02)


def test_we_beat_the_recorded_hosted_service_on_its_own_questions(decider, recorded):
    """The comparison neither party chose: same states, same wording, same computed truth."""
    items = [(r["state"], [Question(r["question"], r["options"])]) for r in recorded]
    out = decider.decide_batch(items, batch_size=8, temperature=1.3, max_state_tokens=2048)
    gold = [r["answer_index"] for r in recorded]
    ours = accuracy([d[0].index for d in out], gold)
    service = accuracy([1 if r["recorded_service_p_yes"] >= 0.5 else 0 for r in recorded], gold)
    assert service == pytest.approx(SERVICE_ACCURACY, abs=0.01), "the recorded file changed"
    assert ours > service


@pytest.mark.slow
def test_calibration_beats_a_constant_predictor(decider):
    """Where the truth is a probability, distance to it is the measurement that is not capped."""
    from benchmarks.simulator.tasks import stochastic_set
    items = stochastic_set(n=120)
    truth = [it.truth[0] for it in items]
    out = decider.decide_batch([(it.state, it.questions) for it in items], batch_size=8)
    q_l2 = squared_error_to_distribution([list(d[0].probabilities) for d in out], truth)
    constant = constant_predictor_error(truth)
    assert q_l2 < constant / 2, f"qL2 {q_l2:.4f} vs constant {constant:.4f}"

    acc = accuracy([d[0].index for d in out], [it.answer_index[0] for it in items])
    ceiling = accuracy_ceiling(truth)
    # accuracy is measured against one draw from `truth`, so it may sit a little above an
    # expectation-based ceiling; what would be wrong is sitting well below it
    assert acc > ceiling - 0.08, f"accuracy {acc:.3f} against ceiling {ceiling:.3f}"


@pytest.mark.slow
def test_the_window_holds_while_the_whole_state_decays(decider):
    """The decomposition claim: same worlds, same questions, different amount of state shown."""
    from benchmarks.simulator.tasks import context_ladder
    sets = context_ladder(sizes=(32, 80), maps_per_size=3, starts_per_map=2)
    scores = {}
    for key, items in sets.items():
        out = decider.decide_batch([(it.state, it.questions) for it in items], batch_size=4,
                                   max_state_tokens=32768)
        picks = [d.index for row in out for d in row]
        gold = [g for it in items for g in it.answer_index]
        scores[key] = accuracy(picks, gold)
    assert scores["sim_local_32"] > 0.95 and scores["sim_local_80"] > 0.95
    assert scores["sim_local_80"] > scores["sim_global_80"] + 0.2
    # flat in the size of the world: that is the claim, not merely "high"
    assert abs(scores["sim_local_80"] - scores["sim_local_32"]) < 0.1

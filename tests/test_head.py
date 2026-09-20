"""The typed head's guarantees, checked on the real model rather than argued for."""
import pytest

from thisthat import Question


def test_probabilities_are_a_distribution_over_exactly_the_options(decider):
    q = Question("Is the sky overcast?", ["no", "yes", "cannot tell"])
    d = decider.decide("cloud cover: 8 oktas, visibility 2 km", q)
    assert len(d.probabilities) == 3, "one probability per declared option, no more"
    assert abs(sum(d.probabilities) - 1.0) < 1e-5
    assert all(p >= 0.0 for p in d.probabilities)
    assert d.choice in q.options


@pytest.mark.parametrize("n", [2, 3, 5, 10, 12, 26])
def test_option_count_is_respected_at_every_width(decider, n):
    """Ten options or fewer use the (A)..(J) rendering; more switch to single-token labels."""
    opts = [f"option {i}" for i in range(n)]
    d = decider.decide("some state", Question("Which one?", opts))
    assert len(d.probabilities) == n
    assert 0 <= d.index < n
    assert abs(sum(d.probabilities) - 1.0) < 1e-5


def test_several_questions_share_one_pass_and_keep_their_order(decider):
    qs = [Question("Is the value above ten?", ["no", "yes"]),
          Question("Is the value even?", ["no", "yes"]),
          Question("Sign", ["negative", "zero", "positive"])]
    out = decider.decide("value = 42", qs)
    assert [d.question for d in out] == [q.text for q in qs]
    assert [len(d.probabilities) for d in out] == [2, 2, 3]


def test_a_single_question_returns_a_single_decision(decider):
    from thisthat import Decision
    d = decider.decide("x", Question("Is x a letter?", ["no", "yes"]))
    assert isinstance(d, Decision)


def test_both_layouts_agree_on_an_easy_question(decider):
    """The layouts move the state, not the meaning; they should not disagree where truth is plain."""
    state = ("Agent coordinate: (2,2); zero-based row and column. Rows increase south; columns "
             "increase east. The local window is centered on A. '#': wall; '.': open; 'A': agent; "
             "'X': outside.\nLocal map:\n#####\n#...#\n#.A.#\n#...#\n#####")
    q = Question("Is the cell immediately north of the agent a wall?", ["no", "yes"])
    a = decider.decide(state, q, layout="state_first")
    b = decider.decide(state, q, layout="schema_first")
    assert a.index == b.index

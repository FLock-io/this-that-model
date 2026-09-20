"""Evaluation sets built from the simulator, as (state, questions, truth) rather than as labels.

The point of a simulator is that the answer is *computed*, so two things are available that a
human-annotated set cannot give:

  * for the deterministic questions, the truth is certain, and a model answering 0.8 to a question
    whose answer is 1.0 is miscalibrated -- the set can say so.
  * for the noisy-actuator question, the truth is a genuine probability, derived from the
    transition rules.  Accuracy against one sampled draw has a ceiling below 1; distance to the
    true distribution has none, and is the more informative measurement.

Every item therefore carries `truth` (the analytic distribution over the options) as well as
`answer_index` (one sampled outcome, for the deterministic families the same thing).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from thisthat import Question

from .game_outcomes import event_record
from .scaled_maze import (DIRECTIONS, SCORE_CRITERIA, TOPOLOGIES, _distance_level, make_maze,
                          render_request, solve, split_for_state, valid_actions)

WINDOW = 5
DELTAS = {"north": "row-1", "south": "row+1", "east": "col+1", "west": "col-1"}


@dataclass(frozen=True)
class Item:
    state: str
    questions: tuple[Question, ...]
    answer_index: tuple[int, ...]
    truth: tuple[tuple[float, ...], ...]      # analytic distribution per question
    task: str


def local_window(state, window: int = WINDOW) -> str:
    """The agent-centred ASCII window, worded exactly as the upstream renderer words it."""
    row, col = state["position"]
    size, radius = state["size"], window // 2
    walls = set(map(tuple, state["walls"]))
    lines = []
    for r in range(row - radius, row + radius + 1):
        line = []
        for c in range(col - radius, col + radius + 1):
            ch = "X" if not (0 <= r < size and 0 <= c < size) else "#" if (r, c) in walls else "."
            line.append("A" if (r, c) == (row, col) else ch)
        lines.append("".join(line))
    return (f"Agent coordinate: ({row},{col}); zero-based row and column. "
            "Rows increase south; columns increase east. "
            "The local window is centered on A. '#': wall; '.': open; 'A': agent; 'X': outside.\n"
            "Local map:\n" + "\n".join(lines))


def _clear(state, action: str) -> bool:
    dr, dc = DIRECTIONS[action]
    r, c = state["position"]
    dest = (r + dr, c + dc)
    return (0 <= dest[0] < state["size"] and 0 <= dest[1] < state["size"]
            and dest not in set(map(tuple, state["walls"])))


_MOVE_Q = ("If the agent attempts one cell {d} ({delta}), will the destination be inside the maze "
           "and not a wall? Use the local map: '.' and 'A' are traversable, '#' is a wall, and "
           "'X' is outside the maze.")


def _geometry_questions(state) -> tuple[list[Question], list[int], list[tuple[float, ...]]]:
    qs, gold, truth = [], [], []
    for action in DIRECTIONS:
        ok = _clear(state, action)
        qs.append(Question(_MOVE_Q.format(d=action, delta=DELTAS[action]), ["no", "yes"]))
        gold.append(int(ok))
        truth.append((float(not ok), float(ok)))
    return qs, gold, truth


def local_item(state) -> Item:
    """One state as the agent-centred window; four independent geometry questions."""
    qs, gold, truth = _geometry_questions(state)
    return Item(local_window(state), tuple(qs), tuple(gold), tuple(truth), "sim_local")


def global_item(state) -> Item:
    """The identical questions about the identical world, shown the whole rendered map."""
    qs, gold, truth = _geometry_questions(state)
    return Item(render_request(state)["state"], tuple(qs), tuple(gold), tuple(truth), "sim_global")


def event_item(state, intended: str, reliability: float, seed: int) -> Item:
    """The noisy actuator: the answer is a probability the transition rules define.

    `answer_index` is one draw from that probability and `truth` is the probability itself.  Scoring
    accuracy against the draw is legitimate but capped; scoring distance to `truth` is not capped
    and is what Section 4.3 of the paper reports.
    """
    rec = event_record(state, "train", intended, reliability, seed)
    p = float(rec["gold_probs"]["survive"]["true"])
    return Item(rec["state"],
                (Question(rec["questions"]["survive"]["instructions"], ["no", "yes"]),),
                (int(bool(rec["gold"]["survive"])),), ((1.0 - p, p),), "sim_event")


def distance_item(state) -> Item:
    """Shortest-path length bucketed into ordered bands, from the whole map."""
    oracle = solve(state)
    level = _distance_level(oracle["distance"])
    opts = [f"{i}: {c}" for i, c in enumerate(SCORE_CRITERIA)]
    t = [0.0] * len(opts)
    t[level] = 1.0
    return Item(render_request(state)["state"],
                (Question("How many moves remain on the shortest path from the agent to the goal?",
                          opts),),
                (level,), (tuple(t),), "sim_distance")


def context_ladder(sizes: Sequence[int] = (32, 50, 80, 128, 200), maps_per_size: int = 12,
                   starts_per_map: int = 4, seed: int = 991) -> dict[str, list[Item]]:
    """The same questions about the same worlds, shown the whole map and shown the window.

    Returns {"sim_global_<size>": [...], "sim_local_<size>": [...]}.  The two halves differ only in
    how much of the state the model is handed, so the gap between them is the cost of not
    decomposing -- and nothing else differs, which is what makes the comparison worth anything.
    """
    rng = random.Random(seed)
    out: dict[str, list[Item]] = {}
    for size in sizes:
        g, l = [], []
        for m in range(maps_per_size):
            state = make_maze(size, seed * 1000 + m, TOPOLOGIES[m % len(TOPOLOGIES)])
            walls = {tuple(w) for w in state["walls"]}
            open_cells = [(r, c) for r in range(size) for c in range(size) if (r, c) not in walls]
            for _ in range(starts_per_map):
                st = dict(state)
                st["position"] = list(rng.choice(open_cells))
                if tuple(st["position"]) == tuple(st["goal"]):
                    continue
                g.append(global_item(st))
                l.append(local_item(st))
        out[f"sim_global_{size}"] = g
        out[f"sim_local_{size}"] = l
    return out


HELDOUT_SIZE = 50            # no map of this size is ever trained on
RELIABILITIES = (0.1, 0.3, 0.7, 0.9)


def heldout_sets(sizes: Sequence[int] = (8, 16, 32), heldout_size: int = HELDOUT_SIZE,
                 maps_per_size: int = 24, starts_per_map: int = 6,
                 reliabilities: Sequence[float] = RELIABILITIES,
                 seed: int = 0) -> dict[str, list[Item]]:
    """The held-out evaluation sets, regenerated exactly as the paper's were.

    Two things here are load-bearing and easy to get wrong by approximating them.

    The reliabilities span 0.1 to 0.9, not just the confident end.  A set built only from reliable
    actuators has answers close to certain, an accuracy ceiling near 0.9, and tells you little;
    including rho = 0.1 and 0.3 is what makes the truth a genuine probability and pulls the ceiling
    down to roughly 0.75.  Evaluating on the confident-only variant scores this model at 0.610
    against a 0.880 ceiling, which is a real number about a different question.

    And the split is by the maze's own D4 wall-layout group, so a map that is a rotation or
    reflection of a training map cannot appear here, with size 50 held out entirely.  A 5x5 window
    is small enough that distinct mazes still render identically, so anything whose rendered state
    also occurs in the training half is then dropped by hash: a window the model has seen is not a
    held-out item.
    """
    rng = random.Random(seed)
    train_contexts: set[str] = set()
    evals: dict[str, list[Item]] = {k: [] for k in
                                    ("sim_local_ood", "sim_event_ood", "sim_distance_ood")}
    for size in tuple(sizes) + (heldout_size,):
        ood_size = size == heldout_size
        for m in range(maps_per_size):
            state = make_maze(size, seed * 1000 + m, TOPOLOGIES[m % len(TOPOLOGIES)])
            walls = [list(w) for w in state["walls"]]
            open_cells = [(r, c) for r in range(size) for c in range(size) if [r, c] not in walls]
            heldout = ood_size or split_for_state(state) in ("test", "ood", "calibration")
            for _ in range(starts_per_map):
                st = dict(state)
                st["position"] = list(rng.choice(open_cells))
                if tuple(st["position"]) == tuple(st["goal"]):
                    continue
                loc, dis = local_item(st), distance_item(st)
                acts = list(valid_actions(st))
                evs = []
                if len(acts) >= 2:
                    for i, rho in enumerate(reliabilities):
                        evs.append(event_item(st, acts[i % len(acts)], rho, rng.randrange(1 << 30)))
                if heldout:
                    evals["sim_local_ood"].append(loc)
                    evals["sim_distance_ood"].append(dis)
                    evals["sim_event_ood"] += evs
                else:
                    # the draws must be consumed in the same order either way, or the held-out
                    # half stops matching the one the paper evaluated
                    if rng.random() >= 0.12:
                        train_contexts.update(e.state for e in [loc, dis] + evs)
    for k in list(evals):
        evals[k] = [e for e in evals[k] if e.state not in train_contexts]
    return evals


def stochastic_set(n: int = 200, seed: int = 0, **kw) -> list[Item]:
    """The noisy-actuator half of the held-out sets: the calibration measurement of the paper."""
    return heldout_sets(seed=seed, **kw)["sim_event_ood"][:n]

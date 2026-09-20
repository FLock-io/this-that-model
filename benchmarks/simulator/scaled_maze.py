# Vendored from NanoJev (https://github.com/TianyuCodings/NanoJev), MIT License,
# Copyright (c) 2026 OpenJev contributors.  Imports adapted for this package;
# the simulator logic is unchanged.
#!/usr/bin/env python3
"""Deterministic, scalable maze decisions with complete geometric observations.

Only the Python standard library is used. Generation/labels may use graph search;
render_request reads geometry and legal one-step transitions only. No distances,
reachability labels, optimal actions, or other metadata enter model observations.

An action distribution is a *policy*: uniform over all shortest-path actions.
It is not the probability that a random future event occurs. Reachability and
distance-bucket questions instead have deterministic, one-hot truth labels.

Source groups identify D4-equivalent wall layouts, ignoring agent, goal, seed,
and topology names. Builders must keep each group in one split. split_for_state
provides a deterministic default; make_record also supports a builder's frozen
group-level assignment. The existing training loader checks group isolation.
"""

from collections import deque
import copy
import hashlib
import json
import random


CURRICULUM_SIZES = (8, 16, 32, 50)
TOPOLOGIES = ("corridor", "tree", "loops", "random_obstacle")
DIRECTIONS = {"north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1)}
SPLITS = ("train", "dev", "calibration", "test", "ood")
SCORE_CRITERIA = (
    "Already at the goal: zero moves remain.",
    "The shortest path requires 1 to 4 moves.",
    "The shortest path requires 5 to 16 moves.",
    "The shortest path requires 17 to 64 moves.",
    "The shortest path requires 65 to 256 moves.",
    "The shortest path requires 257 or more moves.",
    "The goal is unreachable from the agent.",
)
FORMAT_VERSION = "scaled_maze_v1"


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cell(value, size, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a [row, column] pair")
    if any(type(x) is not int or not 0 <= x < size for x in value):
        raise ValueError(f"{label} coordinates must be integers inside the map")
    return tuple(value)


def validate_state(state):
    if not isinstance(state, dict) or state.get("game") != "scaled_maze":
        raise ValueError("Expected a scaled_maze state")
    size = state.get("size")
    if type(size) is not int or size < 2:
        raise ValueError("size must be an integer of at least 2")
    walls = state.get("walls")
    if not isinstance(walls, (list, tuple)):
        raise ValueError("walls must be a sequence of coordinates")
    parsed = [_cell(cell, size, "wall") for cell in walls]
    if len(set(parsed)) != len(parsed):
        raise ValueError("Duplicate wall coordinates")
    position = _cell(state.get("position"), size, "position")
    goal = _cell(state.get("goal"), size, "goal")
    if position in set(parsed) or goal in set(parsed):
        raise ValueError("position and goal must be walkable")
    if "seed" in state and type(state["seed"]) is not int:
        raise ValueError("seed must be an integer when provided")
    if "topology" in state and state["topology"] not in TOPOLOGIES:
        raise ValueError(f"Unknown topology; expected one of {TOPOLOGIES}")


def _neighbors(cell, size, walls):
    row, col = cell
    for action, (dr, dc) in DIRECTIONS.items():
        dest = (row + dr, col + dc)
        if 0 <= dest[0] < size and 0 <= dest[1] < size and dest not in walls:
            yield action, dest


def _distances(size, walls, origin):
    """Internal BFS, used by generation and labels, never by the renderer."""
    distances, queue = {origin: 0}, deque([origin])
    while queue:
        cell = queue.popleft()
        for _, dest in _neighbors(cell, size, walls):
            if dest not in distances:
                distances[dest] = distances[cell] + 1
                queue.append(dest)
    return distances


def _tree_cells(size, rng, straight_bias):
    nodes = [(r, c) for r in range(1, size - 1, 2) for c in range(1, size - 1, 2)]
    root = rng.choice(nodes)
    visited, opened, stack = {root}, {root}, [(root, None)]
    node_set = set(nodes)
    while stack:
        cell, previous = stack[-1]
        candidates = []
        for action, (dr, dc) in DIRECTIONS.items():
            nxt = (cell[0] + 2 * dr, cell[1] + 2 * dc)
            if nxt in node_set and nxt not in visited:
                candidates.append((action, nxt))
        if not candidates:
            stack.pop()
            continue
        straight = [pair for pair in candidates if pair[0] == previous]
        action, nxt = straight[0] if straight and rng.random() < straight_bias else rng.choice(candidates)
        dr, dc = DIRECTIONS[action]
        opened.update((nxt, (cell[0] + dr, cell[1] + dc)))
        visited.add(nxt)
        stack.append((nxt, action))
    return opened


def _add_loops(opened, size, rng):
    # Open closed separators between two passages. Require the perpendicular
    # neighbors to be walls, avoiding diagonal-touch ambiguities at junctions.
    separators = []
    for row in range(1, size - 1):
        for col in range(1, size - 1):
            if (row, col) in opened:
                continue
            north, east = (row - 1, col) in opened, (row, col + 1) in opened
            south, west = (row + 1, col) in opened, (row, col - 1) in opened
            if (north and south and not east and not west) or (east and west and not north and not south):
                separators.append((row, col))
    rng.shuffle(separators)
    # A fixed density range varies with the seed, independently of map size.
    count = min(len(separators), max(1, round(len(separators) * rng.uniform(0.12, 0.30))))
    opened.update(separators[:count])
    return opened


def _random_obstacle_cells(size, rng):
    density = rng.uniform(0.20, 0.38)
    opened = {(r, c) for r in range(size) for c in range(size) if rng.random() >= density}
    # Retain the largest component. Ties have deterministic sorted traversal.
    # Unlike retry-until-success generation, every seed has a bounded execution.
    all_cells = {(r, c) for r in range(size) for c in range(size)}
    walls, unseen, largest = all_cells - opened, set(opened), set()
    while unseen:
        component = set(_distances(size, walls, min(unseen)))
        unseen.difference_update(component)
        if len(component) > len(largest):
            largest = component
    if len(largest) < 2:
        # Explicit deterministic boundary fallback; never returns an invalid map.
        largest = {(r, 0) for r in range(size)}
    return largest


def make_maze(size, seed, topology="corridor"):
    """Create a complete connected maze; sizes 8, 16, 32 and 50 are curriculum stages.

    Any integer size >= 5 is supported. corridor/tree use DFS spanning trees;
    corridor favors straight runs, loops opens extra tree separators, and
    random_obstacle retains a sampled obstacle grid's largest free component.
    Two graph sweeps choose separated endpoints, not a claimed exact diameter.
    All fields are JSON serializable, and identical arguments reproduce a state.
    """
    if type(size) is not int or size < 5:
        raise ValueError("Generated mazes require an integer size of at least 5")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if topology not in TOPOLOGIES:
        raise ValueError(f"Unknown topology; expected one of {TOPOLOGIES}")
    rng = random.Random(f"{FORMAT_VERSION}:{size}:{seed}:{topology}")
    if topology == "random_obstacle":
        opened = _random_obstacle_cells(size, rng)
    else:
        opened = _tree_cells(size, rng, 0.88 if topology == "corridor" else 0.0)
        if topology == "loops":
            opened = _add_loops(opened, size, rng)
    walls = {(r, c) for r in range(size) for c in range(size)} - opened
    origin = rng.choice(sorted(opened))
    first = _distances(size, walls, origin)
    position = max(first, key=lambda cell: (first[cell], cell))
    second = _distances(size, walls, position)
    goal = max(second, key=lambda cell: (second[cell], cell))
    state = {"game": "scaled_maze", "size": size, "walls": [list(x) for x in sorted(walls)],
             "position": list(position), "goal": list(goal), "seed": seed, "topology": topology}
    validate_state(state)
    return state


def valid_actions(state):
    """Legal orthogonal moves in north/east/south/west order; goal is terminal."""
    validate_state(state)
    if tuple(state["position"]) == tuple(state["goal"]):
        return []
    return [action for action, _ in _neighbors(tuple(state["position"]), state["size"],
                                               {tuple(cell) for cell in state["walls"]})]


def step(state, action):
    """Apply one legal move, returning a deep copy; blocked/terminal moves raise."""
    if not isinstance(action, str) or action not in valid_actions(state):
        raise ValueError("Illegal maze action")
    result = copy.deepcopy(state)
    dr, dc = DIRECTIONS[action]
    result["position"] = [state["position"][0] + dr, state["position"][1] + dc]
    return result


def solve(state):
    """Exact BFS oracle for labels. Unreachable states tie all legal actions."""
    validate_state(state)
    actions = valid_actions(state)
    distances = _distances(state["size"], {tuple(cell) for cell in state["walls"]}, tuple(state["goal"]))
    distance = distances.get(tuple(state["position"]))
    costs = {}
    for action in actions:
        dr, dc = DIRECTIONS[action]
        dest = (state["position"][0] + dr, state["position"][1] + dc)
        costs[action] = None if dest not in distances else 1 + distances[dest]
    best = min((cost for cost in costs.values() if cost is not None), default=None)
    at_goal = tuple(state["position"]) == tuple(state["goal"])
    return {"terminal": not actions, "at_goal": at_goal, "reachable": distance is not None,
            "distance": distance, "action_costs": costs,
            "optimal_actions": [action for action in actions if costs[action] == best],
            "objective": "minimize moves to the goal; if unreachable, all legal actions tie"}


def transform_cell(cell, size, symmetry):
    if type(symmetry) is not int or not 0 <= symmetry < 8:
        raise ValueError("symmetry must be an integer from 0 to 7")
    row, col = cell
    if symmetry >= 4:
        col = size - 1 - col
    for _ in range(symmetry % 4):
        row, col = col, size - 1 - row
    return row, col


def transform_state(state, symmetry):
    validate_state(state)
    result = copy.deepcopy(state)
    result["walls"] = [list(cell) for cell in sorted(transform_cell(x, state["size"], symmetry)
                                                   for x in state["walls"])]
    for name in ("position", "goal"):
        result[name] = list(transform_cell(state[name], state["size"], symmetry))
    return result


def source_group_id(state):
    """D4 canonical wall-layout group, independent of both agent and goal."""
    validate_state(state)
    size = state["size"]
    variants = [tuple(sorted(transform_cell(cell, size, i) for cell in state["walls"])) for i in range(8)]
    return f"maze:{size}:" + _digest({"size": size, "walls": min(variants)})


def state_id(state):
    """Physical observation identity; seed/topology/labels/split never affect it."""
    validate_state(state)
    return "maze_state:" + _digest({"size": state["size"], "walls": sorted(map(tuple, state["walls"])),
                                    "position": list(state["position"]), "goal": list(state["goal"])})


def split_for_state(state):
    """Stable default hash split; OOD here is a held-out group, not a size claim.

    For size/topology OOD experiments, builders should preassign whole groups and
    pass those assignments to make_record instead. No per-row random splitting.
    """
    bucket = int(_digest([FORMAT_VERSION, "split", source_group_id(state)]), 16) % 100
    for upper, name in ((70, "train"), (80, "dev"), (85, "calibration"), (95, "test"), (100, "ood")):
        if bucket < upper:
            return name
    raise AssertionError("Unreachable split bucket")


def difficulty_metrics(state):
    """Offline geometry statistics, deliberately excluded from model inputs."""
    validate_state(state)
    size, walls = state["size"], {tuple(cell) for cell in state["walls"]}
    opened = {(r, c) for r in range(size) for c in range(size)} - walls
    degrees = {cell: sum(1 for _ in _neighbors(cell, size, walls)) for cell in opened}
    edges = sum(degrees.values()) // 2
    components, unseen = 0, set(opened)
    while unseen:
        unseen.difference_update(_distances(size, walls, min(unseen)))
        components += 1
    turns, straight = 0, 0
    for cell, degree in degrees.items():
        if degree == 2:
            neighbors = [dest for _, dest in _neighbors(cell, size, walls)]
            is_straight = neighbors[0][0] == neighbors[1][0] or neighbors[0][1] == neighbors[1][1]
            straight += int(is_straight)
            turns += int(not is_straight)
    return {"free_cells": len(opened), "wall_fraction": len(walls) / (size * size),
            "connected_components": components, "edges": edges,
            "cycle_rank": edges - len(opened) + components,
            "dead_ends": sum(degree == 1 for degree in degrees.values()),
            "junctions": sum(degree >= 3 for degree in degrees.values()),
            "degree_two_turns": turns, "degree_two_straight": straight}


def render_request(state):
    """Complete compact ASCII plus explicit coordinates; no graph search or labels.

    Dynamic Choice exists only at 2+ legal moves. A caller executes a sole legal
    action directly, and stops at zero legal moves. Boolean/Score remain present
    at terminal, forced, and unreachable states. No map cell is cropped.
    """
    validate_state(state)
    size, walls = state["size"], {tuple(cell) for cell in state["walls"]}
    position, goal = tuple(state["position"]), tuple(state["goal"])
    rows = []
    for row in range(size):
        line = []
        for col in range(size):
            cell = row, col
            char = "#" if cell in walls else "."
            if cell == goal:
                char = "G"
            if cell == position:
                char = "@" if position == goal else "A"
            line.append(char)
        rows.append("".join(line))
    observation = (f"Maze {size}x{size}. Coordinates are zero-based (row,column). "
                   f"Agent=({position[0]},{position[1]}), goal=({goal[0]},{goal[1]}).\n"
                   "#=wall, .=free, A=agent, G=goal, @=agent at goal. "
                   "Move one cell north/east/south/west; no diagonals or wraparound. "
                   "Stop on the goal.\n" + "\n".join(rows))
    questions = {}
    actions = valid_actions(state)
    if len(actions) >= 2:
        criteria = {}
        for action in actions:
            dr, dc = DIRECTIONS[action]
            criteria[action] = f"Move {action} to ({position[0] + dr},{position[1] + dc})."
        questions["action"] = {"type": "choice", "instructions":
            "Choose a legal next move on a shortest path to the goal. "
            "All equally short next moves tie; if the goal is unreachable, all legal moves tie.",
            "criteria": criteria}
    questions["solvable"] = {"type": "boolean", "instructions":
        "The agent can reach the goal using legal moves, including zero moves if already there."}
    questions["value"] = {"type": "score", "instructions":
        "Classify the minimum number of legal moves required to reach the goal.",
        "criteria": list(SCORE_CRITERIA)}
    return {"state": observation, "questions": questions}


def _distance_level(distance):
    if distance is None:
        return 6
    if distance == 0:
        return 0
    for upper, level in ((4, 1), (16, 2), (64, 3), (256, 4)):
        if distance <= upper:
            return level
    return 5


def make_record(state, split):
    """Create a training row with independently computed labels and source IDs.

    The explicit split is a builder's group-level assignment, not inferred from
    answers. Use split_for_state for the default hash protocol. Repeated starts
    must not be presented as new maps; state_id also detects exact repetitions.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split; expected one of {SPLITS}")
    rendered, oracle, actions = render_request(state), solve(state), valid_actions(state)
    level = _distance_level(oracle["distance"])
    gold = {"solvable": oracle["reachable"], "value": level}
    probs = {"solvable": {"false": float(not oracle["reachable"]), "true": float(oracle["reachable"])},
             "value": {str(i): float(i == level) for i in range(len(SCORE_CRITERIA))}}
    kinds = {qid: "deterministic_truth" for qid in gold}
    label_kinds = dict(kinds)
    if "action" in rendered["questions"]:
        optimal = oracle["optimal_actions"]
        gold["action"] = optimal[0]
        probs["action"] = {action: (1.0 / len(optimal) if action in optimal else 0.0) for action in actions}
        kinds["action"] = "optimal_action_policy"
        label_kinds["action"] = "reference_argmax_compatibility"
    sid = state_id(state)
    # Store only the actual geometry, not arbitrary caller-supplied fields.
    environment = {name: copy.deepcopy(state[name]) for name in
                   ("game", "size", "walls", "position", "goal", "seed", "topology") if name in state}
    metrics = difficulty_metrics(state)
    manhattan = sum(abs(a - b) for a, b in zip(state["position"], state["goal"]))
    metrics.update({"shortest_path_length": oracle["distance"], "manhattan_distance": manhattan,
                    "detour_ratio": (oracle["distance"] / manhattan)
                    if oracle["distance"] is not None and manhattan else None})
    return {"id": sid, "state_id": sid, "family_id": "scaled_maze", "split": split,
            **rendered, "gold": gold, "gold_probs": probs, "gold_probs_kind": kinds,
            "gold_label_kind": label_kinds,
            "optimal_actions": {"action": list(oracle["optimal_actions"])},
            "metadata": {"source": "self_authored_programmatic", "format_version": FORMAT_VERSION,
                         "source_group_id": source_group_id(state),
                         "split_assignment": "caller_frozen_map_group",
                         "environment_state": environment, "oracle": oracle,
                         "difficulty": metrics, "legal_actions": actions,
                         "forced_action": actions[0] if len(actions) == 1 else None,
                         "action_target_semantics": "uniform_optimal_action_policy_not_event_probability",
                         "hard_action_label_semantics": "first_optimal_action_for_compatibility_only"}}

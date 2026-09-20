# Vendored from NanoJev (https://github.com/TianyuCodings/NanoJev), MIT License,
# Copyright (c) 2026 OpenJev contributors.  Imports adapted for this package;
# the simulator logic is unchanged.
"""Stochastic event questions with independent, verifiable one-step outcomes.

Training may consume the sampled event only. The exact Bernoulli probability is
retained for held-out probability-recovery evaluation, separate from game policy.
"""
import copy
import hashlib
import json
import random

DIRECTIONS = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}


def event_record(state, split, intended, reliability, seed):
    """A noisy actuator executes intended with rho, each alternative with (1-rho)/(K-1)."""
    if not 0 <= reliability <= 1:
        raise ValueError("Actuator reliability must be in [0,1]")
    if state["game"] == "scaled_maze":
        from .scaled_maze import render_request, source_group_id
        actions = list(DIRECTIONS)
        walls = set(map(tuple, state["walls"]))
        r, c = state["position"]
        safe = {}
        for action, (dr, dc) in DIRECTIONS.items():
            target = (r + dr, c + dc)
            safe[action] = (0 <= target[0] < state["size"] and 0 <= target[1] < state["size"] and target not in walls)
        group = source_group_id(state)
        geometry = render_request(state)["state"]
        criterion = "The attempted move stays inside the maze and does not enter a wall."
    elif state["game"] == "snake":
        from .snake_game import render_request, valid_actions, step
        actions = list(valid_actions(state))
        next_states = {action: step(state, action) for action in actions}
        safe = {action: not following["done"] or following.get("outcome") == "win" for action, following in next_states.items()}
        geometry = render_request(state)["state"]
        episode_key = {"size": state["size"], "seed": state["seed"] & ((1 << 64) - 1)}
        group = "snake_episode:" + hashlib.sha256(json.dumps(episode_key, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
        criterion = "The snake survives this move without a wall or body collision (filling the board also counts as survival)."
    else:
        raise ValueError("Unsupported game")
    if intended not in actions or len(actions) < 2:
        raise ValueError("Expected an offered intended action and at least two actuator outcomes")
    action_probability = {a: reliability if a == intended else (1 - reliability) / (len(actions) - 1) for a in actions}
    probability = sum(action_probability[a] for a in actions if safe[a])
    probability = min(1.0, max(0.0, probability))
    rng = random.Random(seed)
    draw = rng.random()
    cumulative = 0.0
    realized = actions[-1]
    for action in actions:
        cumulative += action_probability[action]
        if draw < cumulative:
            realized = action
            break
    observed = safe[realized]
    event = {"intended_action": intended, "reliability": reliability,
             "actuator_actions": actions, "event": "one_step_collision_free"}
    visible = geometry + "\nActuator: execute " + intended + " with probability " + format(reliability, '.8g') + "; otherwise choose uniformly among " + ", ".join(a for a in actions if a != intended) + "."
    identity = hashlib.sha256(json.dumps({"geometry": geometry, "event": event}, sort_keys=True).encode()).hexdigest()
    record_id = "event_v4:" + identity[:24]
    return {"id": record_id, "state_id": record_id, "family_id": state["game"] + "_noisy_actuator",
            "split": split, "state": visible,
            "questions": {"survive": {"type": "boolean", "instructions": "Under the stated random actuator, will this one move be collision-free? " + criterion,
                                        "criteria": {"true": "The randomly executed move is collision-free, including a move that completes the board.",
                                                     "false": "The randomly executed move causes a wall, boundary, or body collision."}}},
            "gold": {"survive": observed},
            "gold_label_kind": {"survive": "observed_outcome"},
            "gold_probs": {"survive": {"false": 1-probability, "true": probability}},
            "gold_probs_kind": {"survive": "programmatic_conditional_distribution"},
            "metadata": {"source": "self_authored_stochastic_simulator", "source_group_id": group,
                         "event_spec": event, "environment_state": copy.deepcopy(state),
                         "outcome_seed": seed, "realized_actuator_action": realized,
                         "exact_event_probability": probability, "safe_actions": safe,
                         "probability_meaning": "one-step event probability under the stated actuator; not policy probability",
                         "model_input_oracle_fields": []}}

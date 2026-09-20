# Vendored from NanoJev (https://github.com/TianyuCodings/NanoJev), MIT License,
# Copyright (c) 2026 OpenJev contributors.  Imports adapted for this package;
# the simulator logic is unchanged.
"""Deterministic Snake dynamics and one-step decision questions (standard library).

Body order is head first. Direction records the last attempted direction, including
on collision; a collision preserves the last valid body. Food uses a stored
SplitMix64 state with rejection sampling, so JSON round trips preserve the stream.
Rendered requests omit seed/RNG and exact outcome labels. Seeds differing by a
multiple of 2**64 share a random stream and an episode group.
"""
import copy
import hashlib
import json

DIRECTIONS = {'north': (-1, 0), 'east': (0, 1), 'south': (1, 0), 'west': (0, -1)}
REVERSE = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
SPLITS = {'train', 'dev', 'calibration', 'test', 'ood'}
MASK64 = (1 << 64) - 1


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _random64(rng_state):
    rng_state = (rng_state + 0x9E3779B97F4A7C15) & MASK64
    value = rng_state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return rng_state, value ^ (value >> 31)


def _food(body, size, rng_state):
    occupied = {tuple(cell) for cell in body}
    free = [[r, c] for r in range(size) for c in range(size) if (r, c) not in occupied]
    if not free:
        return None, rng_state
    limit = (1 << 64) - ((1 << 64) % len(free))
    while True:
        rng_state, value = _random64(rng_state)
        if value < limit:
            return free[value % len(free)], rng_state


def validate_state(state):
    required = {'game', 'size', 'seed', 'body', 'direction', 'food', 'rng_state', 'done', 'outcome', 'score', 'steps'}
    if not isinstance(state, dict) or not required <= state.keys() or state['game'] != 'snake':
        raise ValueError('Expected a complete Snake state')
    n = state['size']
    if type(n) is not int or n < 2 or type(state['seed']) is not int:
        raise ValueError('Size must be an integer >= 2 and seed an integer')
    if type(state['rng_state']) is not int or not 0 <= state['rng_state'] <= MASK64:
        raise ValueError('Invalid 64-bit RNG state')
    if type(state['done']) is not bool or any(type(state[k]) is not int or state[k] < 0 for k in ('score', 'steps')):
        raise ValueError('Invalid terminal flag or counters')
    if not isinstance(state['direction'], str) or state['direction'] not in DIRECTIONS:
        raise ValueError('Unknown direction')
    body = state['body']
    def valid_cell(cell):
        return type(cell) is list and len(cell) == 2 and all(type(x) is int and 0 <= x < n for x in cell)
    if type(body) is not list or not body or not all(valid_cell(cell) for cell in body):
        raise ValueError('Body must contain in-bounds JSON coordinate pairs')
    occupied = {tuple(cell) for cell in body}
    if len(occupied) != len(body) or any(sum(abs(a-b) for a, b in zip(x, y)) != 1 for x, y in zip(body, body[1:])):
        raise ValueError('Body cells must be unique and adjacent in head-first order')
    if not state['done'] and len(body) > 1 and tuple(a-b for a, b in zip(body[0], body[1])) != DIRECTIONS[state['direction']]:
        raise ValueError('Direction must match the live head-neck orientation')
    if state['done']:
        if not isinstance(state['outcome'], str) or state['outcome'] not in {'win', 'wall_collision', 'self_collision'}:
            raise ValueError('Invalid terminal outcome')
    elif state['outcome'] is not None:
        raise ValueError('A live state has no terminal outcome')
    full = len(body) == n*n
    if full:
        if not state['done'] or state['outcome'] != 'win' or state['food'] is not None:
            raise ValueError('A full board must be a food-free win')
    elif state['outcome'] == 'win' or not valid_cell(state['food']) or tuple(state['food']) in occupied:
        raise ValueError('Food must occupy a free cell; only a full board is a win')


def make_snake(size: int, seed: int):
    if type(size) is not int or size < 2 or type(seed) is not int:
        raise ValueError('Size must be an integer >= 2 and seed an integer')
    length, row = min(3, size), size // 2
    col = max(length - 1, size // 2)
    body = [[row, col - i] for i in range(length)]
    food, rng_state = _food(body, size, seed & MASK64)
    state = {'game': 'snake', 'size': size, 'seed': seed, 'body': body, 'direction': 'east',
             'food': food, 'rng_state': rng_state, 'done': False, 'outcome': None, 'score': 0, 'steps': 0}
    validate_state(state)
    return state


def valid_actions(state):
    """Non-reverse proposals, including actions that will collide."""
    validate_state(state)
    return [] if state['done'] else [action for action in DIRECTIONS if action != REVERSE[state['direction']]]


def _destination(state, action):
    dr, dc = DIRECTIONS[action]
    return [state['body'][0][0] + dr, state['body'][0][1] + dc]


def _collision(state, destination, growing):
    if any(x < 0 or x >= state['size'] for x in destination):
        return 'wall_collision'
    occupied = state['body'] if growing else state['body'][:-1]
    return 'self_collision' if destination in occupied else None


def one_step_safe(state, action):
    """Exact next-step collision predicate; a full-board win is safe."""
    if action not in valid_actions(state):
        raise ValueError('Action must be a non-reverse move in a live state')
    destination = _destination(state, action)
    return _collision(state, destination, destination == state['food']) is None


def step(state, action):
    if action not in valid_actions(state):
        raise ValueError('Action must be a non-reverse move in a live state')
    result = copy.deepcopy(state)
    destination = _destination(state, action)
    growing = destination == state['food']
    collision = _collision(state, destination, growing)
    result['direction'] = action
    result['steps'] += 1
    if collision:
        result['done'], result['outcome'] = True, collision
    else:
        result['body'] = [destination] + (copy.deepcopy(state['body']) if growing else copy.deepcopy(state['body'][:-1]))
        if growing:
            result['score'] += 1
            result['food'], result['rng_state'] = _food(result['body'], result['size'], result['rng_state'])
            if result['food'] is None:
                result['done'], result['outcome'] = True, 'win'
    validate_state(result)
    return result


def render_request(state):
    """Physical observations only; no RNG, future food or safety computation."""
    actions = valid_actions(state)
    if not actions:
        raise ValueError('Terminal states have no action question')
    text = (f"Snake on a {state['size']}x{state['size']} board. Coordinates are zero-based (row,column); "
            "north decreases row and east increases column. "
            f"Body in head-first order: {json.dumps(state['body'], separators=(',', ':'))}. "
            f"Direction: {state['direction']}. Food: {json.dumps(state['food'])}. "
            "Reverse moves are disallowed. A move onto food grows the body; otherwise the tail vacates. "
            "Entering that vacated tail cell is allowed. Walls and occupied body cells cause collision.")
    questions = {'action': {'type': 'choice', 'instructions':
        'First avoid collision on the next step. Among collision-free moves, choose one minimizing the Manhattan distance '
        'from the next head cell to the current food. Tied moves are equivalent. If every offered move collides, they all tie. '
        'Use this local one-step heuristic.',
        'criteria': {action: f'Move {action} to {json.dumps(_destination(state, action))}.' for action in actions}}}
    for action in actions:
        questions['safe_' + action] = {'type': 'boolean', 'instructions':
            f'If the snake takes {action}, will it avoid wall and body collision on this next step? '
            'Eating food retains the tail; otherwise the tail vacates and that vacated cell is safe to enter. Completing the board counts as safe.',
            'criteria': {'true': 'This specified move avoids wall and body collision, including a full-board win.',
                         'false': 'This specified move collides with a wall or a body cell that does not vacate on this step.'}}
    return {'state': text, 'questions': questions}


def make_record(state, split):
    if not isinstance(split, str) or split not in SPLITS:
        raise ValueError('Unknown split')
    public = render_request(state)
    actions = list(public['questions']['action']['criteria'])
    safe = {action: one_step_safe(state, action) for action in actions}
    safe_actions = [action for action in actions if safe[action]]
    distances = {action: sum(abs(x-y) for x, y in zip(_destination(state, action), state['food'])) for action in safe_actions}
    preferred = [action for action in safe_actions if distances[action] == min(distances.values())] if safe_actions else actions
    physical = {k: state[k] for k in ('game', 'size', 'body', 'direction', 'food')}
    identity = 'snake:' + _hash(physical)[:24]
    gold, probabilities = {'action': preferred[0]}, {'action': {a: float(a in preferred)/len(preferred) for a in actions}}
    kinds, label_kinds = {'action': 'optimal_action_policy'}, {'action': 'reference_argmax_compatibility'}
    events, outcomes = {}, {}
    for action in actions:
        qid, value = 'safe_' + action, safe[action]
        gold[qid], probabilities[qid] = value, {'false': float(not value), 'true': float(value)}
        kinds[qid], label_kinds[qid] = 'deterministic_truth', 'deterministic_truth'
        events[qid] = {'event': 'one_step_survival', 'action': action, 'horizon': 1, 'win_counts_as_survival': True,
                       'conditioning': 'The named action is executed from this physical state.'}
        outcomes[qid] = value
    return {'id': identity, 'state_id': identity, 'family_id': 'snake_one_step_safety_v4', 'split': split, **public,
            'gold': gold, 'gold_probs': probabilities, 'gold_probs_kind': kinds, 'gold_label_kind': label_kinds,
            'metadata': {'source': 'self_authored_programmatic', 'license': 'CC0-1.0', 'language': 'en',
                         'source_group_id': 'snake_episode:' + _hash({'size': state['size'], 'seed': state['seed'] & MASK64})[:24],
                         'source_group_scope': 'All steps from a size/canonical-64-bit-seed episode; this module performs no symmetry grouping. Cross-episode physical duplicates need dataset-level auditing.',
                         'environment_state': copy.deepcopy(state), 'event_spec': events, 'outcomes': outcomes,
                         'target_semantics': 'Local heuristic policy: minimize next-head Manhattan distance to food among one-step-safe moves; uniform over ties, or over all offered moves when none are safe. Global planning and eventual-success probability are separate targets.'}}

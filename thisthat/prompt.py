"""Turning a state and some questions into token ids, and finding the answer slots.

One state, N questions, N answer slots, one forward pass.  No answer letter is ever written
into the prompt, so slot k sees the state and every question but no earlier answer: the N
decisions are conditionally independent given the input, which is what lets them share a pass.

Two layouts are available and they differ only in where the state goes:

    state_first   Context: <state>            the default, and what the model was trained on
                  Question 1: ... Options: ...
                  Answer 1: (

    schema_first  Question 1: ... Options: ...   the questions first, so the token ids of that
                  Context: <state>               prefix are identical for every state and its
                  Answer 1: (                    attention cache can be computed once and reused

Schema-first costs |P| + m(|S| + N) tokens of prefill to ask the same m questions about m states,
against m(|P| + |S| + N) for state-first -- a saving of (m-1)|P| when one schema is applied to a
stream of states.  It is the layout to use for a fixed decision point in a program.
"""
from __future__ import annotations

import string
from typing import Sequence

from .types import MAX_OPTIONS, Question

LETTERS = "ABCDEFGHIJ"
NARROW = len(LETTERS)     # up to ten options keep the plain "(A) ..." rendering
DEFAULT_MAX_STATE_TOKENS = 1536

_LABELS: dict[int, tuple[list[str], list[int], list[int]]] = {}


def label_table(tok):
    """(label strings, their token ids, ids of "\\n(") -- MAX_OPTIONS single-token labels.

    A..Z first, so a narrow question renders exactly as it did before wide questions existed,
    then the two-letter combinations that this tokenizer happens to encode as one token.  Every
    label must be a single token: the head reads one position per answer, so a two-token label
    would put half the answer where nothing is looking.
    """
    key = id(tok)
    if key not in _LABELS:
        names = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase
                                                for b in string.ascii_uppercase]
        out = []
        for n in names:
            t = tok.encode(n, add_special_tokens=False)
            if len(t) == 1:
                out.append((n, t[0]))
            if len(out) == MAX_OPTIONS:
                break
        if len(out) < MAX_OPTIONS or len({i for _, i in out}) != MAX_OPTIONS:
            raise RuntimeError(
                f"this tokenizer yields only {len(out)} distinct single-token labels; "
                f"{MAX_OPTIONS} are needed")
        _LABELS[key] = ([n for n, _ in out], [i for _, i in out],
                        tok.encode("\n(", add_special_tokens=False))
    return _LABELS[key]


def option_label_ids(tok) -> list[int]:
    """The token ids the answer head is restricted to, in option order."""
    ids = label_table(tok)[1]
    for j, letter in enumerate(LETTERS):
        got = tok.encode(letter, add_special_tokens=False)
        if got != [ids[j]]:
            raise RuntimeError(f"label {letter!r} is not a single token in this tokenizer: {got}")
    return ids


def _options_ids(tok, options: Sequence[str]) -> list[int]:
    if len(options) <= NARROW:
        return tok.encode("".join(f"\n({LETTERS[j]}) {o}" for j, o in enumerate(options)),
                          add_special_tokens=False)
    _, lab_ids, open_ids = label_table(tok)
    out: list[int] = []
    for j, o in enumerate(options):
        out += open_ids + [lab_ids[j]] + tok.encode(f") {o}", add_special_tokens=False)
    return out


def _question_block(tok, k: int, q: Question, multi: bool, lead: str) -> list[int]:
    head = f"{lead}Question{' ' + str(k + 1) if multi else ''}: {q.text}\nOptions:"
    return tok.encode(head, add_special_tokens=False) + _options_ids(tok, q.options)


def _answer_slots(tok, n: int, ids: list[int], first_lead: str) -> list[int]:
    slots = []
    for k in range(n):
        lead = first_lead if k == 0 else "\n"
        ids += tok.encode(f"{lead}Answer{' ' + str(k + 1) if n > 1 else ''}: (",
                          add_special_tokens=False)
        slots.append(len(ids) - 1)
    return slots


def build(tok, state: str, questions: Sequence[Question], *, layout: str = "state_first",
          max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS) -> dict:
    """-> {ids, slots, n_options, prefix_len}.

    `slots[k]` is the position in `ids` whose hidden state answers question k.  `prefix_len` is
    the length of the state-independent prefix under schema_first, and 0 otherwise.
    """
    questions = list(questions)
    if not questions:
        raise ValueError("no questions")
    if layout not in ("state_first", "schema_first"):
        raise ValueError(f"layout must be 'state_first' or 'schema_first', got {layout!r}")
    multi = len(questions) > 1

    if layout == "schema_first":
        ids: list[int] = []
        for k, q in enumerate(questions):
            ids += _question_block(tok, k, q, multi, "\n\n" if k else "")
        prefix_len = len(ids)
        ids += tok.encode("\n\nContext:\n", add_special_tokens=False)
        ids += tok.encode(state, add_special_tokens=False)[:max_state_tokens]
        slots = _answer_slots(tok, len(questions), ids, "\n\n")
    else:
        # state_first interleaves: each question is followed by its own answer slot, so the
        # slots are collected as the blocks are appended rather than all at the end
        ids = tok.encode("Context:\n" + state, add_special_tokens=False)[:max_state_tokens + 3]
        slots, prefix_len = [], 0
        for k, q in enumerate(questions):
            ids += _question_block(tok, k, q, multi, "\n\n")
            ids += tok.encode(f"\nAnswer{' ' + str(k + 1) if multi else ''}: (",
                              add_special_tokens=False)
            slots.append(len(ids) - 1)

    return dict(ids=ids, slots=slots, n_options=[len(q.options) for q in questions],
                prefix_len=prefix_len)


def render(tok, state: str, questions: Sequence[Question], **kw) -> str:
    """The prompt as text, for looking at."""
    return tok.decode(build(tok, state, questions, **kw)["ids"])

"""v1 agent brain: evaluator-guided 1-ply lookahead via the native Search API.

For each decision, determinize the hidden state once (search_begin), then branch one
search_step per legal option, score each resulting state with the heuristic evaluator, and
pick the argmax. Perspective is handled: a resulting state where it's the opponent's turn is
scored from their view and negated. Everything is wrapped so it degrades to a legal move on
any error or clock pressure — main.py adds the outer watchdog.

Determinization note (v1): the opponent's hidden zones are filled with our own decklist as
valid filler. Our-turn option outcomes don't depend on the opponent's hidden cards, so this
is fine for 1-ply option ranking; deeper opponent-turn rollouts will want the real archetype
template (belief/opponent model) instead.
"""
import os
import random
import sys
import time

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

from cg.api import (  # noqa: E402
    to_observation_class, all_card_data, all_attack,
    search_begin, search_step, search_end,
)
from ptcg.perception import build_game_state  # noqa: E402
from ptcg.evaluator import evaluate, WIN_SCORE  # noqa: E402

_CARD_INDEX = None
_ATTACK_INDEX = None
_DECK = None

MOVE_BUDGET_S = 2.0      # soft wall-clock cap per decision (1-ply is ~ms; this is slack)
LOW_CLOCK_S = 15.0       # below this remaining bank, skip search and move instantly


def _lazy_init():
    global _CARD_INDEX, _ATTACK_INDEX
    if _CARD_INDEX is None:
        _CARD_INDEX = {c.cardId: c for c in all_card_data()}
        _ATTACK_INDEX = {a.attackId: a for a in all_attack()}


def set_deck(deck):
    global _DECK
    _DECK = list(deck)


def _first_basic(deck):
    for cid in deck:
        cd = _CARD_INDEX.get(cid)
        if cd and getattr(cd, "basic", False) and getattr(cd, "cardType", None) == 0:
            return cid
    return deck[0]


def _determinize(obs, deck):
    """Args for search_begin: our decklist fills every hidden zone (valid ids, has Basics)."""
    st = obs.current
    opp = 1 - st.yourIndex
    opp_active = []
    ops = st.players[opp]
    if ops.active and ops.active[0] is None:  # opponent's active is face-down → must predict it
        opp_active = [_first_basic(deck)]
    d = list(deck)
    return (d, d, d, d, d, opp_active, False)


def _legal_random(sel):
    n = len(sel.option)
    if n == 0:
        return []
    return random.sample(range(n), min(sel.maxCount, n))


def _score_state(search_state, my_index):
    obs = search_state.observation
    cur = getattr(obs, "current", None)
    if cur is None:
        return 0.0
    if getattr(cur, "result", -1) != -1:  # terminal
        if cur.result == my_index:
            return WIN_SCORE
        return -WIN_SCORE if cur.result in (0, 1) else 0.0
    gs = build_game_state(obs, _CARD_INDEX)
    s = evaluate(gs, card_index=_CARD_INDEX, attack_index=_ATTACK_INDEX).score
    # resulting state may be the opponent's turn → score is from their view; flip to ours.
    return s if getattr(cur, "yourIndex", my_index) == my_index else -s


def choose(obs_dict, time_left=None):
    """Return the list of option indices for this decision."""
    obs = to_observation_class(obs_dict)
    sel = obs.select
    if sel is None:
        return list(_DECK) if _DECK else []
    n = len(sel.option)
    if n == 0:
        return []
    my_index = obs.current.yourIndex if obs.current else 0

    # Clock pressure → instant legal move (never risk the bank).
    if time_left is not None and time_left < LOW_CLOCK_S:
        return _legal_random(sel)

    try:
        _lazy_init()
        root = search_begin(obs, *_determinize(obs, _DECK))
        sid = root.searchId
        deadline = time.monotonic() + MOVE_BUDGET_S
        try:
            scored = []
            for i in range(n):
                if time.monotonic() > deadline:
                    break
                try:
                    st = search_step(sid, [i])
                    scored.append((_score_state(st, my_index), i))
                except Exception:
                    continue

            if sel.maxCount <= 1:
                # optionally selecting nothing (minCount 0)
                if sel.minCount == 0:
                    try:
                        s0 = _score_state(search_step(sid, []), my_index)
                        if not scored or s0 >= max(scored)[0]:
                            return []
                    except Exception:
                        pass
                if scored:
                    return [max(scored)[1]]
                return _legal_random(sel)

            # multi-select: greedy top-maxCount by single-option score (valid when minCount<=1)
            if scored and sel.minCount <= 1:
                scored.sort(reverse=True)
                pick = sorted({i for _, i in scored[:sel.maxCount]})
                if len(pick) >= sel.minCount:
                    return pick[:sel.maxCount]
            return _legal_random(sel)
        finally:
            search_end()
    except Exception:
        return _legal_random(sel)

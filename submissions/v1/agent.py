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

MOVE_BUDGET_S = 2.0      # soft wall-clock cap per decision
LOW_CLOCK_S = 15.0       # below this remaining bank, skip search and move instantly
ROOT_BEAM = 6            # how many root options to fully roll out to end-of-turn
DEPTH_CAP = 24           # max our-turn actions explored in a rollout
NODE_BUDGET = 7000       # max search_step calls per decision (anytime cap; ~<1s)


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


def _is_our_turn(state, my_index):
    obs = getattr(state, "observation", None)
    cur = getattr(obs, "current", None)
    return (cur is not None and getattr(cur, "result", -1) == -1
            and getattr(obs, "select", None) is not None
            and getattr(cur, "yourIndex", None) == my_index)


def _greedy_rollout(state, my_index, ctx):
    """From an our-turn state, greedily play the rest of our turn (each step = the option that
    maximizes the resulting state's score), until the turn ends / game ends / budget hit.
    Returns the reached end-of-turn (or cut) state; the caller scores it as the line's value."""
    depth = 0
    while _is_our_turn(state, my_index) and depth < DEPTH_CAP:
        if ctx["n"] >= NODE_BUDGET or time.monotonic() > ctx["deadline"]:
            break
        sel = state.observation.select
        nopt = len(sel.option)
        if nopt == 0:
            break
        if sel.maxCount == 1 and sel.minCount <= 1:
            best = None
            for j in range(nopt):
                if ctx["n"] >= NODE_BUDGET or time.monotonic() > ctx["deadline"]:
                    break
                try:
                    cj = search_step(state.searchId, [j])
                    ctx["n"] += 1
                except Exception:
                    continue
                s = _score_state(cj, my_index)
                if best is None or s > best[0]:
                    best = (s, cj)
            if sel.minCount == 0:  # option of taking no more actions
                try:
                    c0 = search_step(state.searchId, [])
                    ctx["n"] += 1
                    if best is None or _score_state(c0, my_index) >= best[0]:
                        best = (0, c0)
                except Exception:
                    pass
            if best is None:
                break
            state = best[1]
        else:  # multi-select mid-turn: take a valid default set and continue
            k = max(sel.minCount, min(sel.maxCount, nopt))
            try:
                state = search_step(state.searchId, list(range(k)))
                ctx["n"] += 1
            except Exception:
                break
        depth += 1
    return state


def choose(obs_dict, time_left=None):
    """Turn-aware search: for the top root options, roll our turn to its end and pick the option
    whose best end-of-turn state scores highest. Falls back to a legal move on any trouble."""
    obs = to_observation_class(obs_dict)
    sel = obs.select
    if sel is None:
        return list(_DECK) if _DECK else []
    n = len(sel.option)
    if n == 0:
        return []
    my_index = obs.current.yourIndex if obs.current else 0

    if time_left is not None and time_left < LOW_CLOCK_S:
        return _legal_random(sel)

    try:
        _lazy_init()
        root = search_begin(obs, *_determinize(obs, _DECK))
        ctx = {"deadline": time.monotonic() + MOVE_BUDGET_S, "n": 0}
        try:
            # Rank root options by immediate score, keep their child states.
            imm = []
            for i in range(n):
                if ctx["n"] >= NODE_BUDGET or time.monotonic() > ctx["deadline"]:
                    break
                try:
                    ci = search_step(root.searchId, [i])
                    ctx["n"] += 1
                    imm.append((_score_state(ci, my_index), i, ci))
                except Exception:
                    continue
            if not imm:
                return _legal_random(sel)

            # Multi-select root: greedy top-maxCount by immediate score (no turn rollout).
            if sel.maxCount > 1:
                if sel.minCount <= 1:
                    imm.sort(key=lambda t: t[0], reverse=True)
                    pick = sorted({i for _, i, _ in imm[:sel.maxCount]})
                    if len(pick) >= sel.minCount:
                        return pick[:sel.maxCount]
                return _legal_random(sel)

            # Single-select: fully roll out the top-BEAM options to end-of-turn.
            imm.sort(key=lambda t: t[0], reverse=True)
            best_i, best_v = None, float("-inf")
            for _s, i, ci in imm[:ROOT_BEAM]:
                if best_i is not None and time.monotonic() > ctx["deadline"]:
                    break
                end = ci if not _is_our_turn(ci, my_index) else _greedy_rollout(ci, my_index, ctx)
                v = _score_state(end, my_index)
                if v > best_v:
                    best_v, best_i = v, i

            # Selecting nothing (minCount 0): compare its turn value too.
            if sel.minCount == 0:
                try:
                    c0 = search_step(root.searchId, [])
                    ctx["n"] += 1
                    end0 = c0 if not _is_our_turn(c0, my_index) else _greedy_rollout(c0, my_index, ctx)
                    if _score_state(end0, my_index) > best_v:
                        return []
                except Exception:
                    pass

            return [best_i] if best_i is not None else _legal_random(sel)
        finally:
            search_end()
    except Exception:
        return _legal_random(sel)

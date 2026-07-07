"""v3 agent brain: beam / best-first search over the turn (replaces v1's greedy rollout).

v1/v2 rolled our turn *greedily* — at each mid-turn decision they committed to the option with
the best immediate resulting-state score. That finds *a* reasonable turn, not the best one: a
setup move that looks weak now but enables a much stronger end-of-turn (evolve→attach→attack
sequencing, discard-to-draw, bench development) is never found (experiment #15: the heuristic
pilot fumbles combo/evolution archetypes).

Within our own turn there is no opponent to interleave, so this is a single-agent deterministic
search over action sequences — a best-first/beam problem, not a game tree. v3 keeps `BEAM_WIDTH`
partial lines alive at each ply (v1 = width 1), scoring end-of-turn states only for the answer
and partial states only to order the frontier. Each line is tagged with the *first* root option
on its path, so the search returns the first move of the best end-of-turn line it finds — i.e.
it can pick a first move that is not greedily best. Anytime + budget-bounded, same watchdog.

Determinization (unchanged from v1): opponent hidden zones filled with our decklist; fine for
ranking our own turn's outcomes. Reply-blind like v1 (opponent-response was neutral in #10 —
re-test it on top of this stronger search later).
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
BEAM_WIDTH = 8           # partial turn-lines kept alive per ply (v1 greedy == 1)
DEPTH_CAP = 24           # max our-turn plies explored
NODE_BUDGET = 7000       # max search_step calls per decision (anytime cap)
_PASS = -1               # first-option tag for the "take no more actions" line


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
    st = obs.current
    opp = 1 - st.yourIndex
    opp_active = []
    ops = st.players[opp]
    if ops.active and ops.active[0] is None:
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
    return s if getattr(cur, "yourIndex", my_index) == my_index else -s


def _is_our_turn(state, my_index):
    obs = getattr(state, "observation", None)
    cur = getattr(obs, "current", None)
    return (cur is not None and getattr(cur, "result", -1) == -1
            and getattr(obs, "select", None) is not None
            and getattr(cur, "yourIndex", None) == my_index)


def _out_of_budget(ctx):
    return ctx["n"] >= NODE_BUDGET or time.monotonic() > ctx["deadline"]


def _expand(state, ctx):
    """Children of a mid-turn state's current decision. Single-select → branch every option (+ the
    pass line when minCount==0). Multi-select mid-turn → one default set, to bound branching."""
    sel = state.observation.select
    nopt = len(sel.option)
    kids = []
    if nopt == 0:
        return kids
    if sel.maxCount == 1 and sel.minCount <= 1:
        for j in range(nopt):
            if _out_of_budget(ctx):
                break
            try:
                kids.append(search_step(state.searchId, [j]))
                ctx["n"] += 1
            except Exception:
                continue
        if sel.minCount == 0 and not _out_of_budget(ctx):
            try:
                kids.append(search_step(state.searchId, []))
                ctx["n"] += 1
            except Exception:
                pass
    else:
        k = max(sel.minCount, min(sel.maxCount, nopt))
        try:
            kids.append(search_step(state.searchId, list(range(k))))
            ctx["n"] += 1
        except Exception:
            pass
    return kids


def _beam_search_turn(root, my_index, ctx):
    """Beam search over the turn. Returns (best_value, best_first_option) — the first root option
    on the path to the best end-of-turn state found. best_first_option is an int, `_PASS`, or None
    (no line evaluated)."""
    best_v, best_first = float("-inf"), None
    root_sel = root.observation.select
    n = len(root_sel.option)

    # ply 0: branch the root options, tagging each line by its first option.
    pool = []  # (partial_eval, state, first_option)
    opts = [[i] for i in range(n)]
    if root_sel.minCount == 0:
        opts.append([])  # the pass / end-turn-now line
    for opt in opts:
        if _out_of_budget(ctx):
            break
        try:
            child = search_step(root.searchId, opt)
            ctx["n"] += 1
        except Exception:
            continue
        first = opt[0] if opt else _PASS
        if not _is_our_turn(child, my_index):  # completed turn (or terminal)
            v = _score_state(child, my_index)
            if v > best_v:
                best_v, best_first = v, first
        else:
            pool.append((_score_state(child, my_index), child, first))

    pool.sort(key=lambda t: t[0], reverse=True)
    frontier = [(s, f) for _v, s, f in pool[:BEAM_WIDTH]]

    # deeper plies: expand the beam, scoring completed turns for the answer.
    depth = 1
    while frontier and depth < DEPTH_CAP and not _out_of_budget(ctx):
        pool = []
        for state, first in frontier:
            for child in _expand(state, ctx):
                if not _is_our_turn(child, my_index):
                    v = _score_state(child, my_index)
                    if v > best_v:
                        best_v, best_first = v, first
                else:
                    pool.append((_score_state(child, my_index), child, first))
            if _out_of_budget(ctx):
                break
        pool.sort(key=lambda t: t[0], reverse=True)
        frontier = [(s, f) for _v, s, f in pool[:BEAM_WIDTH]]
        depth += 1

    # budget cut before any line finished the turn → fall back to the best partial line.
    if best_first is None and frontier:
        best_first = frontier[0][1]
    return best_v, best_first


def choose(obs_dict, time_left=None):
    """Beam-search the turn and return the first move of the best end-of-turn line. Multi-select
    root decisions keep v1's greedy top-k. Degrades to a legal move on any trouble."""
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
            # Multi-select root (mulligan / prize / discard): greedy top-maxCount by immediate score.
            if sel.maxCount > 1:
                if sel.minCount <= 1:
                    scored = []
                    for i in range(n):
                        if _out_of_budget(ctx):
                            break
                        try:
                            ci = search_step(root.searchId, [i])
                            ctx["n"] += 1
                            scored.append((_score_state(ci, my_index), i))
                        except Exception:
                            continue
                    if scored:
                        scored.sort(key=lambda t: t[0], reverse=True)
                        pick = sorted({i for _s, i in scored[:sel.maxCount]})
                        if len(pick) >= sel.minCount:
                            return pick[:sel.maxCount]
                return _legal_random(sel)

            # Single-select: beam-search the turn.
            _v, first = _beam_search_turn(root, my_index, ctx)
            if first is None:
                return _legal_random(sel)
            return [] if first == _PASS else [first]
        finally:
            search_end()
    except Exception:
        return _legal_random(sel)

"""v4 agent: MCTS-H — determinized PUCT MCTS with heuristic leaf evaluation (docs §3).

The decision pipeline:
  perception → governor (clock budget + prize-path critical trigger) → determinize K worlds →
  PUCT MCTS over the native Search API (heuristic leaf eval) → most-visited root action.

Deep MCTS runs only on critical turns; routine turns get a shallow search; low clock skips search
for an instant legal move. Multi-select root decisions (mulligan / prize / discard) use a one-step
heuristic (MCTS is single-select-root). Everything is wrapped so any error or clock pressure
degrades to a legal move — main.py adds the outer watchdog (crash = timeout = instant loss).
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
    to_observation_class, all_card_data, all_attack, search_begin, search_step, search_end,
)
from ptcg.perception import build_game_state  # noqa: E402
from ptcg.opponent import OpponentTracker      # noqa: E402
from search.determinizer import determinize    # noqa: E402
from search import mcts, governor               # noqa: E402

_CI = _AI = _DECK = None
_TRACKER = None
_ARCHS = None
MAX_SIMS = 200_000   # cap; the governor deadline is the real limit


def _lazy_init():
    global _CI, _AI, _TRACKER, _ARCHS
    if _CI is None:
        _CI = {c.cardId: c for c in all_card_data()}
        _AI = {a.attackId: a for a in all_attack()}
        try:
            from ptcg.archetypes import default_archetypes
            _ARCHS = default_archetypes()
        except Exception:
            _ARCHS = None
    if _TRACKER is None:
        _TRACKER = OpponentTracker()


def set_deck(deck):
    """Called once per game at deck selection — also resets the per-game opponent tracker so its
    seen-card evidence never leaks across games (e.g. in a multi-game validation harness)."""
    global _DECK, _TRACKER
    _DECK = list(deck)
    _TRACKER = OpponentTracker()


def _legal_random(sel):
    n = len(sel.option)
    return random.sample(range(n), min(sel.maxCount, n)) if n else []


def _multi_pick(obs, my_index):
    """Heuristic for a multi-select root: score each option's resulting state, take top-maxCount."""
    sel = obs.select
    n = len(sel.option)
    worlds = determinize(obs, _DECK, _CI, k=1, archetypes=_ARCHS, opp_tracker=_TRACKER)
    root = search_begin(obs, *worlds[0])
    try:
        scored = []
        for i in range(n):
            try:
                child = search_step(root.searchId, [i])
                scored.append((mcts.leaf_value(child, my_index, _CI, _AI), i))
            except Exception:
                continue
        if not scored:
            return _legal_random(sel)
        scored.sort(reverse=True)
        pick = sorted({i for _s, i in scored[:sel.maxCount]})
        if len(pick) < sel.minCount:
            pick = sorted(range(n))[:sel.minCount]
        return pick[:sel.maxCount]
    finally:
        search_end()


def choose(obs_dict, time_left=None):
    obs = to_observation_class(obs_dict)
    sel = obs.select
    if sel is None:
        return list(_DECK) if _DECK else []
    try:
        _lazy_init()
        try:
            _TRACKER.observe(obs)
        except Exception:
            pass
        n = len(sel.option)
        if n == 0:
            return []
        my_index = obs.current.yourIndex if obs.current else 0

        if sel.maxCount > 1:                       # multi-select root → heuristic
            return _multi_pick(obs, my_index)

        gs = build_game_state(obs, _CI)
        budget = governor.allocate(gs, time_left, _CI, _AI)
        if budget.mode == "instant":
            return _legal_random(sel)

        worlds = determinize(obs, _DECK, _CI, k=budget.k, archetypes=_ARCHS, opp_tracker=_TRACKER)
        deadline = time.monotonic() + budget.seconds
        pick, _stats = mcts.search(obs, worlds, _CI, _AI, sims=MAX_SIMS, deadline=deadline)
        return pick if pick is not None else _legal_random(sel)
    except Exception:
        return _legal_random(sel)

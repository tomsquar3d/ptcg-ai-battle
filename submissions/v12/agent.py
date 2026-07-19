"""v12 agent: v11 (net_ladder2 + Alakazam + raised search budget) + a hand-authored turn-plan /
KO-target SKILL PRIOR applied at the MCTS *root* nodes only, blended with the net prior by `lam`
(experiment #63; the stale docs/v8 brief's "#45/v8" collided with already-shipped work).

The skill prior biases root first-visit order toward prize-trade-correct actions without touching
`_select` or the leaf evaluator (`mcts.search(root_prior_fn=...)`). `lam` (env `PTCG_SKILL_LAM`,
default 0.5) is the single knob; **`lam == 0` reproduces v11 exactly** (root_prior_fn=None), so the
change is a zero-risk experiment — rollback is setting one number.

Everything else is v11 verbatim (deck-conditioned ONNX net-guided MCTS-H on the ladder Alakazam deck,
net_ladder2, `governor.ROUTINE_S = 1.5`). Graceful degradation unchanged: a missing net/onnxruntime,
a skill_prior import error, or any exception in the prior degrades to the plain v11 search / a legal
move — the crash==loss guarantee holds. main.py adds the outer watchdog.
"""
import os
import random
import sys
import time
from functools import partial

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

# v11: raise the routine-turn search budget (we were using ~6% of the 600 s clock; #61 shows 4× search =
# +13–23%). Critical turns already deep (5 s). The governor's CRIT_FRACTION cap (≤10% of remaining clock)
# + LOW_CLOCK_S floor still bound each move, so this stays provably timeout-safe (timing_v11 validates).
governor.ROUTINE_S = 1.5        # was 0.4 → ~4× routine search depth

# v12: skill-prior blend weight at the MCTS roots. 0.0 == v11 exactly (root_prior_fn=None). Read once
# from the env; the A/B harness can override the module global per-seat to sweep lam in one process.
_SKILL_LAM = float(os.environ.get("PTCG_SKILL_LAM", "0.5"))

_CI = _AI = _DECK = None
_TRACKER = None
_ARCHS = None
_GUIDE = None
_GUIDE_TRIED = False
_MODEL_DIR = None
MAX_SIMS = 200_000   # cap; the governor deadline is the real limit


def _model_dir():
    """Locate value.onnx/policy.onnx: alongside this file, else the Kaggle agent dir."""
    global _MODEL_DIR
    if _MODEL_DIR is None:
        try:
            _MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            _MODEL_DIR = "."
        if not os.path.exists(os.path.join(_MODEL_DIR, "value.onnx")):
            alt = "/kaggle_simulations/agent"
            if os.path.exists(os.path.join(alt, "value.onnx")):
                _MODEL_DIR = alt
    return _MODEL_DIR


def _condition_guide():
    """Tell the deck-conditioned net which deck it is piloting (no-op for a v1 net / missing guide).
    archetype_name=None → archetype_id 0, matching how ladder decks were trained."""
    if _GUIDE is not None and _DECK and hasattr(_GUIDE, "set_deck"):
        try:
            _GUIDE.set_deck(_DECK, None)
        except Exception:
            pass


def _lazy_init():
    global _CI, _AI, _TRACKER, _ARCHS, _GUIDE, _GUIDE_TRIED
    if _CI is None:
        _CI = {c.cardId: c for c in all_card_data()}
        _AI = {a.attackId: a for a in all_attack()}
        try:
            from ptcg.archetypes import default_archetypes
            _ARCHS = default_archetypes()
        except Exception:
            _ARCHS = None
    if not _GUIDE_TRIED:                    # try once; degrade to heuristic MCTS if unavailable
        _GUIDE_TRIED = True
        try:
            from nn.onnx_guide import OnnxGuide
            _GUIDE = OnnxGuide(_model_dir(), _CI, _AI)
        except Exception:
            _GUIDE = None
        _condition_guide()                 # deck-condition the net once it (and _DECK) are available
    if _TRACKER is None:
        _TRACKER = OpponentTracker()


def set_deck(deck):
    """Called once per game at deck selection — also resets the per-game opponent tracker and
    re-conditions the net on the piloted deck."""
    global _DECK, _TRACKER
    _DECK = list(deck)
    _TRACKER = OpponentTracker()
    _condition_guide()


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
        prior_fn = _GUIDE.priors if _GUIDE is not None else mcts.uniform_prior
        value_fn = _GUIDE.value if _GUIDE is not None else None
        root_prior_fn = None                       # lam==0 → exactly v11; import lazily & never crash
        if _SKILL_LAM > 0.0:
            try:
                from search.skill_prior import skill_root_prior
                root_prior_fn = partial(skill_root_prior, lam=_SKILL_LAM,
                                        leaf_value=(_GUIDE.value if _GUIDE is not None else None))
            except Exception:
                root_prior_fn = None
        pick, _stats = mcts.search(obs, worlds, _CI, _AI, sims=MAX_SIMS, deadline=deadline,
                                   prior_fn=prior_fn, value_fn=value_fn, root_prior_fn=root_prior_fn)
        return pick if pick is not None else _legal_random(sel)
    except Exception:
        return _legal_random(sel)

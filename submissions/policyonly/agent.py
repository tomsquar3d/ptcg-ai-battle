"""policy-only ship candidate: v6's net POLICY prior drives MCTS, but the leaf value is the
prize-difference heuristic — the net value head is dropped (docs §3d; experiments #69/#70/#71/#72).

Pipeline: perception → governor → determinize K worlds → PUCT MCTS → most-visited action. Priors come
from the deck-conditioned `net_ladder2` policy head served via the FUSED graph `ptcgnet.onnx` (#69 merged,
MAX_OPTIONS=56 per #71). The leaf value is NOT the net value head — #72 showed that head is no better than
counting prizes on held-out data (AUC 0.66 vs 0.71) and is dead weight in the mirror ablation — so the
value is `_prize_value`, a prize-difference heuristic. The policy prior is the component #72 found is
actually carrying the agent.

Clock: routine turns run at v6's original budget — ROUTINE_S=0.4s, hard 563-sim cap. #74 showed search
amplifies its leaf (prize-diff), so more routine sims win (+26pp @563 vs 64); the earlier 0.1s/64 was dropped
because its only support (#64/#70) used the broken net value leaf. Held at v6's ladder-validated 563 (not
deeper: v11's 0.4→1.5 won offline yet lost ladder). Critical turns keep v6's governor escalation (5s/K=8).
Net effect: a clean one-variable change vs v6 — the leaf evaluator only.

**Graceful degradation** (unchanged): if onnxruntime / ptcgnet.onnx is missing, `_GUIDE` stays None and
search falls back to plain heuristic MCTS-H — a missing runtime never becomes a loss. All wrapped so any
error / clock pressure yields a legal move; main.py adds the outer watchdog.
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
_GUIDE = None
_GUIDE_TRIED = False
_MODEL_DIR = None
MAX_SIMS = 200_000   # cap; the governor deadline is the real limit (kept for critical turns)
ROUTINE_SIMS = 563   # #74: search AMPLIFIES the (prize-diff) leaf -> more routine sims win (+26pp @563 vs 64).
                     # 0.1s/64 dropped: its only support was #64/#70, offline gates run with the BROKEN net
                     # value leaf that #74 refuted. Held at v6's ladder-validated ~563; NOT deeper (v11's
                     # 0.4->1.5 won +19.6pp offline @z=4.33 yet lost 84.6 ladder — offline over-shoots anti-predict).
governor.ROUTINE_S = 0.4   # v6's original routine budget -> one-variable change vs v6 (leaf evaluator only)


def _model_dir():
    """Locate ptcgnet.onnx (merged graph): alongside this file, else the Kaggle agent dir."""
    global _MODEL_DIR
    if _MODEL_DIR is None:
        try:
            _MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            _MODEL_DIR = "."
        if not os.path.exists(os.path.join(_MODEL_DIR, "ptcgnet.onnx")):
            alt = "/kaggle_simulations/agent"
            if os.path.exists(os.path.join(alt, "ptcgnet.onnx")):
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


def _prize_value(state, root_player):
    """POLICY-ONLY leaf evaluator: prize-difference heuristic in (-1, 1), ROOT-player perspective.
    Replaces the net value head (which #72 showed is no better than counting prizes on held-out data)."""
    obs = state.observation
    gs = build_game_state(obs, _CI)
    v = (gs.opponent.prize_count - gs.me.prize_count) / 6.0   # acting-player perspective
    v = 1.0 if v > 1.0 else (-1.0 if v < -1.0 else v)
    cur = getattr(obs, "current", None)
    if cur is not None and getattr(cur, "yourIndex", root_player) != root_player:
        v = -v
    return v


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
        # POLICY-ONLY (#72): net policy prior drives MCTS; leaf value = prize-difference heuristic,
        # NOT the net value head (which #72 found is ~prize-level and dead weight). Degrade to plain
        # heuristic MCTS-H if the net/runtime is unavailable.
        if _GUIDE is not None:
            prior_fn = _GUIDE.priors
            value_fn = _prize_value
        else:
            prior_fn = mcts.uniform_prior
            value_fn = None
        sims = ROUTINE_SIMS if budget.k == 1 else MAX_SIMS   # routine: hard 563-sim cap (#74)
        pick, _stats = mcts.search(obs, worlds, _CI, _AI, sims=sims, deadline=deadline,
                                   prior_fn=prior_fn, value_fn=value_fn)
        return pick if pick is not None else _legal_random(sel)
    except Exception:
        return _legal_random(sel)

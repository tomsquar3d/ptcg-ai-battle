"""Determinized PUCT MCTS over the native Search API (docs §3 arbitration, §3b/3c).

One tree per determinized world (from `determinizer`), statistics aggregated across the K worlds
at the root → determinized / information-set MCTS: the hidden state is integrated out over the K
samples. Leaf evaluation is the existing heuristic evaluator (the value-network slot); action
priors are injectable (uniform now, a policy network later).

Design notes tied to this engine:
  - **Perspective.** Values are stored from the ROOT player's view (evaluator score, squashed to
    (-1,1) with tanh so PUCT's exploration term is on the same scale — raw eval would dwarf it).
    At our-turn nodes we maximise Q; at opponent-turn nodes we maximise −Q (minimax under UCT).
  - **Forced / multi-select decisions.** The tree only branches on genuine single-select choices;
    forced single-option steps and multi-select "set" decisions (mulligan/prize/discard) are
    auto-resolved by a one-step heuristic to the next real branch. (Full chained-single-pick MCTS
    over multi-select sets is a documented refinement — this keeps the branching factor sane.)
  - **Chance.** Within a determinized world a `search_step` is effectively deterministic bar coin
    flips; resampling stochastic transitions is a refinement (manual_coin / re-step). Not needed
    to benchmark throughput.
  - **Memory.** `search_end()` frees every world's tree between moves.

`search()` expects a *single-select* root (the common MAIN decision); the agent routes multi-select
root decisions to the heuristic. Pure-Python control; all game logic is the native engine.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any, Callable, Mapping, Optional, Sequence

from cg.api import search_begin, search_step, search_end
from ptcg.perception import build_game_state
from ptcg.evaluator import evaluate, WIN_SCORE

_PASS = -1                 # pseudo-action: "take no more actions" (search_step([]))
VALUE_SCALE = 3000.0       # eval units per tanh-unit; ~3 prizes of lead → tanh≈0.76
C_PUCT = 1.5
_MAX_FORCED = 64           # guard on the forced/multi auto-resolution loop


def _value(state: Any, root_player: int, ci: Mapping, ai: Mapping) -> float:
    """Leaf value in (-1, 1) from the ROOT player's perspective."""
    obs = state.observation
    cur = getattr(obs, "current", None)
    if cur is None:
        return 0.0
    if getattr(cur, "result", -1) != -1:  # terminal
        if cur.result == root_player:
            return 1.0
        return -1.0 if cur.result in (0, 1) else 0.0
    gs = build_game_state(obs, ci)
    s = evaluate(gs, card_index=ci, attack_index=ai).score
    if getattr(cur, "yourIndex", root_player) != root_player:
        s = -s
    return math.tanh(s / VALUE_SCALE)


leaf_value = _value  # public alias: root-perspective leaf value in (-1, 1), for heuristic picks


def _heuristic_pick(state: Any, ci: Mapping, ai: Mapping) -> list[int]:
    """A legal pick for a forced / multi-select decision. Single forced → [0]; multi-select →
    greedy top-maxCount by the acting player's resulting-state score (their choice benefits them)."""
    sel = state.observation.select
    n = len(sel.option)
    if n == 0:
        return []
    if sel.maxCount == 1:
        return [0] if sel.minCount >= 1 else []
    actor = state.observation.current.yourIndex
    scored = []
    for i in range(n):
        try:
            c = search_step(state.searchId, [i])
            scored.append((_value(c, actor, ci, ai), i))
        except Exception:
            continue
    if not scored:
        k = max(sel.minCount, min(sel.maxCount, n))
        return list(range(k))
    scored.sort(reverse=True)
    pick = sorted({i for _s, i in scored[:sel.maxCount]})
    if len(pick) < sel.minCount:
        pick = sorted(range(n))[:max(sel.minCount, 1)]
    return pick[:sel.maxCount]


def _is_branch(state: Any) -> bool:
    """A genuine single-select decision with >1 option (or a pass) — where MCTS should branch."""
    obs = state.observation
    cur = getattr(obs, "current", None)
    sel = getattr(obs, "select", None)
    if cur is None or getattr(cur, "result", -1) != -1 or sel is None:
        return False
    return sel.maxCount == 1 and (len(sel.option) > 1 or sel.minCount == 0)


def _advance_forced(state: Any, ci: Mapping, ai: Mapping) -> Any:
    """Auto-resolve forced single-option steps and multi-select sets until the next branch point
    (or terminal), so every tree node is a real single-select decision."""
    for _ in range(_MAX_FORCED):
        obs = state.observation
        cur = getattr(obs, "current", None)
        sel = getattr(obs, "select", None)
        if cur is None or getattr(cur, "result", -1) != -1 or sel is None or _is_branch(state):
            return state
        try:
            state = search_step(state.searchId, _heuristic_pick(state, ci, ai))
        except Exception:
            return state
    return state


class _Node:
    def __init__(self, state: Any, root_player: int, ci: Mapping, ai: Mapping):
        self.state = state
        obs = state.observation
        cur = getattr(obs, "current", None)
        sel = getattr(obs, "select", None)
        self.terminal = cur is None or getattr(cur, "result", -1) != -1 or sel is None
        self.value = _value(state, root_player, ci, ai)
        if self.terminal:
            self.actions: list[int] = []
            self.is_our_turn = True
        else:
            self.is_our_turn = (cur.yourIndex == root_player)
            self.actions = list(range(len(sel.option)))
            if sel.minCount == 0 and sel.maxCount == 1:
                self.actions.append(_PASS)
            self.N = {a: 0 for a in self.actions}
            self.W = {a: 0.0 for a in self.actions}
            self.children: dict[int, "_Node"] = {}
            self.P: Optional[dict[int, float]] = None


def uniform_prior(node: "_Node") -> dict[int, float]:
    p = 1.0 / len(node.actions) if node.actions else 0.0
    return {a: p for a in node.actions}


def _select(node: "_Node", c_puct: float) -> int:
    sign = 1.0 if node.is_our_turn else -1.0
    total = sum(node.N.values())
    sq = math.sqrt(total + 1)
    best_a, best_s = node.actions[0], -1e30
    for a in node.actions:
        na = node.N[a]
        q = (node.W[a] / na) if na > 0 else 0.0
        u = c_puct * node.P[a] * sq / (1 + na)
        s = sign * q + u
        if s > best_s:
            best_s, best_a = s, a
    return best_a


def _step_child(node: "_Node", a: int, root_player: int, ci: Mapping, ai: Mapping) -> "_Node":
    pick = [] if a == _PASS else [a]
    child_state = _advance_forced(search_step(node.state.searchId, pick), ci, ai)
    return _Node(child_state, root_player, ci, ai)


def _simulate(root: "_Node", root_player: int, c_puct: float,
              prior_fn: Callable, ci: Mapping, ai: Mapping) -> None:
    node = root
    path: list[tuple[_Node, int]] = []
    while True:
        if node.terminal:
            value = node.value
            break
        if node.P is None:
            node.P = prior_fn(node)
        a = _select(node, c_puct)
        path.append((node, a))
        if a in node.children:
            node = node.children[a]
            continue
        try:
            child = _step_child(node, a, root_player, ci, ai)
        except Exception:
            value = node.value  # illegal/edge: fall back to this node's estimate
            break
        node.children[a] = child
        value = child.value
        break
    for n, a in path:
        n.N[a] += 1
        n.W[a] += value


def search(obs: Any, worlds: Sequence[tuple], card_index: Mapping, attack_index: Mapping, *,
           sims: int = 800, c_puct: float = C_PUCT, prior_fn: Callable = uniform_prior,
           deadline: Optional[float] = None) -> tuple[Optional[list[int]], dict]:
    """Run determinized MCTS over the K `worlds` and return (pick, stats). `pick` is the engine
    select list for the root decision (a single option index, or [] for pass). Aggregates root
    visit counts across worlds and returns the most-visited (robust) action."""
    root_player = obs.current.yourIndex
    roots = [_Node(search_begin(obs, *w), root_player, card_index, attack_index) for w in worlds]
    try:
        done = 0
        for i in range(sims):
            if deadline is not None and time.monotonic() > deadline:
                break
            root = roots[i % len(roots)]
            if root.terminal or not root.actions:
                continue
            _simulate(root, root_player, c_puct, prior_fn, card_index, attack_index)
            done += 1

        agg_N: Counter = Counter()
        agg_W: Counter = Counter()
        for root in roots:
            if root.terminal:
                continue
            for a in root.actions:
                agg_N[a] += root.N[a]
                agg_W[a] += root.W[a]
        if not agg_N:
            return None, {"sims": done}
        best = max(agg_N, key=lambda a: (agg_N[a], agg_W[a] / max(agg_N[a], 1)))
        pick = [] if best == _PASS else [best]
        return pick, {"sims": done, "root_visits": dict(agg_N),
                      "root_q": {a: round(agg_W[a] / max(agg_N[a], 1), 3) for a in agg_N}}
    finally:
        search_end()

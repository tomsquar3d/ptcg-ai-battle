"""Self-play recorder — generate AlphaZero training tuples from MCTS-H (v4) games (docs §3).

Plays full self-play games (both seats driven by the same MCTS-H pipeline) over the native engine
and, at every genuine single-select decision, emits a training tuple:

  inputs   : ``encode_state(gs)`` + ``encode_actions(select)`` — the network's featurized state/action.
  policy π : the MCTS *root visit distribution* over the legal options, plus a trailing PASS slot
             (``_PASS`` = "take no more actions"). Width ``MAX_OPTIONS + 1``; stored as raw counts so
             the trainer chooses how to normalize / weight and how to treat PASS.
  value  z : the eventual game outcome from the *deciding player's* perspective (+1 win / −1 loss /
             0 draw), filled in once the game finishes.

Multi-select decisions (mulligan / prize / discard) and truly-forced single-option steps carry no
MCTS policy target, so they are resolved by a light heuristic and **not** recorded — only the value
signal flows through them via the eventual outcome.

Why this can drive the live battle *and* search between moves: the native Search API runs on a
separate ``agent_ptr`` (``AgentStart``), fully isolated from the live battle's ``battle_ptr``. A real
game already interleaves them (engine drives the battle, agent searches inside each decision); the
recorder just drives both from the outside.

torch-free — emits numpy ``.npz`` shards the trainer consumes. One game per process (the SDK holds a
single global battle); parallelize by launching multiple processes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from cg.api import to_observation_class, search_begin, search_step, search_end
from cg.game import battle_start, battle_select, battle_finish
from ptcg.perception import build_game_state
from ptcg.opponent import OpponentTracker
from search.determinizer import determinize
from search import mcts
from nn import features as F

_PASS_SLOT = F.MAX_OPTIONS          # trailing policy slot for the MCTS PASS action (mcts._PASS = -1)
PI_DIM = F.MAX_OPTIONS + 1

# per-decision policy-target provenance (docs §3d): a soft MCTS visit distribution vs a one-hot
# behavioral-cloning target from a scripted expert. The trainer weights / splits on this.
SOURCE_MCTS = 0
SOURCE_BC = 1


# --- per-game buffer -----------------------------------------------------------------

@dataclass
class GameStats:
    decisions: int          # single-select branch decisions recorded
    turns: int              # engine decisions stepped (recorded + skipped)
    result: int             # winning player index (0/1), or -1 if draw/unknown
    seconds: float


@dataclass
class _Buffer:
    """Accumulate recorded decisions during one game; ``finalize`` stacks them + assigns z."""
    st: list = field(default_factory=list)     # StateFeatures
    ac: list = field(default_factory=list)     # ActionFeatures
    visits: list = field(default_factory=list)  # np.int32 [PI_DIM]
    mover: list = field(default_factory=list)   # int

    source: list = field(default_factory=list)  # SOURCE_MCTS (soft π) | SOURCE_BC (one-hot expert)

    def add(self, sf, af, root_visits: Mapping[int, int], mover: int, source: int = 0) -> None:
        v = np.zeros(PI_DIM, dtype=np.int32)
        for a, c in root_visits.items():
            if a == mcts._PASS:
                v[_PASS_SLOT] = c
            elif 0 <= a < F.MAX_OPTIONS:
                v[a] = c
        self.st.append(sf)
        self.ac.append(af)
        self.visits.append(v)
        self.mover.append(mover)
        self.source.append(source)

    def __len__(self) -> int:
        return len(self.st)

    def finalize(self, result: int) -> Optional[dict]:
        if not self.st:
            return None
        mover = np.asarray(self.mover, dtype=np.int8)
        if result in (0, 1):
            z = np.where(mover == result, 1.0, -1.0).astype(np.float32)
        else:
            z = np.zeros(len(mover), dtype=np.float32)   # draw / unknown → neutral
        return {
            "pokemon_ids": np.stack([s.pokemon_ids for s in self.st]),
            "pokemon_scalars": np.stack([s.pokemon_scalars for s in self.st]),
            "pokemon_mask": np.stack([s.pokemon_mask for s in self.st]),
            "hand_ids": np.stack([s.hand_ids for s in self.st]),
            "hand_mask": np.stack([s.hand_mask for s in self.st]),
            "stadium_id": np.asarray([int(s.stadium_id) for s in self.st], dtype=np.int32),
            "globals": np.stack([s.globals for s in self.st]),
            # schema v2 deck-conditioning (docs §3d) — pulled off the StateFeatures directly
            "deck_summary": np.stack([s.deck_summary for s in self.st]),
            "archetype_id": np.asarray([int(s.archetype_id) for s in self.st], dtype=np.int32),
            "option_card_ids": np.stack([a.option_card_ids for a in self.ac]),
            "option_scalars": np.stack([a.option_scalars for a in self.ac]),
            "option_mask": np.stack([a.option_mask for a in self.ac]),
            "n_options": np.asarray([a.n_options for a in self.ac], dtype=np.int32),
            "context": np.asarray([a.context for a in self.ac], dtype=np.int32),
            "visits": np.stack(self.visits),
            "mover": mover,
            "z": z,
            "source": np.asarray(self.source, dtype=np.int8),
        }


# --- decision helpers ----------------------------------------------------------------

def _legal_min(sel: Any) -> list[int]:
    """A always-legal fallback pick (lowest indices satisfying minCount)."""
    n = len(sel.option)
    k = min(max(sel.minCount, 0), n)
    return list(range(k))


def _heuristic_multi(obs: Any, my_index: int, deck: Sequence[int], ci: Mapping, ai: Mapping,
                     archetypes: Any, tracker: Any) -> list[int]:
    """Resolve a multi-select decision by one-world leaf scoring (mirrors v4 `_multi_pick`)."""
    sel = obs.select
    n = len(sel.option)
    if n == 0:
        return []
    try:
        worlds = determinize(obs, deck, ci, k=1, archetypes=archetypes, opp_tracker=tracker)
        root = search_begin(obs, *worlds[0])
    except Exception:
        return _legal_min(sel)
    try:
        scored = []
        for i in range(n):
            try:
                child = search_step(root.searchId, [i])
                scored.append((mcts.leaf_value(child, my_index, ci, ai), i))
            except Exception:
                continue
        if not scored:
            return _legal_min(sel)
        scored.sort(reverse=True)
        pick = sorted({i for _s, i in scored[:sel.maxCount]})
        if len(pick) < sel.minCount:
            pick = sorted(range(n))[:sel.minCount]
        return pick[:sel.maxCount]
    finally:
        search_end()


def _is_legal(pick: Any, sel: Any) -> bool:
    if not isinstance(pick, list):
        return False
    n = len(sel.option)
    return (sel.minCount <= len(pick) <= sel.maxCount
            and len(set(pick)) == len(pick)
            and all(isinstance(i, int) and 0 <= i < n for i in pick))


# --- pilots (docs §3d step 3: an opponent/teacher pool over the same game loop) --------

class Pilot:
    """A seat driver. `act` returns (pick, record_visits|None, source): record_visits is a policy
    target ONLY for a recordable single-select 'branch' decision (else None)."""
    name = "base"

    def set_deck(self, deck) -> None:  # experts need it; MCTS pilots read the deck via determinize
        pass

    def act(self, obs, obs_dict, sel, my_index, deck, ci, ai, archetypes, tracker,
            *, kind, k, sims, move_time_cap, seed):
        raise NotImplementedError


class MctsPilot(Pilot):
    """Determinized PUCT MCTS. Heuristic leaf by default; pass a net's `prior_fn`/`value_fn` for the
    net pilot. Branch decisions emit the root visit distribution (soft AlphaZero π target)."""

    def __init__(self, name: str = "heuristic", prior_fn=None, value_fn=None):
        self.name = name
        self.prior_fn = prior_fn
        self.value_fn = value_fn

    def act(self, obs, obs_dict, sel, my_index, deck, ci, ai, archetypes, tracker,
            *, kind, k, sims, move_time_cap, seed):
        if kind == "multi":
            return _heuristic_multi(obs, my_index, deck, ci, ai, archetypes, tracker), None, SOURCE_MCTS
        if kind == "forced":
            return [0], None, SOURCE_MCTS
        try:
            worlds = determinize(obs, deck, ci, k=k, archetypes=archetypes, opp_tracker=tracker, seed=seed)
            deadline = time.monotonic() + move_time_cap
            kw = {}
            if self.prior_fn is not None:
                kw["prior_fn"] = self.prior_fn
            if self.value_fn is not None:
                kw["value_fn"] = self.value_fn
            pick, stats = mcts.search(obs, worlds, ci, ai, sims=sims, deadline=deadline, **kw)
            rv = stats.get("root_visits")
            if pick is not None and rv:
                return pick, rv, SOURCE_MCTS
            return (pick if pick is not None else _legal_min(sel)), None, SOURCE_MCTS
        except Exception:
            return (_legal_min(sel) or [0]), None, SOURCE_MCTS


class ExpertPilot(Pilot):
    """A scripted line-expert (src/experts). Branch decisions emit a one-hot BC target on the chosen
    option (source=BC). Multi/forced steps are played by the expert but not recorded."""

    def __init__(self, expert):
        self.expert = expert
        self.name = "expert:" + getattr(expert, "name", "?")

    def set_deck(self, deck) -> None:
        self.expert.set_deck(deck)

    def act(self, obs, obs_dict, sel, my_index, deck, ci, ai, archetypes, tracker,
            *, kind, k, sims, move_time_cap, seed):
        try:
            pick = self.expert.choose(obs_dict, time_left=None)
        except Exception:
            pick = None
        if not _is_legal(pick, sel):
            pick = _legal_min(sel)
        rv = {pick[0]: 1} if (kind == "branch" and len(pick) == 1) else None
        return pick, rv, SOURCE_BC


def build_pilot(spec: str, card_index: Mapping, attack_index: Mapping, *, onnx_dir: Optional[str] = None) -> Pilot:
    """Construct a pilot from a spec: 'heuristic'|'v4' · 'net'|'v5' (needs `onnx_dir`) ·
    'expert:<archetype>'."""
    s = spec.strip().lower()
    if s in ("heuristic", "v4", "mcts"):
        return MctsPilot("heuristic")
    if s in ("net", "v5"):
        if not onnx_dir:
            raise ValueError("net pilot needs onnx_dir (value.onnx/policy.onnx)")
        from nn.onnx_guide import OnnxGuide
        g = OnnxGuide(onnx_dir, card_index, attack_index)
        return MctsPilot("net", prior_fn=g.priors, value_fn=g.value)
    if s.startswith("expert:"):
        from experts import get_expert
        return ExpertPilot(get_expert(s.split(":", 1)[1]))
    raise ValueError(f"unknown pilot spec {spec!r}")


def record_mixed_game(pilots: Mapping[int, Pilot], decks: Mapping[int, Sequence[int]],
                      names: Mapping[int, str], card_index: Mapping, attack_index: Mapping, *,
                      archetypes: Any = None, k: int = 2, sims: int = 160,
                      move_time_cap: float = 8.0, max_decisions: int = 4000,
                      seed: Optional[int] = None) -> tuple[Optional[dict], GameStats]:
    """Play one game with a (possibly different) deck AND pilot per seat, recording deck-conditioned
    branch decisions (docs §3d step 3). `pilots`/`decks`/`names` are keyed by seat 0/1; `names` gives
    each seat's archetype name for `archetype_id`. z is still labelled from each mover's perspective."""
    decks = {0: list(decks[0]), 1: list(decks[1])}
    deckcond = {s: (F.summarize_deck(decks[s], card_index), F.archetype_id_for(names.get(s)))
                for s in (0, 1)}
    for s in (0, 1):
        pilots[s].set_deck(decks[s])
    trackers = {0: OpponentTracker(), 1: OpponentTracker()}
    buf = _Buffer()
    result = -1
    turns = 0
    t0 = time.monotonic()

    obs_dict, _ = battle_start(decks[0], list(decks[1]))
    if obs_dict is None:
        return None, GameStats(0, 0, -1, time.monotonic() - t0)
    try:
        for _ in range(max_decisions):
            obs = to_observation_class(obs_dict)
            cur = getattr(obs, "current", None)
            if cur is not None and getattr(cur, "result", -1) != -1:
                result = cur.result
                break
            sel = obs.select
            if sel is None:
                break
            turns += 1
            mi = cur.yourIndex if cur is not None else 0
            n = len(sel.option)
            if n == 0:
                obs_dict = battle_select([])
                continue
            try:
                trackers[mi].observe(obs)
            except Exception:
                pass

            kind = "multi" if sel.maxCount > 1 else ("forced" if (n == 1 and sel.minCount == 1) else "branch")
            pick, rv, source = pilots[mi].act(
                obs, obs_dict, sel, mi, decks[mi], card_index, attack_index, archetypes, trackers[mi],
                kind=kind, k=k, sims=sims, move_time_cap=move_time_cap, seed=seed)
            if kind == "branch" and rv:
                ds, aid = deckcond[mi]
                try:
                    sf = F.encode_state(build_game_state(obs, card_index), card_index, attack_index,
                                        deck_summary=ds, archetype_id=aid)
                    af = F.encode_actions(sel, mi, attack_index)
                    buf.add(sf, af, rv, mi, source)
                except Exception:
                    pass
            if pick is None:
                pick = _legal_min(sel) or [0]
            obs_dict = battle_select(pick)
    finally:
        battle_finish()

    stats = GameStats(len(buf), turns, result, time.monotonic() - t0)
    return buf.finalize(result), stats


def record_game(deck: Sequence[int], card_index: Mapping, attack_index: Mapping, *,
                archetypes: Any = None, k: int = 2, sims: int = 160,
                move_time_cap: float = 8.0, max_decisions: int = 4000,
                seed: Optional[int] = None) -> tuple[Optional[dict], GameStats]:
    """Play one MCTS-H self-play game (both seats) and return (training-tuple dict, GameStats).

    `k` determinized worlds and `sims` MCTS simulations per recorded decision; `move_time_cap`
    bounds a single search (safety only — throughput comes from `sims`). Returns (None, stats) if
    no branch decision occurred (degenerate game).
    """
    deck = list(deck)
    trackers = {0: OpponentTracker(), 1: OpponentTracker()}
    buf = _Buffer()
    result = -1
    turns = 0
    t0 = time.monotonic()

    obs_dict, start = battle_start(deck, list(deck))
    if obs_dict is None:
        return None, GameStats(0, 0, -1, time.monotonic() - t0)
    try:
        for _ in range(max_decisions):
            obs = to_observation_class(obs_dict)
            cur = getattr(obs, "current", None)
            if cur is not None and getattr(cur, "result", -1) != -1:
                result = cur.result
                break
            sel = obs.select
            if sel is None:
                break
            turns += 1
            my_index = cur.yourIndex if cur is not None else 0
            n = len(sel.option)
            if n == 0:
                obs_dict = battle_select([])
                continue

            try:
                trackers[my_index].observe(obs)
            except Exception:
                pass

            if sel.maxCount > 1:                                   # multi-select → heuristic, no record
                pick = _heuristic_multi(obs, my_index, deck, card_index, attack_index,
                                        archetypes, trackers[my_index])
            elif n == 1 and sel.minCount == 1:                     # truly forced → no choice, no record
                pick = [0]
            else:                                                   # single-select branch → MCTS + record
                pick = None
                try:
                    worlds = determinize(obs, deck, card_index, k=k, archetypes=archetypes,
                                         opp_tracker=trackers[my_index], seed=seed)
                    deadline = time.monotonic() + move_time_cap
                    pick, stats = mcts.search(obs, worlds, card_index, attack_index,
                                              sims=sims, deadline=deadline)
                    rv = stats.get("root_visits")
                    if pick is not None and rv:
                        sf = F.encode_state(build_game_state(obs, card_index), card_index, attack_index)
                        af = F.encode_actions(sel, my_index, attack_index)
                        buf.add(sf, af, rv, my_index)
                except Exception:
                    pick = None
                if pick is None:
                    pick = _legal_min(sel) or [0]

            obs_dict = battle_select(pick)
    finally:
        battle_finish()

    stats = GameStats(len(buf), turns, result, time.monotonic() - t0)
    return buf.finalize(result), stats


def save_shard(path: str, tuples: dict) -> None:
    """Write one game's stacked training tuples to a compressed .npz shard."""
    np.savez_compressed(path, **tuples)

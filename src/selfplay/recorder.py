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

    def add(self, sf, af, root_visits: Mapping[int, int], mover: int) -> None:
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
            "option_card_ids": np.stack([a.option_card_ids for a in self.ac]),
            "option_scalars": np.stack([a.option_scalars for a in self.ac]),
            "option_mask": np.stack([a.option_mask for a in self.ac]),
            "n_options": np.asarray([a.n_options for a in self.ac], dtype=np.int32),
            "context": np.asarray([a.context for a in self.ac], dtype=np.int32),
            "visits": np.stack(self.visits),
            "mover": mover,
            "z": z,
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

"""Turn Kaggle ladder replays into behavioral-cloning shards (docs §3d follow-up; #34/#40).

A replay (`scripts/harvest_ladder.py`) is a full game between two ladder agents. We clone the WINNER:
for each of the winner's groundable single-select decisions we emit one training tuple with a one-hot
policy target on the move they chose (`SOURCE_BC`), z = +1 (they won), the deck-conditioned state
features (the winner's own decklist — recovered from their 60-id deck-selection action per #34 — as the
`deck_summary`), and the same aux targets as self-play. Output shards are the exact `recorder` format,
so `nn.dataset` / `train_net` consume them unchanged.

Grounding rule: only the winner's ACTIVE, single-select (minCount==maxCount==1) decisions whose chosen
index lands inside the shown option list. The deck-return step (action length 60) and value/count/multi
selects are skipped (their action isn't a plain option index) — see #34.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional

import numpy as np

from cg.api import to_observation_class
from ptcg.perception import build_game_state
from nn import features as F
from selfplay.recorder import _Buffer, _aux_targets, SOURCE_BC, _OT_ATTACK

DECK_LEN = 60


def _winner_seat(rewards) -> Optional[int]:
    # ladder replays can carry null rewards (unfinished / errored games) — treat as undecided
    if (not rewards or len(rewards) != 2 or rewards[0] is None or rewards[1] is None
            or rewards[0] == rewards[1]):
        return None
    return 0 if rewards[0] > rewards[1] else 1


def _recover_deck(steps, seat: int) -> Optional[list[int]]:
    """The 60-card deck-selection action is the player's decklist (card ids) — our deck conditioning."""
    for st in steps[:6]:
        a = st[seat].get("action")
        if isinstance(a, list) and len(a) == DECK_LEN and all(isinstance(x, int) for x in a):
            return list(a)
    return None


def replay_to_tuples(path: str, card_index: Mapping, attack_index: Mapping,
                     winner_rating: float = 0.0) -> Optional[dict]:
    """Parse one replay → a BC shard (winner's decisions), or None if unusable. `winner_rating` is
    stored per row so training can weight/filter by teacher strength."""
    try:
        d = json.load(open(path))
    except Exception:
        return None
    steps = d.get("steps")
    if not steps:
        return None
    winner = _winner_seat(d.get("rewards"))
    if winner is None:
        return None
    wdeck = _recover_deck(steps, winner)
    ds = F.summarize_deck(wdeck, card_index) if wdeck else None

    buf = _Buffer()
    timeline: list = []
    first_attack = {0: None, 1: None}
    for st in steps:
        # prize timeline from whichever agent is ACTIVE this step (both prize counts are visible)
        for seat in (0, 1):
            obs_d = st[seat].get("observation") or {}
            cur = obs_d.get("current")
            if cur and cur.get("players"):
                turn = int(cur.get("turn") or 0)
                try:
                    rem0 = len(cur["players"][0].get("prize") or [])
                    rem1 = len(cur["players"][1].get("prize") or [])
                    if not timeline or timeline[-1][0] != turn:
                        timeline.append((turn, rem0, rem1))
                    else:
                        timeline[-1] = (turn, rem0, rem1)
                except Exception:
                    pass
                break

        ae = st[winner]
        if ae.get("status") != "ACTIVE":
            continue
        obs_d = ae.get("observation") or {}
        sel = obs_d.get("select")
        act = ae.get("action")
        if not sel or act is None:
            continue
        opts = sel.get("option") or []
        n = len(opts)
        turn = int((obs_d.get("current") or {}).get("turn") or 0)
        # first attack (any groundable single index selecting an ATTACK option)
        if (isinstance(act, list) and len(act) == 1 and 0 <= act[0] < n
                and first_attack[winner] is None
                and int((opts[act[0]].get("type") if isinstance(opts[act[0]], dict) else -1) or -1) == _OT_ATTACK):
            first_attack[winner] = turn
        # groundable single-select decision → a BC row
        if not (sel.get("minCount") == 1 and sel.get("maxCount") == 1
                and isinstance(act, list) and len(act) == 1 and 0 <= act[0] < n):
            continue
        try:
            obs = to_observation_class(obs_d)
            gs = build_game_state(obs, card_index)
            sf = F.encode_state(gs, card_index, attack_index, deck_summary=ds, archetype_id=0)
            af = F.encode_actions(obs.select, gs.my_index, attack_index)
            buf.add(sf, af, {int(act[0]): 1}, winner, SOURCE_BC, turn=turn)
        except Exception:
            continue

    if len(buf) == 0:
        return None
    tup = buf.finalize(winner)                       # z = +1 for every winner row
    if tup is None:
        return None
    tup.update(_aux_targets(tup["turn"], tup["mover"], timeline, first_attack))
    tup["rating"] = np.full(len(buf), float(winner_rating), dtype=np.float32)
    tup["deck_key"] = np.full(len(buf), _deck_key(wdeck), dtype=np.int64)   # for per-deck row counts
    return tup


def _deck_key(deck: Optional[list[int]]) -> int:
    """A stable small hash of the sorted decklist — groups rows by deck for reporting (not a feature)."""
    if not deck:
        return 0
    return abs(hash(tuple(sorted(deck)))) % (10 ** 9)

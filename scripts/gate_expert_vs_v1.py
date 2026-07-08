#!/usr/bin/env python3
"""Validate a scripted line-expert (docs §3d step 2): it must BEAT v1 piloting its own archetype.

Expert (seat A) vs v1 (seat B), BOTH piloting the same archetype deck (a mirror — isolates pilot
skill, not deck strength). Seats alternate each game to cancel first-player bias. If the expert
can't out-play v1 on the deck it was written for, its teacher signal is worthless (§3c).

    .venv/bin/python scripts/gate_expert_vs_v1.py --archetype dragapult-dusknoir --benchmark
    .venv/bin/python scripts/gate_expert_vs_v1.py --archetype dragapult-dusknoir --games 20 \
        --seed 0 --out logs/expert_0.json         # fan out several --seed for N
"""
import argparse
import importlib
import json
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))   # cg + v1 agent
sys.path.insert(0, os.path.join(REPO, "src"))                 # experts + ptcg

from cg.api import to_observation_class                        # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from experts import get_expert                                 # noqa: E402

_LOG_RESULT = 23
TIME_LEFT = 200.0        # generous per-decision clock (both pilots search fully; not the A/B axis)


def _read_deck(path):
    return [int(x) for x in open(path).read().split() if x.strip()]


def _winner(obs):
    if obs.current is not None and obs.current.result in (0, 1, 2):
        return obs.current.result
    for lg in obs.logs or []:
        if lg.type == _LOG_RESULT and lg.result is not None:
            return lg.result
    return None


def _legal_min(sel):
    n = len(sel.option)
    return list(range(min(max(sel.minCount, 0), n)))


def play_game(deck, expert, v1, a_seat):
    """One mirror game; seat a_seat = expert, the other = v1. Returns winning SEAT (0/1), 2, or None."""
    expert.set_deck(deck)
    v1.set_deck(deck)
    obs_dict, _ = battle_start(list(deck), list(deck))
    winner = None
    try:
        for _ in range(4000):
            obs = to_observation_class(obs_dict)
            if obs.select is None or (obs.current is not None and obs.current.result != -1):
                winner = _winner(obs)
                break
            mi = obs.current.yourIndex if obs.current else 0
            brain = expert if mi == a_seat else v1
            try:
                pick = brain.choose(obs_dict, time_left=TIME_LEFT)
            except Exception:
                pick = _legal_min(obs.select)
            obs_dict = battle_select(pick)
    finally:
        battle_finish()
    return winner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archetype", default="dragapult-dusknoir")
    ap.add_argument("--deck", default=None, help="deck csv (default: experiments/candidate_decks/<archetype>.csv)")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    deck_path = args.deck or os.path.join(REPO, "experiments", "candidate_decks", f"{args.archetype}.csv")
    deck = _read_deck(deck_path)
    expert = get_expert(args.archetype)
    v1 = importlib.import_module("agent")

    games = 2 if args.benchmark else args.games
    e_w = e_l = draw = 0
    t0 = time.monotonic()
    for g in range(games):
        a_seat = (args.seed + g) % 2                 # alternate expert's seat
        w = play_game(deck, expert, v1, a_seat)
        if w == 2 or w is None:
            draw += 1
        elif w == a_seat:
            e_w += 1
        else:
            e_l += 1
        if args.benchmark:
            print(f"game {g}: winner={w} expert_seat={a_seat}  cum {time.monotonic()-t0:.1f}s")
    dt = time.monotonic() - t0

    decided = e_w + e_l
    wr = e_w / decided if decided else 0.0
    se = (wr * (1 - wr) / decided) ** 0.5 if decided else 0.0
    print(f"expert({args.archetype}) vs v1 | mirror | {games} games in {dt:.0f}s ({dt/max(games,1):.1f}s/game)\n"
          f"  expert W/L/D = {e_w}/{e_l}/{draw} | win rate {wr:.1%} ± {se:.1%} (of decided)")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"e_w": e_w, "e_l": e_l, "draw": draw, "games": games,
                       "archetype": args.archetype, "seconds": dt}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())

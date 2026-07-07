#!/usr/bin/env python3
"""A/B two pilots on the SAME deck: head-to-head in the raw SDK, seats alternated.

Isolates pilot skill (both sides play an identical deck), so it answers "does brain A play deck X
better than brain B?" — the question behind every pilot upgrade (v1→v3 search, v3→v4 prize race).

Run:  .venv/bin/python scripts/pilot_ab.py <brainA_dir> <brainB_dir> <deck> [N]
      <deck> = a path to a deck CSV, or 'sample' for submissions/v1/deck.csv.
"""
import importlib.util
import os
import random
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))  # shared cg/ + ptcg/

from cg.api import to_observation_class                       # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402


def _load(brain_dir, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(brain_dir, "agent.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rand_pick(sel):
    n = len(sel["option"])
    return random.sample(range(n), min(sel["maxCount"], n)) if n else []


def _winner(obs):
    if obs.current is not None and obs.current.result in (0, 1, 2):
        return obs.current.result
    for lg in obs.logs or []:
        if lg.type == 23 and lg.result is not None:
            return lg.result
    return None


def main() -> int:
    a_dir = os.path.join(REPO, sys.argv[1])
    b_dir = os.path.join(REPO, sys.argv[2])
    deck_arg = sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    deck_path = os.path.join(REPO, "submissions", "v1", "deck.csv") if deck_arg == "sample" \
        else os.path.join(REPO, deck_arg)
    deck = [int(x) for x in open(deck_path).read().split() if x.strip()]

    a = _load(a_dir, "brain_a")
    b = _load(b_dir, "brain_b")
    a.set_deck(deck)
    b.set_deck(deck)
    a_name, b_name = os.path.basename(a_dir), os.path.basename(b_dir)

    aw = bw = dr = errors = 0
    a_times = []
    for g in range(n):
        a_seat = g % 2
        obs_dict, _ = battle_start(deck, list(deck))
        winner = None
        try:
            for _ in range(4000):
                obs = to_observation_class(obs_dict)
                if obs.select is None or (obs.current is not None and obs.current.result != -1):
                    winner = _winner(obs)
                    break
                brain = a if obs.current.yourIndex == a_seat else b
                t0 = time.perf_counter()
                try:
                    pick = brain.choose(obs_dict, time_left=600)
                except Exception:
                    errors += 1
                    pick = _rand_pick(obs_dict["select"])
                if brain is a:
                    a_times.append(time.perf_counter() - t0)
                obs_dict = battle_select(pick)
        finally:
            battle_finish()
        if winner == a_seat:
            aw += 1
        elif winner in (0, 1):
            bw += 1
        elif winner == 2:
            dr += 1

    tot = aw + bw + dr
    print(f"=== {a_name} vs {b_name}, both piloting {os.path.basename(deck_path)} (N={n}) ===")
    print(f"{a_name} W/L/D : {aw}/{bw}/{dr}  -> {a_name} win rate {aw/max(tot,1):.0%} "
          f"(decisive {aw}/{max(aw+bw,1)} = {aw/max(aw+bw,1):.0%})")
    print(f"{a_name} move time: mean {statistics.mean(a_times)*1e3:.0f} ms, "
          f"max {max(a_times)*1e3:.0f} ms | exceptions {errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

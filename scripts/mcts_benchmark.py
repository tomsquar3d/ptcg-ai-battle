#!/usr/bin/env python3
"""BENCHMARK GATE (task 4): search_step throughput + MCTS simulations/sec at K=1.

Drives a game to a representative mid-game single-select decision, then reports:
  - determinizer validity (search_begin accepts its output → counts/discard handling correct),
  - raw search_step/sec (pure engine transition cost, no evaluation),
  - MCTS simulations/sec at K=1 (includes leaf evaluation — the real budget driver),
so we can size K (4-8) and sims to a ~5 s critical-turn budget before building further.

Run:  .venv/bin/python scripts/mcts_benchmark.py [deck]
"""
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))  # cg
sys.path.insert(0, os.path.join(REPO, "src"))                # search, ptcg (wins for ptcg)

from cg.api import (to_observation_class, all_card_data, all_attack,  # noqa: E402
                    search_begin, search_step, search_end)
from cg.game import battle_start, battle_select, battle_finish       # noqa: E402
from search.determinizer import determinize                          # noqa: E402
from search import mcts                                              # noqa: E402


def _rand_pick(sel):
    n = len(sel.option)
    if n == 0:
        return []
    return random.sample(range(n), min(sel.maxCount, n))


def _drive_to_decision(deck, ci, ai):
    """Play random moves until a mid-game single-select branch (turn≥2, >2 options)."""
    obs_dict, _ = battle_start(deck, list(deck))
    for _ in range(4000):
        obs = to_observation_class(obs_dict)
        cur = obs.current
        if obs.select is None or (cur is not None and cur.result != -1):
            return None
        sel = obs.select
        if (cur is not None and cur.turn >= 2 and sel.maxCount == 1
                and len(sel.option) > 2 and sel.context == 0):
            return obs
        obs_dict = battle_select(_rand_pick(sel))
    return None


def main() -> int:
    deck_arg = sys.argv[1] if len(sys.argv) > 1 else "experiments/candidate_decks/dragapult-ex.csv"
    deck = [int(x) for x in open(os.path.join(REPO, deck_arg)).read().split() if x.strip()]
    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}

    obs = None
    for _ in range(20):
        obs = _drive_to_decision(deck, ci, ai)
        if obs is not None:
            break
        battle_finish()
    if obs is None:
        print("could not reach a suitable decision")
        return 1

    print(f"decision: turn {obs.current.turn}, {len(obs.select.option)} options, "
          f"my deck {obs.current.players[obs.current.yourIndex].deckCount}, "
          f"opp hand {obs.current.players[1-obs.current.yourIndex].handCount}")

    # 1. determinizer validity — search_begin must accept it (counts + discard handling correct)
    t0 = time.perf_counter()
    worlds = determinize(obs, deck, ci, k=8, seed=1)
    t_det = (time.perf_counter() - t0) / 8
    try:
        root = search_begin(obs, *worlds[0])
        search_end()
        ok = True
    except Exception as e:
        ok = False
        print(f"  determinizer INVALID: {e}")
    print(f"1. determinize   : {t_det*1e3:.2f} ms/world, search_begin accepts it: {ok}")
    if not ok:
        battle_finish()
        return 1

    # 2. raw search_step throughput (branch from root, no eval)
    root = search_begin(obs, *worlds[0])
    n = len(obs.select.option)
    steps = 0
    t1 = time.perf_counter()
    while time.perf_counter() - t1 < 2.0:
        for _ in range(50):
            try:
                search_step(root.searchId, [random.randrange(n)])
            except Exception:
                pass
            steps += 1
    step_rate = steps / (time.perf_counter() - t1)
    search_end()
    print(f"2. search_step   : {step_rate:,.0f} steps/sec ({1e6/step_rate:.1f} µs/step)")

    # 3. MCTS sims/sec at K=1 (includes leaf eval)
    for k in (1, 8):
        w = determinize(obs, deck, ci, k=k, seed=2)
        t2 = time.perf_counter()
        pick, stats = mcts.search(obs, w, ci, ai, sims=1000)
        dt = time.perf_counter() - t2
        print(f"3. MCTS K={k:<2}      : {stats['sims']/dt:,.0f} sims/sec "
              f"({stats['sims']} sims in {dt:.2f}s) -> pick {pick}")

    battle_finish()
    print("\n(sizing) at ~X sims/sec: a 5s budget = ~5X sims; pick K and sims/turn accordingly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

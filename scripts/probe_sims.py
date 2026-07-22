#!/usr/bin/env python3
"""Measure ACHIEVED MCTS sims-per-move, split by routine vs critical budget.

Why this matters. Our search budget is defined in WALL-CLOCK (ROUTINE_S=0.4s), not in simulations.
So the actual search DEPTH the agent reaches is a function of how fast the CPU is and how contended
it is. Depth is exactly the variable the v13/v14 question turns on (the PH-governor lever wins when
routine search is shallow, vanishes when it is deep), so the operating point our gates run at is a
first-class experimental parameter -- not an implementation detail.

Two things this checks:
  1. SOLO vs CONTENDED. Our gates run 10 workers on 10 cores (observed load 12+), so each worker gets
     less than a full core and searches SHALLOWER than a solo run. If the gate is artificially shallow,
     it biases the measurement toward the regime where the PH lever looks good.
  2. LOCAL vs DEPLOY. The shipped agent runs on a Kaggle 2-vCPU runtime, single-threaded, so it gets
     ~1 mostly-uncontended core = closer to the SOLO number here. sims/sec is the hardware-portable
     figure to compare against.

Usage:
  .venv/bin/python scripts/probe_sims.py [GAMES] [ROUTINE_S] [CRITICAL_S]
  PTCG_PH_CRITICAL=0 ... to probe the v6 (blind-governor) side instead.
"""
import importlib.util
import os
import statistics
import sys
import time
import random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V14 = os.path.join(REPO, "submissions", "v14")
sys.path.insert(0, _V14)

from cg.api import to_observation_class                          # noqa: E402
from cg.game import battle_start, battle_select, battle_finish   # noqa: E402
from search import governor, mcts                                # noqa: E402

RECORDS = []
_PENDING = {}


def _install_probes():
    orig_alloc = governor.allocate
    orig_search = mcts.search

    def alloc(*a, **k):
        b = orig_alloc(*a, **k)
        _PENDING["b"] = b
        return b

    def search(*a, **k):
        t0 = time.perf_counter()
        pick, stats = orig_search(*a, **k)
        dt = time.perf_counter() - t0
        b = _PENDING.get("b")
        RECORDS.append({
            "kind": ("critical" if (b is not None and b.k > 1) else "routine"),
            "budget_s": (b.seconds if b is not None else None),
            "k": (b.k if b is not None else None),
            "sims": stats.get("sims"),
            "wall": dt,
        })
        return pick, stats

    governor.allocate = alloc
    mcts.search = search


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_V14, "agent.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rand_pick(sel):
    n = len(sel["option"])
    return random.sample(range(n), min(sel["maxCount"], n)) if n else []


def _summarise(rows, label):
    if not rows:
        print(f"  {label:9s}  (none)")
        return
    sims = [r["sims"] for r in rows if r["sims"] is not None]
    wall = [r["wall"] for r in rows]
    budg = [r["budget_s"] for r in rows if r["budget_s"] is not None]
    sps = [r["sims"] / r["wall"] for r in rows if r["sims"] and r["wall"] > 0]
    print(f"  {label:9s}  n={len(rows):5d}  budget={statistics.median(budg):.2f}s  "
          f"wall_med={statistics.median(wall):.2f}s  "
          f"sims_med={statistics.median(sims):7.0f}  sims_mean={statistics.mean(sims):7.0f}  "
          f"sims/sec={statistics.median(sps):7.0f}")


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    routine_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    critical_s = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    governor.ROUTINE_S = routine_s
    governor.CRITICAL_S = critical_s
    governor.CRITICAL_K = int(os.environ.get("CRIT_K", "8"))

    _install_probes()
    a = _load("brain_a"); b = _load("brain_b")
    a.governor.ROUTINE_S = b.governor.ROUTINE_S = routine_s
    a.governor.CRITICAL_S = b.governor.CRITICAL_S = critical_s
    a.governor.CRITICAL_K = b.governor.CRITICAL_K = governor.CRITICAL_K

    deck = [int(x) for x in open(os.path.join(_V14, "deck.csv")).read().split() if x.strip()][:60]
    a.set_deck(deck); b.set_deck(deck)

    ph = os.environ.get("PTCG_PH_CRITICAL", "1") != "0"
    print(f"=== sims probe: PH_CRIT={ph}  ROUTINE_S={routine_s}s CRITICAL_S={critical_s}s "
          f"K={governor.CRITICAL_K}  games={games} ===")
    sys.stdout.flush()

    t_start = time.perf_counter()
    for g in range(games):
        obs_dict, _ = battle_start(deck, list(deck))
        try:
            for _ in range(4000):
                obs = to_observation_class(obs_dict)
                if obs.select is None or (obs.current is not None and obs.current.result != -1):
                    break
                brain = a if obs.current.yourIndex == (g % 2) else b
                try:
                    pick = brain.choose(obs_dict, time_left=600)
                except Exception:
                    pick = _rand_pick(obs_dict["select"])
                obs_dict = battle_select(pick)
        finally:
            battle_finish()
        print(f"  game {g+1}/{games} done ({time.perf_counter()-t_start:.0f}s elapsed, "
              f"{len(RECORDS)} searched moves)")
        sys.stdout.flush()

    rout = [r for r in RECORDS if r["kind"] == "routine"]
    crit = [r for r in RECORDS if r["kind"] == "critical"]
    tot = len(RECORDS)
    print(f"\n--- searched moves: {tot}  routine={len(rout)} ({100*len(rout)/max(tot,1):.0f}%)  "
          f"critical={len(crit)} ({100*len(crit)/max(tot,1):.0f}%) ---")
    _summarise(rout, "ROUTINE")
    _summarise(crit, "CRITICAL")
    print("\nsims/sec is the hardware-portable number -- compare across machines/contention.")


if __name__ == "__main__":
    main()

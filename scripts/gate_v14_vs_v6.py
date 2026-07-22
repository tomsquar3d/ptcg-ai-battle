#!/usr/bin/env python3
"""A/B the v14 Powerful-Hand-aware governor against v6's blind governor, Alakazam mirror.

v14 = v6 + PH-aware `governor.is_critical`, at **v6's ORIGINAL budget** (ROUTINE_S=0.4, CRITICAL_S=5.0,
CRITICAL_K=8 — the module defaults, deliberately NOT overridden).

Why this gate exists (#65 follow-up): the SAME mechanism gated +13.1% (Welch t=+2.99, N=720) at a shallow
routine budget but -1.5% (t=-0.26) when stacked on v11's raised ROUTINE_S=1.5 — with 1.5 s on every turn
there is nothing left to reallocate to the decisive ones. v13 was gated against v11 because v11 was the
presumed champion; v11 in fact converged to 599.1 over 217 live games vs v6's 683.7. So v6 is both the
better base AND the shallow-routine regime the mechanism actually wins in. This gate tests it there.

Loads submissions/v14/agent.py TWICE as independent modules and toggles each seat's `_PH_CRIT`:
  True  = is_critical resolves Powerful Hand (20xhand) -> our KO turns flagged critical -> deep search
  False = v6's printed-damage behaviour (our KO turns read as 0 damage -> routine -> shallow search)
Everything else identical (same net md5, deck, budget) so this isolates the governor lever.

METHODOLOGY (#65): gates here are ~1.4-1.65x over-dispersed vs binomial, so a single run's binomial CI is
over-confident. Run this as >=8 independent replicates PER ARM and compare replicate means with a Welch
t-test. Always run the (False,False) anchor arm too: it validates the harness (must average ~50%) and is
the correct baseline for the delta. Launch test and anchor workers in BALANCED numbers concurrently --
the governor budget is WALL-CLOCK, so CPU contention changes effective search depth and must hit both
arms symmetrically.

Run:  CFG=test|anchor .venv/bin/python scripts/gate_v14_vs_v6.py [N] [ROUTINE_S] [CRITICAL_S]
"""
import importlib.util
import math
import os
import sys
import time
import random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V14 = os.path.join(REPO, "submissions", "v14")
sys.path.insert(0, _V14)

from cg.api import to_observation_class                          # noqa: E402
from cg.game import battle_start, battle_select, battle_finish   # noqa: E402
from search import governor                                      # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_V14, "agent.py"))
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


def _wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - h) / d, (c + h) / d)


def play(a, b, deck, n):
    a.set_deck(deck); b.set_deck(deck)
    aw = bw = dr = errors = 0
    a_times, b_times = [], []
    for g in range(n):
        a_seat = g % 2          # alternate seats so first-move advantage cancels
        obs_dict, _ = battle_start(deck, list(deck))
        winner = None
        try:
            for _ in range(4000):
                obs = to_observation_class(obs_dict)
                if obs.select is None or (obs.current is not None and obs.current.result != -1):
                    winner = _winner(obs); break
                brain = a if obs.current.yourIndex == a_seat else b
                t0 = time.perf_counter()
                try:
                    pick = brain.choose(obs_dict, time_left=600)
                except Exception:
                    errors += 1; pick = _rand_pick(obs_dict["select"])
                (a_times if brain is a else b_times).append(time.perf_counter() - t0)
                obs_dict = battle_select(pick)
        finally:
            battle_finish()
        if winner == a_seat:
            aw += 1
        elif winner in (0, 1):
            bw += 1
        elif winner == 2:
            dr += 1
    return aw, bw, dr, errors, a_times, b_times


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    # DEFAULTS = v6's shipped settings. Do not "helpfully" raise these: running the mechanism at a
    # budget v6 does not ship is exactly the error that produced v13's phantom +13%.
    routine_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    critical_s = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    _k = int(os.environ.get("CRIT_K", "8"))
    governor.ROUTINE_S = routine_s
    governor.CRITICAL_S = critical_s
    governor.CRITICAL_K = _k
    deck = [int(x) for x in open(os.path.join(_V14, "deck.csv")).read().split() if x.strip()][:60]

    a = _load("brain_a"); b = _load("brain_b")
    a.governor.ROUTINE_S = b.governor.ROUTINE_S = routine_s     # same shared module, re-assert
    a.governor.CRITICAL_S = b.governor.CRITICAL_S = critical_s
    a.governor.CRITICAL_K = b.governor.CRITICAL_K = _k

    # (PH_a, PH_b): (False,False) = both v6 (anchor, must tie); (True,False) = v14 vs v6 (the test)
    configs = [(True, False)] if os.environ.get("CFG") == "test" else [(False, False)]
    print(f"=== v14 vs v6 Powerful-Hand governor A/B, Alakazam mirror, "
          f"ROUTINE_S={routine_s}s CRITICAL_S={critical_s}s K={_k}, N={n}/config ===")
    print(f"{'A_ph':>5} {'B_ph':>5}  {'A W/L/D':>10}  {'A win%':>7}  {'95% CI':>13}  "
          f"{'A ms':>6}  {'B ms':>6}  err")
    for pa, pb in configs:
        a._PH_CRIT = pa; b._PH_CRIT = pb
        aw, bw, dr, errs, at, bt = play(a, b, deck, n)
        p, lo, hi = _wilson(aw, aw + bw)
        ams = (sum(at) / len(at) * 1e3) if at else 0.0
        bms = (sum(bt) / len(bt) * 1e3) if bt else 0.0
        tag = "  <- anchor (must tie)" if pa == pb else "  <- v14 vs v6 (the test)"
        print(f"{str(pa):>5} {str(pb):>5}  {aw:>3}/{bw:<3}/{dr:<2}  {100*p:>6.1f}%  "
              f"[{100*lo:>4.0f},{100*hi:>4.0f}]  {ams:>6.0f}  {bms:>6.0f}  {errs}{tag}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

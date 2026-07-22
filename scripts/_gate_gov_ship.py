#!/usr/bin/env python3
"""A/B the v13 Powerful-Hand-aware governor against v11's blind governor, Alakazam mirror (#63c).

Loads submissions/v13/agent.py TWICE as independent modules and toggles each seat's `_PH_CRIT`:
  True  = is_critical resolves Powerful Hand (20×hand) → our KO turns flagged critical → deep search
  False = v11's printed-damage behaviour (our KO turns look like 0 damage → routine → shallow search)
Everything else is identical (same net, deck, budget), so this isolates the governor lever.

The lever redirects the clock to decisive turns, so it only shows at budgets where critical > routine.
Run at a MODERATE budget (criticals get ~5× the routine search) — directional, not the full 5 s real
budget, but faster. Configs: (False,False) sanity anchor (must tie), then (True,False) the test.

Run:  .venv/bin/python scripts/gate_governor.py [N] [ROUTINE_S] [CRITICAL_S]
"""
import importlib.util
import math
import os
import sys
import time
import random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V13 = os.path.join(REPO, "submissions", "v13")
sys.path.insert(0, _V13)

from cg.api import to_observation_class                          # noqa: E402
from cg.game import battle_start, battle_select, battle_finish   # noqa: E402
from search import governor                                      # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_V13, "agent.py"))
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
    a_times = []
    for g in range(n):
        a_seat = g % 2
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
    return aw, bw, dr, errors, a_times


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    routine_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    critical_s = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    governor.ROUTINE_S = routine_s
    governor.CRITICAL_S = critical_s
    _k = int(os.environ.get("CRIT_K", "4"))
    governor.CRITICAL_K = _k
    deck = [int(x) for x in open(os.path.join(_V13, "deck.csv")).read().split() if x.strip()][:60]

    a = _load("brain_a"); b = _load("brain_b")
    a.governor.ROUTINE_S = b.governor.ROUTINE_S = routine_s     # same shared module, re-assert
    a.governor.CRITICAL_S = b.governor.CRITICAL_S = critical_s
    a.governor.CRITICAL_K = b.governor.CRITICAL_K = _k

    # (PH_CRIT_a, PH_CRIT_b): (False,False)=both v11 (anchor, must tie), (True,False)=v13 vs v11
    configs = [(True, False)] if os.environ.get("CFG")=="test" else [(False, False)]
    print(f"=== v13 Powerful-Hand governor A/B, Alakazam mirror, ROUTINE_S={routine_s}s "
          f"CRITICAL_S={critical_s}s K=4, N={n}/config ===")
    print(f"{'A_ph':>5} {'B_ph':>5}  {'A W/L/D':>10}  {'A win%':>7}  {'95% CI':>13}  {'A ms':>6}  err")
    for pa, pb in configs:
        a._PH_CRIT = pa; b._PH_CRIT = pb
        aw, bw, dr, errs, at = play(a, b, deck, n)
        p, lo, hi = _wilson(aw, aw + bw)
        ms = (sum(at) / len(at) * 1e3) if at else 0.0
        tag = "  <- anchor (must tie)" if pa == pb else "  <- v13 vs v11 (the test)"
        print(f"{str(pa):>5} {str(pb):>5}  {aw:>3}/{bw:<3}/{dr:<2}  {100*p:>6.1f}%  "
              f"[{100*lo:>4.0f},{100*hi:>4.0f}]  {ms:>6.0f}  {errs}{tag}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

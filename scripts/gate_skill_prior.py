#!/usr/bin/env python3
"""A/B the v12 root skill-prior against itself at different `lam`, on the Alakazam mirror (#63).

Loads submissions/v12/agent.py TWICE as two independent modules (so each seat has its own
`_SKILL_LAM` / net / tracker globals) and plays them head-to-head in the raw SDK, seats alternated,
both piloting the same deck. This isolates the prior's effect: everything is identical except `lam`.

`lam=0` reproduces v11 exactly (root_prior_fn=None), so config (0,0) is the sanity anchor and MUST be
a coin flip. The prior is a *root* bias whose influence is largest under shallow search, so we run at a
reduced routine budget (ROUTINE_S, arg) — both faster and the regime the prior targets.

Run:  .venv/bin/python scripts/gate_skill_prior.py [N] [ROUTINE_S] [deck_path]
      configs default to lam_A in {0.0, 0.3, 0.5, 0.7} vs lam_B=0.0.
      deck_path defaults to the shipped Alakazam deck (submissions/v12/deck.csv). Point it at a
      flat-damage ex deck (e.g. experiments/candidate_decks/dragapult-ex.csv) to actually EXERCISE
      the KO-target tree, which is inert on Alakazam (Powerful Hand damage=0). Note: v12's net is
      Alakazam-trained → OOD on other decks, but both seats share it so the lam comparison is fair.
"""
import importlib.util
import math
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V12 = os.path.join(REPO, "submissions", "v12")
sys.path.insert(0, _V12)   # v12's vendored cg/ + ptcg/ + search/ (has skill_prior) for BOTH brains

from cg.api import to_observation_class                          # noqa: E402
from cg.game import battle_start, battle_select, battle_finish   # noqa: E402
from search import governor                                      # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_V12, "agent.py"))
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
    """A win/loss/draw over n games, seats alternated. Returns (aw, bw, dr, errors, a_move_ms)."""
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
    return aw, bw, dr, errors, a_times


def _clamp_budget(routine_s):
    """Shallow, tractable, EQUAL budget for both seats — also the prior's max-influence regime.
    (Both sides identical, so this is a fair test of the root prior; not the shipped v11 budget.)"""
    governor.ROUTINE_S = routine_s
    governor.CRITICAL_S = max(routine_s, 0.3)   # was 5.0 — the real cost driver
    governor.CRITICAL_K = 2                      # was 8
    # ROUTINE_K stays 1; LOW_CLOCK_S/CRIT_FRACTION unchanged (still timeout-safe)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    routine_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    deck_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(_V12, "deck.csv")
    if not os.path.isabs(deck_path):
        deck_path = os.path.join(REPO, deck_path)
    _clamp_budget(routine_s)
    deck = [int(x) for x in open(deck_path).read().split() if x.strip()][:60]

    a = _load("brain_a")
    b = _load("brain_b")
    _clamp_budget(routine_s)                      # brains share the same governor module; re-assert

    configs = [(0.0, 0.0), (0.3, 0.0), (0.5, 0.0), (0.7, 0.0)]
    print(f"=== v12 skill-prior lam sweep, deck={os.path.basename(deck_path)} mirror, "
          f"ROUTINE_S={routine_s}s CRITICAL_S={governor.CRITICAL_S}s K={governor.CRITICAL_K}, "
          f"N={n}/config ===")
    print(f"{'lam_A':>6} {'lam_B':>6}  {'A W/L/D':>10}  {'A win%':>7}  {'95% CI':>13}  {'A ms':>6}  err")
    for lam_a, lam_b in configs:
        a._SKILL_LAM = lam_a
        b._SKILL_LAM = lam_b
        aw, bw, dr, errs, at = play(a, b, deck, n)
        p, lo, hi = _wilson(aw, aw + bw)   # decisive-game win rate
        ms = (sum(at) / len(at) * 1e3) if at else 0.0
        tag = "  <- sanity anchor (must tie)" if lam_a == lam_b else ""
        print(f"{lam_a:>6.1f} {lam_b:>6.1f}  {aw:>3}/{bw:<3}/{dr:<2}  {100*p:>6.1f}%  "
              f"[{100*lo:>4.0f},{100*hi:>4.0f}]  {ms:>6.0f}  {errs}{tag}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

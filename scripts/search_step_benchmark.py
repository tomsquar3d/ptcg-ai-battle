#!/usr/bin/env python3
"""Benchmark the native Search API (search_begin / search_step / search_end).

This is the week-1 unknown from docs/ptcg_agent_strategy.md §8/§9: how fast is one
state transition in the engine's lookahead API? The answer gates how deep the
critical-turn search (Layer 5) can go inside a per-move time budget.

Method:
  1. Drive a random self-play game to a mid-game decision and capture that agent
     observation (which carries the `search_begin_input` blob).
  2. Call search_begin() with determinization predictions. In self-play we know both
     full 60-card decklists, so we hand the engine each decklist as the candidate pool
     for the hidden zones (deck / prize / hand). Time search_begin.
  3. Roll out random legal lines with search_step(), timing every step, until we've
     collected a target number of samples. search_end() between rollouts so the engine
     reuses memory (bounded footprint).

Run:  .venv/bin/python scripts/search_step_benchmark.py [target_steps]
"""
import logging
import os
import random
import statistics
import sys
import time

os.environ.setdefault("LITELLM_LOG", "ERROR")
logging.disable(logging.WARNING)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION_DIR = os.path.join(REPO_ROOT, "submissions", "v0")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, SUBMISSION_DIR)
os.chdir(SUBMISSION_DIR)

from cg.api import (  # noqa: E402
    to_observation_class, all_card_data,
    search_begin, search_step, search_end,
)
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from main import read_deck_csv  # noqa: E402

CARD_INDEX = {c.cardId: c for c in all_card_data()}
MAX_ROLLOUT_DEPTH = 80


def random_select(select):
    n = len(select.option)
    return random.sample(range(n), select.maxCount) if n else []


def _terminal(obs):
    return obs.select is None or (obs.current is not None and obs.current.result != -1)


def first_basic_pokemon(deck):
    for cid in deck:
        cd = CARD_INDEX.get(cid)
        if cd and getattr(cd, "basic", False) and getattr(cd, "cardType", None) == 0:
            return cid
    return deck[0]


def capture_mid_game_obs(deck, min_turn=3, max_tries=40):
    """Drive random self-play; return an obs_dict at a decision with turn >= min_turn."""
    for _ in range(max_tries):
        obs_dict, _ = battle_start(deck, list(deck))
        reached = None
        for _step in range(2000):
            obs = to_observation_class(obs_dict)
            if _terminal(obs):
                break
            if obs.current is not None and obs.current.turn >= min_turn:
                reached = obs_dict
                break
            obs_dict = battle_select(random_select(obs.select))
        battle_finish()
        if reached is not None:
            return reached
    raise RuntimeError("could not reach a mid-game decision")


def begin_from(obs_dict, my_deck, opp_deck):
    """search_begin with full-decklist candidate pools for the hidden zones."""
    obs = to_observation_class(obs_dict)
    st = obs.current
    me, op = st.yourIndex, 1 - st.yourIndex
    opp_active = []
    opp_ps = st.players[op]
    if len(opp_ps.active) > 0 and opp_ps.active[0] is None:
        opp_active = [first_basic_pokemon(opp_deck)]
    return search_begin(
        obs,
        list(my_deck),    # your_deck  (ignored if select.deck != None)
        list(my_deck),    # your_prize
        list(opp_deck),   # opponent_deck
        list(opp_deck),   # opponent_prize
        list(opp_deck),   # opponent_hand
        opp_active,       # opponent_active
        False,            # manual_coin
    )


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1))))
    return xs[k]


def run(target_steps: int) -> int:
    deck = read_deck_csv()
    print(f"capturing a mid-game decision (deck of {len(deck)})...")
    obs_dict = capture_mid_game_obs(deck)
    obs = to_observation_class(obs_dict)
    print(f"captured at turn={obs.current.turn}, "
          f"decision={obs.select.context}, options={len(obs.select.option)}")

    begin_times, step_times, depths, errors = [], [], [], 0

    while len(step_times) < target_steps:
        t0 = time.perf_counter()
        state = begin_from(obs_dict, deck, deck)
        begin_times.append(time.perf_counter() - t0)

        sid = state.searchId
        d = 0
        for d in range(1, MAX_ROLLOUT_DEPTH + 1):
            o = state.observation
            if _terminal(o):
                break
            pick = random_select(o.select)
            try:
                t0 = time.perf_counter()
                state = search_step(sid, pick)
                step_times.append(time.perf_counter() - t0)
            except Exception:
                errors += 1
                break
            sid = state.searchId
            if len(step_times) >= target_steps:
                break
        depths.append(d)
        search_end()

    n = len(step_times)
    mean = statistics.mean(step_times)
    med = statistics.median(step_times)
    us = 1e6
    print("\n=== search_step benchmark ===")
    print(f"search_begin calls   : {len(begin_times)}  "
          f"(mean {statistics.mean(begin_times)*1e3:.3f} ms, "
          f"median {statistics.median(begin_times)*1e3:.3f} ms)")
    print(f"search_step samples  : {n}   (errors: {errors})")
    print(f"  mean               : {mean*us:8.1f} us")
    print(f"  median             : {med*us:8.1f} us")
    print(f"  p95                : {pct(step_times,95)*us:8.1f} us")
    print(f"  p99                : {pct(step_times,99)*us:8.1f} us")
    print(f"  min / max          : {min(step_times)*us:.1f} / {max(step_times)*us:.1f} us")
    print(f"  throughput         : {1.0/mean:,.0f} steps/sec")
    print(f"rollout depth (steps to terminal): "
          f"mean {statistics.mean(depths):.1f}, max {max(depths)}")

    # Interpretation: how much lookahead fits a per-move budget?
    sps = 1.0 / mean
    print("\n--- what this buys the lookahead layer ---")
    for budget in (0.5, 1.0, 3.0):
        print(f"  {budget:>4}s budget  ~= {int(sps*budget):,} search_step transitions")
    print(f"  (a full random line to game end is ~{statistics.mean(depths):.0f} steps, "
          f"so ~{int(sps/max(statistics.mean(depths),1)):,} full rollouts/sec)")
    return 0


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    sys.exit(run(target))

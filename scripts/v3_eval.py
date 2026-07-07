#!/usr/bin/env python3
"""Evaluate the v3 agent (beam / best-first turn search) vs v1 (greedy rollout).

  1. QUALITY      — raw SDK: v3 vs a random opponent. Win rate + per-move timing (beam is more
                    expensive — must stay well under the 2 s soft cap) + exception count.
  2. HEAD-TO-HEAD — v3 vs v1, same deck, both brains in one process, seats alternated. Isolates
                    exactly what non-greedy turn search buys. (the money number)
  3. INTEGRATION  — v3 submission (submissions/v3/main.py) in the cabt env vs random: all DONE,
                    win rate, s/game, max move ms.

Run:  .venv/bin/python scripts/v3_eval.py [quality_games] [h2h_games] [env_games]
"""
import importlib.util
import logging
import os
import random
import statistics
import sys
import time

os.environ.setdefault("LITELLM_LOG", "ERROR")
logging.disable(logging.WARNING)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1, V3 = os.path.join(REPO, "submissions", "v1"), os.path.join(REPO, "submissions", "v3")
sys.path.insert(0, V1)  # cg/ + ptcg/ (identical across submissions) resolve from here
os.chdir(V3)

from cg.api import to_observation_class                       # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


v3 = _load(os.path.join(V3, "agent.py"), "v3_agent")
v1 = _load(os.path.join(V1, "agent.py"), "v1_agent")

_LOG_RESULT = 23


def _rand_pick(sel):
    n = len(sel["option"])
    return random.sample(range(n), min(sel["maxCount"], n)) if n else []


def _winner(obs):
    if obs.current is not None and obs.current.result in (0, 1, 2):
        return obs.current.result
    for lg in obs.logs or []:
        if lg.type == _LOG_RESULT and lg.result is not None:
            return lg.result
    return None


def read_deck():
    with open(os.path.join(V3, "deck.csv"), encoding="utf-8") as f:
        return [int(x) for x in f.read().split("\n") if x.strip()]


def quality(num_games, deck):
    v3.set_deck(deck)
    wins = losses = draws = errors = 0
    times = []
    for _g in range(num_games):
        obs_dict, _ = battle_start(deck, list(deck))
        winner = None
        try:
            for _ in range(4000):
                obs = to_observation_class(obs_dict)
                if obs.select is None or (obs.current is not None and obs.current.result != -1):
                    winner = _winner(obs)
                    break
                if obs.current.yourIndex == 0:
                    t0 = time.perf_counter()
                    try:
                        pick = v3.choose(obs_dict, time_left=600)
                    except Exception:
                        errors += 1
                        pick = _rand_pick(obs_dict["select"])
                    times.append(time.perf_counter() - t0)
                else:
                    pick = _rand_pick(obs_dict["select"])
                obs_dict = battle_select(pick)
        finally:
            battle_finish()
        wins += winner == 0
        losses += winner == 1
        draws += winner == 2
    n = wins + losses + draws
    print("=== 1. QUALITY (v3 vs random) ===")
    print(f"v3 W/L/D         : {wins}/{losses}/{draws}  -> {wins/max(n,1):.0%}")
    print(f"v3 move time     : mean {statistics.mean(times)*1e3:.1f} ms, "
          f"p95 {sorted(times)[int(len(times)*0.95)]*1e3:.1f} ms, max {max(times)*1e3:.0f} ms")
    print(f"exceptions       : {errors}")
    return wins / max(n, 1)


def head_to_head(num_games, deck):
    v3.set_deck(deck)
    v1.set_deck(deck)
    v3w = v1w = draws = errors = 0
    for g in range(num_games):
        v3_seat = g % 2
        obs_dict, _ = battle_start(deck, list(deck))
        winner = None
        try:
            for _ in range(4000):
                obs = to_observation_class(obs_dict)
                if obs.select is None or (obs.current is not None and obs.current.result != -1):
                    winner = _winner(obs)
                    break
                brain = v3 if obs.current.yourIndex == v3_seat else v1
                try:
                    pick = brain.choose(obs_dict, time_left=600)
                except Exception:
                    errors += 1
                    pick = _rand_pick(obs_dict["select"])
                obs_dict = battle_select(pick)
        finally:
            battle_finish()
        if winner == v3_seat:
            v3w += 1
        elif winner in (0, 1):
            v1w += 1
        elif winner == 2:
            draws += 1
    n = v3w + v1w + draws
    print("\n=== 2. HEAD-TO-HEAD (v3 vs v1) ===")
    print(f"v3 W/L/D vs v1   : {v3w}/{v1w}/{draws}  -> v3 win rate {v3w/max(n,1):.0%} "
          f"(decisive {v3w}/{max(v3w+v1w,1)} = {v3w/max(v3w+v1w,1):.0%})")
    print(f"exceptions       : {errors}")
    return v3w, v1w, draws


def integration(num_games, deck):
    from kaggle_environments import make
    main_path = os.path.join(V3, "main.py")

    def random_agent(obs, cfg=None):
        d = obs if isinstance(obs, dict) else {}
        sel = d.get("select")
        if sel is None:
            return deck
        n = len(sel["option"])
        return random.sample(range(n), min(sel["maxCount"], n)) if n else []

    wins = 0
    statuses, durations = [], []
    for i in range(num_games):
        env = make("cabt", debug=True)
        t0 = time.perf_counter()
        agents = [main_path, random_agent] if i % 2 == 0 else [random_agent, main_path]
        env.run(agents)
        durations.append(time.perf_counter() - t0)
        final = env.steps[-1]
        statuses.append(tuple(p["status"] for p in final))
        seat = 0 if i % 2 == 0 else 1
        r = [p["reward"] for p in final]
        if r[seat] is not None and r[1 - seat] is not None and r[seat] > r[1 - seat]:
            wins += 1
    ok = all(s == ("DONE", "DONE") for s in statuses)
    print("\n=== 3. INTEGRATION (v3 in cabt env vs random) ===")
    print(f"all DONE/DONE    : {ok}  {set(statuses)}")
    print(f"v3 wins          : {wins}/{num_games}")
    print(f"seconds per game : mean {statistics.mean(durations):.2f}s, max {max(durations):.2f}s")
    assert ok, f"not all games clean: {statuses}"


def main(qg, hg, eg):
    deck = read_deck()
    quality(qg, deck)
    head_to_head(hg, deck)
    integration(eg, deck)
    print("\nOK: v3 evaluated.")
    return 0


if __name__ == "__main__":
    qg = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    hg = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    eg = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    sys.exit(main(qg, hg, eg))

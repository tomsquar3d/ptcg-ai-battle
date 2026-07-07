#!/usr/bin/env python3
"""Evaluate the v1 agent.

  1. QUALITY     — raw SDK: v1 (evaluator-guided search) as P0 vs a random P1. Win rate + per-
                   move timing + exception count. v1 should beat random clearly.
  2. INTEGRATION — the real submission (submissions/v1/main.py, with watchdog + deck load) run
                   in the Kaggle cabt environment vs a random agent: every game must end DONE
                   (no crash / no timeout), and we report win rate + seconds/game.

Run:  .venv/bin/python scripts/v1_eval.py [quality_games] [env_games]
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
V1 = os.path.join(REPO_ROOT, "submissions", "v1")
sys.path.insert(0, V1)
os.chdir(V1)

from cg.api import to_observation_class            # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
import agent as brain                              # noqa: E402
import main as v1main                              # noqa: E402

_LOG_RESULT = 23


def _rand_pick(sel_dict):
    n = len(sel_dict["option"])
    return random.sample(range(n), min(sel_dict["maxCount"], n)) if n else []


def _winner_from_obs(obs):
    if obs.current is not None and obs.current.result in (0, 1, 2):
        return obs.current.result
    for lg in obs.logs or []:
        if lg.type == _LOG_RESULT and lg.result is not None:
            return lg.result
    return None


def quality(num_games):
    deck = v1main.read_deck_csv()
    brain.set_deck(deck)
    wins = losses = draws = 0
    move_times = []
    errors = 0
    V1_SEAT = 0

    for _g in range(num_games):
        obs_dict, _ = battle_start(deck, list(deck))
        winner = None
        try:
            for _ in range(4000):
                obs = to_observation_class(obs_dict)
                if obs.select is None or (obs.current is not None and obs.current.result != -1):
                    winner = _winner_from_obs(obs)
                    break
                me = obs.current.yourIndex
                if me == V1_SEAT:
                    t0 = time.perf_counter()
                    try:
                        pick = brain.choose(obs_dict, time_left=600)
                    except Exception:
                        errors += 1
                        pick = _rand_pick(obs_dict["select"])
                    move_times.append(time.perf_counter() - t0)
                else:
                    pick = _rand_pick(obs_dict["select"])
                obs_dict = battle_select(pick)
        finally:
            battle_finish()
        if winner == V1_SEAT:
            wins += 1
        elif winner in (0, 1):
            losses += 1
        elif winner == 2:
            draws += 1

    n = wins + losses + draws
    print("=== 1. QUALITY (v1 vs random, raw SDK) ===")
    print(f"games            : {num_games}")
    print(f"v1 W / L / D     : {wins} / {losses} / {draws}  -> win rate {wins/max(n,1):.0%}")
    print(f"v1 move time     : mean {statistics.mean(move_times)*1e3:.1f} ms, "
          f"p95 {sorted(move_times)[int(len(move_times)*0.95)]*1e3:.1f} ms, "
          f"max {max(move_times)*1e3:.0f} ms  ({len(move_times)} moves)")
    print(f"brain exceptions : {errors}")
    return wins, losses, draws


def integration(num_games):
    from kaggle_environments import make
    main_path = os.path.join(V1, "main.py")

    def random_agent(obs, cfg=None):
        d = obs if isinstance(obs, dict) else {}
        sel = d.get("select")
        if sel is None:
            return v1main.read_deck_csv()
        n = len(sel["option"])
        return random.sample(range(n), min(sel["maxCount"], n)) if n else []

    wins = 0
    statuses = []
    durations = []
    for i in range(num_games):
        env = make("cabt", debug=True)
        t0 = time.perf_counter()
        # alternate seats so v1 goes first/second evenly
        agents = [main_path, random_agent] if i % 2 == 0 else [random_agent, main_path]
        env.run(agents)
        durations.append(time.perf_counter() - t0)
        final = env.steps[-1]
        st = [p["status"] for p in final]
        statuses.append(tuple(st))
        v1_seat = 0 if i % 2 == 0 else 1
        r = [p["reward"] for p in final]
        if r[v1_seat] is not None and r[1 - v1_seat] is not None and r[v1_seat] > r[1 - v1_seat]:
            wins += 1

    ok = all(s == ("DONE", "DONE") for s in statuses)
    print("\n=== 2. INTEGRATION (v1 submission in cabt env vs random) ===")
    print(f"games            : {num_games}")
    print(f"all DONE/DONE    : {ok}  (statuses: {set(statuses)})")
    print(f"v1 wins          : {wins}/{num_games}")
    print(f"seconds per game : mean {statistics.mean(durations):.2f}s, max {max(durations):.2f}s")
    assert ok, f"not all games ended cleanly: {statuses}"


def main(quality_games, env_games):
    wins, losses, draws = quality(quality_games)
    integration(env_games)
    wr = wins / max(wins + losses + draws, 1)
    assert wr > 0.55, f"v1 did not clearly beat random (win rate {wr:.0%})"
    print("\nOK: v1 beats random and runs cleanly (DONE) in the cabt environment.")
    return 0


if __name__ == "__main__":
    qg = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    eg = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    sys.exit(main(qg, eg))

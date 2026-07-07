#!/usr/bin/env python3
"""Evaluate the v2 agent (opponent-response lookahead / 2-ply search).

  1. QUALITY    — raw SDK: v2 (reply-on) as P0 vs a random P1. Win rate + per-move timing +
                  exception count. Should be >= v1's ~85% vs random.
  2. HEAD-TO-HEAD (the money number) — v2 reply-ON vs v2 reply-OFF (== v1 behaviour), same
                  module in one process, seats alternated. Isolates exactly what the opponent-
                  response rollout buys. Also asserts the opponent rollout actually fired
                  (agent.STATS["opp_rollouts"] > 0) — else the 2-ply is inert.
  3. INTEGRATION — the real submission (submissions/v2/main.py, watchdog + deck load) in the
                  Kaggle cabt env vs random: every game must end DONE (no crash / no timeout),
                  report win rate, s/game, and max move ms (2-ply cost is the risk).

Run:  .venv/bin/python scripts/v2_eval.py [quality_games] [h2h_games] [env_games]
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
V2 = os.path.join(REPO_ROOT, "submissions", "v2")
sys.path.insert(0, V2)
os.chdir(V2)

from cg.api import to_observation_class            # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
import agent as brain                              # noqa: E402
import main as v2main                              # noqa: E402

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
    deck = v2main.read_deck_csv()
    brain.set_deck(deck)
    wins = losses = draws = 0
    move_times = []
    errors = 0
    V2_SEAT = 0

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
                if me == V2_SEAT:
                    t0 = time.perf_counter()
                    try:
                        pick = brain.choose(obs_dict, time_left=600, opponent_reply=True)
                    except Exception:
                        errors += 1
                        pick = _rand_pick(obs_dict["select"])
                    move_times.append(time.perf_counter() - t0)
                else:
                    pick = _rand_pick(obs_dict["select"])
                obs_dict = battle_select(pick)
        finally:
            battle_finish()
        if winner == V2_SEAT:
            wins += 1
        elif winner in (0, 1):
            losses += 1
        elif winner == 2:
            draws += 1

    n = wins + losses + draws
    print("=== 1. QUALITY (v2 reply-on vs random, raw SDK) ===")
    print(f"games            : {num_games}")
    print(f"v2 W / L / D     : {wins} / {losses} / {draws}  -> win rate {wins/max(n,1):.0%}")
    print(f"v2 move time     : mean {statistics.mean(move_times)*1e3:.1f} ms, "
          f"p95 {sorted(move_times)[int(len(move_times)*0.95)]*1e3:.1f} ms, "
          f"max {max(move_times)*1e3:.0f} ms  ({len(move_times)} moves)")
    print(f"brain exceptions : {errors}")
    return wins, losses, draws


def head_to_head(num_games):
    """v2 reply-ON vs v2 reply-OFF, same module, one process. Seat A = reply-on, seat B =
    reply-off; seats alternate each game so first/second is balanced."""
    deck = v2main.read_deck_csv()
    brain.set_deck(deck)
    brain.STATS["our_rollouts"] = 0
    brain.STATS["opp_rollouts"] = 0
    on_wins = off_wins = draws = 0
    errors = 0

    for g in range(num_games):
        on_seat = g % 2  # alternate which seat runs reply-on
        obs_dict, _ = battle_start(deck, list(deck))
        winner = None
        try:
            for _ in range(4000):
                obs = to_observation_class(obs_dict)
                if obs.select is None or (obs.current is not None and obs.current.result != -1):
                    winner = _winner_from_obs(obs)
                    break
                me = obs.current.yourIndex
                reply = (me == on_seat)
                try:
                    pick = brain.choose(obs_dict, time_left=600, opponent_reply=reply)
                except Exception:
                    errors += 1
                    pick = _rand_pick(obs_dict["select"])
                obs_dict = battle_select(pick)
        finally:
            battle_finish()
        if winner == on_seat:
            on_wins += 1
        elif winner in (0, 1):
            off_wins += 1
        elif winner == 2:
            draws += 1

    n = on_wins + off_wins + draws
    print("\n=== 2. HEAD-TO-HEAD (v2 reply-ON vs v2 reply-OFF == v1) ===")
    print(f"games            : {num_games}")
    print(f"reply-ON W/L/D   : {on_wins} / {off_wins} / {draws}  -> reply-on win rate "
          f"{on_wins/max(n,1):.0%}  (decisive {on_wins}/{max(on_wins+off_wins,1)} "
          f"= {on_wins/max(on_wins+off_wins,1):.0%})")
    print(f"rollouts fired   : our {brain.STATS['our_rollouts']}, "
          f"opp {brain.STATS['opp_rollouts']}")
    print(f"brain exceptions : {errors}")
    assert brain.STATS["opp_rollouts"] > 0, (
        "opponent rollout never fired — 2-ply is inert (yourIndex may not flip mid-search)")
    return on_wins, off_wins, draws


def integration(num_games):
    from kaggle_environments import make
    main_path = os.path.join(V2, "main.py")

    def random_agent(obs, cfg=None):
        d = obs if isinstance(obs, dict) else {}
        sel = d.get("select")
        if sel is None:
            return v2main.read_deck_csv()
        n = len(sel["option"])
        return random.sample(range(n), min(sel["maxCount"], n)) if n else []

    wins = 0
    statuses = []
    durations = []
    for i in range(num_games):
        env = make("cabt", debug=True)
        t0 = time.perf_counter()
        # alternate seats so v2 goes first/second evenly
        agents = [main_path, random_agent] if i % 2 == 0 else [random_agent, main_path]
        env.run(agents)
        durations.append(time.perf_counter() - t0)
        final = env.steps[-1]
        st = [p["status"] for p in final]
        statuses.append(tuple(st))
        v2_seat = 0 if i % 2 == 0 else 1
        r = [p["reward"] for p in final]
        if r[v2_seat] is not None and r[1 - v2_seat] is not None and r[v2_seat] > r[1 - v2_seat]:
            wins += 1

    ok = all(s == ("DONE", "DONE") for s in statuses)
    print("\n=== 3. INTEGRATION (v2 submission in cabt env vs random) ===")
    print(f"games            : {num_games}")
    print(f"all DONE/DONE    : {ok}  (statuses: {set(statuses)})")
    print(f"v2 wins          : {wins}/{num_games}")
    print(f"seconds per game : mean {statistics.mean(durations):.2f}s, max {max(durations):.2f}s")
    assert ok, f"not all games ended cleanly: {statuses}"


def main(quality_games, h2h_games, env_games):
    wins, losses, draws = quality(quality_games)
    head_to_head(h2h_games)
    integration(env_games)
    wr = wins / max(wins + losses + draws, 1)
    assert wr > 0.55, f"v2 did not clearly beat random (win rate {wr:.0%})"
    print("\nOK: v2 beats random, opponent rollout fires, and runs cleanly (DONE) in cabt.")
    return 0


if __name__ == "__main__":
    qg = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    hg = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    eg = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    sys.exit(main(qg, hg, eg))

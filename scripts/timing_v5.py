#!/usr/bin/env python3
"""Real-clock timing pass for v5: verify the agent never blows the per-game clock.

Drives v5-vs-v5 through the real `main.agent`, giving EACH seat an independent draining bank (Kaggle
gives each player 600 s/game). Before every move we inject `remainingOverageTime = that seat's bank`
(exactly what Kaggle passes), run the move, and subtract the actual wall time. The governor should
throttle as a bank drains (budget ≤ 10% of remaining, instant below 20 s) so a bank never goes
negative — a negative bank is a TIMEOUT = instant loss.

    .venv/bin/python scripts/timing_v5.py --bank 600 --games 15   # realistic
    .venv/bin/python scripts/timing_v5.py --bank 45  --games 15   # stress the throttle

Reports worst (max) time a seat spent in a game and the min bank ever reached; fails if any bank < 0.
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V5 = os.path.join(REPO, "submissions", "v5")
sys.path.insert(0, V5)
os.chdir(V5)

import main                                                       # noqa: E402
from cg.api import to_observation_class                          # noqa: E402
from cg.game import battle_start, battle_finish, battle_select   # noqa: E402


def _read_deck():
    return [int(x) for x in open(os.path.join(V5, "deck.csv")).read().split() if x.strip()]


def main_timing() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=float, default=600.0, help="per-seat per-game clock (seconds)")
    ap.add_argument("--games", type=int, default=15)
    args = ap.parse_args()
    deck = _read_deck()
    main.agent({"select": None})                                  # set_deck

    worst_used = 0.0          # most time any seat spent in one game
    min_bank = args.bank      # closest any bank came to 0
    max_move = 0.0
    timeouts = 0
    net_loaded = False

    for g in range(args.games):
        obs_dict, _ = battle_start(deck, list(deck))
        bank = {0: args.bank, 1: args.bank}
        result = -1
        try:
            for _ in range(6000):
                obs = to_observation_class(obs_dict)
                cur = getattr(obs, "current", None)
                if cur is not None and getattr(cur, "result", -1) != -1:
                    result = cur.result
                    break
                if obs.select is None:
                    break
                mi = cur.yourIndex if cur is not None else 0
                obs_dict["remainingOverageTime"] = bank[mi]        # what Kaggle passes
                t0 = time.monotonic()
                pick = main.agent(obs_dict)
                dt = time.monotonic() - t0
                bank[mi] -= dt
                max_move = max(max_move, dt)
                min_bank = min(min_bank, bank[mi])
                if bank[mi] < 0:
                    timeouts += 1
                obs_dict = battle_select(pick if isinstance(pick, list) else [0])
        finally:
            battle_finish()
        net_loaded = net_loaded or getattr(main._brain, "_GUIDE", None) is not None
        used = max(args.bank - bank[0], args.bank - bank[1])
        worst_used = max(worst_used, used)
        print(f"game {g}: result={result}  seat-time used ≤ {used:5.1f}s  "
              f"(banks left {bank[0]:.1f}/{bank[1]:.1f})")

    print(f"\nbank={args.bank}s | net loaded: {net_loaded}")
    print(f"  worst seat-time in a game: {worst_used:.1f}s  (of {args.bank}s)")
    print(f"  min bank ever reached: {min_bank:.1f}s | max single move: {max_move*1000:.0f} ms")
    print(f"  TIMEOUTS (bank < 0): {timeouts}")
    ok = timeouts == 0 and min_bank > 0
    print("\n" + ("OK: never exceeded the clock." if ok else "FAIL: clock exceeded — timeout risk."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main_timing())

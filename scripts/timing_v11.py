#!/usr/bin/env python3
"""Real-clock timeout validation for v11 (= v8 + per-world posterior sampling in the determinizer, #49).

v11 == v8 (same net_ladder3, same Alakazam deck, same mined templates, same pilot) EXCEPT the determinizer
now SAMPLES each world's opponent archetype from the classifier posterior (sample_archetype=True) instead of
MAP-collapse. Sampling touches only which template seeds each world — the per-move cost is unchanged in
structure, but re-validate the clock: drive v11-vs-v11 with an independent draining per-seat bank (Kaggle gives
600 s/player/game), inject `remainingOverageTime` before each move, subtract wall time, assert the bank never
goes negative. Also blocks torch and checks the net loaded + is deck-conditioned.

    .venv/bin/python scripts/timing_v9.py --bank 600 --games 6    # realistic
    .venv/bin/python scripts/timing_v9.py --bank 60  --games 6    # stress the throttle
"""
import argparse
import os
import sys
import time

sys.modules["torch"] = None                       # simulate the torch-free Kaggle runtime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V11 = os.path.join(REPO, "submissions", "v11")
sys.path.insert(0, V11)
os.chdir(V11)

import main                                        # noqa: E402
from cg.api import to_observation_class            # noqa: E402
from cg.game import battle_start, battle_finish, battle_select  # noqa: E402


def _read_deck():
    return [int(x) for x in open(os.path.join(V11, "deck.csv")).read().split() if x.strip()]


def main_timing() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=float, default=600.0)
    ap.add_argument("--games", type=int, default=6)
    args = ap.parse_args()
    deck = _read_deck()
    main.agent({"select": None})                   # set_deck

    worst_used = 0.0
    min_bank = args.bank
    max_move = 0.0
    timeouts = done = illegal = move_count = 0

    for g in range(args.games):
        obs_dict, _ = battle_start(deck, list(deck))
        bank = {0: args.bank, 1: args.bank}
        result = -1
        try:
            for _ in range(8000):
                obs = to_observation_class(obs_dict)
                cur = getattr(obs, "current", None)
                if cur is not None and getattr(cur, "result", -1) != -1:
                    result = cur.result; break
                if obs.select is None:
                    break
                mi = cur.yourIndex if cur is not None else 0
                obs_dict["remainingOverageTime"] = bank[mi]    # what Kaggle passes
                t0 = time.monotonic()
                pick = main.agent(obs_dict)
                dt = time.monotonic() - t0
                bank[mi] -= dt
                max_move = max(max_move, dt); move_count += 1
                min_bank = min(min_bank, bank[mi])
                if bank[mi] < 0:
                    timeouts += 1
                sel = obs.select
                if not (isinstance(pick, list) and all(isinstance(i, int) and 0 <= i < len(sel.option) for i in pick)):
                    illegal += 1
                    pick = list(range(min(max(sel.minCount, 0), len(sel.option))))
                obs_dict = battle_select(pick if isinstance(pick, list) else [0])
        finally:
            battle_finish()
        if result in (0, 1):
            done += 1
        used = max(args.bank - bank[0], args.bank - bank[1])
        worst_used = max(worst_used, used)
        print(f"game {g}: result={result} ({'DONE' if result in (0,1) else '?'})  "
              f"seat-time ≤ {used:5.1f}s  banks {bank[0]:.1f}/{bank[1]:.1f}", flush=True)

    gd = getattr(main._brain, "_GUIDE", None)
    conditioned = bool(gd and getattr(gd, "deck_cond", False)
                       and getattr(gd, "_ds", None) is not None and float(abs(gd._ds).sum()) > 0)
    print(f"\nbank={args.bank}s | net loaded: {gd is not None} | deck-conditioned: {conditioned} | torch-free ✓")
    print(f"  games DONE {done}/{args.games} | illegal {illegal} | moves {move_count}")
    print(f"  worst seat-time in a game: {worst_used:.1f}s (of {args.bank}s) | min bank {min_bank:.1f}s | "
          f"max single move {max_move*1000:.0f} ms | TIMEOUTS {timeouts}")
    ok = timeouts == 0 and min_bank > 0 and illegal == 0 and done == args.games and conditioned
    print("\n" + ("OK: v11 timeout-safe, deck-conditioned, all DONE/legal." if ok else "FAIL: see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main_timing())

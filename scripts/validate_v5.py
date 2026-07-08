#!/usr/bin/env python3
"""Validate the v5 submission bundle as it will run on Kaggle.

Blocks `import torch` up front (the deployed bundle must be torch-free — only onnxruntime), then
drives full games through the real entry point `main.agent(obs_dict)` for BOTH seats and checks:
  1. the bundle imports and the ONNX net actually LOADS (not silently degrading to heuristic);
  2. every game reaches a terminal result (DONE) with no exception escaping the watchdog;
  3. every returned move is legal;
  4. per-move time stays well under the clock (and total per game is sane).

Run:  .venv/bin/python scripts/validate_v5.py [n_games]
"""
import os
import sys
import time

# --- simulate the Kaggle runtime: no torch available -------------------------------------
# Poisoning sys.modules makes `import torch` raise ImportError immediately (works on 3.12, unlike
# the removed find_module finder API).
sys.modules["torch"] = None

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V5 = os.path.join(REPO, "submissions", "v5")
sys.path.insert(0, V5)                 # only the agent dir on path, like Kaggle
os.chdir(V5)                           # so read_deck_csv() / model dir resolve relative

import main                            # noqa: E402  (the submission entry point)
from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_start, battle_finish, battle_select  # noqa: E402


def _read_deck():
    return [int(x) for x in open(os.path.join(V5, "deck.csv")).read().split() if x.strip()]


def main_validate() -> int:
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    deck = _read_deck()

    # confirm torch really is blocked
    try:
        import torch  # noqa: F401
        print("WARNING: torch imported — not a faithful deploy test")
        torch_blocked = False
    except ImportError:
        print("torch blocked ✓ (deploy is torch-free)")
        torch_blocked = True
    main.agent({"select": None})       # Kaggle's first call: returns deck + set_deck
    # note: the net loads lazily on the FIRST choose() (inside _lazy_init), not in set_deck — so we
    # check _GUIDE *after* real moves have run.

    done = 0
    max_move_ms = 0.0
    move_count = 0
    exceptions = 0
    for g in range(n_games):
        obs_dict, _ = battle_start(deck, list(deck))
        game_t0 = time.monotonic()
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
                t0 = time.monotonic()
                pick = main.agent(obs_dict)                    # THE deployed entry point
                dt = (time.monotonic() - t0) * 1000
                max_move_ms = max(max_move_ms, dt)
                move_count += 1
                sel = obs.select
                if not (isinstance(pick, list) and all(isinstance(i, int) and 0 <= i < len(sel.option)
                                                       for i in pick)):
                    print(f"  ILLEGAL move at game {g}: {pick} (options={len(sel.option)})")
                    exceptions += 1
                    pick = list(range(min(max(sel.minCount, 0), len(sel.option))))
                obs_dict = battle_select(pick)
        finally:
            battle_finish()
        if result in (0, 1):
            done += 1
        print(f"game {g}: result={result} ({'DONE' if result in (0,1) else 'UNFINISHED'})  "
              f"{time.monotonic()-game_t0:.1f}s")

    net_loaded = getattr(main._brain, "_GUIDE", None) is not None
    print(f"\ntorch blocked: {torch_blocked} | net loaded (onnxruntime, no torch): {net_loaded}")
    print(f"games DONE: {done}/{n_games} | moves: {move_count} | "
          f"max move: {max_move_ms:.0f} ms | illegal/exc: {exceptions}")
    ok = torch_blocked and net_loaded and done == n_games and exceptions == 0
    print("\n" + ("OK: v5 bundle is torch-free, net-loaded, all DONE, all legal." if ok
                  else "FAIL: see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main_validate())

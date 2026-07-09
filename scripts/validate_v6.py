#!/usr/bin/env python3
"""Validate the v6 submission bundle as it will run on Kaggle (deck-conditioned Alakazam agent).

Blocks `import torch` (deploy is onnxruntime-only), drives full Alakazam-mirror games through the real
`main.agent`, and checks: torch blocked; the ONNX net LOADS and is **deck-conditioned** on the Alakazam
deck (deck_cond True + nonzero deck_summary — else it'd pilot blind); every game DONE; every move legal;
per-move time sane.

Run:  .venv/bin/python scripts/validate_v6.py [n_games]
"""
import os
import sys
import time

sys.modules["torch"] = None                      # simulate the torch-free Kaggle runtime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V6 = os.path.join(REPO, "submissions", "v6")
sys.path.insert(0, V6)
os.chdir(V6)

import main                                       # noqa: E402
from cg.api import to_observation_class           # noqa: E402
from cg.game import battle_start, battle_finish, battle_select  # noqa: E402


def _read_deck():
    return [int(x) for x in open(os.path.join(V6, "deck.csv")).read().split() if x.strip()]


def main_validate() -> int:
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    deck = _read_deck()
    try:
        import torch  # noqa: F401
        print("WARNING: torch imported — not a faithful deploy test"); torch_blocked = False
    except ImportError:
        print("torch blocked ✓ (deploy is torch-free)"); torch_blocked = True
    main.agent({"select": None})                 # set_deck

    done = max_ms = move_count = exceptions = 0
    max_ms = 0.0
    for g in range(n_games):
        obs_dict, _ = battle_start(deck, list(deck))
        result = -1
        try:
            for _ in range(6000):
                obs = to_observation_class(obs_dict)
                cur = getattr(obs, "current", None)
                if cur is not None and getattr(cur, "result", -1) != -1:
                    result = cur.result; break
                if obs.select is None:
                    break
                t0 = time.monotonic()
                pick = main.agent(obs_dict)
                max_ms = max(max_ms, (time.monotonic() - t0) * 1000)
                move_count += 1
                sel = obs.select
                if not (isinstance(pick, list) and all(isinstance(i, int) and 0 <= i < len(sel.option)
                                                       for i in pick)):
                    print(f"  ILLEGAL move g{g}: {pick} (options={len(sel.option)})"); exceptions += 1
                    pick = list(range(min(max(sel.minCount, 0), len(sel.option))))
                obs_dict = battle_select(pick)
        finally:
            battle_finish()
        if result in (0, 1):
            done += 1
        print(f"game {g}: result={result} ({'DONE' if result in (0,1) else 'UNFINISHED'})")

    g = getattr(main._brain, "_GUIDE", None)
    net_loaded = g is not None
    deck_cond = bool(getattr(g, "deck_cond", False)) if g else False
    ds = getattr(g, "_ds", None)
    conditioned = bool(deck_cond and ds is not None and float(abs(ds).sum()) > 0)
    print(f"\ntorch blocked: {torch_blocked} | net loaded: {net_loaded} | deck-conditioned on Alakazam: {conditioned}")
    print(f"games DONE: {done}/{n_games} | moves: {move_count} | max move: {max_ms:.0f} ms | illegal: {exceptions}")
    ok = torch_blocked and net_loaded and conditioned and done == n_games and exceptions == 0
    print("\n" + ("OK: v6 is torch-free, net-loaded, deck-conditioned, all DONE, all legal." if ok else "FAIL: see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main_validate())

#!/usr/bin/env python3
"""Validate v4 (MCTS-H) vs the heuristic pilots (v3 beam, v1 turn-aware), per matchup.

The decisive test of the whole search arc: does real multi-turn MCTS out-pilot the greedy/beam
heuristics on an EVOLUTION-LINE deck (dragapult-ex), where the planning gap (#15/#17) lives? Also
runs the simple sample deck as a no-regression check. Both sides pilot the SAME deck; seats
alternated; parallel across processes (MCTS games are slow — CPU cores are the only lever).

Run:  .venv/bin/python scripts/validate_v4.py [N_per_matchup]
"""
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter
from itertools import product

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))  # shared cg/ptcg

from cg.api import to_observation_class                       # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402


def _load(name, sub):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "submissions", sub, "agent.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


V4 = _load("brain_v4", "v4")
V3 = _load("brain_v3", "v3")
V1 = _load("brain_v1", "v1")
BRAINS = {"v3": V3, "v1": V1}
for _b in (V4, V3, V1):          # warm indices in the parent so forked workers inherit them
    try:
        _b._lazy_init()
    except Exception:
        pass

DECKS = {
    "dragapult(evo)": os.path.join(REPO, "experiments", "candidate_decks", "dragapult-ex.csv"),
    "sample": os.path.join(REPO, "submissions", "v1", "deck.csv"),
}


def _rand(sel):
    n = len(sel["option"])
    return random.sample(range(n), min(sel["maxCount"], n)) if n else []


def _winner(obs):
    if obs.current is not None and obs.current.result in (0, 1, 2):
        return obs.current.result
    for lg in obs.logs or []:
        if lg.type == 23 and lg.result is not None:
            return lg.result
    return None


def play_spec(spec):
    """(opp_key, deck_name, deck_tuple, v4_seat) → (opp_key, deck_name, 'v4'|'opp'|'draw')."""
    opp_key, deck_name, deck_t, v4_seat = spec
    deck = list(deck_t)
    opp = BRAINS[opp_key]
    V4.set_deck(deck)
    opp.set_deck(deck)
    obs_dict, _ = battle_start(deck, list(deck))
    winner = None
    try:
        for _ in range(4000):
            obs = to_observation_class(obs_dict)
            if obs.select is None or (obs.current is not None and obs.current.result != -1):
                winner = _winner(obs)
                break
            brain = V4 if obs.current.yourIndex == v4_seat else opp
            try:
                pick = brain.choose(obs_dict, time_left=600)
            except Exception:
                pick = _rand(obs_dict["select"])
            obs_dict = battle_select(pick)
    finally:
        battle_finish()
    if winner == v4_seat:
        r = "v4"
    elif winner in (0, 1):
        r = "opp"
    else:
        r = "draw"
    return opp_key, deck_name, r


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    decks = {name: tuple(int(x) for x in open(p).read().split() if x.strip())
             for name, p in DECKS.items()}
    specs = []
    for opp_key, deck_name in product(("v3", "v1"), decks):
        for g in range(n):
            specs.append((opp_key, deck_name, decks[deck_name], g % 2))
    random.shuffle(specs)  # mix slow (evo) and fast (sample) games across workers

    import multiprocessing as mp
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"validate v4: {len(specs)} games ({n}/matchup), {workers} workers", flush=True)

    tally = {(o, d): Counter() for o, d in product(("v3", "v1"), decks)}
    t0 = time.perf_counter()
    with mp.Pool(workers) as pool:
        for i, (opp_key, deck_name, r) in enumerate(pool.imap_unordered(play_spec, specs, chunksize=1), 1):
            tally[(opp_key, deck_name)][r] += 1
            if i % 20 == 0 or i == len(specs):
                el = time.perf_counter() - t0
                eta = el / i * (len(specs) - i)
                print(f"  [{i}/{len(specs)}] {el/60:.1f} min, ~{eta/60:.1f} min left", flush=True)

    lines = [f"# v4 (MCTS-H) validation — {n} games/matchup", "",
             "| opponent | deck | v4 win rate | v4 W/L/D |", "|---|---|--:|--:|"]
    result = {}
    print("\n=== per-matchup v4 win rate ===")
    for (opp_key, deck_name), c in tally.items():
        tot = c["v4"] + c["opp"] + c["draw"] or 1
        wr = (c["v4"] + 0.5 * c["draw"]) / tot
        result[f"v4_vs_{opp_key}|{deck_name}"] = {"win_rate": round(wr, 3), **dict(c)}
        lines.append(f"| {opp_key} | {deck_name} | {wr:.0%} | {c['v4']}/{c['opp']}/{c['draw']} |")
        print(f"  v4 vs {opp_key:3} on {deck_name:14}: {wr:.0%}  (W/L/D {c['v4']}/{c['opp']}/{c['draw']})")

    out_md = os.path.join(REPO, "experiments", "validation_v4.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(REPO, "experiments", "validation_v4.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

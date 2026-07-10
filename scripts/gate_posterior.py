#!/usr/bin/env python3
"""Gate per-world posterior sampling in determinize() (#49): does hedging opponent-deck uncertainty across
the K worlds (sample_archetype=True) beat MAP-collapse (=False)?

"Us" = Alakazam pilot (heuristic MCTS — isolates the search change, no net/OOD confound). The opponent plays
a real ladder deck (heuristic, determinizer held FIXED at MAP=False as a constant baseline). We vary ONLY
our determinizer's sample_archetype flag, conditions interleave, our seat alternates → the deck-matchup bias
cancels. Templates are the mined set (both conditions). A SMP win = posterior sampling produced better search.

    .venv/bin/python scripts/gate_posterior.py --opp mega-kangaskhan-ex --games 200 --seed 0
    .venv/bin/python scripts/gate_posterior.py --opp alakazam --games 200          # mirror control
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
sys.path.insert(0, os.path.join(REPO, "src"))

from cg.api import (to_observation_class, all_card_data, all_attack,                # noqa: E402
                    search_begin, search_step, search_end)
from cg.game import battle_start, battle_finish, battle_select                       # noqa: E402
from ptcg.opponent import OpponentTracker                                            # noqa: E402
from search.determinizer import determinize                                          # noqa: E402
from search import mcts                                                              # noqa: E402

CAND = os.path.join(REPO, "experiments", "candidate_decks")
ALAKAZAM_DECK = os.path.join(CAND, "alakazam-ladder.csv")
OPP_SAMPLE = False        # opponent determinizer held fixed at MAP-collapse (constant baseline)


def _read_deck(path):
    return [int(x) for x in open(path).read().split() if x.strip()]


def _legal_min(sel):
    n = len(sel.option)
    return list(range(min(max(sel.minCount, 0), n)))


def _multi_pick(obs, mi, deck, ci, ai, tracker, sample):
    sel = obs.select
    n = len(sel.option)
    if n == 0:
        return []
    try:
        worlds = determinize(obs, deck, ci, k=1, opp_tracker=tracker, sample_archetype=sample)
        root = search_begin(obs, *worlds[0])
    except Exception:
        return _legal_min(sel)
    try:
        scored = []
        for i in range(n):
            try:
                child = search_step(root.searchId, [i])
                scored.append((mcts.leaf_value(child, mi, ci, ai), i))
            except Exception:
                continue
        if not scored:
            return _legal_min(sel)
        scored.sort(reverse=True)
        pick = sorted({i for _s, i in scored[:sel.maxCount]})
        if len(pick) < sel.minCount:
            pick = sorted(range(n))[:sel.minCount]
        return pick[:sel.maxCount]
    finally:
        search_end()


def play_game(us_deck, opp_deck, ci, ai, us_seat, us_sample, sims, k, move_cap=12.0, budget=None):
    """us_seat determinizes with sample_archetype=us_sample; opponent fixed at OPP_SAMPLE. Heuristic both.
    Returns True iff 'us' won, None if undecided."""
    MAX_SIMS = 200_000
    decks = {us_seat: us_deck, 1 - us_seat: opp_deck}
    samples = {us_seat: us_sample, 1 - us_seat: OPP_SAMPLE}
    trackers = {0: OpponentTracker(), 1: OpponentTracker()}
    result = -1
    obs_dict, _ = battle_start(list(decks[0]), list(decks[1]))
    if obs_dict is None:
        return None
    try:
        for _ in range(6000):
            obs = to_observation_class(obs_dict)
            cur = getattr(obs, "current", None)
            if cur is not None and getattr(cur, "result", -1) != -1:
                result = cur.result
                break
            sel = obs.select
            if sel is None:
                break
            mi = cur.yourIndex if cur is not None else 0
            deck, sample = decks[mi], samples[mi]
            n = len(sel.option)
            if n == 0:
                obs_dict = battle_select([])
                continue
            try:
                trackers[mi].observe(obs)
            except Exception:
                pass
            if sel.maxCount > 1:
                pick = _multi_pick(obs, mi, deck, ci, ai, trackers[mi], sample)
            elif n == 1 and sel.minCount == 1:
                pick = [0]
            else:
                pick = None
                try:
                    worlds = determinize(obs, deck, ci, k=k, opp_tracker=trackers[mi], sample_archetype=sample)
                    n_sims = MAX_SIMS if budget else sims
                    deadline = time.monotonic() + (budget if budget else move_cap)
                    pick, _st = mcts.search(obs, worlds, ci, ai, sims=n_sims, deadline=deadline)
                except Exception:
                    pick = None
                if pick is None:
                    pick = _legal_min(sel) or [0]
            obs_dict = battle_select(pick)
    finally:
        battle_finish()
    if result == -1:
        return None
    return result == us_seat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--us", default="alakazam")
    ap.add_argument("--opp", required=True, help="opponent deck slug in candidate_decks (or 'alakazam')")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    us_deck = _read_deck(ALAKAZAM_DECK if args.us == "alakazam" else os.path.join(CAND, f"{args.us}.csv"))
    opp_deck = _read_deck(ALAKAZAM_DECK if args.opp == "alakazam" else os.path.join(CAND, f"{args.opp}.csv"))
    print(f"US={args.us} vs OPP={args.opp} (heuristic, k={args.k}) | vary OUR determinizer: "
          f"SMP (posterior-sample) vs MAP (collapse); opponent fixed MAP\n")

    w = {"SMP": 0, "MAP": 0}
    n = {"SMP": 0, "MAP": 0}
    t0 = time.monotonic()
    for gi in range(args.games):
        cond = "SMP" if gi % 2 == 0 else "MAP"
        us_seat = (args.seed + gi) % 2
        won = play_game(us_deck, opp_deck, ci, ai, us_seat, cond == "SMP", args.sims, args.k, budget=args.budget)
        if won is None:
            continue
        n[cond] += 1
        w[cond] += int(won)
        if (gi + 1) % 20 == 0:
            ps = w["SMP"] / n["SMP"] if n["SMP"] else 0
            pm = w["MAP"] / n["MAP"] if n["MAP"] else 0
            print(f"  [{gi+1}/{args.games}] our winrate  SMP {w['SMP']}/{n['SMP']}={ps:.1%}  "
                  f"MAP {w['MAP']}/{n['MAP']}={pm:.1%}  ({time.monotonic()-t0:.0f}s)", flush=True)
    dt = time.monotonic() - t0

    ps = w["SMP"] / n["SMP"] if n["SMP"] else 0.0
    pm = w["MAP"] / n["MAP"] if n["MAP"] else 0.0
    pooled = (w["SMP"] + w["MAP"]) / (n["SMP"] + n["MAP"]) if (n["SMP"] + n["MAP"]) else 0.0
    se = (pooled * (1 - pooled) * (1 / max(n["SMP"], 1) + 1 / max(n["MAP"], 1))) ** 0.5
    z = (ps - pm) / se if se else 0.0
    verdict = ("SMP BETTER" if z > 1.64 else ("SMP WORSE" if z < -1.64 else "no sig. diff"))
    print(f"\nPOSTERIOR-SAMPLING GATE | us={args.us} vs opp={args.opp} | {args.games} games in {dt:.0f}s\n"
          f"  posterior-sample (SMP): {w['SMP']}/{n['SMP']} = {ps:.1%}\n"
          f"  MAP-collapse    (MAP):  {w['MAP']}/{n['MAP']} = {pm:.1%}\n"
          f"  delta (SMP - MAP) = {ps-pm:+.1%}  (z={z:+.2f})  ->  {verdict}")
    if args.out:
        json.dump({"us": args.us, "opp": args.opp, "w": w, "n": n, "ps": ps, "pm": pm,
                   "delta": ps - pm, "z": z, "games": args.games, "seconds": dt}, open(args.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

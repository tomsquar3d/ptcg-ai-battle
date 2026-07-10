#!/usr/bin/env python3
"""Cross-deck gate: do the mined templates (#46) help our pilot when it FACES a deck the old set lacks?

The mirror gate (gate_templates.py) under-tests the fix: a mirror never makes our agent face an archetype
the old TrainerHill set can't represent. Here "us" pilots Alakazam and the OPPONENT plays a real non-Alakazam
ladder deck (Kangaskhan / Grimmsnarl / Garchomp — none of which the old set has a template for). We measure
OUR win rate under two conditions, holding the opponent fixed:
  NEW: our determinizer models the opponent with the mined templates (correct archetype)
  OLD: our determinizer models the opponent with the stale TrainerHill templates (phantom deck)
The opponent's own determinizer is FIXED (new templates) in both conditions, so the ONLY thing that varies
is the quality of OUR opponent model. Both pilots are heuristic MCTS (no net → no OOD confound). Conditions
interleave game-by-game and our seat alternates, so the deck-matchup bias is a constant offset that cancels
in the NEW-vs-OLD comparison.

    .venv/bin/python scripts/gate_cross_templates.py --opp mega-kangaskhan-ex --games 200 --seed 0
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
from ptcg.archetypes import load_archetypes                                          # noqa: E402
from search.determinizer import determinize                                          # noqa: E402
from search import mcts                                                              # noqa: E402

NEW_PATH = os.path.join(REPO, "src", "ptcg", "archetype_templates.json")
OLD_PATH = os.path.join(REPO, "src", "ptcg", "archetype_templates.trainerhill.json")
CAND = os.path.join(REPO, "experiments", "candidate_decks")
ALAKAZAM_DECK = os.path.join(CAND, "alakazam-ladder.csv")


def _read_deck(path):
    return [int(x) for x in open(path).read().split() if x.strip()]


def _legal_min(sel):
    n = len(sel.option)
    return list(range(min(max(sel.minCount, 0), n)))


def _multi_pick(obs, mi, deck, ci, ai, tracker, arch):
    sel = obs.select
    n = len(sel.option)
    if n == 0:
        return []
    try:
        worlds = determinize(obs, deck, ci, k=1, archetypes=arch, opp_tracker=tracker)
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


def play_game(us_deck, opp_deck, ci, ai, us_seat, us_arch, opp_arch, sims, k, move_cap=12.0, budget=None):
    """us_seat pilots us_deck (determinizer=us_arch); other seat pilots opp_deck (determinizer=opp_arch).
    Heuristic MCTS both. Returns True iff 'us' won."""
    MAX_SIMS = 200_000
    decks = {us_seat: us_deck, 1 - us_seat: opp_deck}
    archs = {us_seat: us_arch, 1 - us_seat: opp_arch}
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
            deck, arch = decks[mi], archs[mi]
            n = len(sel.option)
            if n == 0:
                obs_dict = battle_select([])
                continue
            try:
                trackers[mi].observe(obs)
            except Exception:
                pass
            if sel.maxCount > 1:
                pick = _multi_pick(obs, mi, deck, ci, ai, trackers[mi], arch)
            elif n == 1 and sel.minCount == 1:
                pick = [0]
            else:
                pick = None
                try:
                    worlds = determinize(obs, deck, ci, k=k, archetypes=arch, opp_tracker=trackers[mi])
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
    ap.add_argument("--us", default="alakazam", help="our deck (name in candidate_decks or path)")
    ap.add_argument("--opp", required=True, help="opponent archetype slug (candidate_decks/<slug>.csv) or path")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=None, help="seconds/move — equal-WALL-CLOCK mode")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    us_path = ALAKAZAM_DECK if args.us == "alakazam" else (
        os.path.join(CAND, f"{args.us}.csv") if not os.path.sep in args.us else args.us)
    opp_path = args.opp if os.path.sep in args.opp else os.path.join(CAND, f"{args.opp}.csv")
    us_deck, opp_deck = _read_deck(us_path), _read_deck(opp_path)

    arch_new = load_archetypes(NEW_PATH)
    arch_old = load_archetypes(OLD_PATH)
    print(f"US={args.us} ({len(us_deck)}c) vs OPP={args.opp} ({len(opp_deck)}c) | heuristic MCTS both | "
          f"{'wall '+str(args.budget)+'s' if args.budget else f'sims={args.sims}'} k={args.k}")
    print(f"opponent model fixed = NEW; we vary OUR determinizer: NEW (mined) vs OLD (trainerhill)\n")

    w = {"NEW": 0, "OLD": 0}
    n = {"NEW": 0, "OLD": 0}
    t0 = time.monotonic()
    for g in range(args.games):
        cond = "NEW" if g % 2 == 0 else "OLD"
        us_arch = arch_new if cond == "NEW" else arch_old
        us_seat = (args.seed + g) % 2
        won = play_game(us_deck, opp_deck, ci, ai, us_seat, us_arch, arch_new, args.sims, args.k, budget=args.budget)
        if won is None:
            continue
        n[cond] += 1
        w[cond] += int(won)
        if (g + 1) % 20 == 0:
            pn = w["NEW"] / n["NEW"] if n["NEW"] else 0
            po = w["OLD"] / n["OLD"] if n["OLD"] else 0
            print(f"  [{g+1}/{args.games}] our winrate  NEW {w['NEW']}/{n['NEW']}={pn:.1%}  "
                  f"OLD {w['OLD']}/{n['OLD']}={po:.1%}  ({time.monotonic()-t0:.0f}s)", flush=True)
    dt = time.monotonic() - t0

    pn = w["NEW"] / n["NEW"] if n["NEW"] else 0.0
    po = w["OLD"] / n["OLD"] if n["OLD"] else 0.0
    pooled = (w["NEW"] + w["OLD"]) / (n["NEW"] + n["OLD"]) if (n["NEW"] + n["OLD"]) else 0.0
    se = (pooled * (1 - pooled) * (1 / max(n["NEW"], 1) + 1 / max(n["OLD"], 1))) ** 0.5
    z = (pn - po) / se if se else 0.0
    verdict = "NEW BETTER" if z > 1.64 else ("NEW WORSE" if z < -1.64 else "no sig. diff")
    print(f"\nCROSS-DECK GATE | us={args.us} vs opp={args.opp} | {args.games} games in {dt:.0f}s\n"
          f"  our win rate with NEW templates: {w['NEW']}/{n['NEW']} = {pn:.1%}\n"
          f"  our win rate with OLD templates: {w['OLD']}/{n['OLD']} = {po:.1%}\n"
          f"  delta = {pn-po:+.1%}  (z={z:+.2f})  ->  {verdict}")
    if args.out:
        json.dump({"us": args.us, "opp": args.opp, "w": w, "n": n, "pn": pn, "po": po,
                   "delta": pn - po, "z": z, "games": args.games, "seconds": dt}, open(args.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

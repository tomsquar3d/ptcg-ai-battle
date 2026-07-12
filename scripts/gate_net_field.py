#!/usr/bin/env python3
"""Diagnose net_ladder3's live regression: does it beat net_ladder2 vs the FIELD, or only in the mirror?

net_ladder3 (v7) beat net_ladder2 (v6) 72.9% in the Alakazam MIRROR offline (#44), yet converged ~45 pts
BELOW v6 on the live ladder. Hypothesis: ladder3 overfit the Alakazam mirror (its training + its only gate),
so it's worse against the ~2/3 non-Alakazam field. Test it directly: our Alakazam pilot uses each net in
turn, FACING a real field deck (heuristic-piloted, held fixed). Templates are held constant (both conditions
use the mined set), so the NET is the only varied factor. Conditions interleave, our seat alternates, so the
deck-matchup bias cancels in the ladder3-vs-ladder2 comparison.

If ladder3's edge over ladder2 (clear in the mirror) shrinks or reverses vs the field → mirror overfit,
which explains the live result (the ladder is mostly field, not mirror).

    .venv/bin/python scripts/gate_net_field.py --opp mega-kangaskhan-ex --games 120 --budget 0.35
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
from nn.onnx_guide import OnnxGuide                                                   # noqa: E402

CKPT = os.path.join(REPO, "experiments", "checkpoints")
LADDER3 = os.path.join(CKPT, "ladder3")     # v7 net (the suspect)
LADDER2 = os.path.join(CKPT, "ladder2")     # v6 net (the live champion)
CAND = os.path.join(REPO, "experiments", "candidate_decks")
ALAKAZAM_DECK = os.path.join(CAND, "alakazam-ladder.csv")


def _read_deck(path):
    return [int(x) for x in open(path).read().split() if x.strip()]


def _legal_min(sel):
    n = len(sel.option)
    return list(range(min(max(sel.minCount, 0), n)))


def _multi_pick(obs, mi, deck, ci, ai, tracker):
    sel = obs.select
    n = len(sel.option)
    if n == 0:
        return []
    try:
        worlds = determinize(obs, deck, ci, k=1, opp_tracker=tracker)
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


def play_game(us_deck, opp_deck, ci, ai, us_seat, us_guide, sims, k, move_cap=12.0, budget=None):
    """us_seat pilots us_deck with us_guide (a net); other seat pilots opp_deck with HEURISTIC MCTS.
    Returns True iff 'us' won, None if undecided."""
    MAX_SIMS = 200_000
    decks = {us_seat: us_deck, 1 - us_seat: opp_deck}
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
            deck = decks[mi]
            n = len(sel.option)
            if n == 0:
                obs_dict = battle_select([])
                continue
            try:
                trackers[mi].observe(obs)
            except Exception:
                pass
            if sel.maxCount > 1:
                pick = _multi_pick(obs, mi, deck, ci, ai, trackers[mi])
            elif n == 1 and sel.minCount == 1:
                pick = [0]
            else:
                pick = None
                try:
                    worlds = determinize(obs, deck, ci, k=k, opp_tracker=trackers[mi])
                    n_sims = MAX_SIMS if budget else sims
                    deadline = time.monotonic() + (budget if budget else move_cap)
                    if mi == us_seat:                       # our seat: net-guided MCTS
                        pick, _st = mcts.search(obs, worlds, ci, ai, sims=n_sims, deadline=deadline,
                                                prior_fn=us_guide.priors, value_fn=us_guide.value)
                    else:                                   # opponent: heuristic MCTS (fixed)
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
    ap.add_argument("--opp", required=True, help="opponent deck slug in candidate_decks (or 'alakazam')")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=0.35, help="seconds/move (net is slow — wall-clock cap)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--net-a", default=LADDER3, help="net dir for condition A (default ladder3)")
    ap.add_argument("--net-b", default=LADDER2, help="net dir for condition B (default ladder2)")
    ap.add_argument("--label-a", default="L3")
    ap.add_argument("--label-b", default="L2")
    args = ap.parse_args()
    LA, LB = args.label_a, args.label_b

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    us_deck = _read_deck(ALAKAZAM_DECK)
    opp_path = ALAKAZAM_DECK if args.opp == "alakazam" else os.path.join(CAND, f"{args.opp}.csv")
    opp_deck = _read_deck(opp_path)

    g3 = OnnxGuide(args.net_a, ci, ai); g3.set_deck(us_deck, "alakazam")   # condition A
    g2 = OnnxGuide(args.net_b, ci, ai); g2.set_deck(us_deck, "alakazam")   # condition B
    print(f"US=alakazam piloted by {{{LA}, {LB}}} | OPP={args.opp} heuristic | "
          f"{'wall '+str(args.budget)+'s' if args.budget else f'sims={args.sims}'} k={args.k}")
    print("varying ONLY our net (templates held constant); opponent fixed heuristic\n")

    w = {"L3": 0, "L2": 0}
    n = {"L3": 0, "L2": 0}
    t0 = time.monotonic()
    for gi in range(args.games):
        cond = "L3" if gi % 2 == 0 else "L2"
        guide = g3 if cond == "L3" else g2
        us_seat = (args.seed + gi) % 2
        won = play_game(us_deck, opp_deck, ci, ai, us_seat, guide, args.sims, args.k, budget=args.budget)
        if won is None:
            continue
        n[cond] += 1
        w[cond] += int(won)
        if (gi + 1) % 20 == 0:
            p3 = w["L3"] / n["L3"] if n["L3"] else 0
            p2 = w["L2"] / n["L2"] if n["L2"] else 0
            print(f"  [{gi+1}/{args.games}] our winrate  {LA} {w['L3']}/{n['L3']}={p3:.1%}  "
                  f"{LB} {w['L2']}/{n['L2']}={p2:.1%}  ({time.monotonic()-t0:.0f}s)", flush=True)
    dt = time.monotonic() - t0

    p3 = w["L3"] / n["L3"] if n["L3"] else 0.0
    p2 = w["L2"] / n["L2"] if n["L2"] else 0.0
    pooled = (w["L3"] + w["L2"]) / (n["L3"] + n["L2"]) if (n["L3"] + n["L2"]) else 0.0
    se = (pooled * (1 - pooled) * (1 / max(n["L3"], 1) + 1 / max(n["L2"], 1))) ** 0.5
    z = (p3 - p2) / se if se else 0.0
    verdict = (f"{LA} BETTER vs field" if z > 1.64 else
               (f"{LA} WORSE vs field" if z < -1.64 else "no sig. diff"))
    print(f"\nNET-vs-FIELD | us=alakazam vs opp={args.opp} | {args.games} games in {dt:.0f}s\n"
          f"  {LA} as our pilot: {w['L3']}/{n['L3']} = {p3:.1%}\n"
          f"  {LB} as our pilot: {w['L2']}/{n['L2']} = {p2:.1%}\n"
          f"  delta ({LA} - {LB}) = {p3-p2:+.1%}  (z={z:+.2f})  ->  {verdict}")
    if args.out:
        json.dump({"opp": args.opp, "label_a": LA, "label_b": LB, "w": w, "n": n,
                   "p3": p3, "p2": p2, "delta": p3 - p2, "z": z,
                   "games": args.games, "budget": args.budget, "seconds": dt}, open(args.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

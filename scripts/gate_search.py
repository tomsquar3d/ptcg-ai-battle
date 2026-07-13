#!/usr/bin/env python3
"""#3 deeper search: does MORE compute (higher sims) alone lift win rate, or does it cap at leaf quality? (#60)

The study (#60) says the top-tier edge is in-game decision quality (compute), not tempo. Test whether simply
searching DEEPER helps: our Alakazam pilot = net_ladder2 at HIGH sims vs the SAME net at LOW sims, facing a
fixed heuristic field opponent. If HIGH wins materially, search depth is a lever; if not, play caps at the
net's value/leaf quality (the #10/#18/#30 lesson) and deeper search is not the answer.

    .venv/bin/python scripts/gate_search.py --opp mega-kangaskhan-ex --sims-hi 384 --sims-lo 96 --games 120
"""
import argparse, json, os, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
sys.path.insert(0, os.path.join(REPO, "src"))

from cg.api import to_observation_class, all_card_data, all_attack       # noqa: E402
from cg.game import battle_start, battle_finish, battle_select           # noqa: E402
from ptcg.opponent import OpponentTracker                                # noqa: E402
from search.determinizer import determinize                             # noqa: E402
from search import mcts                                                 # noqa: E402
from nn.onnx_guide import OnnxGuide                                      # noqa: E402

LADDER2 = os.path.join(REPO, "experiments", "checkpoints", "ladder2")
CAND = os.path.join(REPO, "experiments", "candidate_decks")
ALK = os.path.join(CAND, "alakazam-ladder.csv")


def _read(p): return [int(x) for x in open(p).read().split() if x.strip()]
def _legal_min(sel): return list(range(min(max(sel.minCount, 0), len(sel.option))))


def play_game(us_deck, opp_deck, ci, ai, guide, us_seat, us_sims, opp_sims, k, move_cap=20.0):
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
                result = cur.result; break
            sel = obs.select
            if sel is None:
                break
            mi = cur.yourIndex if cur is not None else 0
            deck = decks[mi]
            n = len(sel.option)
            if n == 0:
                obs_dict = battle_select([]); continue
            try:
                trackers[mi].observe(obs)
            except Exception:
                pass
            if sel.maxCount > 1:
                pick = _legal_min(sel)
            elif n == 1 and sel.minCount == 1:
                pick = [0]
            else:
                pick = None
                try:
                    worlds = determinize(obs, deck, ci, k=k, opp_tracker=trackers[mi])
                    deadline = time.monotonic() + move_cap
                    if mi == us_seat:
                        pick, _ = mcts.search(obs, worlds, ci, ai, sims=us_sims, deadline=deadline,
                                              prior_fn=guide.priors, value_fn=guide.value)
                    else:
                        pick, _ = mcts.search(obs, worlds, ci, ai, sims=opp_sims, deadline=deadline)
                except Exception:
                    pick = None
                if pick is None:
                    pick = _legal_min(sel) or [0]
            obs_dict = battle_select(pick)
    finally:
        battle_finish()
    return None if result == -1 else (result == us_seat)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", required=True)
    ap.add_argument("--sims-hi", type=int, default=384)
    ap.add_argument("--sims-lo", type=int, default=96)
    ap.add_argument("--opp-sims", type=int, default=96)
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    us = _read(ALK)
    opp = _read(ALK if args.opp == "alakazam" else os.path.join(CAND, f"{args.opp}.csv"))
    g = OnnxGuide(LADDER2, ci, ai); g.set_deck(us, "alakazam")
    print(f"net_ladder2 Alakazam vs {args.opp} | HI sims={args.sims_hi} vs LO sims={args.sims_lo} "
          f"(opp heuristic sims={args.opp_sims})\n")
    w = {"HI": 0, "LO": 0}; n = {"HI": 0, "LO": 0}; t0 = time.monotonic()
    for gi in range(args.games):
        cond = "HI" if gi % 2 == 0 else "LO"
        us_sims = args.sims_hi if cond == "HI" else args.sims_lo
        us_seat = (args.seed + gi) % 2
        won = play_game(us, opp, ci, ai, g, us_seat, us_sims, args.opp_sims, args.k)
        if won is None:
            continue
        n[cond] += 1; w[cond] += int(won)
        if (gi + 1) % 20 == 0:
            print(f"  [{gi+1}/{args.games}] HI {w['HI']}/{n['HI']}  LO {w['LO']}/{n['LO']}  "
                  f"({time.monotonic()-t0:.0f}s)", flush=True)
    ph = w["HI"]/max(n["HI"], 1); pl = w["LO"]/max(n["LO"], 1)
    pool = (w["HI"]+w["LO"])/max(n["HI"]+n["LO"], 1)
    se = (pool*(1-pool)*(1/max(n["HI"], 1)+1/max(n["LO"], 1)))**0.5
    z = (ph-pl)/se if se else 0
    print(f"\nDEEPER-SEARCH GATE | vs {args.opp} | {args.games} games\n"
          f"  HI sims {args.sims_hi}: {w['HI']}/{n['HI']} = {ph:.1%}\n"
          f"  LO sims {args.sims_lo}: {w['LO']}/{n['LO']} = {pl:.1%}\n"
          f"  delta = {ph-pl:+.1%} (z={z:+.2f}) -> {'DEEPER HELPS' if z>1.64 else ('deeper HURTS' if z<-1.64 else 'no sig diff (search caps at leaf quality)')}")
    if args.out:
        json.dump({"opp": args.opp, "w": w, "n": n, "ph": ph, "pl": pl, "delta": ph-pl, "z": z}, open(args.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

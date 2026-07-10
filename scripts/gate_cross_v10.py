#!/usr/bin/env python3
"""Decisive pre-ship gate for v10: does v10's determinizer beat v6's when the NET pilot FACES the field?

The mirror gate (gate_v10_vs_v6.py) came back 50/50 — but a mirror under-tests the determinizer change (both
sets have an Alakazam; the opponent IS Alakazam so MAP is usually right). The determinizer wins live vs the
FIELD. And crucially those wins were measured on the HEURISTIC pilot; the net may be more robust to
determinizer quality. So test the real thing: our Alakazam pilot = net_ladder2 (v6's net), FACING a real
field deck (heuristic-fixed), with our determinizer swapping v10-config (mined templates + posterior
sampling) vs v6-config (TrainerHill templates + MAP-collapse). Net + opponent held constant; only our
determinizer varies; conditions interleave; our seat alternates.

    .venv/bin/python scripts/gate_cross_v10.py --opp mega-kangaskhan-ex --games 160 --seed 0
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
from nn.onnx_guide import OnnxGuide                                                   # noqa: E402

LADDER2 = os.path.join(REPO, "experiments", "checkpoints", "ladder2")
NEW_PATH = os.path.join(REPO, "src", "ptcg", "archetype_templates.json")
OLD_PATH = os.path.join(REPO, "src", "ptcg", "archetype_templates.trainerhill.json")
CAND = os.path.join(REPO, "experiments", "candidate_decks")
ALAKAZAM_DECK = os.path.join(CAND, "alakazam-ladder.csv")


def _read_deck(path):
    return [int(x) for x in open(path).read().split() if x.strip()]


def _legal_min(sel):
    n = len(sel.option)
    return list(range(min(max(sel.minCount, 0), n)))


def _multi_pick(obs, mi, deck, ci, ai, tracker, arch, sample):
    sel = obs.select
    n = len(sel.option)
    if n == 0:
        return []
    try:
        worlds = determinize(obs, deck, ci, k=1, archetypes=arch, opp_tracker=tracker, sample_archetype=sample)
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


def play_game(us_deck, opp_deck, ci, ai, guide, us_seat, us_arch, us_sample,
              opp_arch, sims, k, move_cap=12.0, budget=None):
    """us_seat = net_ladder2 pilot with determinizer (us_arch, us_sample); opponent = heuristic with
    (opp_arch, MAP). Returns True iff 'us' won, None if undecided."""
    MAX_SIMS = 200_000
    decks = {us_seat: us_deck, 1 - us_seat: opp_deck}
    archs = {us_seat: us_arch, 1 - us_seat: opp_arch}
    samples = {us_seat: us_sample, 1 - us_seat: False}
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
            deck, arch, sample = decks[mi], archs[mi], samples[mi]
            n = len(sel.option)
            if n == 0:
                obs_dict = battle_select([])
                continue
            try:
                trackers[mi].observe(obs)
            except Exception:
                pass
            if sel.maxCount > 1:
                pick = _multi_pick(obs, mi, deck, ci, ai, trackers[mi], arch, sample)
            elif n == 1 and sel.minCount == 1:
                pick = [0]
            else:
                pick = None
                try:
                    worlds = determinize(obs, deck, ci, k=k, archetypes=arch, opp_tracker=trackers[mi],
                                         sample_archetype=sample)
                    n_sims = MAX_SIMS if budget else sims
                    deadline = time.monotonic() + (budget if budget else move_cap)
                    if mi == us_seat:                       # net-guided (net_ladder2)
                        pick, _st = mcts.search(obs, worlds, ci, ai, sims=n_sims, deadline=deadline,
                                                prior_fn=guide.priors, value_fn=guide.value)
                    else:                                   # opponent heuristic
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
    ap.add_argument("--opp", required=True)
    ap.add_argument("--games", type=int, default=160)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    us_deck = _read_deck(ALAKAZAM_DECK)
    opp_deck = _read_deck(ALAKAZAM_DECK if args.opp == "alakazam" else os.path.join(CAND, f"{args.opp}.csv"))
    guide = OnnxGuide(LADDER2, ci, ai); guide.set_deck(us_deck, "alakazam")
    arch_new, arch_old = load_archetypes(NEW_PATH), load_archetypes(OLD_PATH)
    print(f"US=alakazam/net_ladder2 vs OPP={args.opp}/heuristic | vary OUR determinizer: "
          f"V10(mined+sample) vs V6(trainerhill+MAP) | sims={args.sims} k={args.k}\n")

    w = {"V10": 0, "V6": 0}
    n = {"V10": 0, "V6": 0}
    t0 = time.monotonic()
    for gi in range(args.games):
        cond = "V10" if gi % 2 == 0 else "V6"
        us_arch = arch_new if cond == "V10" else arch_old
        us_sample = (cond == "V10")
        us_seat = (args.seed + gi) % 2
        won = play_game(us_deck, opp_deck, ci, ai, guide, us_seat, us_arch, us_sample, arch_new,
                        args.sims, args.k, budget=args.budget)
        if won is None:
            continue
        n[cond] += 1
        w[cond] += int(won)
        if (gi + 1) % 20 == 0:
            pv, p6 = w["V10"] / max(n["V10"], 1), w["V6"] / max(n["V6"], 1)
            print(f"  [{gi+1}/{args.games}] winrate  V10 {w['V10']}/{n['V10']}={pv:.1%}  "
                  f"V6 {w['V6']}/{n['V6']}={p6:.1%}  ({time.monotonic()-t0:.0f}s)", flush=True)
    dt = time.monotonic() - t0

    pv = w["V10"] / n["V10"] if n["V10"] else 0.0
    p6 = w["V6"] / n["V6"] if n["V6"] else 0.0
    pooled = (w["V10"] + w["V6"]) / (n["V10"] + n["V6"]) if (n["V10"] + n["V6"]) else 0.0
    se = (pooled * (1 - pooled) * (1 / max(n["V10"], 1) + 1 / max(n["V6"], 1))) ** 0.5
    z = (pv - p6) / se if se else 0.0
    verdict = ("V10 BETTER" if z > 1.64 else ("V10 WORSE" if z < -1.64 else "no sig. diff"))
    print(f"\nGATE-CROSS-v10 | net_ladder2 vs opp={args.opp} | {args.games} games in {dt:.0f}s\n"
          f"  V10 determinizer (mined+sample): {w['V10']}/{n['V10']} = {pv:.1%}\n"
          f"  V6  determinizer (trainerhill+MAP): {w['V6']}/{n['V6']} = {p6:.1%}\n"
          f"  delta (V10 - V6) = {pv-p6:+.1%}  (z={z:+.2f})  ->  {verdict}")
    if args.out:
        json.dump({"opp": args.opp, "w": w, "n": n, "pv": pv, "p6": p6, "delta": pv - p6,
                   "z": z, "games": args.games, "seconds": dt}, open(args.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

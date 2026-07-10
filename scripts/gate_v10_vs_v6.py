#!/usr/bin/env python3
"""Pre-ship gate: does v10's determinizer beat v6's, holding the net fixed at net_ladder2?

v10 = v6's net (net_ladder2) + mined templates (#46) + per-world posterior sampling (#49). v6 = net_ladder2
+ old TrainerHill templates + MAP-collapse. This mirror pits them with the NET held constant on both seats
(net_ladder2, deck-conditioned Alakazam), varying ONLY the determinizer config:
  seat A (v10): archetypes = mined set,      sample_archetype = True   (posterior sampling)
  seat B (v6):  archetypes = TrainerHill set, sample_archetype = False  (MAP-collapse)
Same Alakazam deck both seats; seats alternate. An A-win = v10's determinizer (both search wins together)
produced better play than v6's, on v6's own net — the offline stand-in for the v10-vs-v6 ladder question.

    .venv/bin/python scripts/gate_v10_vs_v6.py --games 200 --seed 0 --out g.json
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

LADDER2 = os.path.join(REPO, "experiments", "checkpoints", "ladder2")   # v6's net
NEW_PATH = os.path.join(REPO, "src", "ptcg", "archetype_templates.json")
OLD_PATH = os.path.join(REPO, "src", "ptcg", "archetype_templates.trainerhill.json")
ALAKAZAM_DECK = os.path.join(REPO, "experiments", "candidate_decks", "alakazam-ladder.csv")


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


def play_game(deck, ci, ai, guide, a_seat, arch_new, arch_old, sims, k, move_cap=12.0, budget=None):
    """Mirror; net_ladder2 (guide) both seats. Seat a_seat determinizes with (mined, sample=True) = v10;
    the other with (trainerhill, sample=False) = v6. Returns winner seat (0/1) or -1."""
    MAX_SIMS = 200_000
    trackers = {0: OpponentTracker(), 1: OpponentTracker()}
    result = -1
    obs_dict, _ = battle_start(list(deck), list(deck))
    if obs_dict is None:
        return -1
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
            is_v10 = (mi == a_seat)
            arch = arch_new if is_v10 else arch_old
            sample = is_v10                       # v10 samples the posterior; v6 MAP-collapses
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
                    pick, _st = mcts.search(obs, worlds, ci, ai, sims=n_sims, deadline=deadline,
                                            prior_fn=guide.priors, value_fn=guide.value)
                except Exception:
                    pick = None
                if pick is None:
                    pick = _legal_min(sel) or [0]
            obs_dict = battle_select(pick)
    finally:
        battle_finish()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    deck = _read_deck(ALAKAZAM_DECK)
    guide = OnnxGuide(LADDER2, ci, ai); guide.set_deck(deck, "alakazam")   # v6's net, both seats
    arch_new, arch_old = load_archetypes(NEW_PATH), load_archetypes(OLD_PATH)
    print(f"v10 (net_ladder2 + mined + sampling) vs v6 (net_ladder2 + trainerhill + MAP) | Alakazam mirror | "
          f"{'wall '+str(args.budget)+'s' if args.budget else f'sims={args.sims}'} k={args.k}\n")

    v10_w = v10_l = draw = 0
    t0 = time.monotonic()
    for g in range(args.games):
        a_seat = (args.seed + g) % 2
        winner = play_game(deck, ci, ai, guide, a_seat, arch_new, arch_old, args.sims, args.k, budget=args.budget)
        if winner == a_seat:
            v10_w += 1
        elif winner in (0, 1):
            v10_l += 1
        else:
            draw += 1
        if (g + 1) % 20 == 0:
            dec = v10_w + v10_l
            print(f"  [{g+1}/{args.games}] v10 {v10_w}/{v10_l}/{draw} = {v10_w/dec:.1%}  "
                  f"({time.monotonic()-t0:.0f}s)", flush=True)
    dt = time.monotonic() - t0

    dec = v10_w + v10_l
    wr = v10_w / dec if dec else 0.0
    se = (wr * (1 - wr) / dec) ** 0.5 if dec else 0.0
    z = (wr - 0.5) / se if se else 0.0
    verdict = ("v10 BETTER" if z > 1.64 else ("v10 WORSE" if z < -1.64 else "no sig. diff"))
    print(f"\nv10-vs-v6 DETERMINIZER GATE (net_ladder2 fixed) | {args.games} games in {dt:.0f}s\n"
          f"  v10 (mined+sampling) W/L/D = {v10_w}/{v10_l}/{draw} | win rate {wr:.1%} ± {se:.1%} (z={z:+.2f})\n"
          f"  -> {verdict}")
    if args.out:
        json.dump({"v10_w": v10_w, "v10_l": v10_l, "draw": draw, "games": args.games,
                   "wr": wr, "z": z, "seconds": dt}, open(args.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

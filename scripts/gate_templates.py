#!/usr/bin/env python3
"""Gate: do the RE-MINED archetype templates (#46) improve opponent modeling vs the stale TrainerHill set?

MIRROR game, same deck on both seats, **same heuristic MCTS pilot** (no net → no OOD confound). The ONLY
difference between the seats is which archetype template set their determinizer uses to model the opponent:
  seat A (NEW)  = src/ptcg/archetype_templates.json            (mined from the real ladder)
  seat B (OLD)  = src/ptcg/archetype_templates.trainerhill.json (the stale TrainerHill scrape)
Seats alternate. A NEW-side win = the mined templates produced a better opponent model → better search.

Run it on the decks we actually care about:
  --deck alakazam : our SHIP deck. Both sets contain an Alakazam template → conservative test.
  --deck sample   : the old ship (Mega Abomasnow). OLD set has NO Abomasnow template → phantom-deck fix.

    .venv/bin/python scripts/gate_templates.py --deck sample   --games 80 --seed 0
    .venv/bin/python scripts/gate_templates.py --deck alakazam --games 80 --seed 1 --budget 0.4
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
SAMPLE_DECK = os.path.join(REPO, "submissions", "v5", "deck.csv")
ALAKAZAM_DECK = os.path.join(REPO, "experiments", "candidate_decks", "alakazam-ladder.csv")


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


def play_game(deck, ci, ai, arch_new, arch_old, a_seat, sims, k, move_cap=12.0, budget=None):
    """Mirror; seat a_seat determinizes with arch_new, the other with arch_old. Heuristic MCTS both.
    Returns winner seat (0/1) or -1."""
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
            arch = arch_new if mi == a_seat else arch_old      # <-- the ONLY per-seat difference
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
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="sample", help="sample | alakazam | <path to deck.csv>")
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=None, help="seconds/move — equal-WALL-CLOCK mode")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    deck_path = {"sample": SAMPLE_DECK, "alakazam": ALAKAZAM_DECK}.get(args.deck, args.deck)
    deck = _read_deck(deck_path)

    arch_new = load_archetypes(NEW_PATH)
    arch_old = load_archetypes(OLD_PATH)
    print(f"NEW templates: {len(arch_new)} archetypes | OLD templates: {len(arch_old)} archetypes")
    print(f"deck={args.deck} ({len(deck)} cards) | mirror, heuristic MCTS both | "
          f"{'equal-wall-clock '+str(args.budget)+'s' if args.budget else f'sims={args.sims}'} k={args.k}\n")

    new_w = new_l = draw = 0
    t0 = time.monotonic()
    for g in range(args.games):
        a_seat = (args.seed + g) % 2
        winner = play_game(deck, ci, ai, arch_new, arch_old, a_seat, args.sims, args.k, budget=args.budget)
        if winner == a_seat:
            new_w += 1
        elif winner in (0, 1):
            new_l += 1
        else:
            draw += 1
        if (g + 1) % 10 == 0:
            dec = new_w + new_l
            wr = new_w / dec if dec else 0
            print(f"  [{g+1}/{args.games}] NEW {new_w}/{new_l}/{draw}  wr={wr:.1%}  "
                  f"({time.monotonic()-t0:.0f}s)", flush=True)
    dt = time.monotonic() - t0

    dec = new_w + new_l
    wr = new_w / dec if dec else 0.0
    se = (wr * (1 - wr) / dec) ** 0.5 if dec else 0.0
    z = (wr - 0.5) / se if se else 0.0
    print(f"\nTEMPLATES GATE | deck={args.deck} | {args.games} games in {dt:.0f}s\n"
          f"  NEW (mined) vs OLD (trainerhill): W/L/D = {new_w}/{new_l}/{draw}\n"
          f"  NEW win rate = {wr:.1%} ± {se:.1%}  (z={z:+.2f})   "
          f"{'NEW BETTER' if z > 1.64 else ('NEW WORSE' if z < -1.64 else 'TIE (no sig. diff)')}")
    if args.out:
        json.dump({"deck": args.deck, "new_w": new_w, "new_l": new_l, "draw": draw,
                   "games": args.games, "sims": args.sims, "k": args.k, "budget": args.budget,
                   "seconds": dt, "wr": wr, "z": z}, open(args.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

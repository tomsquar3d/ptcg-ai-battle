#!/usr/bin/env python3
"""Cross-deck / cross-pilot gate (docs §3d follow-up): a different DECK AND NET per seat.

Seat A pilots `--deck-a` with the deck-conditioned net at `--a-onnx` (e.g. net_ladder on Alakazam);
seat B pilots `--deck-b` with `--b-onnx` (e.g. v5 on sample). Each seat determinizes with its own deck
and searches under the same budget; seats alternate. This is the ship-decision test: does the ladder's
best deck piloted by a net that knows it beat our current ship (sample + v5)?

    .venv/bin/python scripts/gate_cross.py --deck-a experiments/candidate_decks/alakazam-ladder.csv \
        --archetype-a alakazam-ladder --a-onnx experiments/checkpoints/ladder \
        --deck-b submissions/v5/deck.csv --b-onnx experiments/checkpoints --budget 0.4 --games 20
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


def _read_deck(p):
    return [int(x) for x in open(p).read().split() if x.strip()]


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
                scored.append((mcts.leaf_value(search_step(root.searchId, [i]), mi, ci, ai), i))
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


def play_game(decks, guides, ci, ai, a_seat, sims, k, budget=None, move_cap=12.0):
    trackers = {0: OpponentTracker(), 1: OpponentTracker()}
    result = -1
    MAX_SIMS = 200_000
    obs_dict, _ = battle_start(decks[0], list(decks[1]))
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
            n = len(sel.option)
            if n == 0:
                obs_dict = battle_select([])
                continue
            try:
                trackers[mi].observe(obs)
            except Exception:
                pass
            if sel.maxCount > 1:
                pick = _multi_pick(obs, mi, decks[mi], ci, ai, trackers[mi])
            elif n == 1 and sel.minCount == 1:
                pick = [0]
            else:
                pick = None
                try:
                    worlds = determinize(obs, decks[mi], ci, k=k, opp_tracker=trackers[mi])
                    n_sims = MAX_SIMS if budget else sims
                    deadline = time.monotonic() + (budget if budget else move_cap)
                    pick, _st = mcts.search(obs, worlds, ci, ai, sims=n_sims, deadline=deadline,
                                            prior_fn=guides[mi].priors, value_fn=guides[mi].value)
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
    ap.add_argument("--deck-a", required=True)
    ap.add_argument("--archetype-a", default=None)
    ap.add_argument("--a-onnx", required=True)
    ap.add_argument("--deck-b", default=os.path.join(REPO, "submissions", "v5", "deck.csv"))
    ap.add_argument("--archetype-b", default="sample")
    ap.add_argument("--b-onnx", default=os.path.join(REPO, "experiments", "checkpoints"))
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    deck_a, deck_b = _read_deck(args.deck_a), _read_deck(args.deck_b)

    from nn.onnx_guide import OnnxGuide
    ga = OnnxGuide(args.a_onnx, ci, ai); ga.set_deck(deck_a, args.archetype_a)
    gb = OnnxGuide(args.b_onnx, ci, ai); gb.set_deck(deck_b, args.archetype_b)

    a_w = a_l = draw = 0
    t0 = time.monotonic()
    for g in range(args.games):
        a_seat = (args.seed + g) % 2
        decks = {a_seat: deck_a, 1 - a_seat: deck_b}
        guides = {a_seat: ga, 1 - a_seat: gb}
        w = play_game(decks, guides, ci, ai, a_seat, args.sims, args.k, budget=args.budget)
        if w == a_seat:
            a_w += 1
        elif w in (0, 1):
            a_l += 1
        else:
            draw += 1
    dt = time.monotonic() - t0
    dec = a_w + a_l
    wr = a_w / dec if dec else 0.0
    se = (wr * (1 - wr) / dec) ** 0.5 if dec else 0.0
    z = (wr - 0.5) / se if se else 0.0
    an = os.path.basename(args.deck_a).replace(".csv", "")
    print(f"A[{an}+{os.path.basename(args.a_onnx)}] vs B[{args.archetype_b}+v5] | {args.games} games in {dt:.0f}s\n"
          f"  A W/L/D = {a_w}/{a_l}/{draw} | win rate {wr:.1%} ± {se:.1%} (z={z:+.2f})")
    if args.out:
        json.dump({"a_w": a_w, "a_l": a_l, "draw": draw, "games": args.games, "seconds": dt},
                  open(args.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

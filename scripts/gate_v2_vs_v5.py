#!/usr/bin/env python3
"""Phase-5 gate (docs §3d step 5): the deck-conditioned v2 net vs the shipped v5 net, MIRROR.

Same deck on both seats, same determinize + MCTS budget (same sims, k, worlds) — the ONLY difference
is the guiding net: seat A = the v2 NetGuide (deck-conditioned, aux-trained, torch), seat B = the v5
OnnxGuide (the shipped net). Seats alternate. Isolates exactly the v2 net's contribution while piloting
`--archetype`'s deck.

Gate criteria:
  (a) --archetype dragapult-dusknoir : v2 must beat v5 MATERIALLY (the OOD deck it was trained on).
  (b) --archetype sample             : v2 must NOT regress vs v5 (the deck we ship).

EQUAL SIMS (not wall-clock): v2 is torch and v5 is ONNX, so equal wall-clock would handicap v2 on
inference speed, not play quality. This gate asks "does the v2 net play better per decision"; a v2
ONNX export for the equal-wall-clock deploy test is future work.

    .venv/bin/python scripts/gate_v2_vs_v5.py --archetype dragapult-dusknoir --benchmark
    .venv/bin/python scripts/gate_v2_vs_v5.py --archetype sample --games 20 --seed 0 --out logs/g_0.json
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

SAMPLE_DECK = os.path.join(REPO, "submissions", "v5", "deck.csv")
CAND = os.path.join(REPO, "experiments", "candidate_decks")


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


def play_game(deck, ci, ai, v2, v5, a_seat, sims, k, move_cap=12.0, budget=None):
    """Mirror game; seat a_seat guided by v2, other by v5. Returns winner seat (0/1)/-1 + sims tally.
    `budget` (seconds/move) switches to equal-WALL-CLOCK mode (both search to a deadline)."""
    MAX_SIMS = 200_000
    trackers = {0: OpponentTracker(), 1: OpponentTracker()}
    result = -1
    a_d = b_d = 0
    obs_dict, _ = battle_start(list(deck), list(deck))
    if obs_dict is None:
        return -1, {"ad": 0, "bd": 0}
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
                pick = _multi_pick(obs, mi, deck, ci, ai, trackers[mi])
            elif n == 1 and sel.minCount == 1:
                pick = [0]
            else:
                pick = None
                guide = v2 if mi == a_seat else v5
                try:
                    worlds = determinize(obs, deck, ci, k=k, opp_tracker=trackers[mi])
                    n_sims = MAX_SIMS if budget else sims
                    deadline = time.monotonic() + (budget if budget else move_cap)
                    pick, _st = mcts.search(obs, worlds, ci, ai, sims=n_sims, deadline=deadline,
                                            prior_fn=guide.priors, value_fn=guide.value)
                    if mi == a_seat:
                        a_d += 1
                    else:
                        b_d += 1
                except Exception:
                    pick = None
                if pick is None:
                    pick = _legal_min(sel) or [0]
            obs_dict = battle_select(pick)
    finally:
        battle_finish()
    return result, {"ad": a_d, "bd": b_d}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archetype", default="dragapult-dusknoir", help="deck both seats pilot")
    ap.add_argument("--deck", default=None)
    ap.add_argument("--v2-onnx", default=os.path.join(REPO, "experiments", "checkpoints", "v2"))
    ap.add_argument("--v5-onnx", default=os.path.join(REPO, "experiments", "checkpoints"))
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=float, default=None, help="seconds/move — equal-WALL-CLOCK mode")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    deck_path = args.deck or (SAMPLE_DECK if args.archetype == "sample"
                              else os.path.join(CAND, f"{args.archetype}.csv"))
    deck = _read_deck(deck_path)

    from nn.onnx_guide import OnnxGuide
    v2 = OnnxGuide(args.v2_onnx, ci, ai)                  # deck-conditioned net (ONNX, fast)
    v2.set_deck(deck, args.archetype)                     # deck-condition it on the piloted deck
    v5 = OnnxGuide(args.v5_onnx, ci, ai)                  # baseline net
    v5.set_deck(deck, args.archetype)                     # no-op for a v1 net; conditions a deck-cond one

    games = 2 if args.benchmark else args.games
    v2_w = v2_l = draw = 0
    t0 = time.monotonic()
    for g in range(games):
        a_seat = (args.seed + g) % 2
        winner, s = play_game(deck, ci, ai, v2, v5, a_seat, args.sims, args.k, budget=args.budget)
        if winner == a_seat:
            v2_w += 1
        elif winner in (0, 1):
            v2_l += 1
        else:
            draw += 1
        if args.benchmark:
            print(f"game {g}: winner={winner} v2_seat={a_seat} (v2 {s['ad']}d/v5 {s['bd']}d)  "
                  f"cum {time.monotonic()-t0:.1f}s")
    dt = time.monotonic() - t0

    decided = v2_w + v2_l
    wr = v2_w / decided if decided else 0.0
    se = (wr * (1 - wr) / decided) ** 0.5 if decided else 0.0
    z = (wr - 0.5) / se if se else 0.0
    print(f"v2 vs v5 | mirror {args.archetype} | sims={args.sims} k={args.k} | {games} games in {dt:.0f}s\n"
          f"  v2 W/L/D = {v2_w}/{v2_l}/{draw} | win rate {wr:.1%} ± {se:.1%} (z={z:+.2f})")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"v2_w": v2_w, "v2_l": v2_l, "draw": draw, "games": games,
                       "archetype": args.archetype, "sims": args.sims, "k": args.k, "seconds": dt}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())

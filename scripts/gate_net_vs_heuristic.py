#!/usr/bin/env python3
"""Acceptance gate: net-guided MCTS vs plain heuristic MCTS (the make-or-break test, docs §3).

Both seats run the SAME determinize + MCTS budget (same sims, same k, same worlds each turn) — the
ONLY difference is the leaf/prior source: the net seat uses `NetGuide` (policy prior + value head),
the heuristic seat uses the defaults (uniform prior + evaluator leaf). Seats alternate each game to
cancel first-player bias. This isolates exactly the trained net's contribution to playing strength.

    # time a couple of games, pick N:
    .venv/bin/python scripts/gate_net_vs_heuristic.py --benchmark --sims 128 --k 2

    # one worker's slice (writes a JSON tally to --out); fan out several with different --seed:
    .venv/bin/python scripts/gate_net_vs_heuristic.py --games 25 --sims 128 --k 2 \
        --seed 0 --out logs/gate_0.json

A crash/timeout in either pilot degrades to a legal fallback (mirrors the real agent's contract), so
the gate never aborts a game — a pilot that errors just plays a weak move and likely loses.
"""
import argparse
import json
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
sys.path.insert(0, os.path.join(REPO, "src"))

from cg.api import to_observation_class, all_card_data, all_attack, search_begin, search_step, search_end  # noqa: E402
from cg.game import battle_start, battle_finish, battle_select                                           # noqa: E402
from ptcg.opponent import OpponentTracker                                                                # noqa: E402
from search.determinizer import determinize                                                              # noqa: E402
from search import mcts                                                                                  # noqa: E402
from nn.guide import NetGuide                                                                            # noqa: E402


def _read_deck(path):
    return [int(x) for x in open(path).read().split() if x.strip()]


def _legal_min(sel):
    n = len(sel.option)
    return list(range(min(max(sel.minCount, 0), n)))


def _multi_pick(obs, my_index, deck, ci, ai, tracker):
    """Shared multi-select heuristic (identical for both seats — not part of the A/B)."""
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
                scored.append((mcts.leaf_value(child, my_index, ci, ai), i))
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


def play_game(deck, ci, ai, guide, net_seat, sims, k, move_cap=12.0):
    """Play one game; net_seat uses the net pilot, the other uses heuristic MCTS. Return winner idx
    (0/1) or -1, plus (net_decisions, heur_decisions) counts."""
    trackers = {0: OpponentTracker(), 1: OpponentTracker()}
    result = -1
    nd = hd = 0
    obs_dict, _ = battle_start(deck, list(deck))
    if obs_dict is None:
        return -1, (0, 0)
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
                try:
                    worlds = determinize(obs, deck, ci, k=k, opp_tracker=trackers[mi])
                    deadline = time.monotonic() + move_cap
                    if mi == net_seat:
                        pick, _ = mcts.search(obs, worlds, ci, ai, sims=sims, deadline=deadline,
                                              prior_fn=guide.priors, value_fn=guide.value)
                        nd += 1
                    else:
                        pick, _ = mcts.search(obs, worlds, ci, ai, sims=sims, deadline=deadline)
                        hd += 1
                except Exception:
                    pick = None
                if pick is None:
                    pick = _legal_min(sel) or [0]
            obs_dict = battle_select(pick)
    finally:
        battle_finish()
    return result, (nd, hd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--ckpt", default=os.path.join(REPO, "experiments", "checkpoints", "net.pt"))
    ap.add_argument("--deck", default=os.path.join(REPO, "submissions", "v4", "deck.csv"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    # Parallelism here comes from running many of these processes at once; each must be single-
    # threaded or torch's default intra-op pool (one thread per core, per process) oversubscribes
    # the machine and collapses throughput. (Also the deploy contract: 2 vCPU, few threads.)
    import torch
    torch.set_num_threads(1)

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    deck = _read_deck(args.deck)
    guide = NetGuide(args.ckpt, ci, ai)

    games = 2 if args.benchmark else args.games
    net_w = net_l = draw = 0
    t0 = time.monotonic()
    for g in range(games):
        net_seat = (args.seed + g) % 2                    # alternate seats
        winner, (nd, hd) = play_game(deck, ci, ai, guide, net_seat, args.sims, args.k)
        if winner == net_seat:
            net_w += 1
        elif winner in (0, 1):
            net_l += 1
        else:
            draw += 1
        if args.benchmark:
            dt = time.monotonic() - t0
            print(f"game {g}: winner={winner} net_seat={net_seat} "
                  f"(net {nd} / heur {hd} decisions)  cum {dt:.1f}s")
    dt = time.monotonic() - t0

    decided = net_w + net_l
    wr = net_w / decided if decided else 0.0
    se = (wr * (1 - wr) / decided) ** 0.5 if decided else 0.0
    line = (f"net-MCTS vs heuristic-MCTS | sims={args.sims} k={args.k} | "
            f"{games} games in {dt:.0f}s ({dt/games:.1f}s/game)\n"
            f"  net W/L/D = {net_w}/{net_l}/{draw} | win rate {wr:.1%} ± {se:.1%} (of decided)")
    print(line)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"net_w": net_w, "net_l": net_l, "draw": draw, "games": games,
                       "sims": args.sims, "k": args.k, "seconds": dt}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())

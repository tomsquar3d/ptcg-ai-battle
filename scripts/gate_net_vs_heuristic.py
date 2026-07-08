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


def play_game(deck, ci, ai, guide, net_seat, sims, k, move_cap=12.0, budget=None):
    """Play one game; net_seat uses the net pilot, the other uses heuristic MCTS. `budget` (seconds)
    switches to equal-WALL-CLOCK mode: both pilots search to a deadline (fixed sims otherwise). Return
    winner idx (0/1) or -1, plus a stats dict (decisions + sims each side)."""
    trackers = {0: OpponentTracker(), 1: OpponentTracker()}
    result = -1
    nd = hd = 0
    ns = hs = 0                          # total sims each side (to expose the net's speed cost)
    MAX_SIMS = 100_000
    obs_dict, _ = battle_start(deck, list(deck))
    if obs_dict is None:
        return -1, {"nd": 0, "hd": 0, "ns": 0, "hs": 0}
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
                    n_sims = MAX_SIMS if budget else sims
                    deadline = time.monotonic() + (budget if budget else move_cap)
                    if mi == net_seat:
                        pick, st = mcts.search(obs, worlds, ci, ai, sims=n_sims, deadline=deadline,
                                               prior_fn=guide.priors, value_fn=guide.value)
                        nd += 1; ns += st.get("sims", 0)
                    else:
                        pick, st = mcts.search(obs, worlds, ci, ai, sims=n_sims, deadline=deadline)
                        hd += 1; hs += st.get("sims", 0)
                except Exception:
                    pick = None
                if pick is None:
                    pick = _legal_min(sel) or [0]
            obs_dict = battle_select(pick)
    finally:
        battle_finish()
    return result, {"nd": nd, "hd": hd, "ns": ns, "hs": hs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--ckpt", default=os.path.join(REPO, "experiments", "checkpoints", "net.pt"))
    ap.add_argument("--deck", default=os.path.join(REPO, "submissions", "v4", "deck.csv"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--onnx", default=None, help="dir with value.onnx/policy.onnx (use OnnxGuide)")
    ap.add_argument("--budget", type=float, default=None,
                    help="seconds/move — equal-WALL-CLOCK mode (both pilots search to a deadline)")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    # Single-threaded: parallelism comes from many processes, and for a net this small intra-op
    # threads are pure overhead (also the deploy contract: 2 vCPU).
    import torch
    torch.set_num_threads(1)

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    deck = _read_deck(args.deck)
    if args.onnx:
        from nn.onnx_guide import OnnxGuide
        guide = OnnxGuide(args.onnx, ci, ai)
    else:
        guide = NetGuide(args.ckpt, ci, ai)

    games = 2 if args.benchmark else args.games
    net_w = net_l = draw = 0
    ns = hs = nd = hd = 0
    t0 = time.monotonic()
    for g in range(games):
        net_seat = (args.seed + g) % 2                    # alternate seats
        winner, s = play_game(deck, ci, ai, guide, net_seat, args.sims, args.k, budget=args.budget)
        ns += s["ns"]; hs += s["hs"]; nd += s["nd"]; hd += s["hd"]
        if winner == net_seat:
            net_w += 1
        elif winner in (0, 1):
            net_l += 1
        else:
            draw += 1
        if args.benchmark:
            dt = time.monotonic() - t0
            print(f"game {g}: winner={winner} net_seat={net_seat} "
                  f"(net {s['nd']}d/{s['ns']}sims  heur {s['hd']}d/{s['hs']}sims)  cum {dt:.1f}s")
    dt = time.monotonic() - t0

    decided = net_w + net_l
    wr = net_w / decided if decided else 0.0
    se = (wr * (1 - wr) / decided) ** 0.5 if decided else 0.0
    mode = f"budget={args.budget}s/move" if args.budget else f"sims={args.sims}"
    net_spm = ns / nd if nd else 0
    heur_spm = hs / hd if hd else 0
    print(f"net-MCTS vs heuristic-MCTS | {mode} k={args.k} | {games} games in {dt:.0f}s "
          f"({dt/games:.1f}s/game)\n"
          f"  avg sims/move: net {net_spm:.0f}  heuristic {heur_spm:.0f}\n"
          f"  net W/L/D = {net_w}/{net_l}/{draw} | win rate {wr:.1%} ± {se:.1%} (of decided)")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"net_w": net_w, "net_l": net_l, "draw": draw, "games": games,
                       "sims": args.sims, "budget": args.budget, "k": args.k, "seconds": dt,
                       "net_spm": net_spm, "heur_spm": heur_spm}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())

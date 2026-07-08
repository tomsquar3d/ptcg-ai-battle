#!/usr/bin/env python3
"""Deck A/B gate: the SAME net-MCTS pilot on both seats, a DIFFERENT deck each seat (docs §5).

The deck-construction arc (experiments #15-#23) closed under *heuristic* pilots: no constructed
meta deck beat the SDK `sample` deck, even under the v4 MCTS pilot (sample won ~73-81%). This
re-opens exactly that question under the STRONGER trained pilot: hold the pilot fixed (net-guided
MCTS, identical budget/k/worlds on both seats) and vary only the DECK. Seat A plays `--deck-a`
(the real meta list), seat B plays `--deck-b` (default: sample). Seats swap every game to cancel
first-player bias. Deck-A win rate is the number.

IMPORTANT confound to remember when reading the result: the net was trained on self-play with the
SAMPLE deck only, so piloting deck-a is OUT-OF-DISTRIBUTION for the policy/value heads. A deck-a
LOSS is therefore confounded (weaker deck OR a pilot that doesn't know the deck); a deck-a WIN,
despite that handicap, is a strong signal the deck is genuinely better.

    # time a couple of games first:
    .venv/bin/python scripts/gate_deck_ab.py --onnx experiments/checkpoints --benchmark --budget 0.4

    # one worker's slice (writes a JSON tally); fan out several --seed for N:
    .venv/bin/python scripts/gate_deck_ab.py --onnx experiments/checkpoints \
        --deck-a experiments/candidate_decks/dragapult-dusknoir.csv \
        --games 25 --budget 0.4 --seed 0 --out logs/deckab_0.json

A crash/timeout in either pilot degrades to a legal fallback (mirrors the agent contract), so the
gate never aborts a game.
"""
import argparse
import json
import os
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


def play_game(decks, ci, ai, guide, a_seat, sims, k, move_cap=12.0, budget=None):
    """Play one game. `decks` maps seat -> decklist; `a_seat` is the seat playing deck-a. BOTH seats
    use the net pilot (same budget/k). Return winner idx (0/1) or -1, plus sims/decisions per side."""
    trackers = {0: OpponentTracker(), 1: OpponentTracker()}
    result = -1
    ad = bd = 0
    a_s = b_s = 0                        # total sims per side (should be ~equal: same pilot+budget)
    MAX_SIMS = 100_000
    obs_dict, _ = battle_start(decks[0], decks[1])
    if obs_dict is None:
        return -1, {"ad": 0, "bd": 0, "as": 0, "bs": 0}
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

            deck = decks[mi]             # each seat determinizes with ITS OWN decklist
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
                    pick, st = mcts.search(obs, worlds, ci, ai, sims=n_sims, deadline=deadline,
                                           prior_fn=guide.priors, value_fn=guide.value)
                    if mi == a_seat:
                        ad += 1; a_s += st.get("sims", 0)
                    else:
                        bd += 1; b_s += st.get("sims", 0)
                except Exception:
                    pick = None
                if pick is None:
                    pick = _legal_min(sel) or [0]
            obs_dict = battle_select(pick)
    finally:
        battle_finish()
    return result, {"ad": ad, "bd": bd, "as": a_s, "bs": b_s}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--ckpt", default=os.path.join(REPO, "experiments", "checkpoints", "net.pt"))
    ap.add_argument("--onnx", default=None, help="dir with value.onnx/policy.onnx (use OnnxGuide)")
    ap.add_argument("--deck-a", default=os.path.join(REPO, "experiments", "candidate_decks",
                                                     "dragapult-dusknoir.csv"),
                    help="the real meta list (seat A)")
    ap.add_argument("--deck-b", default=os.path.join(REPO, "submissions", "v5", "deck.csv"),
                    help="the baseline deck (seat B) — default: v5 sample deck")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=float, default=None,
                    help="seconds/move — equal-WALL-CLOCK mode (both pilots search to a deadline)")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    deck_a = _read_deck(args.deck_a)
    deck_b = _read_deck(args.deck_b)
    if args.onnx:
        from nn.onnx_guide import OnnxGuide
        guide = OnnxGuide(args.onnx, ci, ai)
    else:
        from nn.guide import NetGuide
        guide = NetGuide(args.ckpt, ci, ai)

    games = 2 if args.benchmark else args.games
    a_w = a_l = draw = 0
    a_s = b_s = ad = bd = 0
    t0 = time.monotonic()
    for g in range(games):
        a_seat = (args.seed + g) % 2                     # alternate which seat plays deck-a
        decks = {a_seat: deck_a, 1 - a_seat: deck_b}
        winner, s = play_game(decks, ci, ai, guide, a_seat, args.sims, args.k, budget=args.budget)
        a_s += s["as"]; b_s += s["bs"]; ad += s["ad"]; bd += s["bd"]
        if winner == a_seat:
            a_w += 1
        elif winner in (0, 1):
            a_l += 1
        else:
            draw += 1
        if args.benchmark:
            dt = time.monotonic() - t0
            print(f"game {g}: winner={winner} a_seat={a_seat} "
                  f"(A {s['ad']}d/{s['as']}sims  B {s['bd']}d/{s['bs']}sims)  cum {dt:.1f}s")
    dt = time.monotonic() - t0

    decided = a_w + a_l
    wr = a_w / decided if decided else 0.0
    se = (wr * (1 - wr) / decided) ** 0.5 if decided else 0.0
    mode = f"budget={args.budget}s/move" if args.budget else f"sims={args.sims}"
    a_spm = a_s / ad if ad else 0
    b_spm = b_s / bd if bd else 0
    print(f"deck-A ({os.path.basename(args.deck_a)}) vs deck-B ({os.path.basename(args.deck_b)}) "
          f"| SAME net pilot | {mode} k={args.k} | {games} games in {dt:.0f}s ({dt/max(games,1):.1f}s/game)\n"
          f"  avg sims/move: A {a_spm:.0f}  B {b_spm:.0f}\n"
          f"  deck-A W/L/D = {a_w}/{a_l}/{draw} | win rate {wr:.1%} ± {se:.1%} (of decided)")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"a_w": a_w, "a_l": a_l, "draw": draw, "games": games, "sims": args.sims,
                       "budget": args.budget, "k": args.k, "seconds": dt, "a_spm": a_spm,
                       "b_spm": b_spm, "deck_a": args.deck_a, "deck_b": args.deck_b}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())

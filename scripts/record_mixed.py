#!/usr/bin/env python3
"""Drive the MIXED self-play recorder (docs §3d step 3): generate deck-conditioned training shards
across a deck pool × pilot pool, over all seat pairings.

A "side" = (archetype name, deck, pilot). Each game pairs two sides (seat 0 vs seat 1) and records
deck-conditioned branch decisions: MCTS pilots emit the visit distribution (soft π), scripted experts
emit a one-hot BC target (the `source` field tags which). Matchups cycle deterministically per
--seed so fanned-out workers cover different pairings.

Default pool (edit SIDES / pass --with-net):
  decks   : sample, dragapult-dusknoir
  pilots  : heuristic (v4 MCTS), net (v5 ONNX, with --onnx-dir), expert:dragapult-dusknoir
(basic-box-m is not yet built — add its deck under experiments/candidate_decks/ + register an expert.)

    .venv/bin/python scripts/record_mixed.py --benchmark --onnx-dir experiments/checkpoints
    .venv/bin/python scripts/record_mixed.py --games 30 --sims 160 --k 2 --seed 0 \
        --onnx-dir experiments/checkpoints --out-dir experiments/selfplay/mixed
One game per process — fan out several --seed for a buffer.
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
sys.path.insert(0, os.path.join(REPO, "src"))

from cg.api import all_card_data, all_attack                                         # noqa: E402
from selfplay.recorder import (record_mixed_game, build_pilot, save_shard, PI_DIM,   # noqa: E402
                               SOURCE_MCTS, SOURCE_BC)

SAMPLE_DECK = os.path.join(REPO, "submissions", "v5", "deck.csv")
CAND = os.path.join(REPO, "experiments", "candidate_decks")


def _read_deck(path):
    return [int(x) for x in open(path).read().split() if x.strip()]


def _default_sides(with_net: bool):
    """(archetype_name, deck_path, pilot_spec). Net sides only if --onnx-dir given."""
    sides = [
        ("sample", SAMPLE_DECK, "heuristic"),
        ("dragapult-dusknoir", os.path.join(CAND, "dragapult-dusknoir.csv"), "heuristic"),
        ("dragapult-dusknoir", os.path.join(CAND, "dragapult-dusknoir.csv"), "expert:dragapult-dusknoir"),
    ]
    if with_net:
        sides += [
            ("sample", SAMPLE_DECK, "net"),
            ("dragapult-dusknoir", os.path.join(CAND, "dragapult-dusknoir.csv"), "net"),
        ]
    return sides


def _check(tup: dict) -> str:
    d = tup["z"].shape[0]
    assert tup["visits"].shape == (d, PI_DIM)
    assert tup["pokemon_ids"].shape == (d, 12)
    assert tup["deck_summary"].shape[1:] == (28,) and tup["archetype_id"].shape == (d,)
    assert tup["source"].shape == (d,)
    assert set(np.unique(tup["z"]).tolist()) <= {-1.0, 0.0, 1.0}
    # aux value targets (docs §3d step 4)
    assert tup["prize_diff_t3"].shape == (d,) and tup["turns_to_attack"].shape == (d,)
    assert np.all(np.abs(tup["prize_diff_t3"]) <= 1.0) and np.all((0.0 <= tup["turns_to_attack"]) & (tup["turns_to_attack"] <= 1.0))
    vt = tup["visits"].sum(1)
    assert (vt > 0).all(), "a recorded decision has zero visits"
    om = tup["option_mask"]
    for i in range(d):
        top = int(tup["visits"][i, :64].argmax())
        if tup["visits"][i, top] >= tup["visits"][i, 64]:
            assert om[i, top], f"decision {i}: most-visited option masked out"
    nb = int((tup["source"] == SOURCE_BC).sum())
    nm = int((tup["source"] == SOURCE_MCTS).sum())
    return (f"{d} dec (MCTS {nm}/BC {nb}), z(+{int((tup['z']>0).sum())}/-{int((tup['z']<0).sum())}"
            f"/={int((tup['z']==0).sum())}), arch{sorted(set(tup['archetype_id'].tolist()))}, "
            f"prizeΔt3∈[{tup['prize_diff_t3'].min():+.2f},{tup['prize_diff_t3'].max():+.2f}] "
            f"tta μ={tup['turns_to_attack'].mean():.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--onnx-dir", default=None, help="dir with value.onnx/policy.onnx → enables net sides")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "experiments", "selfplay", "mixed"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}

    sides_cfg = _default_sides(with_net=bool(args.onnx_dir))
    # build each side's (name, deck, pilot) once; reuse pilots across games (net = ONNX session).
    sides = []
    for name, deck_path, spec in sides_cfg:
        pilot = build_pilot(spec, ci, ai, onnx_dir=args.onnx_dir)
        sides.append((name, _read_deck(deck_path), spec, pilot))
    pairs = [(i, j) for i in range(len(sides)) for j in range(len(sides))]   # all ordered seat pairings
    print(f"{len(sides)} sides → {len(pairs)} matchups: "
          + ", ".join(f"{n}/{s.split(':')[0]}" for n, _d, s, _p in sides))

    games = 2 if args.benchmark else args.games
    if not args.benchmark:
        os.makedirs(args.out_dir, exist_ok=True)
    total = 0
    for g in range(games):
        i, j = pairs[(args.seed * max(games, 1) + g) % len(pairs)]
        (na, da, sa, pa), (nb, db, sb, pb) = sides[i], sides[j]
        tup, st = record_mixed_game(
            {0: pa, 1: pb}, {0: da, 1: db}, {0: na, 1: nb}, ci, ai,
            k=args.k, sims=args.sims, seed=args.seed + g)
        tag = f"{na}:{sa.split(':')[0]}_vs_{nb}:{sb.split(':')[0]}"
        summary = _check(tup) if tup else "no branch decisions"
        print(f"game {g}: {st.seconds:5.1f}s winner={st.result} [{tag}] | {summary}")
        if tup and not args.benchmark:
            save_shard(os.path.join(args.out_dir, f"mix_s{args.seed}_{g:04d}.npz"), tup)
            total += st.decisions
    if args.benchmark:
        print("\nOK: mixed recorder produces well-formed deck-conditioned tuples.")
    else:
        print(f"\ndone: {games} games, {total} recorded decisions in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

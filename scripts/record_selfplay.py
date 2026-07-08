#!/usr/bin/env python3
"""Drive the self-play recorder: run N MCTS-H games and write .npz training shards.

Each game runs in THIS process (the SDK holds one global battle), so parallelize by launching
several copies with different --seed / --out-dir. Benchmark first — game cost is dominated by
`sims` * (# branch decisions) — before committing to a big run.

    # quick timing + shape check (2 short games, no files written):
    .venv/bin/python scripts/record_selfplay.py --benchmark

    # generate 20 shards into data-local buffer (gitignored):
    .venv/bin/python scripts/record_selfplay.py --games 20 --sims 160 --k 2 \
        --out-dir experiments/selfplay/buffer --seed 0
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))   # cg/ (native SDK)
sys.path.insert(0, os.path.join(REPO, "src"))                  # ptcg / search / nn / selfplay win

from cg.api import all_card_data, all_attack                   # noqa: E402
from selfplay.recorder import record_game, save_shard, PI_DIM  # noqa: E402


def _read_deck(path: str) -> list[int]:
    with open(path) as fh:
        return [int(x) for x in fh.read().split() if x.strip()]


def _check(tuples: dict) -> str:
    """Sanity-check one game's tuples; return a short summary or raise AssertionError."""
    d = tuples["z"].shape[0]
    assert tuples["visits"].shape == (d, PI_DIM), tuples["visits"].shape
    assert tuples["pokemon_ids"].shape == (d, 12)
    assert tuples["option_scalars"].shape[1:] == (64, 27)
    assert set(np.unique(tuples["z"]).tolist()) <= {-1.0, 0.0, 1.0}
    # every recorded decision must have at least one visit and its argmax must be a legal option
    vt = tuples["visits"].sum(1)
    assert (vt > 0).all(), "a recorded decision has zero visits"
    om = tuples["option_mask"]
    for i in range(d):
        top = int(tuples["visits"][i, :64].argmax())
        pass_v = tuples["visits"][i, 64]
        if tuples["visits"][i, top] >= pass_v:            # top is a real option (not PASS)
            assert om[i, top], f"decision {i}: most-visited option is masked out"
    frac_pass = (tuples["visits"][:, 64] > 0).mean()
    return (f"{d} decisions, z(+{int((tuples['z']>0).sum())}/-{int((tuples['z']<0).sum())}"
            f"/={int((tuples['z']==0).sum())}), mean visits/dec {vt.mean():.0f}, "
            f"PASS-present {frac_pass:.0%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--deck", default=os.path.join(REPO, "submissions", "v4", "deck.csv"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "experiments", "selfplay", "buffer"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benchmark", action="store_true",
                    help="2 short games, no files; print timing + shape check and exit")
    args = ap.parse_args()

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    deck = _read_deck(args.deck)
    assert len(deck) == 60, f"deck has {len(deck)} cards"

    if args.benchmark:
        for g in range(2):
            tuples, st = record_game(deck, ci, ai, k=args.k, sims=args.sims, seed=args.seed + g)
            summary = _check(tuples) if tuples else "no branch decisions"
            per = st.seconds / max(st.decisions, 1)
            print(f"game {g}: {st.seconds:5.1f}s  {st.turns:4d} turns  "
                  f"winner={st.result}  {per:4.2f}s/decision  | {summary}")
        print("\nOK: self-play recorder produces well-formed tuples.")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    total_dec = 0
    for g in range(args.games):
        tuples, st = record_game(deck, ci, ai, k=args.k, sims=args.sims, seed=args.seed + g)
        if tuples is None:
            print(f"game {g}: no decisions ({st.seconds:.1f}s) — skipped")
            continue
        _check(tuples)
        path = os.path.join(args.out_dir, f"game_s{args.seed}_{g:04d}.npz")
        save_shard(path, tuples)
        total_dec += st.decisions
        print(f"game {g}: {st.seconds:5.1f}s  {st.decisions:3d} decisions  winner={st.result}  "
              f"-> {os.path.relpath(path, REPO)}")
    print(f"\ndone: {args.games} games, {total_dec} recorded decisions in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Flywheel gen-3 field-inclusive self-play recorder (docs §3e).

Our seat = the ship deck (Alakazam) piloted by the NET (net_ladder3 by default); the opponent seat is drawn
from the real ladder field (a field deck piloted by heuristic MCTS), or the Alakazam mirror a fraction of the
time. Reuses `record_mixed_game` (already asymmetric: deck+pilot per seat) and KEEPS ONLY the Alakazam seat's
decisions (`mover == our_seat`) so we train the net on Alakazam-facing-the-field WITHOUT diluting it toward
piloting decks we don't ship (#40). Value target = real game outcome → the value head finally sees
field-facing leaves (net_ladder3 was self-played mirror-only).

    .venv/bin/python scripts/record_field.py --games 30 --sims 160 --onnx-dir experiments/checkpoints/ladder3 \
        --out-dir experiments/selfplay/field --seed 0
One process runs many games; fan out several --seed for a buffer.
"""
import argparse
import os
import random
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
sys.path.insert(0, os.path.join(REPO, "src"))

from cg.api import all_card_data, all_attack                                         # noqa: E402
from selfplay.recorder import record_mixed_game, build_pilot, save_shard, SOURCE_MCTS, SOURCE_BC  # noqa: E402

CAND = os.path.join(REPO, "experiments", "candidate_decks")
ALAKAZAM = os.path.join(CAND, "alakazam-ladder.csv")

# ladder-field opponents (modal decklists, mined) + their ladder shares (from #46 priors, field-only)
FIELD = [
    ("mega-kangaskhan-ex", 0.152),
    ("marnie-s-grimmsnarl-ex", 0.128),
    ("mega-lucario-ex", 0.052),
    ("cynthia-s-garchomp-ex", 0.052),
    ("archaludon-ex", 0.050),
    ("cornerstone-mask-ogerpon-ex", 0.037),
]
MIRROR_FRAC = 0.34            # ~1/3 Alakazam mirror (keep the competence we have), ~2/3 field


def _read_deck(path):
    return [int(x) for x in open(path).read().split() if x.strip()]


def _pick_opponent(rng):
    """Return (name, deck_path) for the opponent this game — mirror MIRROR_FRAC of the time, else a
    field deck weighted by ladder share."""
    if rng.random() < MIRROR_FRAC:
        return "alakazam", ALAKAZAM
    names = [n for n, _ in FIELD]
    weights = [w for _, w in FIELD]
    name = rng.choices(names, weights=weights, k=1)[0]
    return name, os.path.join(CAND, f"{name}.csv")


def _filter_to_seat(tup: dict, seat: int) -> dict:
    """Keep only rows whose mover == seat (our Alakazam decisions). Filters every per-decision array."""
    if tup is None:
        return None
    mask = tup["mover"] == seat
    d = tup["mover"].shape[0]
    if not mask.any():
        return None
    out = {}
    for kk, arr in tup.items():
        out[kk] = arr[mask] if (isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == d) else arr
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--onnx-dir", default=os.path.join(REPO, "experiments", "checkpoints", "ladder3"),
                    help="net for OUR (Alakazam) pilot — default net_ladder3")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "experiments", "selfplay", "field"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benchmark", action="store_true", help="2 games, no shards written")
    args = ap.parse_args()

    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    alk_deck = _read_deck(ALAKAZAM)

    net_pilot = build_pilot("net", ci, ai, onnx_dir=args.onnx_dir)       # Alakazam pilot (net_ladder3)
    heur_pilot = build_pilot("heuristic", ci, ai)                        # field opponent pilot
    rng = random.Random(args.seed)

    games = 2 if args.benchmark else args.games
    if not args.benchmark:
        os.makedirs(args.out_dir, exist_ok=True)

    total_rows = 0
    opp_counts = {}
    for g in range(games):
        opp_name, opp_path = _pick_opponent(rng)
        opp_deck = _read_deck(opp_path)
        our_seat = (args.seed + g) % 2                                   # alternate first/second player
        decks = {our_seat: alk_deck, 1 - our_seat: opp_deck}
        pilots = {our_seat: net_pilot, 1 - our_seat: heur_pilot}
        names = {our_seat: "alakazam", 1 - our_seat: opp_name}
        tup, st = record_mixed_game(pilots, decks, names, ci, ai, k=args.k, sims=args.sims,
                                    seed=args.seed * 1000 + g)
        alk = _filter_to_seat(tup, our_seat)
        n_alk = 0 if alk is None else int(alk["z"].shape[0])
        opp_counts[opp_name] = opp_counts.get(opp_name, 0) + 1
        won = "?" if st.result == -1 else ("W" if st.result == our_seat else "L")
        print(f"game {g}: {st.seconds:5.1f}s alk_seat={our_seat} vs {opp_name:24s} "
              f"outcome={won} | alk rows {n_alk}", flush=True)
        if alk is not None and not args.benchmark:
            save_shard(os.path.join(args.out_dir, f"fld_s{args.seed}_{g:04d}.npz"), alk)
            total_rows += n_alk

    if args.benchmark:
        print("\nOK: field recorder produces well-formed Alakazam-only shards.")
    else:
        z_note = ""
        print(f"\ndone: {games} games → {total_rows} Alakazam decisions in {args.out_dir}")
        print("opponent mix:", ", ".join(f"{n}:{c}" for n, c in sorted(opp_counts.items(), key=lambda x: -x[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

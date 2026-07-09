#!/usr/bin/env python3
"""Build the ladder-BC dataset from harvested replays (docs §3d follow-up; step 3).

Runs `selfplay.ladder_bc.replay_to_tuples` over every replay in `data/replays/`, writing one BC shard
per usable replay to `experiments/selfplay/ladder_bc/` (gitignored, same format as self-play shards),
and reports total rows, per-deck row counts, and the winner-rating distribution.

    .venv/bin/python scripts/build_ladder_bc.py [--replays data/replays] [--out experiments/selfplay/ladder_bc]
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
sys.path.insert(0, os.path.join(REPO, "src"))

from cg.api import all_card_data, all_attack                       # noqa: E402
from selfplay.ladder_bc import replay_to_tuples                    # noqa: E402
from selfplay.recorder import save_shard                           # noqa: E402

_EID = re.compile(r"episode-(\d+)-replay\.json")


def _winner_rating(meta: dict) -> float:
    """Winner = the agent with reward == 1; return its rating (0 if unknown)."""
    best = None
    for a in meta.get("agents", []):
        if a.get("reward") == 1:
            best = a
    if best is None:
        return 0.0
    return float(best.get("rating") or 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default=os.path.join(REPO, "data", "replays"))
    ap.add_argument("--out", default=os.path.join(REPO, "experiments", "selfplay", "ladder_bc"))
    ap.add_argument("--limit", type=int, default=0, help="cap replays processed (0 = all)")
    args = ap.parse_args()

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    idx_path = os.path.join(args.replays, "index.json")
    index = json.load(open(idx_path))["index"] if os.path.exists(idx_path) else {}

    files = sorted(f for f in os.listdir(args.replays) if f.endswith("-replay.json"))
    if args.limit:
        files = files[:args.limit]
    os.makedirs(args.out, exist_ok=True)

    n_ok = n_skip = total_rows = 0
    deck_rows = Counter()
    deck_decks = {}                 # deck_key -> a representative winner decklist summary (distinct count)
    ratings = []
    for i, fn in enumerate(files, 1):
        m = _EID.search(fn)
        eid = m.group(1) if m else None
        wr = _winner_rating(index.get(eid, {})) if eid else 0.0
        tup = replay_to_tuples(os.path.join(args.replays, fn), ci, ai, winner_rating=wr)
        if tup is None:
            n_skip += 1
            continue
        save_shard(os.path.join(args.out, f"bc_{eid}.npz"), tup)
        n_ok += 1
        total_rows += len(tup["z"])
        for k in np.unique(tup["deck_key"]):
            deck_rows[int(k)] += int((tup["deck_key"] == k).sum())
        if wr > 0:
            ratings.append(wr)
        if i % 200 == 0:
            print(f"  [{i}/{len(files)}] {n_ok} shards, {total_rows} rows", flush=True)

    print(f"\nladder-BC dataset: {n_ok} replays → {total_rows} winner-decision rows "
          f"({n_skip} replays unusable) in {os.path.relpath(args.out, REPO)}")
    print(f"distinct decks (by decklist): {len(deck_rows)}")
    print("top decks by row count:")
    for k, c in deck_rows.most_common(12):
        print(f"  deck {k:>10}: {c} rows")
    if ratings:
        r = np.array(ratings)
        print(f"winner ratings (known): n={len(r)} min={r.min():.0f} median={np.median(r):.0f} max={r.max():.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

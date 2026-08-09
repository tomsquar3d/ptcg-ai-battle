#!/usr/bin/env python3
"""Pack a directory of one-game-per-shard `.npz` files into a few large `.npz` files.

For Kaggle. The v9 corpus is ~2,500 (soon ~5,000) tiny shards; uploading that as a dataset and then
globbing + `np.load`-ing it per game is slow at both ends. Packing keeps the byte count essentially
the same but turns thousands of files into a handful.

The `game_id` column rides along, which is what keeps the GAME-level train/val split honest after
packing — `dataset.load_buffer` groups on it (it falls back to the file index only for old buffers
that predate the column). Never pack without it: every packed row would land in one group and
correlated decisions from the same game would straddle the split.

    .venv/bin/python scripts/pack_corpus.py \
        --in experiments/selfplay/v9_corpus --out experiments/selfplay/v9_packed --per-file 2000
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--per-file", type=int, default=2000, help="games per packed .npz")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.src, "*.npz")))
    if not paths:
        raise SystemExit(f"no .npz shards in {args.src}")
    os.makedirs(args.dst, exist_ok=True)

    written = rows_total = games_total = skipped = 0
    for part, start in enumerate(range(0, len(paths), args.per_file)):
        cols = defaultdict(list)
        keys = None
        games = 0
        for p in paths[start:start + args.per_file]:
            try:
                d = np.load(p)
            except Exception:
                skipped += 1
                continue
            if int(d["z"].shape[0]) == 0:
                continue
            if "game_id" not in d.files:
                raise SystemExit(f"{os.path.basename(p)} has no game_id column — packing it would "
                                 f"collapse the game-level split; pack only recorder output that has one")
            if keys is None:
                keys = set(d.files)
            elif set(d.files) != keys:
                raise SystemExit(f"{os.path.basename(p)} key set differs from the rest of the pack")
            for k in d.files:
                cols[k].append(d[k])
            games += 1
        if not games:
            continue
        packed = {k: np.concatenate(v) for k, v in cols.items()}
        n = int(packed["z"].shape[0])
        out = os.path.join(args.dst, f"pack_{part:03d}.npz")
        np.savez_compressed(out, **packed)
        u = len(np.unique(packed["game_id"]))
        assert u == games, f"{out}: {u} distinct game_id from {games} shards — ids are not unique"
        print(f"  {os.path.basename(out)}: {games} games, {n:,} rows, "
              f"{os.path.getsize(out)/1e6:.1f} MB")
        written += 1
        rows_total += n
        games_total += games

    src_bytes = sum(os.path.getsize(p) for p in paths)
    dst_bytes = sum(os.path.getsize(p) for p in glob.glob(os.path.join(args.dst, "*.npz")))
    print(f"\npacked {games_total:,} games / {rows_total:,} rows: {len(paths):,} files "
          f"({src_bytes/1e6:.1f} MB) -> {written} files ({dst_bytes/1e6:.1f} MB)"
          + (f"; {skipped} unreadable shards skipped" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

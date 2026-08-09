#!/usr/bin/env python3
"""Build the frozen card-attribute matrix for the v9 compositional card embedding (Option C).

    e_c = W @ attr(c) + r_c

`attr(c)` is what this script produces: a hand-built [VOCAB_SIZE, 100] float32 matrix derived from
`data/cards/EN_Card_Data.csv`. W is learned and shared across every card, so a card the self-play
corpus never shows still gets a meaningful vector from its attributes instead of a dead random
embedding row. `r_c` is the learned per-card residual (see `nn.encoder._CompositionalCardEmbedding`).

It also emits `residual_index`: card id -> row in the residual table, ranked by how often the card
actually appears in a self-play corpus. Cards outside the top-K map to row 0, a frozen zero row, so
they get a pure compositional embedding and cost no parameters. That is how `cfg.n_residual_cards`
gets sized from the corpus frequency curve (K+1).

    .venv/bin/python scripts/build_card_features.py \
        --corpus experiments/selfplay/v9_corpus --top-k 64 \
        --out experiments/checkpoints/card_features.npz

ID MAPPING: the CSV's `Card ID` column is the engine card id — verified 1:1 (both sides are exactly
the 1267 ids 1..1267, no gaps, no extras). The CSV has ~2022 rows because a card gets one row per
move (up to 3), so rows are aggregated per card id.

IP: the CSV is competition-licensed and gitignored, and so is the .npz this writes. It is derived
card data — do NOT commit it or ship it in a submission bundle. The v9 deploy path bakes the
resulting embedding table into the ONNX weights (`PTCGNet.materialize_card_embed`), so no card data
ever leaves this machine.
"""
import argparse
import csv
import glob
import os
import sys
from collections import Counter, defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from nn import features as F                                             # noqa: E402

ATTR_DIM = 100

STAGES = ["Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon", "Item", "Supporter",
          "Pokémon Tool", "Stadium", "Special Energy", "Basic Energy"]
RULES = ["n/a", "Pokémon ex", "Mega Pokémon ex", "ACE SPEC"]
ENERGY = ["{G}", "{R}", "{W}", "{L}", "{P}", "{F}", "{D}", "{M}", "{C}", "竜"]   # + "other"
WEAK = ["{G}", "{R}", "{W}", "{L}", "{P}", "{F}", "{D}", "{M}", "{C}"]
KEYWORDS = ["draw", "search", "discard", "heal", "switch", "attach", "deck", "hand", "prize",
            "knocked out", "paralyz", "asleep", "confus", "poison", "burn", "retreat", "ability",
            "coin", "damage counter", "prevent", "evolv", "bench", "opponent", "cannot",
            "energy", "shuffle", "return", "put"]
HP_BUCKETS = [60, 90, 120, 160, 200, 250, 300]      # 8 buckets with the open top


def _num(s, default=0.0):
    s = (s or "").strip()
    if not s or s == "n/a":
        return default
    keep = "".join(c for c in s if c.isdigit())
    return float(keep) if keep else default


def _onehot_at(v, start, value, table):
    """Set the slot matching `value`; unknown/absent values leave the block all-zero."""
    if value in table:
        v[start + table.index(value)] = 1.0


def card_attr(rows) -> np.ndarray:
    """Aggregate every CSV row for one card id into the fixed ATTR_DIM attribute vector."""
    r0 = rows[0]
    v = []

    def block(n):
        v.extend([0.0] * n)
        return len(v) - n

    stage = r0["Stage (Pokémon)/Type (Energy and Trainer)"].strip()
    i = block(len(STAGES)); _onehot_at(v, i, stage, STAGES)
    i = block(len(RULES)); _onehot_at(v, i, r0["Rule"].strip(), RULES)

    cat = r0["Category"].strip()
    v.extend([float("Trainer's Pokémon" in cat), float(cat.startswith("Tera")),
              float(cat in ("Ancient", "Future")), float(cat == "Fossil")])

    i = block(len(ENERGY) + 1)
    etype = r0["Type"].strip()
    if etype in ENERGY:
        v[i + ENERGY.index(etype)] = 1.0
    elif etype and etype != "n/a":
        v[i + len(ENERGY)] = 1.0                      # "other" (multi-symbol / special)

    i = block(len(WEAK)); _onehot_at(v, i, r0["Weakness"].strip(), WEAK)

    res = r0["Resistance (Type)"].strip()
    v.extend([float(res == "{F}"), float(res == "{G}"),
              float(bool(res) and res not in ("n/a", "{F}", "{G}"))])

    retreat_raw = r0["Retreat"].strip()
    i = block(5)
    if retreat_raw.isdigit():
        v[i + min(int(retreat_raw), 4)] = 1.0
    v.append(min(_num(retreat_raw), 4.0) / 4.0)

    hp = _num(r0["HP"])
    v.append(min(hp, 340.0) / 340.0)
    i = block(len(HP_BUCKETS) + 1)
    if hp > 0:
        b = len(HP_BUCKETS)
        for j, hi in enumerate(HP_BUCKETS):
            if hp <= hi:
                b = j
                break
        v[i + b] = 1.0

    prev = r0["Previous stage"].strip()
    v.append(float(bool(prev) and prev != "n/a"))

    moves = [r for r in rows if (r["Move Name"] or "").strip() not in ("", "n/a")]
    dmgs = [_num(r["Damage"]) for r in moves]
    dmgs = [d for d in dmgs if d > 0]
    costs = [len([c for c in (r["Cost"] or "") if c.strip()]) for r in moves]
    costs = [c for c in costs if c > 0]
    v.append(min(len(moves), 3) / 3.0)
    v.append(min(max(dmgs, default=0.0), 300.0) / 300.0)
    v.append(min(sum(dmgs) / len(dmgs) if dmgs else 0.0, 300.0) / 300.0)
    v.append(float(any(("×" in (r["Damage"] or "") or "+" in (r["Damage"] or "")) for r in moves)))
    v.append(min(max(costs, default=0), 5) / 5.0)
    v.append(min(min(costs, default=0), 5) / 5.0)
    v.append(float(bool(moves) and all(set(r["Cost"] or "") <= set("● ") for r in moves)))

    v.append(float("Pokémon" in stage))
    v.append(float(stage in ("Item", "Supporter", "Pokémon Tool", "Stadium")))
    v.append(float("Energy" in stage))

    text = " ".join((r["Effect Explanation"] or "") for r in rows).lower()
    v.append(float("ability" in text))
    v.append(min(len(text), 600) / 600.0)
    v.append(1.0)                                       # always-on bias feature
    for kw in KEYWORDS:
        v.append(float(kw in text))

    v.append(0.0)                                       # [97] is_unknown — set by the caller, not the CSV
    v.extend([0.0, 0.0])                                # [98:100] reserved

    assert len(v) == ATTR_DIM, f"attr vector is {len(v)}, expected {ATTR_DIM}"
    return np.asarray(v, dtype=np.float32)


def corpus_frequency(corpus_dir: str) -> Counter:
    """Count card-id occurrences across every id field of every shard (masked, PAD excluded)."""
    freq = Counter()
    paths = sorted(glob.glob(os.path.join(corpus_dir, "*.npz")))
    for p in paths:
        try:
            d = np.load(p)
        except Exception:
            continue
        for ids_key, mask_key in (("pokemon_ids", "pokemon_mask"), ("hand_ids", "hand_mask"),
                                  ("option_card_ids", "option_mask")):
            if ids_key not in d.files:
                continue
            x = d[ids_key][d[mask_key]] if mask_key in d.files else d[ids_key].ravel()
            x = x[x != F.PAD_ID]
            if x.size:
                u, c = np.unique(x, return_counts=True)
                freq.update(dict(zip(u.tolist(), c.tolist())))
        if "stadium_id" in d.files:
            s = d["stadium_id"]
            s = s[s != F.PAD_ID]
            if s.size:
                u, c = np.unique(s, return_counts=True)
                freq.update(dict(zip(u.tolist(), c.tolist())))
    return freq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(REPO, "data", "cards", "EN_Card_Data.csv"))
    ap.add_argument("--corpus", default=os.path.join(REPO, "experiments", "selfplay", "v9_corpus"),
                    help="self-play dir used to rank cards for the residual table")
    ap.add_argument("--top-k", type=int, default=64,
                    help="cards that get their own residual row; the rest share the frozen zero row")
    ap.add_argument("--out", default=os.path.join(REPO, "experiments", "checkpoints", "card_features.npz"))
    args = ap.parse_args()

    by_id = defaultdict(list)
    with open(args.csv, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            cid = (row.get("Card ID") or "").strip()
            if cid.isdigit():
                by_id[int(cid)].append(row)
    print(f"csv: {sum(len(v) for v in by_id.values())} rows -> {len(by_id)} distinct card ids "
          f"({min(by_id)}..{max(by_id)})")
    assert max(by_id) < F.VOCAB_SIZE, f"card id {max(by_id)} >= VOCAB_SIZE {F.VOCAB_SIZE}"

    attrs = np.zeros((F.VOCAB_SIZE, ATTR_DIM), dtype=np.float32)
    for cid, rows in by_id.items():
        attrs[cid] = card_attr(rows)
    # PAD stays all-zero (so e_PAD == 0, matching the old padding_idx row). UNKNOWN has no CSV row,
    # so without a marker it would collide with PAD — give it its own indicator dimension.
    attrs[F.PAD_ID] = 0.0
    attrs[F.UNKNOWN_ID] = 0.0
    attrs[F.UNKNOWN_ID, 97] = 1.0

    freq = corpus_frequency(args.corpus) if args.corpus and os.path.isdir(args.corpus) else Counter()
    ranked = [c for c, _ in freq.most_common() if 0 < c < F.VOCAB_SIZE]
    if F.UNKNOWN_ID not in ranked:                 # always worth a residual row
        ranked.append(F.UNKNOWN_ID)
    keep = ranked[:max(0, args.top_k)]
    residual_index = np.zeros(F.VOCAB_SIZE, dtype=np.int64)
    for row, cid in enumerate(keep, start=1):      # row 0 is the frozen zero row
        residual_index[cid] = row
    n_residual_cards = len(keep) + 1

    covered = sum(freq[c] for c in keep)
    total = sum(freq.values()) or 1
    print(f"corpus: {len(freq)} distinct ids observed, {total:,} occurrences")
    print(f"residual table: top-{len(keep)} cards get a row -> n_residual_cards={n_residual_cards} "
          f"({covered/total:.4%} of observed occurrences covered)")
    if len(freq) and args.top_k > len(ranked):
        print(f"  NOTE: --top-k {args.top_k} exceeds the {len(ranked)} ids the corpus has ever shown; "
              f"table truncated to those (an unseen card gets no residual to learn from anyway)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, attrs=attrs, residual_index=residual_index,
                        n_residual_cards=np.int64(n_residual_cards),
                        attr_dim=np.int64(ATTR_DIM), vocab_size=np.int64(F.VOCAB_SIZE))
    nz = int((attrs.any(axis=1)).sum())
    print(f"wrote {os.path.relpath(args.out, REPO)}: attrs {attrs.shape} "
          f"({nz} non-zero rows), residual_index {residual_index.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

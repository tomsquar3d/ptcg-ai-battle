#!/usr/bin/env python3
"""Extract the MODAL real decklist for given archetypes from harvested replays (for cross-deck gating).

For each requested archetype slug, find the single most-frequent exact 60-card decklist among replays
whose marquee (same rule as the miner) matches, and write it to experiments/candidate_decks/<slug>.csv.
These are real submitted decks → guaranteed legal, unlike the averaged templates.

    .venv/bin/python scripts/extract_decks.py mega-kangaskhan-ex marnie-s-grimmsnarl-ex cynthia-s-garchomp-ex
"""
import json
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from cg.api import all_card_data                                      # noqa: E402
from mine_archetypes import _marquee, _recover, _slug, REPLAYS        # noqa: E402

OUTDIR = os.path.join(REPO, "experiments", "candidate_decks")


def main() -> int:
    want = set(sys.argv[1:]) or {"mega-kangaskhan-ex", "marnie-s-grimmsnarl-ex", "cynthia-s-garchomp-ex"}
    ci = {c.cardId: c for c in all_card_data()}
    lists = defaultdict(Counter)          # slug -> Counter of exact deck tuples
    for fn in os.listdir(REPLAYS):
        if not fn.endswith("-replay.json"):
            continue
        try:
            d = json.load(open(os.path.join(REPLAYS, fn)))
        except Exception:
            continue
        steps = d.get("steps")
        if not steps:
            continue
        for seat in (0, 1):
            deck = _recover(steps, seat)
            if not deck:
                continue
            slug = _slug(_marquee(deck, ci))
            if slug in want:
                lists[slug][tuple(deck)] += 1

    os.makedirs(OUTDIR, exist_ok=True)
    for slug in want:
        c = lists.get(slug)
        if not c:
            print(f"  {slug}: NO decks found")
            continue
        deck, freq = c.most_common(1)[0]
        path = os.path.join(OUTDIR, f"{slug}.csv")
        with open(path, "w") as f:
            f.write("\n".join(str(x) for x in deck) + "\n")
        names = Counter(ci[cid].name for cid in deck if cid in ci)
        top = ", ".join(f"{n}×{k.split(' ')[0]}" for k, n in names.most_common(4))
        print(f"  {slug}: modal list x{freq} ({len(deck)} cards, {len(c)} variants) -> {os.path.relpath(path, REPO)}")
        print(f"       headline: {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

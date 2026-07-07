"""Candidate deck generation — Deck Lab, strategy §5 ("template + flex, not free search").

Turns the human-meta archetype priors into legal, playable 60-card decks the round-robin harness
can rank. Each archetype's priors give per-card expected copies; we build a deck by:

  1. Seed counts = round(expected_copies) for every mapped in-pool card (energy included — its
     copy_prob is a point mass at the modal count). Cap non-basic-energy at 4 copies.
  2. Legalize: at most one ACE SPEC (keep the highest-expected one), guarantee ≥1 basic Pokémon.
  3. Reconcile to exactly 60 by a residual-greedy: while short, add the card with the largest
     (expected − current) residual — this restores flex cards that rounded to 0 before it ever
     floods energy; when no positive residual remains, pad with the deck's dominant basic energy.
     While over, trim the most over-represented card (never the last basic Pokémon).

The out-of-pool gap is honest: e.g. mega-greninja-ex is built without its namesake (CRI-022),
so it is legal but weak — exactly what the round-robin will expose. This is the *generator*;
meta-weighted selection and hill-climb over flex slots come next (§5).

Run:  .venv/bin/python src/deck_lab/candidates.py [out_dir]   (default: experiments/candidate_decks)
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PRIORS = os.path.join(REPO, "src", "priors", "archetype_priors.json")
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
from cg.api import all_card_data, CardType, EnergyType  # noqa: E402

DECK_SIZE = 60
MAX_COPIES = 4  # non-basic-energy copy limit


def _engine():
    return {c.cardId: c for c in all_card_data()}


def _basic_energy_by_type(engine) -> dict[int, int]:
    """energyType value -> basic-energy card id (for inferring which energy to pad with)."""
    return {c.energyType: c.cardId for c in engine.values()
            if CardType(c.cardType) == CardType.BASIC_ENERGY}


def _is_basic_pokemon(engine, cid: int) -> bool:
    c = engine.get(cid)
    return bool(c and CardType(c.cardType) == CardType.POKEMON and c.basic)


def _dominant_basic_energy(counts: dict[int, int], engine, be_by_type) -> int:
    """The basic energy to pad with: the one already most present, else inferred from the
    majority Pokémon energy type, else Water as a last resort."""
    be = {cid: n for cid, n in counts.items()
          if CardType(engine[cid].cardType) == CardType.BASIC_ENERGY}
    if be:
        return max(be, key=be.get)
    types = Counter(engine[cid].energyType for cid, n in counts.items()
                    for _ in range(n) if CardType(engine[cid].cardType) == CardType.POKEMON)
    for etype, _ in types.most_common():
        if etype in be_by_type:
            return be_by_type[etype]
    return be_by_type.get(EnergyType.WATER, 3)


def build_candidate(cards_entry: dict, engine, be_by_type) -> list[int]:
    """Build one legal 60-card deck from an archetype's `cards` priors entry."""
    expected = {int(cid): c["expected_copies"] for cid, c in cards_entry.items()}
    is_be = {int(cid): c["basic_energy"] for cid, c in cards_entry.items()}

    # 1. seed counts
    counts: dict[int, int] = {}
    for cid, exp in expected.items():
        n = round(exp)
        if not is_be[cid]:
            n = min(n, MAX_COPIES)
        if n > 0:
            counts[cid] = n

    # 2a. ACE SPEC: at most one, the highest-expected; each capped at 1
    aces = [cid for cid in counts if getattr(engine[cid], "aceSpec", False)]
    if aces:
        keep = max(aces, key=lambda c: expected[c])
        for cid in aces:
            if cid == keep:
                counts[cid] = 1
            else:
                del counts[cid]
    # 2b. guarantee a basic Pokémon (templates always have one; assert the invariant)
    if not any(_is_basic_pokemon(engine, cid) for cid in counts):
        raise ValueError("archetype has no basic Pokémon after seeding — cannot build a legal deck")

    def can_add(cid: int) -> bool:
        if getattr(engine[cid], "aceSpec", False):
            return counts.get(cid, 0) < 1
        if is_be.get(cid) or CardType(engine[cid].cardType) == CardType.BASIC_ENERGY:
            return True
        return counts.get(cid, 0) < MAX_COPIES

    # 3. reconcile to exactly 60
    total = sum(counts.values())
    while total < DECK_SIZE:
        # largest positive (expected − current) residual among addable priors cards…
        best, best_res = None, 0.0
        for cid in expected:
            if can_add(cid):
                res = expected[cid] - counts.get(cid, 0)
                if res > best_res:
                    best, best_res = cid, res
        if best is None:  # …else pad with the dominant basic energy
            best = _dominant_basic_energy(counts, engine, be_by_type)
        counts[best] = counts.get(best, 0) + 1
        total += 1

    while total > DECK_SIZE:
        # trim the most over-represented card, never dropping the last basic Pokémon
        basics = sum(1 for cid, n in counts.items() if _is_basic_pokemon(engine, cid) and n > 0)
        cand = [cid for cid, n in counts.items() if n > 0
                and not (_is_basic_pokemon(engine, cid) and n == 1 and basics == 1)]
        drop = max(cand, key=lambda c: counts[c] - expected.get(c, 0))
        counts[drop] -= 1
        total -= 1

    deck = [cid for cid, n in sorted(counts.items()) for _ in range(n)]
    assert len(deck) == DECK_SIZE
    return deck


def generate_all(priors_path: str = PRIORS) -> dict[str, list[int]]:
    engine = _engine()
    be_by_type = _basic_energy_by_type(engine)
    priors = json.load(open(priors_path, encoding="utf-8"))["archetypes"]
    return {name: build_candidate(entry["cards"], engine, be_by_type)
            for name, entry in priors.items()}


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "experiments", "candidate_decks")
    os.makedirs(out_dir, exist_ok=True)
    decks = generate_all()
    for name, deck in decks.items():
        with open(os.path.join(out_dir, f"{name}.csv"), "w", encoding="utf-8") as f:
            f.write("\n".join(str(c) for c in deck) + "\n")
    print(f"wrote {len(decks)} candidate decks to {out_dir}:")
    for name, deck in decks.items():
        print(f"  {name:24} 60 cards, {len(set(deck))} distinct")


if __name__ == "__main__":
    main()

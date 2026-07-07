#!/usr/bin/env python3
"""Smoke test for candidate deck generation (src/deck_lab/candidates.py).

Generates the 8 archetype candidate decks and checks:
  * CONSTRUCTION — each is exactly 60 cards, ≤4 copies of any non-basic-energy card, ≤1 ACE SPEC,
    and contains ≥1 basic Pokémon (the legality rules the cabt engine enforces).
  * PLAYABLE     — each deck actually runs a game to completion in the SDK (no INVALID / crash),
    piloted by the v1 brain vs the sample deck.

Run:  .venv/bin/python scripts/candidates_smoke.py [games_per_deck]
"""
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))

from cg.api import all_card_data, CardType                      # noqa: E402
from deck_lab.candidates import generate_all                    # noqa: E402
from deck_lab.round_robin import play_game, read_deck           # noqa: E402


def main() -> int:
    gpd = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cards = {c.cardId: c for c in all_card_data()}
    sample = read_deck(os.path.join(REPO, "submissions", "v1", "deck.csv"))

    decks = generate_all()
    assert len(decks) == 8, f"expected 8 candidates, got {len(decks)}"

    # 1. CONSTRUCTION
    print("=== construction ===")
    for name, deck in decks.items():
        counts = Counter(deck)
        assert len(deck) == 60, f"{name}: {len(deck)} cards"
        overs = {cid: n for cid, n in counts.items()
                 if n > 4 and CardType(cards[cid].cardType) != CardType.BASIC_ENERGY}
        assert not overs, f"{name}: >4 copies of {overs}"
        aces = [cid for cid in counts if getattr(cards[cid], "aceSpec", False)]
        assert len(aces) <= 1, f"{name}: {len(aces)} ACE SPECs {[cards[c].name for c in aces]}"
        basics = sum(n for cid, n in counts.items()
                     if CardType(cards[cid].cardType) == CardType.POKEMON and cards[cid].basic)
        assert basics >= 1, f"{name}: no basic Pokémon"
        energy = sum(n for cid, n in counts.items()
                     if CardType(cards[cid].cardType) in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY))
        ace_name = cards[aces[0]].name if aces else "-"
        print(f"  {name:24} distinct={len(counts):2} basics={basics:2} energy={energy:2} ace={ace_name}")

    # 2. PLAYABLE (legality via actually running games)
    print("\n=== playable (vs sample, SDK) ===")
    for name, deck in decks.items():
        outcomes = []
        for g in range(gpd):
            d0, d1 = (deck, sample) if g % 2 == 0 else (sample, deck)
            w = play_game(d0, d1)
            assert w in (0, 1, 2), f"{name}: game {g} did not complete cleanly (winner={w})"
            outcomes.append(w)
        print(f"  {name:24} {gpd} games completed OK  (winners={outcomes})")

    print("\nOK: 8 candidate decks are legal (60 / ≤4 / ≤1 ACE / ≥1 basic) and playable in the SDK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

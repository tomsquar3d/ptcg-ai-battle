#!/usr/bin/env python3
"""Smoke test for the round-robin deck harness (src/deck_lab/round_robin.py).

Self-generates three fixture decks (no files committed) and checks the harness discriminates:
  * sample        — the SDK sample deck (submissions/v1/deck.csv)
  * sample_mirror — a byte-identical copy  → must play ~50% vs sample (sanity on fairness)
  * basics_only   — sample with every non-basic-Pokémon card (evolutions, trainers) swapped for
                    basic energy: it can't evolve and has no draw engine → must lose clearly

Run:  .venv/bin/python scripts/round_robin_smoke.py [games_per_pairing]
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))

from cg.api import all_card_data, CardType           # noqa: E402
from deck_lab.round_robin import round_robin, read_deck  # noqa: E402

BASIC_ENERGY_ID = 3  # Basic {W} Energy — a legal filler card


def main() -> int:
    gpp = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    cards = {c.cardId: c for c in all_card_data()}
    is_basic_pokemon = {cid: (CardType(c.cardType) == CardType.POKEMON and c.basic)
                        for cid, c in cards.items()}
    is_energy = {cid: CardType(c.cardType) in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)
                 for cid, c in cards.items()}

    sample = read_deck(os.path.join(REPO, "submissions", "v1", "deck.csv"))
    # Keep only basic Pokémon + existing energy; everything else → basic energy. The result can
    # never evolve and has no draw engine — a deliberately broken deck the harness must rank last.
    basics_only = [c if (is_basic_pokemon.get(c) or is_energy.get(c)) else BASIC_ENERGY_ID
                   for c in sample]
    swapped = sum(1 for a, b in zip(sample, basics_only) if a != b)
    assert len(basics_only) == 60 and swapped > 0, f"cripple built wrong ({swapped} swaps)"

    decks = {"sample": sample, "sample_mirror": list(sample), "basics_only": basics_only}
    res = round_robin(decks, games_per_pairing=gpp)

    from deck_lab.round_robin import _print
    _print(res)

    wr = {r["deck"]: r["win_rate"] for r in res["ranking"]}
    assert all(r["games"] > 0 for r in res["ranking"]), "some deck played no games"

    # 1. mirror fairness: sample vs its identical copy is a coin flip (loose bound for N).
    mirror = res["matrix"]["sample"]["sample_mirror"]
    assert 0.25 <= mirror <= 0.75, f"mirror not ~50%: sample vs copy = {mirror:.0%} (seat bias?)"

    # 2. discrimination: the broken deck is clearly worse and ranks last.
    assert res["ranking"][-1]["deck"] == "basics_only", f"crippled deck not last: {res['ranking']}"
    assert wr["basics_only"] < 0.40, f"crippled deck not clearly losing: {wr['basics_only']:.0%}"

    print(f"\nOK: harness fair on mirrors ({mirror:.0%}) and ranks the broken deck last "
          f"({wr['basics_only']:.0%}); {swapped} slots crippled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

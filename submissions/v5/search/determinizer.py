"""Determinizer — sample K complete world-states for the native Search API (docs §3, §4).

`search_begin` needs a *concrete* hypothesis of every hidden card: our deck order + prizes, and
the opponent's deck order + prizes + hand (+ a face-down active). MCTS then explores that fully
observable world. Sampling K worlds from the belief distributions and aggregating turns MCTS into
determinized / information-set MCTS: the hidden state is integrated out over K plausible worlds.

Sources of truth (both exact where the rules make them so):
  - OUR side: we know our 60-card decklist, so our hidden pool = decklist − everything we can see
    (`ptcg.belief.build_belief` → `DrawModel.pool`). A uniform shuffle splits it into deck + prizes
    by exchangeability — a correct sample, not a guess.
  - OPPONENT: `ptcg.opponent.build_opponent_belief` gives a hypergeometric zone model over
    `archetype_template − seen`. With archetype priors (`ptcg.archetypes.default_archetypes`) the
    template is the MAP archetype; with none it degrades to the priors' top template. We reconcile
    the template pool to the exact hidden count (pad with basic energy / trim) and guarantee a Basic
    in the deck, so `search_begin`'s count + basic checks always pass — the uniform fallback.

Each world is a tuple ready to splat: `search_begin(obs, *world)`.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Any, Mapping, Optional, Sequence

from ptcg.belief import build_belief
from ptcg.opponent import build_opponent_belief

_FILLER_ENERGY = 3       # Basic {W} Energy — a valid, ownership-neutral padding card
_CT_POKEMON, _CT_BASIC_ENERGY = 0, 5


def _is_basic_pokemon(cid: int, card_index: Mapping[int, Any]) -> bool:
    cd = card_index.get(cid)
    return bool(cd and getattr(cd, "cardType", None) == _CT_POKEMON and getattr(cd, "basic", False))


def _reconcile(pool: Counter, n: int, rng: random.Random) -> list[int]:
    """Expand a card multiset to a flat list of exactly `n` ids: trim the excess at random, or pad
    with filler energy when the template is short of the true hidden count."""
    cards = list(pool.elements())
    rng.shuffle(cards)
    if len(cards) > n:
        return cards[:n]
    cards.extend([_FILLER_ENERGY] * (n - len(cards)))
    return cards


def _split_own(obs: Any, own_decklist: Sequence[int], card_index: Mapping[int, Any],
               own_facedown_ids, rng: random.Random) -> tuple[list[int], list[int]]:
    """Sample (your_deck, your_prize) from our exact hidden pool. Revealed prizes are kept as-is;
    the rest of the pool is shuffled and split deck / hidden-prize by exchangeability."""
    st = obs.current
    me = st.yourIndex
    prize_slots = st.players[me].prize or []
    revealed = [getattr(c, "id") for c in prize_slots if c is not None]
    hidden_prizes = len(prize_slots) - len(revealed)

    belief = build_belief(obs, own_decklist, card_index, own_facedown_ids=own_facedown_ids)
    hidden = _reconcile(belief.draw.pool, belief.my_deck_count + hidden_prizes, rng)
    your_deck = hidden[:belief.my_deck_count]
    your_prize = revealed + hidden[belief.my_deck_count:]
    return your_deck, your_prize


def _split_opponent(obs: Any, card_index: Mapping[int, Any], archetypes, opp_tracker,
                    own_decklist: Sequence[int], rng: random.Random):
    """Sample (opponent_deck, opponent_prize, opponent_hand, opponent_active) from the archetype
    zone model, reconciled to exact zone counts with a guaranteed Basic in the deck."""
    st = obs.current
    me = st.yourIndex
    opp = 1 - me
    ops = st.players[opp]
    hand_n = getattr(ops, "handCount", 0) or 0
    deck_n = getattr(ops, "deckCount", 0) or 0
    prize_n = len(getattr(ops, "prize", None) or [])

    ob = build_opponent_belief(obs, tracker=opp_tracker, archetypes=archetypes, card_index=card_index)
    pool = Counter(ob.zone.pool) if ob.zone is not None else Counter(own_decklist)
    flat = _reconcile(pool, hand_n + deck_n + prize_n, rng)
    hand = flat[:hand_n]
    prize = flat[hand_n:hand_n + prize_n]
    deck = flat[hand_n + prize_n:]

    # guarantee a Basic in the deck (setup rule); pull one from hand/prize or the template pool
    if deck and not any(_is_basic_pokemon(c, card_index) for c in deck):
        donor_pools = [hand, prize]
        for donor in donor_pools:
            for i, c in enumerate(donor):
                if _is_basic_pokemon(c, card_index):
                    deck[0], donor[i] = c, deck[0]
                    break
            else:
                continue
            break

    # face-down opponent active → predict a Basic (from the sampled hidden cards or the template)
    opp_active = []
    active = getattr(ops, "active", None) or []
    if active and active[0] is None:
        cand = next((c for c in flat if _is_basic_pokemon(c, card_index)), None)
        if cand is None:
            cand = next((cid for cid in pool if _is_basic_pokemon(cid, card_index)), None)
        opp_active = [cand] if cand is not None else []
    return deck, prize, hand, opp_active


def determinize(obs: Any, own_decklist: Sequence[int], card_index: Mapping[int, Any], *,
                k: int = 1, archetypes: Optional[Sequence[Any]] = None, opp_tracker: Any = None,
                seed: Optional[int] = None) -> list[tuple]:
    """Return K worlds, each a tuple `(your_deck, your_prize, opponent_deck, opponent_prize,
    opponent_hand, opponent_active)` valid to splat into `search_begin(obs, *world)`.

    `archetypes` defaults to the human-meta templates; pass `opp_tracker` (a persistent
    `OpponentTracker`) for exact seen-card evidence. `seed` makes sampling reproducible.
    """
    if archetypes is None:
        from ptcg.archetypes import default_archetypes
        archetypes = default_archetypes()
    rng = random.Random(seed)
    worlds = []
    for _ in range(k):
        your_deck, your_prize = _split_own(obs, own_decklist, card_index, None, rng)
        opp_deck, opp_prize, opp_hand, opp_active = _split_opponent(
            obs, card_index, archetypes, opp_tracker, own_decklist, rng)
        worlds.append((your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active))
    return worlds

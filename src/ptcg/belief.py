"""Belief layer — Belief Engine Layer 2 (docs/ptcg_agent_strategy.md §3, §4).

This first slice is the *exact*, priors-free part of belief: a hidden-zone tracker for
**our own deck** plus a hypergeometric draw model.

Why it's exact (no learning, no priors): we know our own 60-card decklist, and the engine
shows us every one of our cards except those in our deck and our (face-down) prizes. So our
hidden pool is simply

    pool = decklist − (every card of ours we can currently see)

and its size n equals deck_count + (unknown) prize_count. The physical draw comes off the top
of the deck, but the deck/prize split is a uniform random partition of the pool, so by
exchangeability the next k drawn cards are a uniform k-subset of the whole pool. Hence
"probability the next k draws contain a target" is a plain hypergeometric over the pool —
prizes included in n, never drawn. This is the tracker §4's deck-thinning metric and the
determinization sampler both build on.

The opponent's deck is NOT known — modelling it needs the Bayesian archetype prior (§4,
"Opponent prediction → a distribution"), which is the next piece; `BeliefState.opponent` is a
placeholder for it.

Pure module: no ctypes/native import. Consumes the `Observation` dataclass (from
`cg.api.to_observation_class`) plus our decklist; an optional `card_index`
(``{card_id: CardData}``) enables card-property queries (draw a Basic? a Supporter?).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import comb
from typing import Any, Callable, Iterable, Mapping, Optional, Union

Targets = Union[int, Iterable[int], Callable[[int], bool]]


def _matches(target: Targets, card_index: Optional[Mapping[int, Any]]) -> Callable[[int], bool]:
    """Normalize a target spec (id / ids / predicate) into a card_id -> bool test."""
    if callable(target):
        return target  # predicate over card_id (may consult a closed-over card_index)
    if isinstance(target, int):
        return lambda cid: cid == target
    ids = set(target)
    return lambda cid: cid in ids


@dataclass
class DrawModel:
    """Hypergeometric draw model over our hidden pool (unknown deck + prize cards).

    pool:       card_id -> copies among our hidden (undrawn, unrevealed) cards.
    deck_count: cards currently in our deck (the only zone we actually draw from).
    n = |pool| = deck_count + unknown_prize_count.
    """
    pool: Counter
    deck_count: int
    card_index: Optional[Mapping[int, Any]] = None

    @property
    def n(self) -> int:
        return sum(self.pool.values())

    def copies(self, target: Targets) -> int:
        """How many copies in the hidden pool match the target spec."""
        test = _matches(target, self.card_index)
        return sum(cnt for cid, cnt in self.pool.items() if test(cid))

    # --- core hypergeometric ---------------------------------------------------------
    def _draws(self, k: int) -> int:
        # We can never draw more than the deck holds; prizes are never drawn.
        return max(0, min(k, self.deck_count))

    def prob_at_least(self, target: Targets, k: int = 1, m: int = 1) -> float:
        """P(at least m copies of target among the next k draws)."""
        n = self.n
        K = self.copies(target)
        d = self._draws(k)
        if m <= 0:
            return 1.0
        if n <= 0 or d <= 0 or K < m:
            return 0.0
        # P(X >= 1) via complement for stability.
        if m == 1:
            if n - K < d:
                return 1.0
            return 1.0 - comb(n - K, d) / comb(n, d)
        total = comb(n, d)
        hi = min(d, K)
        acc = 0
        for i in range(m, hi + 1):
            rest = d - i
            if rest < 0 or n - K < rest:
                continue
            acc += comb(K, i) * comb(n - K, rest)
        return acc / total

    def prob_draw(self, target: Targets, k: int = 1) -> float:
        """P(at least one copy of target among the next k draws)."""
        return self.prob_at_least(target, k=k, m=1)

    def expected_count(self, target: Targets, k: int = 1) -> float:
        """Expected number of target copies among the next k draws."""
        n = self.n
        if n <= 0:
            return 0.0
        return self._draws(k) * self.copies(target) / n

    # --- card-property convenience (needs card_index) --------------------------------
    def _prop(self, attr: str, value: Any = True) -> Callable[[int], bool]:
        ci = self.card_index
        return lambda cid: bool(getattr(ci.get(cid), attr, None) == value) if ci else False

    def prob_draw_basic_pokemon(self, k: int = 1) -> float:
        # CardType.POKEMON == 0 and CardData.basic == True
        ci = self.card_index
        def is_basic(cid):
            cd = ci.get(cid) if ci else None
            return bool(cd and getattr(cd, "basic", False) and getattr(cd, "cardType", None) == 0)
        return self.prob_draw(is_basic, k=k)

    def prob_draw_supporter(self, k: int = 1) -> float:
        return self.prob_draw(self._prop("cardType", 3), k=k)  # CardType.SUPPORTER == 3

    def prob_draw_energy(self, k: int = 1) -> float:
        ci = self.card_index
        def is_energy(cid):
            ct = getattr(ci.get(cid), "cardType", None) if ci else None
            return ct in (5, 6)  # BASIC_ENERGY / SPECIAL_ENERGY
        return self.prob_draw(is_energy, k=k)


@dataclass
class BeliefState:
    """Belief snapshot. Own-deck draw model is exact; opponent model is deferred."""
    draw: DrawModel                 # exact model of OUR deck
    my_prizes_left: int
    opp_prizes_left: int
    my_hand_count: int
    opp_hand_count: int
    my_deck_count: int
    known_prize_count: int          # how many of our own prizes we've had revealed
    facedown_in_play: int = 0       # our own face-down in-play Pokémon (setup only)
    opponent: Optional[Any] = None  # placeholder: archetype-prior opponent model (next piece)

    @property
    def prize_race(self) -> tuple[int, int]:
        """(our prizes remaining, opponent's prizes remaining) — lower is closer to winning."""
        return (self.my_prizes_left, self.opp_prizes_left)

    @property
    def prize_diff(self) -> int:
        """Positive => we are ahead (fewer prizes left to take)."""
        return self.opp_prizes_left - self.my_prizes_left


def _own_hidden_pool(state: Any, my_index: int, own_decklist: Iterable[int]) -> tuple[Counter, int, int, int]:
    """Return (pool, deck_count, known_prize_count, facedown_in_play) for our side.

    pool = decklist − (every card of ours currently visible/known), where 'known' covers
    hand, discard, in-play Pokémon + their attachments/pre-evolutions, the stadium if ours,
    cards we're looking at, and any of our prizes that have been revealed. What remains is
    the multiset of our unknown cards: deck + (unknown) prizes + any face-down own Pokémon.

    Face-down own Pokémon occur only during simultaneous setup (the engine hides our own
    active until the reveal), so `pool` momentarily conflates deck + prize + that card and
    draw probabilities are approximate during setup; post-setup `facedown_in_play == 0` and
    the pool is exactly deck + unknown prizes.
    """
    pool: Counter = Counter(own_decklist)

    def take(cards: Optional[Iterable[Any]]) -> None:
        for c in cards or []:
            if c is None:
                continue
            cid = getattr(c, "id", -1)
            if cid is not None and cid >= 0 and getattr(c, "playerIndex", my_index) == my_index:
                pool[cid] -= 1

    players = getattr(state, "players", None) or []
    if my_index < len(players):
        mep = players[my_index]
        take(getattr(mep, "hand", None))
        take(getattr(mep, "discard", None))
        take(getattr(mep, "prize", None))  # only revealed (non-None) prizes are subtracted

    take(getattr(state, "stadium", None))
    take(getattr(state, "looking", None))

    # Pokémon cards belong to the board they sit on; attachments carry their own playerIndex.
    for pidx, ps in enumerate(players):
        for pk in list(getattr(ps, "active", None) or []) + list(getattr(ps, "bench", None) or []):
            if pk is None:
                continue
            if pidx == my_index:
                pkid = getattr(pk, "id", -1)
                if pkid is not None and pkid >= 0:
                    pool[pkid] -= 1
            take(getattr(pk, "energyCards", None))
            take(getattr(pk, "tools", None))
            take(getattr(pk, "preEvolution", None))

    # count revealed own prizes (already subtracted above) and face-down own Pokémon (setup)
    known_prize = 0
    facedown = 0
    if my_index < len(players):
        mep = players[my_index]
        known_prize = sum(1 for c in (getattr(mep, "prize", None) or []) if c is not None)
        facedown = sum(1 for p in list(getattr(mep, "active", None) or [])
                       + list(getattr(mep, "bench", None) or []) if p is None)

    deck_count = getattr(players[my_index], "deckCount", 0) if my_index < len(players) else 0
    return pool, deck_count, known_prize, facedown


def build_belief(
    observation: Any,
    own_decklist: Iterable[int],
    card_index: Optional[Mapping[int, Any]] = None,
) -> BeliefState:
    """Build a BeliefState from an in-game Observation and our known 60-card decklist.

    The hidden-pool reconstruction is exact at stable decision points — the MAIN action
    menu and turn boundaries, which is when draw math is actually used. During multi-step
    effect resolution (e.g. while a "look at the top N" is resolving) a card can be in
    flight (out of hand, not yet in discard), so the pool may transiently include it; treat
    draw probabilities as exact only at MAIN decisions.

    Raises ValueError on the initial deck-selection observation (no `current` state).
    """
    state = getattr(observation, "current", None)
    if state is None:
        raise ValueError("build_belief requires an in-game observation (current is None).")

    my_index = getattr(state, "yourIndex", 0) or 0
    opp_index = 1 - my_index
    players = getattr(state, "players", None) or []

    pool, deck_count, known_prize, facedown = _own_hidden_pool(state, my_index, own_decklist)
    draw = DrawModel(pool=pool, deck_count=deck_count, card_index=card_index)

    def prize_left(idx: int) -> int:
        return len(getattr(players[idx], "prize", None) or []) if idx < len(players) else 0

    def hand_count(idx: int) -> int:
        return getattr(players[idx], "handCount", 0) if idx < len(players) else 0

    return BeliefState(
        draw=draw,
        my_prizes_left=prize_left(my_index),
        opp_prizes_left=prize_left(opp_index),
        my_hand_count=hand_count(my_index),
        opp_hand_count=hand_count(opp_index),
        my_deck_count=deck_count,
        known_prize_count=known_prize,
        facedown_in_play=facedown,
    )

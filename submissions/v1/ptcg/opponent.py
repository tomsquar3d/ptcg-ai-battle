"""Opponent model — Belief Engine Layer 2b (docs/ptcg_agent_strategy.md §4, §6).

"Opponent prediction → a distribution." Three pieces, most-exact first:

1. `OpponentTracker` / `opponent_visible_cards` — the *exact, priors-free* evidence: which
   specific opponent cards we've legitimately seen. The engine reveals the opponent's own
   cards to us only when they actually become visible (board, discard, stadium) or are named
   in a log with a cardId (played / attacked / evolved); their hidden draws are
   MOVE/DRAW_REVERSE with no id. Deduped by serial, this is the minimum multiset their deck
   contains — the input both to the classifier and to subtract-seen hidden-zone estimation.

2. `OpponentZoneModel` — given a predicted decklist *template*, `pool = template − currently
   visible` is the opponent's hidden cards; by exchangeability each is equally likely to sit
   in any hidden zone, so P(they hold X in hand / draw X next / prized X) are hypergeometric
   over the pool (reusing `belief.hyper_at_least`). This is what feeds the threat model and
   the determinization sampler (experiment #3 showed re-determinizing is cheap).

3. `Archetype` + `classify_archetype` — the Bayesian archetype classifier: posterior over a
   set of archetype templates given the revealed cards (naive multinomial bag-of-cards with
   additive smoothing). The *framework* is here and unit-tested; the real archetype templates
   and meta-share priors come from replay mining (§5) — until then callers pass their own.

Pure module (no native import). Consumes the `Observation` dataclass.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from .belief import hyper_at_least, Targets, _matches

# AreaType / LogType constants (mirror cg.api).
_AREA_ACTIVE, _AREA_BENCH = 4, 5


# --- 1. Evidence: what opponent cards have we actually seen? --------------------------

def _add_card(counter: Counter, serials: dict, c: Any, owner: int) -> None:
    if c is None:
        return
    cid = getattr(c, "id", -1)
    serial = getattr(c, "serial", None)
    if cid is None or cid < 0:
        return
    if getattr(c, "playerIndex", owner) != owner:
        return
    counter[cid] += 1
    if serial is not None:
        serials[serial] = cid


def opponent_visible_cards(state: Any, opp_index: int) -> Counter:
    """Multiset of opponent cards currently sitting in zones we can see (board + attachments,
    discard, stadium, looking). Excludes their hand/deck/prizes and face-down Pokémon."""
    vis: Counter = Counter()
    serials: dict = {}
    players = getattr(state, "players", None) or []
    if opp_index < len(players):
        _add_all(vis, serials, getattr(players[opp_index], "discard", None), opp_index)

    for pidx, ps in enumerate(players):
        for pk in list(getattr(ps, "active", None) or []) + list(getattr(ps, "bench", None) or []):
            if pk is None:
                continue
            if pidx == opp_index:  # the Pokémon card belongs to the board it sits on
                pkid = getattr(pk, "id", -1)
                if pkid is not None and pkid >= 0:
                    vis[pkid] += 1
            _add_all(vis, serials, getattr(pk, "energyCards", None), opp_index)
            _add_all(vis, serials, getattr(pk, "tools", None), opp_index)
            _add_all(vis, serials, getattr(pk, "preEvolution", None), opp_index)

    _add_all(vis, serials, getattr(state, "stadium", None), opp_index)
    _add_all(vis, serials, getattr(state, "looking", None), opp_index)
    return vis


def _add_all(counter: Counter, serials: dict, cards: Optional[Iterable[Any]], owner: int) -> None:
    for c in cards or []:
        _add_card(counter, serials, c, owner)


class OpponentTracker:
    """Stateful: accumulates every opponent card we've ever confirmed (deduped by serial).

    Feed every observation in order. Monotonic — a card, once seen, is known to be in the
    opponent's deck for the rest of the match even if it returns to a hidden zone. This is the
    evidence the archetype classifier consumes.

    Identity is harvested only from **visible zones** (board Pokémon + their attachments,
    discard, stadium), where physical location unambiguously proves ownership. We deliberately
    do NOT trust the log stream's `playerIndex` for this: for damage/effect logs that index is
    the acting player, not the card owner, so an opponent action targeting *our* card would
    otherwise mis-attribute it. The trade-off is that a card the opponent reveals and then
    re-hides (rare) isn't captured — correctness over completeness.
    """

    def __init__(self) -> None:
        self._serial_to_id: dict[int, int] = {}

    def _harvest(self, cards: Optional[Iterable[Any]], opp: int) -> None:
        for c in cards or []:
            if c is None:
                continue
            cid, serial = getattr(c, "id", -1), getattr(c, "serial", None)
            if cid is not None and cid >= 0 and serial is not None and getattr(c, "playerIndex", opp) == opp:
                self._serial_to_id[serial] = cid

    def observe(self, observation: Any) -> "OpponentTracker":
        state = getattr(observation, "current", None)
        if state is None:
            return self
        opp = 1 - (getattr(state, "yourIndex", 0) or 0)
        players = getattr(state, "players", None) or []

        if opp < len(players):
            self._harvest(getattr(players[opp], "discard", None), opp)
        for pidx, ps in enumerate(players):
            for pk in list(getattr(ps, "active", None) or []) + list(getattr(ps, "bench", None) or []):
                if pk is None:
                    continue
                if pidx == opp:  # the Pokémon card belongs to the board it sits on
                    s, i = getattr(pk, "serial", None), getattr(pk, "id", -1)
                    if s is not None and i is not None and i >= 0:
                        self._serial_to_id[s] = i
                self._harvest(getattr(pk, "energyCards", None), opp)
                self._harvest(getattr(pk, "tools", None), opp)
                self._harvest(getattr(pk, "preEvolution", None), opp)
        self._harvest(getattr(state, "stadium", None), opp)
        self._harvest(getattr(state, "looking", None), opp)
        return self

    def revealed(self) -> Counter:
        return Counter(self._serial_to_id.values())


# --- 2. Hidden-zone estimation given a decklist template ------------------------------

@dataclass
class OpponentZoneModel:
    """Hypergeometric over the opponent's hidden cards, given a predicted decklist template.

    pool = template − currently-visible; n = |pool| = hand + deck + prize + face-down. Every
    hidden card is equally likely to be in any hidden zone (exchangeability), so hand / deck /
    prize queries are hypergeometric over the pool with the matching draw count.
    """
    pool: Counter
    hand_count: int
    deck_count: int
    prize_count: int
    card_index: Optional[Mapping[int, Any]] = None

    @property
    def n(self) -> int:
        return sum(self.pool.values())

    def copies(self, target: Targets) -> int:
        test = _matches(target, self.card_index)
        return sum(cnt for cid, cnt in self.pool.items() if test(cid))

    def prob_in_hand(self, target: Targets, m: int = 1) -> float:
        return hyper_at_least(self.n, self.copies(target), self.hand_count, m)

    def prob_draw_next(self, target: Targets, k: int = 1) -> float:
        return hyper_at_least(self.n, self.copies(target), min(k, self.deck_count), 1)

    def prob_prized(self, target: Targets, m: int = 1) -> float:
        return hyper_at_least(self.n, self.copies(target), self.prize_count, m)

    def expected_in_hand(self, target: Targets) -> float:
        return self.hand_count * self.copies(target) / self.n if self.n else 0.0


# --- 3. Bayesian archetype classifier -------------------------------------------------

@dataclass
class Archetype:
    """A meta archetype: a decklist template (card_id -> count) and a prior meta share."""
    name: str
    template: Mapping[int, int]
    prior: float = 1.0


def classify_archetype(revealed: Mapping[int, int], archetypes: Sequence[Archetype],
                       alpha: float = 0.5) -> dict[str, float]:
    """Posterior P(archetype | revealed cards) via naive multinomial bag-of-cards with additive
    smoothing. With no evidence it returns the (normalized) priors; a revealed card strongly
    associated with one archetype shifts mass toward it, and a card absent from a template
    penalizes it. Real templates/priors come from meta mining; this is the mechanism."""
    if not archetypes:
        return {}
    vocab = set(revealed)
    for a in archetypes:
        vocab |= set(a.template)
    v = max(1, len(vocab))

    log_post: dict[str, float] = {}
    for a in archetypes:
        size = sum(a.template.values())
        lp = math.log(a.prior) if a.prior > 0 else -math.inf
        for cid, obs in revealed.items():
            p = (a.template.get(cid, 0) + alpha) / (size + alpha * v)
            lp += obs * math.log(p)
        log_post[a.name] = lp

    hi = max(log_post.values())
    exps = {k: math.exp(v_ - hi) for k, v_ in log_post.items()}
    z = sum(exps.values()) or 1.0
    return {k: v_ / z for k, v_ in exps.items()}


# --- Composite opponent belief --------------------------------------------------------

@dataclass
class OpponentBelief:
    revealed: Counter                       # cards we've confirmed the opponent has
    deck_count: int
    hand_count: int
    prize_count: int
    facedown_in_play: int
    posterior: Optional[dict[str, float]] = None   # archetype posterior (if archetypes given)
    zone: Optional[OpponentZoneModel] = None       # hidden-zone model (if a template is available)

    @property
    def map_archetype(self) -> Optional[str]:
        if not self.posterior:
            return None
        return max(self.posterior, key=self.posterior.get)


def build_opponent_belief(
    observation: Any,
    *,
    tracker: Optional[OpponentTracker] = None,
    archetypes: Optional[Sequence[Archetype]] = None,
    deck_template: Optional[Mapping[int, int]] = None,
    card_index: Optional[Mapping[int, Any]] = None,
) -> OpponentBelief:
    """Build the opponent belief from an in-game Observation.

    - `tracker`: a persistent OpponentTracker (already fed this observation) for monotonic
      revealed-card evidence; without it, evidence falls back to what's visible right now.
    - `archetypes`: enables the archetype posterior; its MAP template drives the zone model
      unless `deck_template` is given explicitly (e.g. a known deck for determinization/tests).
    """
    state = getattr(observation, "current", None)
    if state is None:
        raise ValueError("build_opponent_belief requires an in-game observation.")
    me = getattr(state, "yourIndex", 0) or 0
    opp = 1 - me
    players = getattr(state, "players", None) or []
    opp_ps = players[opp] if opp < len(players) else None

    visible = opponent_visible_cards(state, opp)
    revealed = tracker.revealed() if tracker is not None else Counter(visible)

    posterior = classify_archetype(revealed, archetypes) if archetypes else None

    template = deck_template
    if template is None and posterior:
        best = max(posterior, key=posterior.get)
        template = next((a.template for a in archetypes if a.name == best), None)

    zone = None
    if template is not None:
        # Accept a decklist as a mapping (card_id -> count) or a flat list of card ids.
        tmpl = template if isinstance(template, Mapping) else Counter(template)
        pool = Counter()
        for cid, cnt in tmpl.items():
            rem = cnt - visible.get(cid, 0)
            if rem > 0:
                pool[cid] = rem
        zone = OpponentZoneModel(
            pool=pool,
            hand_count=getattr(opp_ps, "handCount", 0) or 0,
            deck_count=getattr(opp_ps, "deckCount", 0) or 0,
            prize_count=len(getattr(opp_ps, "prize", None) or []),
            card_index=card_index,
        )

    facedown = sum(1 for p in list(getattr(opp_ps, "active", None) or [])
                   + list(getattr(opp_ps, "bench", None) or []) if p is None) if opp_ps else 0

    return OpponentBelief(
        revealed=revealed,
        deck_count=getattr(opp_ps, "deckCount", 0) or 0,
        hand_count=getattr(opp_ps, "handCount", 0) or 0,
        prize_count=len(getattr(opp_ps, "prize", None) or []),
        facedown_in_play=facedown,
        posterior=posterior,
        zone=zone,
    )

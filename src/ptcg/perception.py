"""Perception layer — Belief Engine Layer 1 (docs/ptcg_agent_strategy.md §3).

Pure transformation: a raw engine ``Observation`` -> a structured, decision-friendly
``GameState``. It resolves card IDs to card stats, splits the board into *my* vs
*opponent* views, and normalizes the current decision.

Design rules:
- **No native/ctypes import.** This module reads plain attributes off the observation
  object, so it can be unit-tested on fixtures and can never be the layer that fails to
  import inside the agent. The caller converts the raw ``obs`` dict to an ``Observation``
  (``cg.api.to_observation_class``) and passes a ``card_id -> CardData`` mapping.
- **Never raises on a legal observation.** Unknown card IDs, empty boards, face-down
  cards, and enum values added mid-competition all degrade gracefully — the agent's
  "crash = loss" guarantee starts here.

Typical use inside an agent::

    from cg.api import to_observation_class, all_card_data
    from ptcg.perception import build_game_state

    CARD_INDEX = {c.cardId: c for c in all_card_data()}   # once, at startup
    gs = build_game_state(to_observation_class(obs_dict), CARD_INDEX)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

# --- Selection context names (mirrors cg.api.SelectContext) ------------------------
# Duplicated here to keep this module import-pure. Unknown values (the enum may gain
# members mid-competition) fall back to "CONTEXT_<n>" via SELECT_CONTEXT_NAMES.get.
SELECT_CONTEXT_NAMES: dict[int, str] = {
    0: "MAIN", 1: "SETUP_ACTIVE_POKEMON", 2: "SETUP_BENCH_POKEMON", 3: "SWITCH",
    4: "TO_ACTIVE", 5: "TO_BENCH", 6: "TO_FIELD", 7: "TO_HAND", 8: "DISCARD",
    9: "TO_DECK", 10: "TO_DECK_BOTTOM", 11: "TO_PRIZE", 12: "NOT_MOVE",
    13: "DAMAGE_COUNTER", 14: "DAMAGE_COUNTER_ANY", 15: "DAMAGE",
    16: "REMOVE_DAMAGE_COUNTER", 17: "HEAL", 18: "EVOLVES_FROM", 19: "EVOLVES_TO",
    20: "DEVOLVE", 21: "ATTACH_FROM", 22: "ATTACH_TO", 23: "DETACH_FROM", 24: "LOOK",
    25: "EFFECT_TARGET", 26: "DISCARD_ENERGY_CARD", 27: "DISCARD_TOOL_CARD",
    28: "SWITCH_ENERGY_CARD", 29: "DISCARD_CARD_OR_ATTACHED_CARD", 30: "DISCARD_ENERGY",
    31: "TO_HAND_ENERGY", 32: "TO_DECK_ENERGY", 33: "SWITCH_ENERGY", 34: "SKILL_ORDER",
    35: "ATTACK", 36: "DISABLE_ATTACK", 37: "EVOLVE", 38: "DRAW_COUNT",
    39: "DAMAGE_COUNTER_COUNT", 40: "REMOVE_DAMAGE_COUNTER_COUNT", 41: "IS_FIRST",
    42: "MULLIGAN", 43: "ACTIVATE", 44: "FIRST_EFFECT", 45: "MORE_DEVOLVE",
    46: "COIN_HEAD", 47: "AFFECT_SPECIAL_CONDITION", 48: "RECOVER_SPECIAL_CONDITION",
}


def context_name(ctx: Optional[int]) -> Optional[str]:
    if ctx is None:
        return None
    return SELECT_CONTEXT_NAMES.get(ctx, f"CONTEXT_{ctx}")


# --- Views -------------------------------------------------------------------------

@dataclass
class CardView:
    """A card resolved against the card DB (stats present only if the ID is known)."""
    card_id: int
    serial: int
    name: str
    card_type: Optional[int] = None  # cg.api.CardType


@dataclass
class PokemonView:
    """An in-play Pokémon with its stats and attached resources resolved."""
    card_id: int
    serial: int
    name: str
    hp: int                       # current HP
    max_hp: int
    damage: int                   # max_hp - hp (0 if unknown)
    energy_count: int             # total energy units attached
    energy_types: list[int]       # cg.api.EnergyType ints
    tool_names: list[str]
    appeared_this_turn: bool
    is_ex: bool = False
    is_mega_ex: bool = False
    prize_value: int = 1          # prizes the opponent takes on KO (1 / 2 ex / 3 megaEx)

    @classmethod
    def from_pokemon(cls, p: Any, cards: Mapping[int, Any]) -> "PokemonView":
        cd = cards.get(getattr(p, "id", None))
        name = getattr(cd, "name", None) or f"#{getattr(p, 'id', '?')}"
        is_ex = bool(getattr(cd, "ex", False))
        is_mega = bool(getattr(cd, "megaEx", False))
        prize_value = 3 if is_mega else (2 if is_ex else 1)
        hp = getattr(p, "hp", 0) or 0
        max_hp = getattr(p, "maxHp", 0) or 0
        tools = getattr(p, "tools", None) or []
        tool_names = [getattr(cards.get(getattr(t, "id", None)), "name", None)
                      or f"#{getattr(t, 'id', '?')}" for t in tools]
        energies = getattr(p, "energies", None) or []
        return cls(
            card_id=getattr(p, "id", -1),
            serial=getattr(p, "serial", -1),
            name=name,
            hp=hp,
            max_hp=max_hp,
            damage=max(0, max_hp - hp),
            energy_count=len(energies),
            energy_types=list(energies),
            tool_names=tool_names,
            appeared_this_turn=bool(getattr(p, "appearThisTurn", False)),
            is_ex=is_ex,
            is_mega_ex=is_mega,
            prize_value=prize_value,
        )


@dataclass
class PlayerView:
    """One player's zones. Hidden zones are counts; ``hand`` is None for the opponent."""
    index: int
    is_me: bool
    active: Optional[PokemonView]        # None if empty or face-down
    active_facedown: bool                # opponent's active present but hidden
    bench: list[PokemonView]
    bench_max: int
    deck_count: int
    hand_count: int
    hand: Optional[list[CardView]]       # None when not visible (opponent)
    prize_count: int                     # prize cards remaining
    discard_count: int
    poisoned: bool = False
    burned: bool = False
    asleep: bool = False
    paralyzed: bool = False
    confused: bool = False

    @property
    def has_special_condition(self) -> bool:
        return self.poisoned or self.burned or self.asleep or self.paralyzed or self.confused


@dataclass
class DecisionView:
    """The choice the engine is currently asking for."""
    select_type: int                     # cg.api.SelectType
    context: int                         # cg.api.SelectContext
    context_name: str
    min_count: int
    max_count: int
    num_options: int
    remain_damage_counter: int = 0
    remain_energy_cost: int = 0


@dataclass
class GameState:
    """Decision-ready snapshot for one selection point."""
    turn: int
    turn_action_count: int
    my_index: int
    opp_index: int
    first_player: int
    i_go_first: Optional[bool]           # None until the starting player is decided
    supporter_played: bool
    stadium_played: bool
    energy_attached: bool
    retreated: bool
    stadium: Optional[CardView]
    me: PlayerView
    opponent: PlayerView
    decision: Optional[DecisionView]     # None only at initial deck selection
    result: int = -1                     # win player index; -1 while ongoing

    @property
    def is_terminal(self) -> bool:
        return self.result != -1

    @property
    def prize_race(self) -> tuple[int, int]:
        """(my prizes remaining, opponent's prizes remaining) — lower is closer to winning."""
        return (self.me.prize_count, self.opponent.prize_count)


# --- Builders ----------------------------------------------------------------------

def _card_view(c: Any, cards: Mapping[int, Any]) -> CardView:
    cd = cards.get(getattr(c, "id", None))
    return CardView(
        card_id=getattr(c, "id", -1),
        serial=getattr(c, "serial", -1),
        name=getattr(cd, "name", None) or f"#{getattr(c, 'id', '?')}",
        card_type=getattr(cd, "cardType", None),
    )


def _active_view(active: Optional[Sequence[Any]], cards: Mapping[int, Any]) -> tuple[Optional[PokemonView], bool]:
    """Return (active PokemonView or None, is_facedown). ``active`` is a size-0/1 list."""
    if not active:
        return None, False
    first = active[0]
    if first is None:          # opponent's active present but face-down
        return None, True
    return PokemonView.from_pokemon(first, cards), False


def _player_view(idx: int, ps: Any, is_me: bool, cards: Mapping[int, Any]) -> PlayerView:
    active, facedown = _active_view(getattr(ps, "active", None), cards)
    bench = [PokemonView.from_pokemon(p, cards) for p in (getattr(ps, "bench", None) or [])]
    hand_cards = getattr(ps, "hand", None)
    hand = [_card_view(c, cards) for c in hand_cards] if hand_cards is not None else None
    return PlayerView(
        index=idx,
        is_me=is_me,
        active=active,
        active_facedown=facedown,
        bench=bench,
        bench_max=getattr(ps, "benchMax", 0) or 0,
        deck_count=getattr(ps, "deckCount", 0) or 0,
        hand_count=getattr(ps, "handCount", 0) or 0,
        hand=hand,
        prize_count=len(getattr(ps, "prize", None) or []),
        discard_count=len(getattr(ps, "discard", None) or []),
        poisoned=bool(getattr(ps, "poisoned", False)),
        burned=bool(getattr(ps, "burned", False)),
        asleep=bool(getattr(ps, "asleep", False)),
        paralyzed=bool(getattr(ps, "paralyzed", False)),
        confused=bool(getattr(ps, "confused", False)),
    )


def _decision_view(select: Any) -> Optional[DecisionView]:
    if select is None:
        return None
    ctx = getattr(select, "context", None)
    options = getattr(select, "option", None) or []
    return DecisionView(
        select_type=getattr(select, "type", -1),
        context=ctx if ctx is not None else -1,
        context_name=context_name(ctx) or "UNKNOWN",
        min_count=getattr(select, "minCount", 0) or 0,
        max_count=getattr(select, "maxCount", 0) or 0,
        num_options=len(options),
        remain_damage_counter=getattr(select, "remainDamageCounter", 0) or 0,
        remain_energy_cost=getattr(select, "remainEnergyCost", 0) or 0,
    )


def build_game_state(observation: Any, card_index: Mapping[int, Any]) -> GameState:
    """Build a ``GameState`` from an ``Observation`` and a ``card_id -> CardData`` map.

    Requires ``observation.current`` to be present (i.e. not the initial deck-selection
    observation, where ``select`` and ``current`` are both None — the agent handles that
    case by returning its deck). Raises ``ValueError`` if called on that pre-game obs.
    """
    state = getattr(observation, "current", None)
    if state is None:
        raise ValueError(
            "build_game_state requires an in-game observation; got the initial "
            "deck-selection obs (observation.current is None)."
        )

    my_index = getattr(state, "yourIndex", 0) or 0
    opp_index = 1 - my_index
    first_player = getattr(state, "firstPlayer", -1)
    i_go_first = None if first_player == -1 else (first_player == my_index)

    players = getattr(state, "players", None) or []
    # Defensive: the engine always sends exactly 2, but never index past what's there.
    me_ps = players[my_index] if my_index < len(players) else None
    opp_ps = players[opp_index] if opp_index < len(players) else None

    stadium_list = getattr(state, "stadium", None) or []
    stadium = _card_view(stadium_list[0], card_index) if stadium_list else None

    return GameState(
        turn=getattr(state, "turn", 0) or 0,
        turn_action_count=getattr(state, "turnActionCount", 0) or 0,
        my_index=my_index,
        opp_index=opp_index,
        first_player=first_player,
        i_go_first=i_go_first,
        supporter_played=bool(getattr(state, "supporterPlayed", False)),
        stadium_played=bool(getattr(state, "stadiumPlayed", False)),
        energy_attached=bool(getattr(state, "energyAttached", False)),
        retreated=bool(getattr(state, "retreated", False)),
        stadium=stadium,
        me=_player_view(my_index, me_ps, True, card_index),
        opponent=_player_view(opp_index, opp_ps, False, card_index),
        decision=_decision_view(getattr(observation, "select", None)),
        result=getattr(state, "result", -1),
    )

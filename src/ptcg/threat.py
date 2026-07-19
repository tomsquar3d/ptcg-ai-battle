"""Threat model — the belief→decision bridge (docs/ptcg_agent_strategy.md §4).

Turns the board + opponent belief into the danger signals the evaluator and critical-turn
search consume: expected/max incoming damage, whether our active is KO'd now or next turn, and
— using the opponent hidden-zone model — the probability they hold a gust/switch or the energy
they need.

Deterministic core (from the card DB): for the opponent's active, which of its attacks its
attached energy can actually pay for, and how much they do to our active after weakness. The
probabilistic layer (from `OpponentZoneModel`) covers what's in their hidden hand.

Scope / v1 limitations (documented, not hidden):
- Damage uses the printed `Attack.damage` × weakness. Text-based conditional/scaling damage
  ("30× heads", "+50 if…"), abilities, and special-energy multi-type output are NOT parsed yet.
- Weakness is ×2; resistance is −30 if typed (historical default) — validate vs engine later.
- Bench attackers and switch-to-a-charged-attacker lines aren't modelled (only the current
  active's threat + the probability they can gust ours).

Pure module (no native import). Consumes Perception's `GameState` + card/attack DB dicts.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

# EnergyType values (mirror cg.api.EnergyType)
COLORLESS, GRASS, FIRE, WATER, LIGHTNING, PSYCHIC, FIGHTING, DARKNESS, METAL, DRAGON = range(10)
RAINBOW, TEAM_ROCKET = 10, 11
WEAKNESS_MULT = 2
RESISTANCE_REDUCTION = 30

# CardType values
_CT_ITEM, _CT_SUPPORTER = 1, 3
_CT_BASIC_ENERGY, _CT_SPECIAL_ENERGY = 5, 6


def _cost_shortfall(cost: Sequence[int], available: Sequence[int]) -> int:
    """Minimum number of energies we'd have to add to `available` to pay `cost`.

    Specific-type requirements are matched by exact type, then by a wild energy (Rainbow, or
    Team Rocket energy for Psychic/Darkness); the rest of the cost is colorless (any energy).
    A missing specific counts as one to add (assume the ideal type can be attached). 0 = payable.
    """
    avail = Counter(e for e in available)
    short = 0
    specifics = [e for e in cost if e != COLORLESS]
    colorless = sum(1 for e in cost if e == COLORLESS)
    for t in specifics:
        if avail.get(t, 0) > 0:
            avail[t] -= 1
        elif avail.get(RAINBOW, 0) > 0:
            avail[RAINBOW] -= 1
        elif t in (PSYCHIC, DARKNESS) and avail.get(TEAM_ROCKET, 0) > 0:
            avail[TEAM_ROCKET] -= 1
        else:
            short += 1
    remaining = sum(v for v in avail.values() if v > 0)
    if remaining < colorless:
        short += colorless - remaining
    return short


def can_pay(cost: Sequence[int], available: Sequence[int]) -> bool:
    return _cost_shortfall(cost, available) == 0


def attack_damage(base: int, attacker_type: Optional[int],
                  defender_weakness: Optional[int], defender_resistance: Optional[int]) -> int:
    """Printed damage adjusted for the defender's weakness/resistance to the attacker's type."""
    if not base:
        return 0
    dmg = base
    if defender_weakness is not None and attacker_type is not None and defender_weakness == attacker_type:
        dmg *= WEAKNESS_MULT
    if defender_resistance is not None and attacker_type is not None and defender_resistance == attacker_type:
        dmg -= RESISTANCE_REDUCTION
    return max(0, dmg)


# --- text-based / conditional damage the printed `Attack.damage` can't express (#63 prototype) ----
# Keyed by lowercased attack name (reprint-robust). Each rule takes a `context` dict of state inputs
# and returns final damage that ALREADY accounts for weakness/resistance (damage-counter placement
# ignores W/R), or None to fall back to the printed value. Extend this table as more scalers are found.
def _scaled_damage(atk: Any, context: Optional[Mapping]) -> Optional[int]:
    if context is None:
        return None
    name = (getattr(atk, "name", "") or "").strip().lower()
    if name == "powerful hand":            # "Place 2 damage counters per card in your hand" = 20 × hand
        h = context.get("attacker_hand_size")
        if h is not None:
            return max(0, 20 * int(h))     # counter placement → no weakness/resistance
    return None


@dataclass
class ThreatAssessment:
    our_active_hp: int
    incoming_now: int              # max damage their active can deal to ours this turn
    incoming_potential: int        # …if they attach one more energy next turn
    ko_now: bool                   # incoming_now >= our HP
    ko_potential: bool             # incoming_potential >= our HP
    attacker_id: Optional[int] = None
    best_attack_now: Optional[str] = None
    # probabilistic layer (present when an OpponentZoneModel is supplied)
    p_gust_in_hand: Optional[float] = None
    p_needed_energy_in_hand: Optional[float] = None


def _best_damage(attacker_cd: Any, available: Sequence[int], def_weak, def_res,
                 attack_index: Mapping[int, Any], extra_energy: int,
                 context: Optional[Mapping] = None) -> tuple[int, Optional[str]]:
    """Max damage over the attacker's affordable attacks (allowing `extra_energy` more energy).
    `context` (optional) supplies state inputs so text-scaling attacks resolve; None = printed only."""
    attacker_type = getattr(attacker_cd, "energyType", None)
    best, best_name = 0, None
    for aid in getattr(attacker_cd, "attacks", None) or []:
        atk = attack_index.get(aid)
        if atk is None:
            continue
        if _cost_shortfall(getattr(atk, "energies", None) or [], available) > extra_energy:
            continue
        override = _scaled_damage(atk, context)
        if override is not None:
            dmg = override                 # text-scaling (e.g. counter placement): W/R already applied
        else:
            dmg = attack_damage(getattr(atk, "damage", 0) or 0, attacker_type, def_weak, def_res)
        if dmg > best:
            best, best_name = dmg, getattr(atk, "name", None)
    return best, best_name


def attack_potential(attacker: Any, defender: Any, card_index: Mapping[int, Any],
                     attack_index: Mapping[int, Any],
                     context: Optional[Mapping] = None) -> tuple[int, int, Optional[str]]:
    """Direction-agnostic: (damage_now, damage_with_one_more_energy, best_attack_name) that
    `attacker` (a PokemonView) can deal to `defender` (a PokemonView). Used both for the
    opponent's threat to us and for our own offense in the evaluator.

    `context` (optional) carries state inputs (e.g. {"attacker_hand_size": n}) that let text-based
    scaling attacks resolve; default None preserves the printed-damage behaviour exactly, so callers
    that don't pass it (evaluator, governor) are unaffected."""
    if attacker is None or defender is None:
        return 0, 0, None
    attacker_cd = card_index.get(attacker.card_id)
    if attacker_cd is None:
        return 0, 0, None
    defender_cd = card_index.get(defender.card_id)
    def_weak = getattr(defender_cd, "weakness", None)
    def_res = getattr(defender_cd, "resistance", None)
    available = list(getattr(attacker, "energy_types", None) or [])
    now, name = _best_damage(attacker_cd, available, def_weak, def_res, attack_index, 0, context)
    pot, _ = _best_damage(attacker_cd, available, def_weak, def_res, attack_index, 1, context)
    return now, max(pot, now), name


def assess_threat(
    game_state: Any,
    card_index: Mapping[int, Any],
    attack_index: Mapping[int, Any],
    *,
    opponent_belief: Any = None,
    gust_card_ids: Optional[Iterable[int]] = None,
    energy_card_ids: Optional[Iterable[int]] = None,
) -> ThreatAssessment:
    """Assess the danger to our active on the opponent's next turn.

    game_state: a Perception `GameState`. opponent_belief: an `OpponentBelief` (its `.zone`
    enables the gust/energy probabilities). gust/energy card-id sets can come from
    `gust_card_ids(card_index)` / `energy_card_ids(card_index)` below.
    """
    me_active = getattr(game_state.me, "active", None)
    opp_active = getattr(game_state.opponent, "active", None)
    our_hp = getattr(me_active, "hp", 0) if me_active else 0

    now, potential, best_name = attack_potential(opp_active, me_active, card_index, attack_index)
    attacker_id = opp_active.card_id if opp_active is not None else None

    ta = ThreatAssessment(
        our_active_hp=our_hp,
        incoming_now=now,
        incoming_potential=potential,
        ko_now=our_hp > 0 and now >= our_hp,
        ko_potential=our_hp > 0 and potential >= our_hp,
        attacker_id=attacker_id,
        best_attack_now=best_name,
    )

    zone = getattr(opponent_belief, "zone", None) if opponent_belief is not None else None
    if zone is not None:
        if gust_card_ids is not None:
            ta.p_gust_in_hand = zone.prob_in_hand(list(gust_card_ids))
        if energy_card_ids is not None:
            ta.p_needed_energy_in_hand = zone.prob_in_hand(list(energy_card_ids))
    return ta


# --- card-set helpers (derive id sets from the card DB) -------------------------------

def energy_card_ids(card_index: Mapping[int, Any]) -> set[int]:
    """All energy card ids (basic + special)."""
    return {cid for cid, cd in card_index.items()
            if getattr(cd, "cardType", None) in (_CT_BASIC_ENERGY, _CT_SPECIAL_ENERGY)}


def gust_card_ids(card_index: Mapping[int, Any]) -> set[int]:
    """Heuristic: Trainer cards whose skill text switches in the opponent's Pokémon (gust).
    Keyword-based (English) — a starting point until real card tagging exists; callers may
    override with a curated set."""
    ids: set[int] = set()
    for cid, cd in card_index.items():
        if getattr(cd, "cardType", None) not in (_CT_ITEM, _CT_SUPPORTER):
            continue
        for sk in getattr(cd, "skills", None) or []:
            text = (getattr(sk, "text", "") or "").lower()
            if "opponent" in text and "active" in text and ("switch" in text or "bench" in text):
                ids.add(cid)
                break
    return ids

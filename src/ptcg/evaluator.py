"""Heuristic evaluator — Belief Engine Layer 5 leaf evaluation (docs/ptcg_agent_strategy.md
§3, §4).

A single interpretable score for "how good is this position for us", as a weighted sum of
named terms so every term can be ablated on/off (§4 "per-object on/off ablation ladder", §7.1).
Built first regardless of the ML path: it is the week-2 ladder presence, the ablation baseline,
the search leaf evaluation, and the clock fallback.

Master signal is the prize-race differential (§4). The rest are board/threat/tempo corrections.
Consumes Perception's `GameState` plus the card/attack DB (for offense/threat via
`threat.attack_potential`); `BeliefState` is optional (reserved for the deck-quality term).

Pure module (no native import).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .threat import attack_potential

WIN_SCORE = 1_000_000.0


@dataclass
class EvalWeights:
    prize_diff: float = 1000.0       # per prize of lead (master signal)
    ko_them_now: float = 300.0       # × prizes we'd take by KO'ing their active this turn
    ko_us_now: float = 300.0         # × prizes we'd give up if our active is KO'd
    offense_next: float = 90.0       # we can KO their active with one more energy
    threat_next: float = 110.0       # they can KO ours with one more energy (and not already)
    our_output: float = 1.5          # per point of our best affordable attack (setup gradient)
    their_output: float = 1.0        # per point of their best affordable attack (penalty)
    energy_active: float = 12.0      # per energy attached to our active (capped)
    energy_active_cap: int = 4
    active_hp: float = 60.0          # our active HP fraction advantage
    board: float = 40.0              # (our Pokémon in play) − (theirs)
    hand: float = 8.0                # hand-size advantage
    special_us: float = 55.0         # penalty per special condition on our active
    special_opp: float = 35.0        # bonus per special condition on their active
    deckout: float = 25.0            # penalty as our deck runs low (per card under threshold)
    deckout_threshold: int = 4


@dataclass
class EvalResult:
    score: float
    terms: dict[str, float] = field(default_factory=dict)

    def __float__(self) -> float:
        return self.score


def _hp_frac(pv: Any) -> float:
    if pv is None or not getattr(pv, "max_hp", 0):
        return 0.0
    return max(0.0, min(1.0, pv.hp / pv.max_hp))


def evaluate(
    game_state: Any,
    *,
    card_index: Optional[Mapping[int, Any]] = None,
    attack_index: Optional[Mapping[int, Any]] = None,
    belief: Any = None,
    weights: Optional[EvalWeights] = None,
) -> EvalResult:
    """Score `game_state` from the to-move player's perspective (higher = better for us)."""
    w = weights or EvalWeights()

    # Terminal states dominate everything.
    if getattr(game_state, "is_terminal", False):
        res = getattr(game_state, "result", -1)
        if res == getattr(game_state, "my_index", -99):
            return EvalResult(WIN_SCORE, {"terminal": WIN_SCORE})
        if res in (0, 1):
            return EvalResult(-WIN_SCORE, {"terminal": -WIN_SCORE})
        return EvalResult(0.0, {"terminal": 0.0})  # draw

    me, opp = game_state.me, game_state.opponent
    terms: dict[str, float] = {}

    # 1) Prize race — master signal. prize_diff > 0 means we're ahead (opponent has more left).
    terms["prize_diff"] = w.prize_diff * (opp.prize_count - me.prize_count)

    # 2) Immediate KO exchange (weighted by the prizes at stake — ex/mega are worth more).
    my_active, opp_active = me.active, opp.active
    if card_index is not None and attack_index is not None:
        our_now, our_next, _ = attack_potential(my_active, opp_active, card_index, attack_index)
        thr_now, thr_next, _ = attack_potential(opp_active, my_active, card_index, attack_index)

        if opp_active is not None and our_now >= opp_active.hp > 0:
            terms["ko_them_now"] = w.ko_them_now * getattr(opp_active, "prize_value", 1)
        elif opp_active is not None and our_next >= opp_active.hp > 0:
            terms["offense_next"] = w.offense_next

        if my_active is not None and thr_now >= my_active.hp > 0:
            terms["ko_us_now"] = -w.ko_us_now * getattr(my_active, "prize_value", 1)
        elif my_active is not None and thr_next >= my_active.hp > 0:
            terms["threat_next"] = -w.threat_next

        # Continuous offense/defense output — gives 1-ply lookahead a gradient toward
        # attaching energy and building an attacker (otherwise setup moves all look equal).
        if our_now:
            terms["our_output"] = w.our_output * our_now
        if thr_now:
            terms["their_output"] = -w.their_output * thr_now
        if my_active is not None:
            terms["energy_active"] = w.energy_active * min(
                len(getattr(my_active, "energy_types", None) or []), w.energy_active_cap)

    # 3) Board health / presence.
    terms["active_hp"] = w.active_hp * (_hp_frac(my_active) - _hp_frac(opp_active))
    my_board = (1 if my_active else 0) + len(me.bench)
    opp_board = (1 if opp_active else 0) + len(opp.bench)
    terms["board"] = w.board * (my_board - opp_board)

    # 4) Tempo / resources.
    terms["hand"] = w.hand * (me.hand_count - opp.hand_count)
    if me.has_special_condition:
        terms["special_us"] = -w.special_us * sum(
            (me.poisoned, me.burned, me.asleep, me.paralyzed, me.confused))
    if opp.has_special_condition:
        terms["special_opp"] = w.special_opp * sum(
            (opp.poisoned, opp.burned, opp.asleep, opp.paralyzed, opp.confused))

    # 5) Deck-out risk.
    if me.deck_count <= w.deckout_threshold:
        terms["deckout"] = -w.deckout * (w.deckout_threshold - me.deck_count + 1)

    return EvalResult(sum(terms.values()), terms)

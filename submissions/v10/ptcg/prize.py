"""Prize-mapping plan object — strategy §4 ("Prize mapping → a plan object").

The master strategic signal: how many of OUR turns until we take our last prize, vs how many of
THEIRS. A position where we reach our 6th prize first is winning even if the current prize *count*
is even — so this turns-to-6 race, fed to the leaf evaluator, gives a single-turn search
**multi-turn foresight**: it rewards board development that speeds our race (a bigger attacker
coming online, chip/spread damage that cheapens future KOs) and correct trades, without any
multi-turn game-tree search (which #10 showed has poor ROI here).

Model (deliberately a "tiny greedy", per §4):
  - We win by KOing opponent Pokémon worth our remaining `prize_count` prizes in total.
  - Per-turn damage output D = our best set-up attack; turns to KO a target = ceil(hp / D).
  - Greedily take targets by fewest turns-per-prize until the needed prizes are reached; sum turns.

Approximations (documented, not hidden): uniform D across targets (ignores per-bench-target
weakness), assumes we can eventually reach any target (no gust / active-only constraint), and a
fresh-basic tail when the board lacks enough prizes on the field. Symmetric for the opponent.
Pure module (no native import); consumes a Perception `GameState`.
"""
from __future__ import annotations

import math
from typing import Any

TURNS_CAP = 12    # no-offense / unreachable → a large but bounded turns-to-6
TAIL_HP = 70      # assumed HP of extra basics benched for the tail estimate


def turns_to_six(needed_prizes: int, targets: list[tuple[int, int]], damage: float) -> float:
    """Turns to take `needed_prizes` prizes by KOing `targets` (each (hp, prize_value)) at
    `damage` output per turn. Greedy by turns-per-prize; fresh-basic tail if the field is short."""
    if needed_prizes <= 0:
        return 0.0
    if damage <= 0:
        return float(TURNS_CAP)  # no offense → we are not winning the race
    costed = [(max(1, math.ceil(hp / damage)), pv) for hp, pv in targets if pv > 0 and hp > 0]
    costed.sort(key=lambda t: t[0] / t[1])  # cheapest turns-per-prize first
    turns = prizes = 0
    for cost, pv in costed:
        turns += cost
        prizes += pv
        if prizes >= needed_prizes:
            return float(turns)
    # not enough prizes on the field: tail = remaining prizes off freshly-benched basics
    tail_cost = max(1, math.ceil(TAIL_HP / damage))
    turns += (needed_prizes - prizes) * tail_cost
    return float(min(turns, 2 * TURNS_CAP))


def prize_race(game_state: Any, our_output: float, their_output: float) -> tuple[float, float]:
    """(our_turns_to_6, their_turns_to_6) given each side's per-turn damage output. Lower = faster
    to victory; the differential (theirs − ours) is the master evaluation signal."""
    opp_targets = [(p.hp, p.prize_value)
                   for p in [game_state.opponent.active, *game_state.opponent.bench] if p is not None]
    my_targets = [(p.hp, p.prize_value)
                  for p in [game_state.me.active, *game_state.me.bench] if p is not None]
    ours = turns_to_six(game_state.me.prize_count, opp_targets, our_output)
    theirs = turns_to_six(game_state.opponent.prize_count, my_targets, their_output)
    return ours, theirs

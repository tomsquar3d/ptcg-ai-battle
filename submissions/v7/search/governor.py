"""Search governor — time-budget allocator (docs §3 arbitration point 4, §7.4).

The compute budget is a managed strategic resource: spend deep MCTS only on the turns that decide
games, move fast otherwise, and NEVER risk the clock (crash = timeout = instant loss).

Per decision it returns a `Budget(mode, seconds, k)`:
  - **critical turn** → deep MCTS: `seconds ≈ min(CRITICAL_S, CRIT_FRACTION × remaining_clock)`,
    K worlds. Critical = a prize-path trigger fires (a KO is available, our active is about to be
    KO'd — a race-differential swing — or we're in the endgame where every prize is decisive).
  - **routine turn** → shallow MCTS: a small budget, K=1.
  - **low clock** → `instant`: skip search entirely, let the agent play a legal heuristic move.

The budget is a hard ceiling: the agent passes `now + seconds` as the MCTS deadline, which is
checked every simulation, so search can only overshoot by one simulation (bounded ms). Keeping
`CRIT_FRACTION` small means even that overshoot is safe against the 600 s bank.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ptcg.threat import attack_potential

CRITICAL_S = 5.0        # target deep-search budget on a critical turn
ROUTINE_S = 0.4         # shallow-search budget on a routine turn
LOW_CLOCK_S = 20.0      # below this remaining bank → instant heuristic, no search
CRIT_FRACTION = 0.10    # never spend more than this share of the remaining clock on one move
CRITICAL_K = 8          # determinized worlds on a critical turn
ROUTINE_K = 1
ENDGAME_PRIZES = 2      # ≤ this many prizes left (either side) → treat every turn as critical


@dataclass
class Budget:
    mode: str           # "mcts" | "instant"
    seconds: float
    k: int


def is_critical(game_state: Any, card_index: Mapping, attack_index: Mapping) -> bool:
    """Prize-path critical-turn triggers: a KO is on the table, we're under a KO (race swing), or
    the game is in its prize endgame. Cheap — reuses the threat damage model."""
    me, opp = game_state.me, game_state.opponent
    # endgame: few prizes left to take. Guard >0 — prize_count is 0 only pre-deal (setup) or on a
    # win, so 0 would else make every setup decision falsely "critical".
    if 0 < me.prize_count <= ENDGAME_PRIZES or 0 < opp.prize_count <= ENDGAME_PRIZES:
        return True
    my_a, opp_a = me.active, opp.active
    if my_a is None or opp_a is None:
        return False
    our_now, _, _ = attack_potential(my_a, opp_a, card_index, attack_index)
    thr_now, _, _ = attack_potential(opp_a, my_a, card_index, attack_index)
    ko_available = opp_a.hp > 0 and our_now >= opp_a.hp     # we can take a prize this turn
    under_threat = my_a.hp > 0 and thr_now >= my_a.hp       # they can take one back — a swing
    return ko_available or under_threat


def allocate(game_state: Any, time_left: Any, card_index: Mapping, attack_index: Mapping) -> Budget:
    """Decide the search budget for the current decision from the remaining clock + criticality."""
    if time_left is not None and time_left < LOW_CLOCK_S:
        return Budget("instant", 0.0, 0)
    clock = float(time_left) if time_left is not None else 600.0
    cap = clock * CRIT_FRACTION
    if is_critical(game_state, card_index, attack_index):
        return Budget("mcts", max(min(CRITICAL_S, cap), ROUTINE_S), CRITICAL_K)
    return Budget("mcts", min(ROUTINE_S, cap), ROUTINE_K)

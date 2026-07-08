"""Scripted line-expert: `dragapult-dusknoir` (docs §3d step 2).

The archetype's skeleton line: build the Dreepy→Drakloak→Dragapult ex board (Rare Candy skips
straight to Dragapult ex off a Dreepy), load the attacker with Fire+Psychic, play Crispin for the
energy before committing, then swing **Phantom Dive** (200, F+P) — its bench spread is the deck's
whole point. Dusknoir/Munkidori are the damage-move support (left to the v1 fallback for now).

Card ids: Dreepy 119 → Drakloak 120 → Dragapult ex 121 (Rare Candy 1079 = 119→121). Duskull 131 →
Dusclops 132 → Dusknoir 133. Crispin 1198, Boss's Orders 1182. Energies: Fire 2, Psychic 5, Dark 7.
Dragapult ex attacks: Jet Headbutt 153, **Phantom Dive 154**.

Rules fire only on high-confidence patterns; anything else defers to v1 (see base.py). Setup actions
outrank attacking because each non-turn-ending action re-prompts MAIN, so the whole line resolves
before Phantom Dive is taken.
"""
from __future__ import annotations

from typing import Optional

from .base import (
    ScriptedExpert, _Ctx, CTX_MAIN, OT_ATTACH, OT_ENERGY, OT_ENERGY_CARD,
)
from cg.api import SelectContext

# line card ids
DREEPY, DRAKLOAK, DRAGAPULT = 119, 120, 121
DUSKULL, DUSCLOPS, DUSKNOIR = 131, 132, 133
RARE_CANDY, CRISPIN = 1079, 1198
FIRE, PSYCHIC, DARK = 2, 5, 7
PHANTOM_DIVE = 154

# evolution-target preference (best first) and energy-search preference for Phantom Dive
_EVOLVE_PREF = [DRAGAPULT, DUSKNOIR, DRAKLOAK, DUSCLOPS]
_EVOLVE_FROM_PREF = [DREEPY, DUSKULL, DRAKLOAK, DUSCLOPS]
_ATTACH_TARGET_PREF = [DRAGAPULT, DRAKLOAK, DREEPY, DUSKNOIR]
_ENERGY_PREF = [FIRE, PSYCHIC, DARK]

_CTX_EVOLVES_FROM = int(SelectContext.EVOLVES_FROM)
_CTX_EVOLVES_TO = int(SelectContext.EVOLVES_TO)
_CTX_ATTACH_TO = int(SelectContext.ATTACH_TO)
_CTX_TO_ACTIVE = int(SelectContext.TO_ACTIVE)
_CTX_TO_BENCH = int(SelectContext.TO_BENCH)
_CTX_TO_HAND = int(SelectContext.TO_HAND)
_CTX_DAMAGE_COUNTER = int(SelectContext.DAMAGE_COUNTER)
_CTX_DAMAGE_COUNTER_ANY = int(SelectContext.DAMAGE_COUNTER_ANY)


def _prefer(ctx: _Ctx, priority_ids) -> Optional[list[int]]:
    """For a sub-select over cards, pick the highest-priority available card ids up to maxCount.
    Single-select → the single best; multi-select → best..maxCount. None if none present / can't
    fill minCount legally (defer to v1)."""
    chosen: list[int] = []
    for cid in priority_ids:
        idx = ctx.card_option(cid)
        if idx is not None and idx not in chosen:
            chosen.append(idx)
            if len(chosen) >= ctx.sel.maxCount:
                break
    if len(chosen) < ctx.sel.minCount:
        return None
    return chosen[:ctx.sel.maxCount] if chosen else None


class DragapultDusknoirExpert(ScriptedExpert):
    name = "dragapult-dusknoir"

    def script(self, ctx: _Ctx) -> Optional[list[int]]:
        c = ctx.context

        # --- follow-up sub-selects triggered by our scripted plays -----------------------
        if c == _CTX_EVOLVES_TO:
            return _one(_prefer(ctx, _EVOLVE_PREF))
        if c == _CTX_EVOLVES_FROM:
            return _one(_prefer(ctx, _EVOLVE_FROM_PREF))
        if c in (_CTX_ATTACH_TO, _CTX_TO_ACTIVE, _CTX_TO_BENCH):
            # attaching energy / placing a Pokémon → steer onto the attacker line
            return _one(_prefer(ctx, _ATTACH_TARGET_PREF))
        if c == _CTX_TO_HAND and ctx.single:
            # e.g. Crispin's basic-energy search → grab Phantom Dive's colours
            return _one(_prefer(ctx, _ENERGY_PREF))
        if c in (_CTX_DAMAGE_COUNTER_ANY, _CTX_DAMAGE_COUNTER) and ctx.single:
            # Phantom Dive spread: concentrate counters on one opponent BENCH target to push it
            # into KO range next turn (a KO-setup proxy for "within next-turn KO range").
            pool = ctx.opp_targets(prefer_bench=True)
            return [pool[0]] if pool else None

        # --- MAIN turn line (setup first; attack falls out last) -------------------------
        if c != CTX_MAIN or not ctx.single:
            return None

        # 1. Rare Candy the Dreepy straight into Dragapult ex, if the ex is in hand.
        if ctx.in_play(DREEPY) and ctx.in_hand(DRAGAPULT):
            i = ctx.play_card(RARE_CANDY)
            if i is not None:
                return [i]

        # 2. Evolve up the lines (best target first).
        for cid in _EVOLVE_PREF:
            i = ctx.evolve_to(cid)
            if i is not None:
                return [i]

        # 3. Crispin for the energy, before committing, if we haven't used our Supporter.
        if not ctx.supporter_played:
            i = ctx.play_card(CRISPIN)
            if i is not None:
                return [i]

        # 4. Attach energy (target resolved by the ATTACH_TO sub-select above).
        i = ctx.any_attach()
        if i is not None:
            return [i]

        # 5. Payoff: Phantom Dive once the attacker is loaded (only reached after setup).
        i = ctx.attack(PHANTOM_DIVE)
        if i is not None:
            return [i]

        return None   # Ultra Ball / Poffin / draw supporters / Boss's / retreat / pass → v1


def _one(pick):
    return pick if pick else None

"""Scripted line-expert base (docs §3d step 2).

A line-expert is a per-archetype *scripted* pilot that executes the deck's known skeleton lines
(evolve on curve, energy onto the planned attacker, key supporters before attacking, the payoff
attack). It is NOT the shipped agent — it is (a) a behavioral-cloning teacher for the multi-deck
net and (b) the per-archetype validation baseline (§3c's "heuristic cross-check until the net beats
it"). The bar every expert must clear: **beat v1 while piloting its own archetype.**

Design: an expert answers each decision with a short, ordered list of high-confidence scripted
rules. When no rule fires with confidence (an unrecognized context, an ambiguous target), it
**falls back to the v1 evaluator-guided brain** — so an expert is v1-plus-scripted-priorities and
can never score below the v1 baseline or (via the outer try/except) return an illegal move.

The `choose(obs_dict, time_left=None) -> list[int]` / `set_deck(deck)` interface matches the agent
contract, so an expert drops straight into the round-robin / recorder / gate harnesses.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Callable, Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(REPO, "submissions", "v1"), os.path.join(REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cg.api import (  # noqa: E402
    to_observation_class, all_card_data, all_attack, OptionType, SelectContext,
)
from ptcg.perception import build_game_state  # noqa: E402

# option-type / context ints (avoid enum churn in hot predicates)
OT_ENERGY_CARD = int(OptionType.ENERGY_CARD)
OT_ENERGY = int(OptionType.ENERGY)
OT_PLAY = int(OptionType.PLAY)
OT_ATTACH = int(OptionType.ATTACH)
OT_EVOLVE = int(OptionType.EVOLVE)
OT_ABILITY = int(OptionType.ABILITY)
OT_ATTACK = int(OptionType.ATTACK)
OT_CARD = int(OptionType.CARD)
CTX_MAIN = int(SelectContext.MAIN)

_CARD_INDEX: Optional[dict] = None
_ATTACK_INDEX: Optional[dict] = None
_V1 = None                      # the fallback brain (v1 agent module)


def _lazy():
    global _CARD_INDEX, _ATTACK_INDEX, _V1
    if _CARD_INDEX is None:
        _CARD_INDEX = {c.cardId: c for c in all_card_data()}
        _ATTACK_INDEX = {a.attackId: a for a in all_attack()}
    if _V1 is None:
        _V1 = importlib.import_module("agent")   # submissions/v1/agent.py (on path above)


class ScriptedExpert:
    """Base pilot. Subclasses set `name` + `archetype_id` and implement `script(ctx) -> list[int] |
    None`. Everything else (parsing, legality, v1 fallback, crash safety) lives here."""

    name: str = "base"

    def __init__(self):
        self._deck: list[int] = []

    # --- agent contract -----------------------------------------------------------------
    def set_deck(self, deck) -> None:
        self._deck = list(deck)

    def choose(self, obs_dict: dict, time_left: Optional[float] = None) -> list[int]:
        try:
            _lazy()
            obs = to_observation_class(obs_dict)
            sel = obs.select
            if sel is None:
                return list(self._deck)
            n = len(sel.option)
            if n == 0:
                return []
            _V1.set_deck(self._deck)                     # keep the fallback's deck in sync
            gs = None
            try:
                gs = build_game_state(obs, _CARD_INDEX)
            except Exception:
                gs = None
            ctx = _Ctx(self, obs, sel, gs, time_left)
            pick = None
            try:
                pick = self.script(ctx)
            except Exception:
                pick = None
            if pick is not None and _legal(pick, sel):
                return pick
            return _V1.choose(obs_dict, time_left=time_left)   # v1 baseline for the rest
        except Exception:
            return _legal_min(obs_dict)

    # --- to be overridden ---------------------------------------------------------------
    def script(self, ctx: "_Ctx") -> Optional[list[int]]:
        """Return option indices for a recognized scripted line, or None to defer to v1."""
        return None


def _legal(pick, sel) -> bool:
    n = len(sel.option)
    if not isinstance(pick, list):
        return False
    if not (sel.minCount <= len(pick) <= sel.maxCount):
        return False
    return len(set(pick)) == len(pick) and all(isinstance(i, int) and 0 <= i < n for i in pick)


def _legal_min(obs_dict) -> list[int]:
    try:
        sel = obs_dict.get("select") if isinstance(obs_dict, dict) else None
        if not sel:
            return []
        n = len(sel.get("option") or [])
        return list(range(min(max(sel.get("minCount", 0), 0), n)))
    except Exception:
        return [0]


class _Ctx:
    """Everything a `script` needs about one decision + reusable option queries."""

    def __init__(self, expert: ScriptedExpert, obs: Any, sel: Any, gs: Any, time_left):
        self.expert = expert
        self.obs = obs
        self.sel = sel
        self.gs = gs
        self.time_left = time_left
        self.options = list(sel.option or [])
        self.context = int(getattr(sel, "context", 0) or 0)
        self.single = (sel.maxCount == 1 and sel.minCount == 1)
        self.ci = _CARD_INDEX
        self.ai = _ATTACK_INDEX
        cur = getattr(obs, "current", None)
        self.my_index = getattr(cur, "yourIndex", 0) if cur else 0

    # --- option finders (return the option INDEX or None) --------------------------------
    def _find(self, pred: Callable[[Any], bool]) -> Optional[int]:
        for i, o in enumerate(self.options):
            try:
                if pred(o):
                    return i
            except Exception:
                continue
        return None

    def play_card(self, card_id: int) -> Optional[int]:
        """Index of the option that PLAYs a specific hand card (Item/Supporter/Tool)."""
        return self._find(lambda o: int(getattr(o, "type", -1) or -1) == OT_PLAY
                           and getattr(o, "cardId", None) == card_id)

    def evolve_to(self, card_id: int) -> Optional[int]:
        return self._find(lambda o: int(getattr(o, "type", -1) or -1) == OT_EVOLVE
                          and getattr(o, "cardId", None) == card_id)

    def attack(self, attack_id: int) -> Optional[int]:
        return self._find(lambda o: int(getattr(o, "type", -1) or -1) == OT_ATTACK
                          and getattr(o, "attackId", None) == attack_id)

    def any_attach(self) -> Optional[int]:
        return self._find(lambda o: int(getattr(o, "type", -1) or -1) in (OT_ATTACH, OT_ENERGY, OT_ENERGY_CARD))

    def card_option(self, card_id: int) -> Optional[int]:
        """Any option (any type) that references a specific card id — for follow-up sub-selects
        (evolve-from/to, energy search, attach-to a card)."""
        return self._find(lambda o: getattr(o, "cardId", None) == card_id)

    def option_card_id(self, i: int) -> Optional[int]:
        return getattr(self.options[i], "cardId", None) if 0 <= i < len(self.options) else None

    def opp_targets(self, prefer_bench: bool = True) -> list[int]:
        """Option indices that target an opponent Pokémon (e.g. damage-counter placement / gust),
        bench-first when `prefer_bench`. Stable order → repeated single-selects concentrate on the
        same target rather than spreading thin."""
        opp = 1 - self.my_index
        bench, other = [], []
        for i, o in enumerate(self.options):
            pidx = getattr(o, "playerIndex", None)
            if pidx is not None and pidx != opp:
                continue
            loc = getattr(o, "inPlayArea", None)
            if loc is None:
                loc = getattr(o, "area", None)
            (bench if loc == 5 else other).append(i)   # AreaType.BENCH == 5
        return (bench + other) if not prefer_bench else (bench or other)

    # --- board queries -------------------------------------------------------------------
    def in_play(self, card_id: int, side: str = "me") -> bool:
        gs = self.gs
        if gs is None:
            return False
        p = gs.me if side == "me" else gs.opponent
        for pk in ([p.active] if getattr(p, "active", None) else []) + list(getattr(p, "bench", []) or []):
            if getattr(pk, "card_id", None) == card_id:
                return True
        return False

    def in_hand(self, card_id: int) -> bool:
        gs = self.gs
        if gs is None:
            return False
        return any(getattr(c, "card_id", None) == card_id for c in (getattr(gs.me, "hand", None) or []))

    @property
    def turn(self) -> int:
        return int(getattr(self.gs, "turn", 0) or 0) if self.gs is not None else 0

    @property
    def supporter_played(self) -> bool:
        return bool(getattr(self.gs, "supporter_played", False)) if self.gs is not None else True

"""Turn-plan / KO-target skill prior — a hand-authored decision tree as an MCTS *root* prior.

Encodes the competitive turn-planning heuristics (prize map + prize-trade efficiency +
prize-count-as-resource) as an explicit decision tree, then turns its recommendation into a
prior distribution over the root's legal options so MCTS explores skill-consistent lines first.

Design choices (deliberate, tied to the v7 pipeline):
  - **Root only.** `prior_fn` is called on every expanded node; scoring every option one-ply at
    every node would gut sims/move on routine turns (0.4 s, K=1). We apply the skill prior only at
    the K root nodes — one O(#options) native-step pass per move, the same cost as the existing
    `_multi_pick` root heuristic — and let the net/uniform prior run the tree interior unchanged.
  - **Blend, don't replace.** `P = (1-lam)*base + lam*skill`, so a wrong tree recommendation is a
    soft bias PUCT can override, not a hard override. `lam` is the single knob to A/B.
  - **Never raises.** Any failure degrades to `base` (or uniform) — the crash=loss guarantee.

Grounded in: turn-planning framework (prize map + "what will my opponent do"), prize-trade
fundamentals (non-ex = 1 prize, ex = 2, megaEx = 3; favorable trade), efficient-vs-direct KO,
Counter Catcher unlocked when behind, and the "opponent at <=2 prizes -> snipe singles" rule. See
docs/ptcg_agent_strategy.md and experiments/log.md.

VERIFY resolutions (Task 1 — checked against src/ptcg/perception.py + cg/api.py, 2026-07):
  - `GameState.me/.opponent` are `PlayerView`; `.active` is `PokemonView|None`, `.bench` a list of
    `PokemonView`, `.hand` a list of `CardView|None`, `.prize_count` an int (prizes *remaining*).
  - Per-Pokemon fields on `PokemonView`: `.hp`, `.card_id`, `.serial`, `.energy_count`,
    `.energy_types`, `.is_ex`, `.is_mega_ex`, and `.prize_value` (perception already folds the
    CardData `ex`/`megaEx` flags into 1/2/3) — so `_prizes_for` just reads `.prize_value`.
  - `CardView` (hand) exposes `.card_id` (not `.id`).
  - Counter Catcher is NOT in the engine's card pool (1267 cards); gust in the shipped Alakazam
    deck is Boss's Orders (id 1182). Both the gust set and the Counter-Catcher set are resolved by
    name/text from the live card DB at first use (robust to reprints / multiple ids).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ptcg.threat import attack_potential, gust_card_ids   # (attacker, defender, ci, ai) -> (now, pot, name)

# --- leaf intents (mirror the decision-tree leaves; each is an ablation point) --------
DEVELOP = "develop"                 # no KO reachable -> chip / set up the highest-value next target
GUST_THREAT = "gust_threat"         # opp near lethal: pull & KO their lethal attacker
SNIPE = "snipe"                     # opp near lethal: KO the cheapest 1-prize, deny their setup
DIRECT_KO = "direct_ko"             # opp 3+ prizes: KO a 2-prize ex, follow-up secured
GUST_SUPPORT = "gust_support"       # opp 3+ prizes: pull & KO a benched engine piece (1 prize)
EFFICIENT_SINGLE = "efficient_single"  # opp 3+ prizes: cheapest 1-prize, save resources for ex KOs

# Counter Catcher card id(s) in the engine DB. Resolved by name at runtime (see _counter_catcher_ids);
# empty in the current pool. Kept overridable for a known reprint id if it ever appears.
COUNTER_CATCHER_IDS: frozenset[int] = frozenset()
SNIPE_PRIZE_THRESHOLD = 2      # opponent prizes-left at/under which we prefer 1-prize snipes / denial

# runtime-resolved card-id caches (ci is built once at startup; keyed by identity to avoid rescans)
_GUST_CACHE: dict[int, frozenset] = {}
_CC_CACHE: dict[int, frozenset] = {}


@dataclass
class Intent:
    kind: str
    target: Optional[Any] = None   # a PokemonView on the opponent board (or our next attacker)


# --- card-set resolution (name/text over the live DB — robust to reprints) ------------
def _counter_catcher_ids(ci: Mapping) -> frozenset:
    key = id(ci)
    cached = _CC_CACHE.get(key)
    if cached is not None:
        return cached
    ids = set(COUNTER_CATCHER_IDS)
    for cid, cd in ci.items():
        if "counter catcher" in (getattr(cd, "name", "") or "").lower():
            ids.add(cid)
    out = frozenset(ids)
    _CC_CACHE[key] = out
    return out


def _gust_ids(ci: Mapping) -> frozenset:
    """Boss's-Orders-type gust effects (switch an opponent's benched Pokemon into the Active Spot).
    Reuses the text heuristic in threat.gust_card_ids; catches Boss's Orders / Prime Catcher / etc."""
    key = id(ci)
    cached = _GUST_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        out = frozenset(gust_card_ids(ci))
    except Exception:
        out = frozenset()
    _GUST_CACHE[key] = out
    return out


# --- the decision tree (the diagram, as code) ----------------------------------------
def turn_plan(gs: Any, ci: Mapping, ai: Mapping) -> Intent:
    """Top of the tree. With no active on either side there is nothing to attack -> DEVELOP.
    Otherwise the whole 'what do we KO / gust' decision is `ko_target`."""
    my_a, opp_a = gs.me.active, gs.opponent.active
    if my_a is None or opp_a is None:
        return Intent(DEVELOP, target=_chip_target(gs) if opp_a is not None else None)
    return ko_target(gs, ci, ai)


def ko_target(gs: Any, ci: Mapping, ai: Mapping) -> Intent:
    """Attacking/gusting is the plan — choose the target (Task 2).

    Reachable set R = {active if we can KO it} ∪ {bench p : we can gust AND our damage >= p.hp}.
    Gust availability = a Boss's-Orders-type effect in hand, or (when behind) Counter Catcher.
    """
    me, opp = gs.me, gs.opponent
    reach, _behind = _reachable(gs, ci, ai)

    if not reach:                                   # nothing KO-able this turn
        return Intent(DEVELOP, target=_chip_target(gs))   # CHIP / SET UP the best next target

    if opp.prize_count <= SNIPE_PRIZE_THRESHOLD:    # they are near lethal — deny, don't dump
        threat = _lethal_threat_in(reach, me.active, ci, ai)
        if threat is not None:
            return Intent(GUST_THREAT, target=threat)     # remove their last KO, break their map
        return Intent(SNIPE, target=_lowest_cost_single(reach))

    # opponent has 3+ prizes ----------------------------------------------------------
    ex = _best_ex_in(reach)
    if ex is not None and _follow_up_secured(gs):
        return Intent(DIRECT_KO, target=ex)               # max prize velocity, advance the 2-2-2 map
    sup = _support_in(reach, opp, ci)
    if sup is not None:
        return Intent(GUST_SUPPORT, target=sup)           # 1 prize but cripples their engine
    return Intent(EFFICIENT_SINGLE, target=_lowest_cost_single(reach))


# --- helpers (heuristic; each is a small, individually ablatable rule) ----------------
def _prizes_for(poke: Any) -> int:
    """Prizes the opponent takes on KO. Perception precomputes this as `.prize_value` (1/2/3)."""
    pv = getattr(poke, "prize_value", None)
    if isinstance(pv, int) and pv > 0:
        return pv
    if getattr(poke, "is_mega_ex", False):
        return 3
    return 2 if getattr(poke, "is_ex", False) else 1


def _damage_to(my_active: Any, target: Any, ci: Mapping, ai: Mapping,
               hand_size: Optional[int] = None) -> int:
    """Damage our active can deal to `target` this turn (weakness/resistance-aware). `hand_size`, when
    given, lets hand-scaling attacks (e.g. Alakazam's Powerful Hand = 20×hand) resolve (#63 fix)."""
    if my_active is None or target is None:
        return 0
    ctx = {"attacker_hand_size": hand_size} if hand_size is not None else None
    now, _pot, _name = attack_potential(my_active, target, ci, ai, context=ctx)
    return now


def _has_gust(me: Any, behind: bool, ci: Mapping) -> bool:
    """Do we hold something that pulls an opponent's benched Pokemon active this turn?"""
    hand = me.hand or []
    gset = _gust_ids(ci)
    if any(getattr(c, "card_id", None) in gset for c in hand):
        return True
    if behind and any(getattr(c, "card_id", None) in _counter_catcher_ids(ci) for c in hand):
        return True
    return False


def _reachable(gs: Any, ci: Mapping, ai: Mapping) -> tuple[list, bool]:
    """R = targets we can KO this turn; plus the `behind` flag (more prizes LEFT == further behind)."""
    me, opp = gs.me, gs.opponent
    my_a = me.active
    behind = me.prize_count > opp.prize_count
    hand_size = len(me.hand or [])                       # for hand-scaling attacks (Powerful Hand)
    reach: list = []
    if my_a is None:
        return reach, behind
    if opp.active is not None and opp.active.hp > 0 \
            and _damage_to(my_a, opp.active, ci, ai, hand_size) >= opp.active.hp:
        reach.append(opp.active)
    if _has_gust(me, behind, ci):
        for p in (opp.bench or []):
            if p.hp > 0 and _damage_to(my_a, p, ci, ai, hand_size) >= p.hp:
                reach.append(p)
    return reach, behind


def _lowest_cost_single(reach: list) -> Optional[Any]:
    """Cheapest-to-KO 1-prize target in R (tie-break hp asc); any target if no 1-prize is reachable."""
    singles = [p for p in reach if _prizes_for(p) == 1] or reach
    return min(singles, key=lambda p: p.hp, default=None)


def _best_ex_in(reach: list) -> Optional[Any]:
    """Highest-prize ex/megaEx in R (prizes desc, then hp asc)."""
    exs = [p for p in reach if _prizes_for(p) >= 2]
    return max(exs, key=lambda p: (_prizes_for(p), -p.hp), default=None)


def _can_ko(attacker: Any, defender: Any, ci: Mapping, ai: Mapping) -> bool:
    if attacker is None or defender is None or defender.hp <= 0:
        return False
    now, _pot, _name = attack_potential(attacker, defender, ci, ai)
    return now >= defender.hp


def _lethal_threat_in(reach: list, my_active: Any, ci: Mapping, ai: Mapping) -> Optional[Any]:
    """The p ∈ R that is their likely lethal attacker next turn (can KO our active now).
    Prefer the biggest hitter."""
    if my_active is None:
        return None
    threats = [p for p in reach if _can_ko(p, my_active, ci, ai)]
    return max(threats, key=lambda p: _damage_to(p, my_active, ci, ai), default=None)


def _is_support(poke: Any, ci: Mapping) -> bool:
    """Detect a draw/setup engine piece by its ability text (VERIFY: ability-text heuristic, mirrors
    threat.gust_card_ids' keyword approach — no card tagging exists in the DB)."""
    cd = ci.get(getattr(poke, "card_id", None))
    if cd is None:
        return False
    for sk in getattr(cd, "skills", None) or []:
        t = (getattr(sk, "text", "") or "").lower()
        if any(k in t for k in ("draw", "search your deck", "into your hand", "look at the top")):
            return True
    return False


def _support_in(reach: list, opp: Any, ci: Mapping) -> Optional[Any]:
    """A reachable benched engine piece to gust & KO (opportunity-cost play). Bench only — sniping the
    active isn't a 'gust support'. Prefer the cheapest to KO."""
    bench_serials = {getattr(p, "serial", None) for p in (opp.bench or [])}
    cands = [p for p in reach if getattr(p, "serial", None) in bench_serials and _is_support(p, ci)]
    return min(cands, key=lambda p: p.hp, default=None)


def _next_attacker(gs: Any) -> Optional[Any]:
    """Bench Pokemon closest to online (crude: most energy attached)."""
    bench = gs.me.bench or []
    return max(bench, key=lambda p: getattr(p, "energy_count", 0) or 0, default=None)


def _follow_up_secured(gs: Any) -> bool:
    """Do we have a second attacker that can KO again next turn? Crude proxy: a benched attacker
    with any energy already attached."""
    nxt = _next_attacker(gs)
    return nxt is not None and (getattr(nxt, "energy_count", 0) or 0) > 0


def _chip_target(gs: Any) -> Optional[Any]:
    """No KO this turn -> put damage on the highest-value target reachable next turn
    (prizes desc, then hp asc)."""
    opp = gs.opponent
    targets = ([opp.active] if opp.active is not None else []) + list(opp.bench or [])
    return max(targets, key=lambda p: (_prizes_for(p), -p.hp), default=None)


# --- prior construction: intent -> distribution over the root's legal options ---------
def skill_root_prior(obs: Any, node: Any, ci: Mapping, ai: Mapping, *,
                     base_prior: Optional[dict] = None, lam: float = 0.5,
                     leaf_value=None) -> dict:
    """Return {action: prob} over `node.actions`, biased toward options consistent with the
    turn-plan intent. Steps each option one ply (like `_heuristic_pick`) and scores the resulting
    state; options whose result matches the intent target get an additive bonus. Blends with
    `base_prior` (net or uniform). Falls back to base/uniform on any error.
    """
    from cg.api import search_step
    from search.mcts import _value as _leaf_value          # root-perspective leaf value in (-1,1)
    from ptcg.perception import build_game_state

    actions = list(node.actions)
    base = base_prior or ({a: 1.0 / len(actions) for a in actions} if actions else {})
    if lam == 0.0 or not actions:
        return dict(base)
    try:
        gs = build_game_state(obs, ci)
        intent = turn_plan(gs, ci, ai)
        root_player = obs.current.yourIndex
        opp_index = 1 - root_player
        scores: dict = {}
        for a in actions:
            pick = [] if a < 0 else [a]                    # a == _PASS (-1) -> []
            try:
                child = search_step(node.state.searchId, pick)
                v = _leaf_value(child, root_player, ci, ai, leaf_value)
                scores[a] = v + _intent_bonus(child, intent, opp_index)
            except Exception:
                scores[a] = -1.0
        skill = _softmax(scores)
        return {a: (1.0 - lam) * base.get(a, 0.0) + lam * skill.get(a, 0.0) for a in actions}
    except Exception:
        return dict(base)


def _intent_bonus(child_state: Any, intent: Intent, opp_index: int) -> float:
    """Reward the option whose resulting state realises the intent — the intended target is gone
    (KO'd) from the opponent's board. Identity by serial (unique per match)."""
    if intent is None or intent.target is None:
        return 0.0
    try:
        tgt_serial = getattr(intent.target, "serial", None)
        if tgt_serial is None:
            return 0.0
        cur = getattr(child_state.observation, "current", None)
        players = getattr(cur, "players", None) or []
        if opp_index >= len(players):
            return 0.0
        opp = players[opp_index]
        in_play = list(opp.active or []) + list(opp.bench or [])
        still_there = any(pk is not None and getattr(pk, "serial", None) == tgt_serial
                          for pk in in_play)
        return 0.6 if not still_there else 0.0            # target removed by this option -> boost
    except Exception:
        return 0.0


def _softmax(scores: dict) -> dict:
    import math
    if not scores:
        return {}
    m = max(scores.values())
    exp = {a: math.exp(s - m) for a, s in scores.items()}
    z = sum(exp.values()) or 1.0
    return {a: v / z for a, v in exp.items()}

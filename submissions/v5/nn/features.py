"""State featurization — Belief Engine → network interface (docs §3 Layer 3).

Turns a Perception `GameState` into a **fixed-schema** set of numpy arrays the encoder network
consumes: per-zone card-id sequences (for the embedding table), per-Pokémon scalar features, our
hand, the stadium, and a global scalar vector that folds in the strategy objects (§4) — the
prize race, threat/offense output — so the net gets them as *inputs* (safer than reward shaping).

Design rules (mirror `perception.py`):
  - **Pure Python + numpy, no torch** — this is the interface; the torch encoder imports it, and it
    stays unit-testable and importable inside the agent.
  - **Never raises on a legal GameState** — unknown ids, empty boards, face-down Pokémon all degrade
    gracefully (the agent's crash=loss guarantee). Card ids are emitted raw (1..1267); the embedding
    table reserves PAD_ID=0 (empty slot) and UNKNOWN_ID (opponent face-down Pokémon).
  - **Fixed shapes** — every state yields identical array shapes (padded + masked), so batches stack.

Perspective is always "me": zone 0 = our active, 1..5 our bench, 6 opp active, 7..11 opp bench.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from ptcg.threat import attack_potential
from ptcg.prize import prize_race

# --- vocabulary + shapes (the encoder reads these) -----------------------------------
PAD_ID = 0                 # empty slot / padding
UNKNOWN_ID = 1268          # opponent face-down Pokémon (its id is hidden from us)
VOCAB_SIZE = 1269          # engine ids 1..1267 + PAD(0) + UNKNOWN(1268)

MAX_BENCH = 5
MAX_HAND = 12              # hand card-id slots (padded/truncated)
POKEMON_SLOTS = 2 * (1 + MAX_BENCH)      # 12: our active+bench then opp active+bench
POKEMON_SCALAR_DIM = 10
GLOBAL_DIM = 36

MAX_OPTIONS = 64          # legal options per decision (padded/truncated for the policy head)
N_OPTION_TYPES = 17       # cg.api.OptionType values 0..16
OPTION_SCALAR_DIM = N_OPTION_TYPES + 10   # type one-hot + 10 scalar features

# AreaType values (mirror cg.api.AreaType — kept native-free like perception)
_AREA_HAND, _AREA_DISCARD, _AREA_ACTIVE, _AREA_BENCH = 2, 3, 4, 5

_DMG_NORM = 300.0          # a big attack ~300 damage
_T6_NORM = 12.0            # turns-to-6 cap
_COST_NORM = 5.0


@dataclass
class ActionFeatures:
    option_card_ids: np.ndarray   # int32   [MAX_OPTIONS]  card the option refers to (PAD if none)
    option_scalars: np.ndarray    # float32 [MAX_OPTIONS, OPTION_SCALAR_DIM]
    option_mask: np.ndarray       # bool    [MAX_OPTIONS]  real (non-padding) option
    n_options: int                # true option count (may exceed MAX_OPTIONS → truncated)
    context: int                  # SelectContext of the decision


@dataclass
class StateFeatures:
    pokemon_ids: np.ndarray       # int32   [POKEMON_SLOTS]
    pokemon_scalars: np.ndarray   # float32 [POKEMON_SLOTS, POKEMON_SCALAR_DIM]
    pokemon_mask: np.ndarray      # bool    [POKEMON_SLOTS]  (slot occupied)
    hand_ids: np.ndarray          # int32   [MAX_HAND]
    hand_mask: np.ndarray         # bool    [MAX_HAND]
    stadium_id: np.int32          # scalar card id (PAD_ID if none)
    globals: np.ndarray           # float32 [GLOBAL_DIM]


def _pk_scalars(pv: Any, is_active: bool, side: int) -> list[float]:
    """Per-Pokémon scalar features (side: 0=ours, 1=opponent)."""
    mh = getattr(pv, "max_hp", 0) or 0
    hp = getattr(pv, "hp", 0) or 0
    energy = getattr(pv, "energy_count", 0) or 0
    etypes = len(set(getattr(pv, "energy_types", None) or []))
    return [
        (hp / mh) if mh else 0.0,                       # hp fraction
        (getattr(pv, "damage", 0) / mh) if mh else 0.0,  # damage fraction
        min(energy, 8) / 8.0,                            # energy attached
        min(etypes, 4) / 4.0,                            # distinct energy types
        getattr(pv, "prize_value", 1) / 3.0,             # prizes on KO
        float(getattr(pv, "is_ex", False)),
        float(getattr(pv, "is_mega_ex", False)),
        float(getattr(pv, "appeared_this_turn", False)),
        float(is_active),
        float(side),
    ]


def _fill_side(pv_active, active_facedown, bench, side, ids, scal, mask, base):
    """Write one player's active+bench into the shared arrays starting at slot `base`."""
    if pv_active is not None:
        ids[base] = getattr(pv_active, "card_id", UNKNOWN_ID) or UNKNOWN_ID
        scal[base] = _pk_scalars(pv_active, True, side)
        mask[base] = True
    elif active_facedown:                 # opponent's active present but hidden
        ids[base] = UNKNOWN_ID
        mask[base] = True
    for j, pv in enumerate(bench[:MAX_BENCH]):
        slot = base + 1 + j
        ids[slot] = getattr(pv, "card_id", UNKNOWN_ID) or UNKNOWN_ID
        scal[slot] = _pk_scalars(pv, False, side)
        mask[slot] = True


def _globals(gs: Any, ci: Mapping, ai: Mapping) -> np.ndarray:
    me, opp = gs.me, gs.opponent
    my_a, opp_a = me.active, opp.active
    our_now, our_next, _ = attack_potential(my_a, opp_a, ci, ai)
    thr_now, thr_next, _ = attack_potential(opp_a, my_a, ci, ai)
    our_t6, their_t6 = prize_race(gs, our_next, thr_next)
    my_hp = getattr(my_a, "hp", 0) or 0
    opp_hp = getattr(opp_a, "hp", 0) or 0

    g = [
        min(getattr(gs, "turn", 0) or 0, 20) / 20.0,
        0.5 if gs.i_go_first is None else float(gs.i_go_first),
        me.prize_count / 6.0, opp.prize_count / 6.0, (opp.prize_count - me.prize_count) / 6.0,
        min(me.hand_count, 12) / 12.0, min(opp.hand_count, 12) / 12.0,
        me.deck_count / 60.0, opp.deck_count / 60.0,
        me.discard_count / 60.0, opp.discard_count / 60.0,
        (1 + len(me.bench)) / 6.0, (1 + len(opp.bench)) / 6.0,
        float(gs.energy_attached), float(gs.supporter_played),
        float(gs.stadium_played), float(gs.retreated),
        float(me.poisoned), float(me.burned), float(me.asleep), float(me.paralyzed), float(me.confused),
        float(opp.poisoned), float(opp.burned), float(opp.asleep), float(opp.paralyzed), float(opp.confused),
        min(our_now, _DMG_NORM) / _DMG_NORM, min(our_next, _DMG_NORM) / _DMG_NORM,
        min(thr_now, _DMG_NORM) / _DMG_NORM, min(thr_next, _DMG_NORM) / _DMG_NORM,
        float(opp_a is not None and our_now >= opp_hp > 0),   # can KO their active now
        float(my_a is not None and thr_now >= my_hp > 0),     # they can KO ours now
        our_t6 / _T6_NORM, their_t6 / _T6_NORM, (their_t6 - our_t6) / _T6_NORM,
    ]
    return np.asarray(g, dtype=np.float32)


def encode_state(game_state: Any, card_index: Mapping[int, Any],
                 attack_index: Mapping[int, Any]) -> StateFeatures:
    """Featurize a Perception `GameState` into fixed-shape arrays for the encoder network."""
    ids = np.full(POKEMON_SLOTS, PAD_ID, dtype=np.int32)
    scal = np.zeros((POKEMON_SLOTS, POKEMON_SCALAR_DIM), dtype=np.float32)
    mask = np.zeros(POKEMON_SLOTS, dtype=bool)

    me, opp = game_state.me, game_state.opponent
    _fill_side(me.active, False, me.bench, 0, ids, scal, mask, 0)
    _fill_side(opp.active, opp.active_facedown, opp.bench, 1, ids, scal, mask, 1 + MAX_BENCH)

    hand_ids = np.full(MAX_HAND, PAD_ID, dtype=np.int32)
    hand_mask = np.zeros(MAX_HAND, dtype=bool)
    hand = getattr(me, "hand", None) or []
    for i, c in enumerate(hand[:MAX_HAND]):
        hand_ids[i] = getattr(c, "card_id", PAD_ID) or PAD_ID
        hand_mask[i] = True

    stadium_id = np.int32(getattr(getattr(game_state, "stadium", None), "card_id", PAD_ID) or PAD_ID)

    return StateFeatures(
        pokemon_ids=ids, pokemon_scalars=scal, pokemon_mask=mask,
        hand_ids=hand_ids, hand_mask=hand_mask, stadium_id=stadium_id,
        globals=_globals(game_state, card_index, attack_index),
    )


# --- action featurization (docs §3 Layer 4 policy head) ------------------------------

def _option_scalars(opt: Any, my_index: int, attack_index: Mapping[int, Any]) -> list[float]:
    """Per-option scalar features: option-type one-hot + attack output + target-location flags +
    numerics. The option's card id (emitted separately) carries the rest via the shared embedding."""
    typ = int(getattr(opt, "type", 0) or 0)
    onehot = [0.0] * N_OPTION_TYPES
    if 0 <= typ < N_OPTION_TYPES:
        onehot[typ] = 1.0

    dmg = cost = 0.0
    aid = getattr(opt, "attackId", None)
    if aid is not None:
        atk = attack_index.get(aid)
        if atk is not None:
            dmg = min(getattr(atk, "damage", 0) or 0, _DMG_NORM) / _DMG_NORM
            cost = min(len(getattr(atk, "energies", None) or []), _COST_NORM) / _COST_NORM

    area = getattr(opt, "area", None)
    inplay = getattr(opt, "inPlayArea", None)
    pidx = getattr(opt, "playerIndex", None)
    scal = [
        dmg, cost,
        float(pidx is None or pidx == my_index),                    # option acts on our card
        float(area == _AREA_ACTIVE or inplay == _AREA_ACTIVE),      # targets an Active
        float(area == _AREA_BENCH or inplay == _AREA_BENCH),        # targets a Bench
        float(area == _AREA_HAND),
        float(area == _AREA_DISCARD),
        min(getattr(opt, "count", 0) or 0, 8) / 8.0,
        min(getattr(opt, "index", 0) or 0, 12) / 12.0,
        min(getattr(opt, "number", 0) or 0, 8) / 8.0,
    ]
    return onehot + scal


def encode_actions(select: Any, my_index: int, attack_index: Mapping[int, Any]) -> ActionFeatures:
    """Featurize a decision's legal options for the pointer policy head. Each option → its card id
    (shared embedding table) + `OPTION_SCALAR_DIM` scalars. Fixed [MAX_OPTIONS] shape, mask-padded;
    options beyond MAX_OPTIONS are truncated (the agent scores them with a uniform fallback)."""
    options = getattr(select, "option", None) or []
    ids = np.full(MAX_OPTIONS, PAD_ID, dtype=np.int32)
    scal = np.zeros((MAX_OPTIONS, OPTION_SCALAR_DIM), dtype=np.float32)
    mask = np.zeros(MAX_OPTIONS, dtype=bool)
    for i, opt in enumerate(options[:MAX_OPTIONS]):
        cid = getattr(opt, "cardId", None)
        ids[i] = cid if (cid is not None and 0 < cid < VOCAB_SIZE) else PAD_ID
        scal[i] = _option_scalars(opt, my_index, attack_index)
        mask[i] = True
    return ActionFeatures(option_card_ids=ids, option_scalars=scal, option_mask=mask,
                          n_options=len(options), context=int(getattr(select, "context", 0) or 0))

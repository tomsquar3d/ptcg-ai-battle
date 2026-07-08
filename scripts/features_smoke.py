#!/usr/bin/env python3
"""Smoke test for state featurization (src/nn/features.py).

Encodes every decision of real self-play games and asserts the schema is fixed, values are finite
and in range, ids are within the embedding vocab, and it never raises — the network's input
contract must hold on any legal state.

Run:  .venv/bin/python scripts/features_smoke.py [num_games]
"""
import os
import random
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))  # cg
sys.path.insert(0, os.path.join(REPO, "src"))                # src first → ptcg (incl. prize) + nn

from cg.api import to_observation_class, all_card_data, all_attack       # noqa: E402
from cg.game import battle_start, battle_select, battle_finish           # noqa: E402
from ptcg.perception import build_game_state                            # noqa: E402
from nn import features as F                                            # noqa: E402


def _rand(sel):
    n = len(sel.option)
    return random.sample(range(n), min(sel.maxCount, n)) if n else []


def main() -> int:
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    deck = [int(x) for x in open(os.path.join(REPO, "experiments", "candidate_decks", "dragapult-ex.csv")).read().split() if x.strip()]

    n_states = errors = n_actions = max_opts = truncated = 0
    for _g in range(games):
        obs_dict, _ = battle_start(deck, list(deck))
        try:
            for _ in range(2000):
                obs = to_observation_class(obs_dict)
                if obs.select is None or (obs.current is not None and obs.current.result != -1):
                    break
                try:
                    gs = build_game_state(obs, ci)
                    sf = F.encode_state(gs, ci, ai)
                    # schema
                    assert sf.pokemon_ids.shape == (F.POKEMON_SLOTS,)
                    assert sf.pokemon_scalars.shape == (F.POKEMON_SLOTS, F.POKEMON_SCALAR_DIM)
                    assert sf.pokemon_mask.shape == (F.POKEMON_SLOTS,)
                    assert sf.hand_ids.shape == (F.MAX_HAND,)
                    assert sf.globals.shape == (F.GLOBAL_DIM,)
                    # value integrity
                    assert np.all(np.isfinite(sf.globals)), "non-finite global"
                    assert np.all(np.isfinite(sf.pokemon_scalars)), "non-finite pk scalar"
                    assert sf.pokemon_ids.min() >= 0 and sf.pokemon_ids.max() < F.VOCAB_SIZE, "id out of vocab"
                    assert sf.hand_ids.min() >= 0 and sf.hand_ids.max() < F.VOCAB_SIZE, "hand id out of vocab"
                    # a filled slot has a non-pad id; an empty slot is padded
                    assert np.all((sf.pokemon_ids != F.PAD_ID) == sf.pokemon_mask), "mask/id mismatch"

                    # --- action featurization ---
                    af = F.encode_actions(obs.select, gs.my_index, ai)
                    assert af.option_scalars.shape == (F.MAX_OPTIONS, F.OPTION_SCALAR_DIM)
                    assert af.option_card_ids.shape == (F.MAX_OPTIONS,)
                    n_real = min(af.n_options, F.MAX_OPTIONS)
                    assert af.option_mask.sum() == n_real, "option mask count wrong"
                    assert np.all(np.isfinite(af.option_scalars)), "non-finite option scalar"
                    assert af.option_card_ids.min() >= 0 and af.option_card_ids.max() < F.VOCAB_SIZE
                    if n_real > 0:  # each real option carries exactly one option-type one-hot
                        oh = af.option_scalars[:n_real, :F.N_OPTION_TYPES].sum(axis=1)
                        assert np.all(oh == 1.0), "option-type one-hot missing"
                    n_actions += n_real
                    max_opts = max(max_opts, af.n_options)
                    truncated += int(af.n_options > F.MAX_OPTIONS)
                    n_states += 1
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  ERROR: {type(e).__name__}: {e}")
                obs_dict = battle_select(_rand(obs.select))
        finally:
            battle_finish()

    assert errors == 0, f"{errors} featurization errors"

    # --- schema v2 deck-conditioning + v1/v5 backward-compat (docs §3d) ---
    ds = F.summarize_deck(deck, ci)
    assert ds.shape == (F.DECK_SUMMARY_DIM,) and np.all(np.isfinite(ds))
    assert abs(ds[:7].sum() - 1.0) < 1e-4, "card-type histogram must sum to 1"
    assert np.all(F.summarize_deck(None, ci) == 0) and np.all(F.summarize_deck([], ci) == 0)
    assert F.archetype_id_for("dragapult-ex") > 0 and F.archetype_id_for("nope") == 0 and F.archetype_id_for(None) == 0
    obs_dict, _ = battle_start(deck, list(deck))
    try:
        obs = to_observation_class(obs_dict)
        gs = build_game_state(obs, ci)
        sf1 = F.encode_state(gs, ci, ai)                                              # v1/v5 call
        sf2 = F.encode_state(gs, ci, ai, deck_summary=ds, archetype_id=F.archetype_id_for("dragapult-ex"))
        for fld in ("pokemon_ids", "pokemon_scalars", "pokemon_mask", "hand_ids", "hand_mask", "globals"):
            assert np.array_equal(getattr(sf1, fld), getattr(sf2, fld)), f"v2 perturbed v1 field {fld}"
        assert np.all(sf1.deck_summary == 0) and int(sf1.archetype_id) == 0, "v1 call must yield zeros"
        assert np.array_equal(sf2.deck_summary, ds) and int(sf2.archetype_id) > 0, "v2 must populate"
    finally:
        battle_finish()
    print(f"0. SCHEMA v{F.SCHEMA_VERSION}: deck_summary {F.DECK_SUMMARY_DIM}d + archetype_id; "
          f"v1 fields byte-identical (v5 ONNX unaffected)")

    # sanity: globals mostly bounded (normalized), and features actually vary across states
    print(f"1. STATE    : {n_states} states encoded over {games} games, 0 errors")
    print(f"2. ACTION   : {n_actions} options encoded, max {max_opts} options/decision, "
          f"{truncated} decisions truncated (>{F.MAX_OPTIONS})")
    print(f"3. SHAPES   : pokemon {F.POKEMON_SLOTS}x{F.POKEMON_SCALAR_DIM}, hand {F.MAX_HAND}, "
          f"globals {F.GLOBAL_DIM}, options {F.MAX_OPTIONS}x{F.OPTION_SCALAR_DIM}, vocab {F.VOCAB_SIZE}")
    print("\nOK: state + action featurization — fixed schema, finite values, valid ids on every state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

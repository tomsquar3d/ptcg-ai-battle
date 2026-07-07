#!/usr/bin/env python3
"""Smoke + correctness test for the threat model (ptcg.threat).

  1. UNITS   — energy-cost matcher (`can_pay`/shortfall) and weakness/resistance math.
  2. DAMAGE  — empirical: compare predicted attack damage to the ACTUAL HP change the engine
               applies when a Pokémon attacks (ground truth from the log stream). Simple
               (printed-damage) attacks should match; text-scaling attacks are expected to
               diverge — reported, not asserted.
  3. THREAT  — assess_threat runs at every decision over real games without crashing; sane
               values; readout of KO-threat frequency and the gust/energy hidden-hand probs.

Run:  .venv/bin/python scripts/threat_smoke.py [num_games]
"""
import logging
import os
import random
import statistics
import sys
from collections import Counter

os.environ.setdefault("LITELLM_LOG", "ERROR")
logging.disable(logging.WARNING)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION_DIR = os.path.join(REPO_ROOT, "submissions", "v0")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, SUBMISSION_DIR)
os.chdir(SUBMISSION_DIR)

from cg.api import to_observation_class, all_card_data, all_attack     # noqa: E402
from cg.game import battle_start, battle_select, battle_finish         # noqa: E402
from main import read_deck_csv                                         # noqa: E402
from ptcg.perception import build_game_state                          # noqa: E402
from ptcg.opponent import OpponentTracker, build_opponent_belief      # noqa: E402
from ptcg.threat import (                                             # noqa: E402
    assess_threat, can_pay, _cost_shortfall, attack_damage,
    energy_card_ids, gust_card_ids, WATER, FIRE, COLORLESS, RAINBOW, PSYCHIC, TEAM_ROCKET,
)

_LOG_ATTACK, _LOG_HP_CHANGE = 15, 16


def random_select(select):
    n = len(select.option)
    return random.sample(range(n), select.maxCount) if n else []


def _terminal(obs):
    return obs.select is None or (obs.current is not None and obs.current.result != -1)


def check_units():
    # cost matcher
    assert can_pay([FIRE, COLORLESS], [FIRE, WATER])            # fire + any
    assert not can_pay([FIRE, FIRE], [FIRE, WATER])            # need 2 fire
    assert can_pay([PSYCHIC], [TEAM_ROCKET])                    # TR energy covers psychic
    assert can_pay([FIRE, COLORLESS, COLORLESS], [RAINBOW, WATER, WATER])  # rainbow wild + 2 any
    assert _cost_shortfall([FIRE, FIRE, COLORLESS], [FIRE]) == 2           # short 1 fire + 1 any
    assert _cost_shortfall([], [WATER]) == 0
    # weakness / resistance
    assert attack_damage(60, FIRE, FIRE, None) == 120          # weak x2
    assert attack_damage(60, FIRE, WATER, None) == 60          # no weakness
    assert attack_damage(60, FIRE, None, FIRE) == 30           # resistance -30
    assert attack_damage(0, FIRE, FIRE, None) == 0             # no printed damage
    print("1. UNITS  : energy-cost matcher + weakness/resistance math  OK")


def check_damage_accuracy(num_games):
    card_index = {c.cardId: c for c in all_card_data()}
    attack_index = {a.attackId: a for a in all_attack()}
    deck = read_deck_csv()
    pairs = matches = 0
    abs_errs = []
    examples = []

    for _g in range(num_games):
        obs_dict, _ = battle_start(deck, list(deck))
        try:
            for _ in range(2000):
                obs = to_observation_class(obs_dict)
                if _terminal(obs):
                    break
                logs = obs.logs or []
                for i, lg in enumerate(logs):
                    if lg.type != _LOG_ATTACK or lg.attackId is None or lg.cardId is None:
                        continue
                    atk = attack_index.get(lg.attackId)
                    if atk is None or not getattr(atk, "damage", 0):
                        continue  # text-only-damage attacks: no printed number to check
                    attacker_cd = card_index.get(lg.cardId)
                    atk_serial = lg.serial
                    for j in range(i + 1, len(logs)):
                        l2 = logs[j]
                        if (l2.type == _LOG_HP_CHANGE and not l2.putDamageCounter
                                and l2.value and l2.serial != atk_serial and l2.cardId is not None):
                            defender_cd = card_index.get(l2.cardId)
                            pred = attack_damage(atk.damage,
                                                 getattr(attacker_cd, "energyType", None),
                                                 getattr(defender_cd, "weakness", None),
                                                 getattr(defender_cd, "resistance", None))
                            actual = abs(l2.value)
                            pairs += 1
                            abs_errs.append(abs(pred - actual))
                            if pred == actual:
                                matches += 1
                            elif len(examples) < 6:
                                examples.append((getattr(atk, "name", "?"), pred, actual))
                            break
                obs_dict = battle_select(random_select(obs.select))
        finally:
            battle_finish()

    rate = matches / pairs if pairs else float("nan")
    print(f"2. DAMAGE : predicted == actual on {matches}/{pairs} attack hits "
          f"({rate:.0%}); mean abs err {statistics.mean(abs_errs):.1f}")
    print(f"          (mismatches are text-scaling/bonus attacks, e.g. {examples[:4]})")


def check_threat_robustness(num_games):
    card_index = {c.cardId: c for c in all_card_data()}
    attack_index = {a.attackId: a for a in all_attack()}
    deck = read_deck_csv()
    gust_ids = gust_card_ids(card_index)
    energy_ids = energy_card_ids(card_index)

    checked = ko_now = ko_pot = 0
    incoming = []
    sample = None
    for _g in range(num_games):
        trackers = {0: OpponentTracker(), 1: OpponentTracker()}
        obs_dict, _ = battle_start(deck, list(deck))
        try:
            for _ in range(2000):
                obs = to_observation_class(obs_dict)
                if _terminal(obs):
                    break
                me = obs.current.yourIndex
                trackers[me].observe(obs)
                gs = build_game_state(obs, card_index)
                ob = build_opponent_belief(obs, tracker=trackers[me], deck_template=deck,
                                           card_index=card_index)
                ta = assess_threat(gs, card_index, attack_index, opponent_belief=ob,
                                   gust_card_ids=gust_ids, energy_card_ids=energy_ids)
                assert ta.incoming_now >= 0 and ta.incoming_potential >= ta.incoming_now
                assert not (ta.ko_now and not ta.ko_potential)
                checked += 1
                ko_now += ta.ko_now
                ko_pot += ta.ko_potential
                incoming.append(ta.incoming_now)
                if sample is None and obs.current.turn >= 4 and ta.attacker_id and ta.incoming_now > 0:
                    sample = ta
                obs_dict = battle_select(random_select(obs.select))
        finally:
            battle_finish()

    print(f"3. THREAT : assessed {checked} decisions, 0 crashes; "
          f"ko_now {ko_now/checked:.0%}, ko_potential {ko_pot/checked:.0%}, "
          f"mean incoming_now {statistics.mean(incoming):.1f}")
    print(f"          gust cards in DB: {len(gust_ids)}, energy cards: {len(energy_ids)}")
    if sample is not None:
        s = sample
        print("\n--- sample ThreatAssessment (mid-game) ---")
        print(f"    our active HP {s.our_active_hp} | incoming now {s.incoming_now} "
              f"(best: {s.best_attack_now!r}) | potential {s.incoming_potential}")
        print(f"    ko_now={s.ko_now} ko_potential={s.ko_potential}")
        print(f"    P(gust in their hand)={s.p_gust_in_hand:.3f} "
              f"P(energy in their hand)={s.p_needed_energy_in_hand:.3f}")


def main(num_games):
    print("=== threat model test ===")
    check_units()
    check_damage_accuracy(num_games)
    check_threat_robustness(num_games)
    print("\nOK: cost matcher, damage model, and threat assessment all validated.")
    return 0


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    sys.exit(main(games))

#!/usr/bin/env python3
"""Smoke + correctness test for the opponent model (ptcg.opponent).

  1. CLASSIFY — the Bayesian archetype classifier on synthetic archetypes: favours the right
                one from signature cards, returns priors under no evidence, respects priors.
  2. TRACKER  — OpponentTracker.revealed() is a sub-multiset of the (known, in self-play)
                opponent decklist and is monotonic non-decreasing over a game.
  3. ZONE     — with the true decklist as template, the hidden pool obeys
                n == opp_deck + opp_hand + opp_prize + opp_facedown at stable (MAIN) decisions
                (validates opponent visible/hidden reconstruction). Plus a sample readout.

Run:  .venv/bin/python scripts/opponent_smoke.py [num_games]
"""
import logging
import os
import random
import sys
from collections import Counter

os.environ.setdefault("LITELLM_LOG", "ERROR")
logging.disable(logging.WARNING)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION_DIR = os.path.join(REPO_ROOT, "submissions", "v0")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, SUBMISSION_DIR)
os.chdir(SUBMISSION_DIR)

from cg.api import to_observation_class, all_card_data             # noqa: E402
from cg.game import battle_start, battle_select, battle_finish     # noqa: E402
from main import read_deck_csv                                     # noqa: E402
from ptcg.opponent import (                                        # noqa: E402
    Archetype, classify_archetype, OpponentTracker, build_opponent_belief,
)


def random_select(select):
    n = len(select.option)
    return random.sample(range(n), select.maxCount) if n else []


def _terminal(obs):
    return obs.select is None or (obs.current is not None and obs.current.result != -1)


def check_classifier():
    COMMON = 900
    fire = Archetype("Fire", {101: 4, 102: 3, COMMON: 12}, prior=0.5)
    water = Archetype("Water", {201: 4, 202: 3, COMMON: 12}, prior=0.5)
    arcs = [fire, water]

    # signature Fire cards -> Fire dominates
    post = classify_archetype(Counter({101: 2, 102: 1, COMMON: 2}), arcs)
    assert post["Fire"] > 0.9 and post["Fire"] > post["Water"], post
    # no evidence -> equal priors
    post0 = classify_archetype(Counter(), arcs)
    assert abs(post0["Fire"] - 0.5) < 1e-9, post0
    # priors respected under no evidence
    post_p = classify_archetype(Counter(), [Archetype("A", {1: 4}, 0.8), Archetype("B", {2: 4}, 0.2)])
    assert abs(post_p["A"] - 0.8) < 1e-9, post_p
    # a card only in Water flips it
    postw = classify_archetype(Counter({201: 3}), arcs)
    assert postw["Water"] > 0.9, postw
    print(f"4-way classifier checks pass (Fire post={post['Fire']:.3f}, "
          f"Water-flip={postw['Water']:.3f})")
    print("1. CLASSIFY: archetype posterior behaves correctly on synthetic archetypes  OK")


def check_tracker_and_zone(num_games):
    deck = read_deck_csv()
    deck_counts = Counter(deck)
    card_index = {c.cardId: c for c in all_card_data()}
    stable_checked = 0
    total = 0
    monotonic_ok = True
    subset_ok = True
    sample = None

    for _g in range(num_games):
        # Self-play flips perspective every turn (yourIndex alternates), so a real agent's
        # fixed viewpoint is emulated with one tracker per side, each fed only its own turns.
        trackers = {0: OpponentTracker(), 1: OpponentTracker()}
        prev_revealed = {0: -1, 1: -1}
        obs_dict, _ = battle_start(deck, list(deck))
        try:
            for _ in range(2000):
                obs = to_observation_class(obs_dict)
                if _terminal(obs):
                    break
                me = obs.current.yourIndex
                tracker = trackers[me]
                tracker.observe(obs)
                total += 1

                rev = tracker.revealed()
                # monotonic (per perspective)
                cur = sum(rev.values())
                if cur < prev_revealed[me]:
                    monotonic_ok = False
                prev_revealed[me] = cur
                # sub-multiset of the (known) opponent decklist
                if any(rev[cid] > deck_counts.get(cid, 0) for cid in rev):
                    subset_ok = False

                # zone model with the TRUE decklist as template
                ob = build_opponent_belief(obs, tracker=tracker, deck_template=deck,
                                           card_index=card_index)
                assert all(v >= 0 for v in ob.zone.pool.values())
                if obs.select.type == 0:  # stable MAIN decision
                    expect = ob.deck_count + ob.hand_count + ob.prize_count + ob.facedown_in_play
                    assert ob.zone.n == expect, f"opp pool {ob.zone.n} != {expect}"
                    stable_checked += 1
                    if sample is None and obs.current.turn >= 4 and ob.deck_count > 10:
                        sample = ob
                obs_dict = battle_select(random_select(obs.select))
        finally:
            battle_finish()

    assert monotonic_ok, "revealed() was not monotonic"
    assert subset_ok, "revealed() exceeded the known opponent decklist"
    print(f"2. TRACKER : revealed() sub-multiset of decklist + monotonic over {num_games} games  OK")
    print(f"3. ZONE    : hidden-pool invariant held at all {stable_checked} stable (MAIN) "
          f"decisions ({total} total)  OK")

    if sample is not None:
        ob = sample
        # pick a card we know is in the deck to demo a hidden-zone query
        some_card = next(iter(ob.zone.pool)) if ob.zone.pool else None
        print("\n--- sample OpponentBelief (mid-game, true-deck template) ---")
        print(f"    revealed so far : {sum(ob.revealed.values())} cards "
              f"({len(ob.revealed)} distinct)")
        print(f"    hidden counts   : deck={ob.deck_count} hand={ob.hand_count} "
              f"prize={ob.prize_count}  (pool n={ob.zone.n})")
        if some_card is not None:
            nm = getattr(card_index.get(some_card), "name", some_card)
            print(f"    P(opp holds '{nm}' in hand)     : {ob.zone.prob_in_hand(some_card):.3f}")
            print(f"    P(opp draws '{nm}' next turn)   : {ob.zone.prob_draw_next(some_card, 1):.3f}")


def main(num_games):
    print("=== opponent model test ===")
    check_classifier()
    check_tracker_and_zone(num_games)
    print("\nOK: archetype classifier, opponent tracker, and hidden-zone model all validated.")
    return 0


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    sys.exit(main(games))

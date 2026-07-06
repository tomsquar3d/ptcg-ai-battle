#!/usr/bin/env python3
"""Smoke + correctness test for the Belief layer (ptcg.belief).

Three checks:
  1. MATH   — DrawModel.prob_at_least vs brute-force enumeration of all k-subsets.
  2. MODEL  — the pool->draw model vs a Monte-Carlo shuffle-and-draw simulation
              (validates the deck/prize exchangeability assumption + the k<=deck cap).
  3. TRACKER— across thousands of real decisions, the reconstructed hidden pool obeys
              the invariant  pool_size == deck_count + unknown_prizes  with no negative
              counts (validates the zone reconstruction — the bug-prone part).

Run:  .venv/bin/python scripts/belief_smoke.py [num_games]
"""
import itertools
import logging
import math
import os
import random
import sys

os.environ.setdefault("LITELLM_LOG", "ERROR")
logging.disable(logging.WARNING)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION_DIR = os.path.join(REPO_ROOT, "submissions", "v0")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, SUBMISSION_DIR)
os.chdir(SUBMISSION_DIR)

from collections import Counter                                    # noqa: E402
from cg.api import to_observation_class, all_card_data             # noqa: E402
from cg.game import battle_start, battle_select, battle_finish     # noqa: E402
from main import read_deck_csv                                     # noqa: E402
from ptcg.belief import DrawModel, build_belief                    # noqa: E402


def random_select(select):
    n = len(select.option)
    return random.sample(range(n), select.maxCount) if n else []


def _terminal(obs):
    return obs.select is None or (obs.current is not None and obs.current.result != -1)


# ---- check 1: hypergeometric vs brute force ------------------------------------------
def check_math():
    cases = [(10, 3, 4, 1), (10, 3, 4, 2), (20, 5, 6, 1), (20, 5, 6, 3),
             (8, 8, 3, 2), (15, 0, 5, 1), (12, 4, 12, 1), (12, 4, 12, 4)]
    worst = 0.0
    for n, K, k, m in cases:
        # pool: K "targets" (id 1) + (n-K) blanks (id 0); deck_count large so cap doesn't bite
        pool = Counter({1: K, 0: n - K})
        dm = DrawModel(pool=pool, deck_count=n)
        got = dm.prob_at_least(1, k=k, m=m)
        # brute force over all k-subsets of a labelled n-card multiset
        cards = [1] * K + [0] * (n - K)
        total = math.comb(n, k)
        succ = sum(1 for combo in itertools.combinations(range(n), k)
                   if sum(cards[i] for i in combo) >= m)
        exp = succ / total
        worst = max(worst, abs(got - exp))
        assert abs(got - exp) < 1e-9, f"math mismatch n={n} K={K} k={k} m={m}: {got} vs {exp}"
    print(f"1. MATH   : prob_at_least matches brute force on {len(cases)} cases "
          f"(max err {worst:.2e})  OK")


# ---- check 2: model vs Monte-Carlo shuffle -------------------------------------------
def check_model(trials=200_000):
    n, K, deck_count, k = 40, 6, 30, 5      # 40 hidden = 30 deck + 10 prize; draw 5
    pool = Counter({1: K, 0: n - K})
    dm = DrawModel(pool=pool, deck_count=deck_count)
    analytic = dm.prob_draw(1, k=k)
    cards = [1] * K + [0] * (n - K)
    rng = random.Random(12345)
    hits = 0
    for _ in range(trials):
        rng.shuffle(cards)
        # first deck_count are the deck; we draw its top k
        if any(cards[i] == 1 for i in range(min(k, deck_count))):
            hits += 1
    empirical = hits / trials
    err = abs(analytic - empirical)
    print(f"2. MODEL  : draw P(>=1 of {K}/{n}, k={k}) analytic={analytic:.4f} "
          f"MonteCarlo={empirical:.4f} (err {err:.4f})  "
          f"{'OK' if err < 0.005 else 'FAIL'}")
    assert err < 0.005


# ---- check 3: reconstruction invariant over real games -------------------------------
def check_tracker(num_games):
    deck = read_deck_csv()
    card_index = {c.cardId: c for c in all_card_data()}
    stable_checked = 0
    total_checked = 0
    sample = None
    for _g in range(num_games):
        obs_dict, _ = battle_start(deck, list(deck))
        try:
            for _ in range(2000):
                obs = to_observation_class(obs_dict)
                if _terminal(obs):
                    break
                b = build_belief(obs, deck, card_index)
                total_checked += 1
                # no negative counts, ever
                assert all(v >= 0 for v in b.draw.pool.values()), "negative pool count"
                # Exact invariant at STABLE points (MAIN action menu, SelectType.MAIN == 0):
                # no card is mid-resolution, so pool == deck + unknown prizes + face-down.
                if obs.select.type == 0:
                    unknown_prizes = b.my_prizes_left - b.known_prize_count
                    assert b.draw.n == b.my_deck_count + unknown_prizes + b.facedown_in_play, (
                        f"pool {b.draw.n} != deck {b.my_deck_count} + unk_prize {unknown_prizes} "
                        f"+ facedown {b.facedown_in_play}")
                    stable_checked += 1
                    if sample is None and obs.current.turn >= 4 and b.my_deck_count > 10:
                        sample = b
                obs_dict = battle_select(random_select(obs.select))
        finally:
            battle_finish()

    print(f"3. TRACKER: exact invariant held at all {stable_checked} stable (MAIN) decisions; "
          f"no negative pool counts across {total_checked} total decisions "
          f"over {num_games} games  OK")
    if sample is not None:
        b = sample
        print("\n--- sample BeliefState (mid-game) ---")
        print(f"    hidden pool: {b.draw.n} cards ({b.my_deck_count} deck + "
              f"{b.my_prizes_left - b.known_prize_count} unknown prize)")
        print(f"    prize race (me,opp): {b.prize_race}   diff: {b.prize_diff:+d}")
        print(f"    P(draw a Basic Pokemon in next 1) : {b.draw.prob_draw_basic_pokemon(1):.3f}")
        print(f"    P(draw any Energy in next 2)      : {b.draw.prob_draw_energy(2):.3f}")
        print(f"    P(draw a Supporter in next 3)     : {b.draw.prob_draw_supporter(3):.3f}")
        print(f"    E[energy cards in next 3 draws]   : {b.draw.expected_count(_energy_pred(card_index), 3):.2f}")


def _energy_pred(card_index):
    return lambda cid: getattr(card_index.get(cid), "cardType", None) in (5, 6)


def main(num_games):
    print("=== belief layer test ===")
    check_math()
    check_model()
    check_tracker(num_games)
    print("\nOK: belief math, draw model, and hidden-zone tracker all validated.")
    return 0


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    sys.exit(main(games))

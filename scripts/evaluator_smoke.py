#!/usr/bin/env python3
"""Smoke + predictive test for the heuristic evaluator (ptcg.evaluator).

  1. UNITS      — terminal states, prize-race sign, terms breakdown present.
  2. PREDICTIVE — the real test: over random self-play, does a higher evaluation of a position
                  actually predict winning that game? Reports Pearson correlation of eval with
                  outcome and win-rate by eval quartile; asserts a clear positive relationship.

Run:  .venv/bin/python scripts/evaluator_smoke.py [num_games]
"""
import logging
import math
import os
import random
import sys
from types import SimpleNamespace as NS

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
from ptcg.evaluator import evaluate, EvalWeights, WIN_SCORE           # noqa: E402

_LOG_RESULT = 23


def random_select(select):
    n = len(select.option)
    return random.sample(range(n), select.maxCount) if n else []


def _fake_player(prize=6, bench=0, hand=5):
    active = NS(hp=100, max_hp=100, card_id=1, energy_types=[], prize_value=1)
    return NS(prize_count=prize, bench=[NS() for _ in range(bench)], active=active,
              hand_count=hand, deck_count=30, has_special_condition=False,
              poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False)


def _fake_gs(me_prize, opp_prize):
    return NS(is_terminal=False, result=-1, my_index=0,
              me=_fake_player(prize=me_prize), opponent=_fake_player(prize=opp_prize))


def check_units():
    assert evaluate(NS(is_terminal=True, result=0, my_index=0)).score == WIN_SCORE
    assert evaluate(NS(is_terminal=True, result=1, my_index=0)).score == -WIN_SCORE
    assert evaluate(NS(is_terminal=True, result=2, my_index=0)).score == 0.0
    ahead = evaluate(_fake_gs(me_prize=1, opp_prize=5)).score   # we've taken more prizes
    behind = evaluate(_fake_gs(me_prize=5, opp_prize=1)).score
    assert ahead > 0 > behind, (ahead, behind)
    terms = evaluate(_fake_gs(2, 4)).terms
    assert "prize_diff" in terms
    print("1. UNITS  : terminal scores, prize-race sign, term breakdown  OK")


def _winner(obs):
    if obs.current is not None and obs.current.result in (0, 1, 2):
        return obs.current.result
    for lg in obs.logs or []:
        if lg.type == _LOG_RESULT and lg.result is not None:
            return lg.result
    return None


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0


def check_predictive(num_games):
    card_index = {c.cardId: c for c in all_card_data()}
    attack_index = {a.attackId: a for a in all_attack()}
    deck = read_deck_csv()
    records = []  # (eval_score, won)

    for _g in range(num_games):
        obs_dict, _ = battle_start(deck, list(deck))
        per_game = []  # (yourIndex, eval)
        winner = None
        try:
            for _ in range(2000):
                obs = to_observation_class(obs_dict)
                if obs.select is None or (obs.current is not None and obs.current.result != -1):
                    winner = _winner(obs)
                    break
                gs = build_game_state(obs, card_index)
                ev = evaluate(gs, card_index=card_index, attack_index=attack_index).score
                per_game.append((obs.current.yourIndex, ev))
                obs_dict = battle_select(random_select(obs.select))
        finally:
            battle_finish()
        if winner in (0, 1):
            for yi, ev in per_game:
                records.append((ev, 1 if winner == yi else 0))

    evals = [r[0] for r in records]
    wins = [r[1] for r in records]
    corr = _pearson(evals, wins)
    pos = [w for e, w in records if e > 0]
    neg = [w for e, w in records if e < 0]
    wr_pos = sum(pos) / len(pos) if pos else float("nan")
    wr_neg = sum(neg) / len(neg) if neg else float("nan")

    # win rate by eval quartile
    order = sorted(records, key=lambda r: r[0])
    q = len(order) // 4
    quartiles = [order[:q], order[q:2 * q], order[2 * q:3 * q], order[3 * q:]]
    wr_q = [sum(w for _, w in blk) / len(blk) for blk in quartiles if blk]

    print(f"2. PREDICT: {len(records)} decisions over {num_games} games")
    print(f"          Pearson(eval, win) = {corr:+.3f}")
    print(f"          win rate: eval>0 {wr_pos:.0%}  vs  eval<0 {wr_neg:.0%}")
    print(f"          win rate by eval quartile (low→high): "
          f"{'  '.join(f'{r:.0%}' for r in wr_q)}")

    assert corr > 0.15, f"eval barely predicts outcome (corr={corr:.3f})"
    assert wr_q[-1] > wr_q[0] + 0.15, f"top quartile not clearly better ({wr_q})"


def main(num_games):
    print("=== heuristic evaluator test ===")
    check_units()
    check_predictive(num_games)
    print("\nOK: evaluator sanity + it meaningfully predicts game outcome under random play.")
    return 0


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    sys.exit(main(games))

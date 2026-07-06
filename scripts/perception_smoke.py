#!/usr/bin/env python3
"""Smoke test for the Perception layer (ptcg.perception).

Drives N random self-play games through the raw SDK and builds a GameState at *every*
in-game decision, asserting it never raises. This exercises the "perception never crashes
on a legal observation" guarantee across as many SelectContexts as random play reaches,
and prints a coverage summary plus one sample GameState.

Run:  .venv/bin/python scripts/perception_smoke.py [num_games]
"""
import logging
import os
import random
import sys

os.environ.setdefault("LITELLM_LOG", "ERROR")
logging.disable(logging.WARNING)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION_DIR = os.path.join(REPO_ROOT, "submissions", "v0")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))   # ptcg
sys.path.insert(0, SUBMISSION_DIR)                    # cg
os.chdir(SUBMISSION_DIR)                              # cg lib + deck.csv are here

from cg.api import to_observation_class, all_card_data          # noqa: E402
from cg.game import battle_start, battle_select, battle_finish   # noqa: E402
from main import read_deck_csv                                   # noqa: E402
from ptcg.perception import build_game_state, GameState          # noqa: E402

MAX_DECISIONS_PER_GAME = 100_000  # safety cap against a pathological loop


def _summarize(gs: GameState) -> str:
    d = gs.decision
    me, opp = gs.me, gs.opponent
    act = me.active.name if me.active else ("<facedown>" if me.active_facedown else "<none>")
    oact = opp.active.name if opp.active else ("<facedown>" if opp.active_facedown else "<none>")
    return (
        f"turn={gs.turn} action#{gs.turn_action_count} me_first={gs.i_go_first} "
        f"decision={d.context_name}({d.min_count}-{d.max_count} of {d.num_options})\n"
        f"    me : active={act!r} bench={len(me.bench)} hand={me.hand_count} "
        f"deck={me.deck_count} prizes={me.prize_count}\n"
        f"    opp: active={oact!r} bench={len(opp.bench)} hand={opp.hand_count} "
        f"deck={opp.deck_count} prizes={opp.prize_count}"
    )


def run(num_games: int) -> int:
    card_index = {c.cardId: c for c in all_card_data()}
    deck = read_deck_csv()

    contexts_seen: dict[str, int] = {}
    total_decisions = 0
    finished = 0
    sample = None  # a GameState from mid-game to print

    for g in range(num_games):
        obs_dict, _ = battle_start(deck, list(deck))
        try:
            for _ in range(MAX_DECISIONS_PER_GAME):
                obs = to_observation_class(obs_dict)
                if obs.current is not None and obs.current.result != -1:
                    finished += 1
                    break
                if obs.select is None:
                    break  # only the pre-deck obs; not expected mid-game here

                gs = build_game_state(obs, card_index)  # <-- the thing under test
                total_decisions += 1
                contexts_seen[gs.decision.context_name] = \
                    contexts_seen.get(gs.decision.context_name, 0) + 1
                if sample is None and gs.turn >= 3 and gs.me.active is not None:
                    sample = gs

                n = len(obs.select.option)
                pick = random.sample(range(n), obs.select.maxCount) if n else []
                obs_dict = battle_select(pick)
        finally:
            battle_finish()

    print("=== perception smoke test ===")
    print(f"games run            : {num_games}  (finished cleanly: {finished})")
    print(f"GameStates built     : {total_decisions}  (0 exceptions)")
    print(f"distinct contexts hit: {len(contexts_seen)}")
    for ctx, cnt in sorted(contexts_seen.items(), key=lambda kv: -kv[1]):
        print(f"    {ctx:<28} {cnt}")
    if sample is not None:
        print("\n--- sample GameState (mid-game) ---")
        print(_summarize(sample))

    assert finished == num_games, f"only {finished}/{num_games} games finished cleanly"
    assert total_decisions > 0, "no decisions were exercised"
    print("\nOK: perception built a GameState for every decision without raising.")
    return 0


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sys.exit(run(games))

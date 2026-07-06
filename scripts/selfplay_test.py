#!/usr/bin/env python3
"""Self-play smoke test for the v0 baseline agent.

Runs N games of submissions/v0/main.py against itself in the Kaggle "cabt"
environment, asserts every game finishes cleanly (both players status DONE),
and reports the win/loss/draw split (from player 0's view) plus average
seconds per game.

Expected for the legal-random baseline: ~50/50 split, ~0.5 s/game.

Run:  .venv/bin/python scripts/selfplay_test.py [num_games]
"""
import logging
import os
import sys
import time

# Importing kaggle_environments pulls in LiteLLM / OpenSpiel, which emit noisy
# import-time logs. Silence them before the import below so test output is clean.
os.environ.setdefault("LITELLM_LOG", "ERROR")
logging.disable(logging.WARNING)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION_DIR = os.path.join(REPO_ROOT, "submissions", "v0")

# The agent (main.py) imports the local `cg` package and reads `deck.csv` with a
# path relative to the process CWD, exactly as it will on Kaggle where main.py,
# deck.csv and cg/ all sit in /kaggle_simulations/agent/. Reproduce that layout
# by running from inside the submission folder.
os.chdir(SUBMISSION_DIR)
sys.path.insert(0, SUBMISSION_DIR)

from kaggle_environments import make  # noqa: E402  (import after chdir/path setup)

AGENT = os.path.join(SUBMISSION_DIR, "main.py")


def _field(agent_state, key):
    """kaggle-environments step entries are dict-like; be defensive."""
    if isinstance(agent_state, dict):
        return agent_state.get(key)
    return getattr(agent_state, key, None)


def run_selfplay(num_games: int) -> int:
    wins = losses = draws = 0
    durations = []
    failures = []

    for i in range(num_games):
        env = make("cabt", debug=True)
        t0 = time.perf_counter()
        env.run([AGENT, AGENT])
        durations.append(time.perf_counter() - t0)

        final = env.steps[-1]
        p0, p1 = final[0], final[1]
        s0, s1 = _field(p0, "status"), _field(p1, "status")

        if s0 != "DONE" or s1 != "DONE":
            failures.append((i, s0, s1))
            print(f"  game {i:>2}: NOT DONE -> p0={s0} p1={s1}")
            continue

        r0 = _field(p0, "reward") or 0
        r1 = _field(p1, "reward") or 0
        if r0 > r1:
            wins += 1
        elif r0 < r1:
            losses += 1
        else:
            draws += 1

    n = len(durations)
    avg = sum(durations) / n if n else float("nan")
    print("\n=== self-play test result ===")
    print(f"games run           : {num_games}")
    print(f"player-0 W / L / D  : {wins} / {losses} / {draws}")
    print(f"avg seconds per game: {avg:.3f}s  (total {sum(durations):.1f}s)")

    assert not failures, (
        f"{len(failures)} game(s) did not end with both players DONE: {failures}"
    )
    print("OK: all games ended with status DONE for both players.")
    return 0


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    sys.exit(run_selfplay(games))

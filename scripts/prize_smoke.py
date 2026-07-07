#!/usr/bin/env python3
"""Unit smoke for the prize-mapping model (src/ptcg/prize.py).

Checks the turns-to-6 arithmetic and the race differential on hand-built fixtures.

Run:  .venv/bin/python scripts/prize_smoke.py
"""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ptcg.prize import turns_to_six, prize_race, TURNS_CAP  # noqa: E402


def _mon(hp, pv):
    return NS(hp=hp, prize_value=pv)


def _player(active, bench, prizes):
    return NS(active=active, bench=bench, prize_count=prizes)


def main() -> int:
    # 1. turns_to_six arithmetic
    assert turns_to_six(0, [(70, 1)], 200) == 0.0, "no prizes needed → 0 turns"
    assert turns_to_six(1, [(70, 1)], 0) == float(TURNS_CAP), "no damage → capped"
    assert turns_to_six(1, [(70, 1)], 200) == 1.0, "one KO, 200≥70 → 1 turn"
    assert turns_to_six(2, [(320, 2)], 200) == 2.0, "ex (2 prizes), ceil(320/200)=2 turns"
    # field short of prizes → fresh-basic tail: 1 on field + 5×ceil(70/200)=5 → 6
    assert turns_to_six(6, [(70, 1)], 200) == 6.0, "tail estimate for missing prizes"
    # value density: KO the 2-prize target first when it's turn-efficient
    assert turns_to_six(2, [(70, 1), (140, 2)], 140) == 1.0, "2-prize target reaches need in 1 turn"
    print("1. turns_to_six arithmetic  OK")

    # 2. race differential: we out-race a slow, high-HP opponent
    me = _player(_mon(200, 1), [_mon(200, 1)], prizes=2)       # our board: tanky, they hit softly
    opp = _player(_mon(70, 1), [_mon(70, 1)], prizes=2)        # their board: fragile, we hit hard
    ours, theirs = prize_race(NS(me=me, opponent=opp), our_output=200, their_output=60)
    assert ours == 2.0 and theirs == 8.0, f"expected (2, 8), got ({ours}, {theirs})"
    assert theirs - ours > 0, "we should be winning the race"
    print(f"2. prize_race differential  OK  (ours={ours}, theirs={theirs}, diff=+{theirs-ours:.0f})")

    # 3. symmetry: swapping outputs/boards flips the sign
    ours2, theirs2 = prize_race(NS(me=opp, opponent=me), our_output=60, their_output=200)
    assert theirs2 - ours2 < 0, "flipped setup should flip the race sign"
    print("3. race sign flips on swap  OK")

    print("\nOK: prize-mapping model validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

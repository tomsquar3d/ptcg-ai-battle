# Candidate Round-Robin (30 games/pairing, pilot=v3)

Decks: 8 archetype candidates (`experiments/candidate_decks/`) + the SDK `sample` deck, all piloted by the `v3` brain. Draws count as half.

## Ranking

| rank | deck | raw win rate | meta-weighted | W/L/D |
|--:|---|--:|--:|--:|
| 1 | sample | 81.2% | 80.9% | 195/45/0 |
| 2 | dragapult-dusknoir | 59.2% | 64.9% | 142/98/0 |
| 3 | dragapult-blaziken | 57.9% | 64.2% | 139/101/0 |
| 4 | dragapult-ex | 52.9% | 57.2% | 127/113/0 |
| 5 | rockets-honchkrow | 52.9% | 56.9% | 127/113/0 |
| 6 | n-zoroark | 42.1% | 45.6% | 101/139/0 |
| 7 | alakazam-dudunsparce | 37.1% | 40.5% | 89/151/0 |
| 8 | mega-greninja-ex | 35.0% | 37.4% | 84/156/0 |
| 9 | slowking-scr | 31.7% | 35.5% | 76/164/0 |

## Win-rate matrix (row vs column, draws as half)

| vs | sample | dragapult- | dragapult- | dragapult- | rockets-ho | n-zoroark | alakazam-d | mega-greni | slowking-s |
|---|---|---|---|---|---|---|---|---|---|
| sample | — | 0.733 | 0.833 | 0.767 | 0.733 | 0.8 | 0.9 | 0.867 | 0.867 |
| dragapult-dusknoir | 0.267 | — | 0.5 | 0.533 | 0.433 | 0.633 | 0.7 | 0.833 | 0.833 |
| dragapult-blaziken | 0.167 | 0.5 | — | 0.533 | 0.567 | 0.833 | 0.567 | 0.733 | 0.733 |
| dragapult-ex | 0.233 | 0.467 | 0.467 | — | 0.633 | 0.667 | 0.567 | 0.667 | 0.533 |
| rockets-honchkrow | 0.267 | 0.567 | 0.433 | 0.367 | — | 0.6 | 0.733 | 0.5 | 0.767 |
| n-zoroark | 0.2 | 0.367 | 0.167 | 0.333 | 0.4 | — | 0.667 | 0.667 | 0.567 |
| alakazam-dudunsparce | 0.1 | 0.3 | 0.433 | 0.433 | 0.267 | 0.333 | — | 0.633 | 0.467 |
| mega-greninja-ex | 0.133 | 0.167 | 0.267 | 0.333 | 0.5 | 0.333 | 0.367 | — | 0.7 |
| slowking-scr | 0.133 | 0.167 | 0.267 | 0.467 | 0.233 | 0.433 | 0.533 | 0.3 | — |

_N=30 per pairing — provisional; coin flips lie at small N (§5 wants ~200). Confirm the top decks at high N before selecting._

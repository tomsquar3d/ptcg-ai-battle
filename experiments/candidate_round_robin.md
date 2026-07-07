# Candidate Round-Robin (40 games/pairing)

Decks: 8 archetype candidates (`experiments/candidate_decks/`) + the SDK `sample` deck, all piloted by the v1 brain. Draws count as half.

## Ranking

| rank | deck | raw win rate | meta-weighted | W/L/D |
|--:|---|--:|--:|--:|
| 1 | sample | 80.0% | 79.6% | 256/64/0 |
| 2 | dragapult-blaziken | 59.4% | 65.7% | 190/130/0 |
| 3 | rockets-honchkrow | 55.9% | 59.3% | 178/140/2 |
| 4 | dragapult-ex | 52.2% | 58.7% | 167/153/0 |
| 5 | dragapult-dusknoir | 47.7% | 51.8% | 152/167/1 |
| 6 | alakazam-dudunsparce | 45.0% | 51.5% | 144/176/0 |
| 7 | slowking-scr | 43.4% | 49.4% | 138/180/2 |
| 8 | n-zoroark | 35.2% | 35.5% | 112/207/1 |
| 9 | mega-greninja-ex | 31.2% | 36.3% | 100/220/0 |

## Win-rate matrix (row vs column, draws as half)

| vs | sample | dragapult- | rockets-ho | dragapult- | dragapult- | alakazam-d | slowking-s | n-zoroark | mega-greni |
|---|---|---|---|---|---|---|---|---|---|
| sample | — | 0.9 | 0.675 | 0.825 | 0.825 | 0.725 | 0.825 | 0.675 | 0.95 |
| dragapult-blaziken | 0.1 | — | 0.65 | 0.675 | 0.55 | 0.7 | 0.5 | 0.725 | 0.85 |
| rockets-honchkrow | 0.325 | 0.35 | — | 0.4 | 0.45 | 0.775 | 0.8 | 0.65 | 0.725 |
| dragapult-ex | 0.175 | 0.325 | 0.6 | — | 0.575 | 0.575 | 0.55 | 0.7 | 0.675 |
| dragapult-dusknoir | 0.175 | 0.45 | 0.55 | 0.425 | — | 0.3 | 0.5 | 0.613 | 0.8 |
| alakazam-dudunsparce | 0.275 | 0.3 | 0.225 | 0.425 | 0.7 | — | 0.575 | 0.675 | 0.425 |
| slowking-scr | 0.175 | 0.5 | 0.2 | 0.45 | 0.5 | 0.425 | — | 0.675 | 0.55 |
| n-zoroark | 0.325 | 0.275 | 0.35 | 0.3 | 0.388 | 0.325 | 0.325 | — | 0.525 |
| mega-greninja-ex | 0.05 | 0.15 | 0.275 | 0.325 | 0.2 | 0.575 | 0.45 | 0.475 | — |

_N=40 per pairing — provisional; coin flips lie at small N (§5 wants ~200). Confirm the top decks at high N before selecting._

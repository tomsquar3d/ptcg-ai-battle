# Meta-Priors Report

Source: TrainerHill skeletons (`data/external/trainerhill/`) + `trainerhill-meta-matchups-2026-07-07.csv`, joined onto the competition pool (EN_Card_Data + engine DB).

## 1. Card mapping coverage

| archetype | skeleton cards | mapped | by-cards | by-copies | flag |
|---|--:|--:|--:|--:|:--:|
| alakazam-dudunsparce | 26 | 26 | 100% | 100% | ok |
| dragapult-blaziken | 27 | 26 | 96% | 98% | ok |
| dragapult-dusknoir | 25 | 24 | 96% | 98% | ok |
| dragapult-ex | 21 | 20 | 95% | 98% | ok |
| mega-greninja-ex | 19 | 17 | 89% | 91% | ok |
| n-zoroark | 25 | 23 | 92% | 93% | ok |
| rockets-honchkrow | 18 | 18 | 100% | 100% | ok |
| slowking-scr | 23 | 23 | 100% | 100% | ok |

### Unmapped skeleton cards (out of pool)

- **dragapult-blaziken**: Special Red Card (CRI-082)
- **dragapult-dusknoir**: Special Red Card (CRI-082)
- **dragapult-ex**: Special Red Card (CRI-082)
- **mega-greninja-ex**: Mega Greninja ex (CRI-022), Special Red Card (CRI-082)
- **n-zoroark**: Transformation Tome (CRI-083), Special Red Card (CRI-082)

## 2. Archetype ranking — constructibility × meta_share × matchup_spread

| rank | archetype | constructibility (copies) | meta_share | matchup_spread | composite |
|--:|---|--:|--:|--:|--:|
| 1 | alakazam-dudunsparce | 100% | 8.1% | 0.150 | 0.0122 |
| 2 | n-zoroark | 93% | 12.0% | 0.089 | 0.0100 |
| 3 | dragapult-ex | 98% | 11.5% | 0.079 | 0.0090 |
| 4 | slowking-scr | 100% | 9.9% | 0.090 | 0.0088 |
| 5 | dragapult-dusknoir | 98% | 11.1% | 0.069 | 0.0075 |
| 6 | rockets-honchkrow | 100% | 5.6% | 0.129 | 0.0072 |
| 7 | mega-greninja-ex | 91% | 7.4% | 0.080 | 0.0054 |
| 8 | dragapult-blaziken | 98% | 5.7% | 0.072 | 0.0040 |

## 3. Vulnerability profiles (expected copies)

| archetype | basic energy | special energy | tools (jamming) | %Pokémon w/ ability (watchtower) | draw supporters (red card/judge) |
|---|--:|--:|--:|--:|--:|
| alakazam-dudunsparce | 1.0 | 4.79 | 2.57 | 62% | 0.12 |
| dragapult-blaziken | 8.0 | 0.0 | 0.25 | 52% | 4.16 |
| dragapult-dusknoir | 8.0 | 0.0 | 0.2 | 51% | 4.12 |
| dragapult-ex | 8.0 | 0.0 | 0.0 | 48% | 4.4 |
| mega-greninja-ex | 6.0 | 1.61 | 1.74 | 27% | 3.6 |
| n-zoroark | 8.0 | 0.0 | 1.93 | 47% | 4.07 |
| rockets-honchkrow | 0.0 | 7.99 | 0.66 | 16% | 8.0 |
| slowking-scr | 4.22 | 5.93 | 1.38 | 47% | 4.3 |

## Notes

- meta_share is over the full 15-deck field in the matchup CSV; the 8 skeleton archetypes cover 71% of field games.
- Ability detection uses the engine `skills` list; Tera passive abilities are not modelled there and are undercounted.
- Unmapped cards are genuinely out of the competition pool (chiefly the 'CRI' set), not mapping failures — they lower constructibility as they should.

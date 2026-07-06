# Experiment Log

Running journal of agent versions and experiments. Each entry: hypothesis, change,
setup, result. This is the spine of the Strategy writeup (Model Score = 70%), so
record numbers as they happen — see `docs/ptcg_agent_strategy.md` §7.

---

## #1 — v0 baseline: legal-random agent + self-play harness

**Date:** 2026-07-06
**Agent:** `submissions/v0/` (frozen)
**Deck:** `submissions/v0/deck.csv` (the SDK sample 60-card deck, unmodified)

### Goal
Stand up the project and validate the end-to-end loop before any real strategy:
the SDK loads, the `cabt` environment runs, a legal move is always returned, and
games terminate cleanly. This is milestone step "build + run a legal-random agent"
from the plan (§9).

### Agent description
The unmodified SDK sample (`main.py`):
- On the initial selection (`obs.select is None`) it returns the 60-card deck.
- Otherwise it returns `random.sample(range(len(options)), maxCount)` — i.e. a
  uniformly random pick among the **legal** options the engine presents. It never
  constructs an illegal move, so it cannot crash on move legality.

No belief engine, no heuristics, no lookahead. This is purely the baseline / clock
fallback reference and the ablation floor.

### Setup
- Env: `kaggle_environments` 1.30.2, `cabt` environment (bundled in the pip package).
- Host: WSL2 aarch64, Python 3.12, 2 vCPU-class. Engine lib: `cg/libcg-arm64.so`.
- Harness: `scripts/selfplay_test.py` — runs N games of `submissions/v0/main.py`
  against itself, asserts both players finish with status `DONE`, reports the
  win/loss/draw split (player-0 view, by final reward ±1/0) and avg seconds/game.
- Command: `.venv/bin/python scripts/selfplay_test.py 20`

### Result
```
games run           : 20
player-0 W / L / D  : 9 / 11 / 0
avg seconds per game: 0.139s  (total 2.8s)
OK: all games ended with status DONE for both players.
```

- **All 20 games terminated cleanly** (`DONE`/`DONE`) — no crash, timeout, or INVALID.
- **9/11/0** ≈ the expected ~50/50 for a mirror of a symmetric random policy (small-N
  coin-flip noise; not a signal about deck or policy quality).
- **0.139 s/game**, well under the ~0.5 s expectation and vastly under the 10-min
  per-player clock. Confirms the native engine + harness are fast enough for the
  large self-play batteries the deck round-robins and RL will need.

### Takeaways / next
- Baseline μ on the ladder is the real reference point — submit v0 to record it (§9 step 3).
- The random mirror ≈50/50 is the control the seven-axis deck experiments measure against.
- Next: benchmark `search_step` speed (gates lookahead depth), stand up the Belief
  Engine skeleton, and start the heuristic evaluator (`src/`).

---

## #2 — Belief Engine Layer 1: Perception (`src/ptcg/perception.py`)

**Date:** 2026-07-06
**Code:** `src/ptcg/perception.py`, `scripts/perception_smoke.py`

### Goal
Build the first Belief Engine layer (§3, Layer 1): a **pure** transformation from a raw
engine `Observation` to a structured, decision-friendly `GameState`. Foundation for every
downstream layer.

### Design
- `build_game_state(observation, card_index) -> GameState`. Resolves card IDs to `CardData`
  (names/HP/ex-status), splits the board into `me` / `opponent` `PlayerView`s, wraps in-play
  Pokémon as `PokemonView` (energy count, tools, damage, prize value 1/2/3 for ex/megaEx),
  and normalizes the current decision as a `DecisionView` (with a human-readable
  `context_name`). Exposes `prize_race` and `is_terminal`.
- **No native/ctypes import** — reads plain attributes off the observation, so it is unit-
  testable on fixtures and can never be the layer that fails to import inside the agent.
- **Graceful degradation** — unknown card IDs (→ `#<id>`), empty/face-down boards, and
  enum values added mid-competition (→ `CONTEXT_<n>`) never raise. This is where the
  "crash = loss" guarantee starts.

### Result (`scripts/perception_smoke.py 60`)
Drives random self-play and builds a `GameState` at *every* in-game decision:
```
games run            : 60  (finished cleanly: 60)
GameStates built     : 3298  (0 exceptions)
distinct contexts hit: 11   (MAIN, TO_HAND, DISCARD_ENERGY, SETUP_ACTIVE/BENCH_POKEMON,
                              ATTACH_TO/FROM, TO_ACTIVE, IS_FIRST, SWITCH, DRAW_COUNT)
sample: turn=3 MAIN(1-1 of 9)  me active='Snover' bench=0 hand=7 deck=45 prizes=6 ...
```
- **0 exceptions over 3298 decisions** — perception is robust across every context random
  play reached; card-ID → name resolution and prize/board summaries verified by eye.

### Caveats / next
- Only 11 of 49 `SelectContext`s are reached by a random mirror of the sample deck (no
  evolutions, attacks-with-effects, damage-counter placement, special conditions). Widen
  coverage later with scripted lines or richer decks; add a fixture-based unit test for the
  rarer contexts so they're not only exercised by chance.
- Next Belief work: hidden-zone tracker + hypergeometric draw model; and the week-1
  `search_step` speed benchmark (still outstanding — gates lookahead depth).

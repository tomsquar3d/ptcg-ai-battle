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

---

## #3 — `search_step` speed benchmark (native Search API)

**Date:** 2026-07-06
**Code:** `scripts/search_step_benchmark.py`

### Goal
Resolve the week-1 unknown (§8/§9 item 4): how fast is one engine state transition via
the native lookahead API (`search_begin`/`search_step`/`search_end`)? This number gates
how deep the critical-turn search (§3, Layer 5) can go within a per-move time budget.

### Method
Drive random self-play to a mid-game decision, capture that agent observation (carries
`search_begin_input`), then `search_begin()` a determinized tree and roll out random legal
lines with `search_step()`, timing each. In self-play both full 60-card decklists are known,
so each decklist is handed to the engine as the candidate pool for the hidden zones
(deck/prize/hand). `search_end()` between rollouts keeps memory bounded.

### Result (5000 steps; WSL2 **aarch64**, single-thread, `libcg-arm64.so`)
```
search_begin : 71 calls,  mean 0.249 ms,  median 0.211 ms
search_step  : 5000 samples, 0 errors
  mean 124.0 us | median 109.3 us | p95 203 us | p99 410 us | max 9.7 ms
  throughput ~8,065 steps/sec (single core)
rollout depth to terminal: mean ~71 steps  ->  ~114 full random games/sec
```
Budget translation: ~4k transitions per 0.5 s, ~8k per 1 s, ~24k per 3 s (single core).

### Takeaways
- **Lookahead is comfortably affordable.** At ~8k steps/sec/core (≈16k on the Kaggle 2-vCPU
  box), a sub-second slice of the 600 s clock explores thousands of states — MCTS-style
  critical-turn search and belief-sampled determinization (§9 Monte Carlo uses #2/#3) are
  well within budget. The step-speed gate is **not** a constraint on depth.
- **`search_begin` is cheap (~0.2 ms)** → re-determinizing per Monte Carlo sample (draw a
  fresh opponent hand from the belief distribution each rollout) costs almost nothing.
- **Tail to watch:** p99 ~410 us and rare ~10 ms transitions (heavy resolutions — attacks /
  reshuffles / end-of-turn). Any fixed-node search budget must be wall-clock-checked, not
  step-count-checked, so a cluster of heavy transitions can't blow the per-move clock.

### Caveats
- Measured single-thread on aarch64 WSL; Kaggle is x86 / 2 vCPU — treat as order-of-magnitude,
  not absolute. Re-run on the target image when available.
- Random rollouts under-sample expensive lines (few big attacks in a random mirror); real
  search will weight toward combat, so expect the effective mean above 124 us. Direction and
  magnitude hold: depth is not step-speed-limited.

---

## #4 — Belief Engine Layer 2a: own-deck hidden-zone tracker + hypergeometric draw model

**Date:** 2026-07-06
**Code:** `src/ptcg/belief.py`, `scripts/belief_smoke.py`

### Goal
The exact, priors-free slice of Belief (§3 Layer 2; §4 deck-thinning): track our own hidden
cards and answer draw questions ("P(draw a Basic / energy / supporter in next k)",
"E[energy in next k]"). Feeds the deck-quality metric, out-counting, deck-out risk, and the
determinization sampler that experiment #3 showed is cheap to run.

### Key idea (why it's exact, no learning)
We know our 60-card decklist, and the engine shows every one of our cards except those in
deck and (face-down) prizes. So `pool = decklist − everything-of-ours-we-can-see`, with
`n = |pool| = deck_count + unknown_prizes`. The deck/prize split is a uniform random
partition, so by **exchangeability the next k draws are a uniform k-subset of the pool** →
"P(next k draws contain a target)" is a plain hypergeometric over the pool (prizes counted
in n, never drawn; effective draws capped at deck_count).

### Validation (`scripts/belief_smoke.py 60`)
```
1. MATH   : prob_at_least matches brute-force enumeration on 8 cases (max err 0)
2. MODEL  : P(>=1 of 6/40, k=5) analytic 0.5771 vs Monte-Carlo 0.5816 (err 0.0045)
3. TRACKER: exact pool invariant held at all 1903 stable (MAIN) decisions;
            no negative pool counts across 2784 total decisions over 60 games
sample: 50 hidden (44 deck+6 prize) | P(Basic,1)=0.08 | P(energy,2)=0.845 |
        P(supporter,3)=0.414 | E[energy in 3]=1.80
```

### Two edge cases found & handled
- **Simultaneous setup:** our own active is placed face-down (`None`) in the state snapshot.
  BUT — like real TCG — the engine still tells *us* which card it is: our own placement is a
  `MOVE_CARD` (type 6) log **with** the cardId, while the opponent's is `MOVE_CARD_REVERSE`
  (type 7, no id). The identifying log only lives in the window since our last selection, so
  it scrolls off ~16% of face-down snapshots (130/154 recoverable from the current obs alone).
  Fixed with `OwnZoneTracker` (§2 Zone Tracker): a stateful replay of the log stream that
  accumulates our own face-down in-play cards (keyed by serial, cleared on reveal / leaving
  play). Feeding it every observation, `build_belief(..., own_facedown_ids=...)` makes the
  pool **exact through setup** — validated: tracker identified 67/67 face-down cards, pool
  exact at all 247 turn-0 decisions. Without a tracker, `facedown_in_play` carries the count.
- **Mid-effect resolution:** while a multi-step effect resolves (e.g. "look at top N"), the
  played card is in flight (out of hand, not yet discarded) and momentarily in no zone. So
  the pool is exact only at **stable points** (MAIN menu / turn boundaries) — exactly where
  draw math is used. The smoke test asserts the exact invariant at MAIN decisions (and, with
  the tracker, at setup) and checks no-negative-counts everywhere.

### Next (Belief Layer 2b)
- Opponent model: Bayesian archetype prior → decklist template → subtract seen cards →
  hypergeometric over their hidden zones (§4 "opponent prediction → distribution"). This is
  the `BeliefState.opponent` placeholder.
- Deck-quality metric = E[usefulness of next k draws] via this tracker (needs per-card
  situational usefulness scoring — pairs with the heuristic evaluator).

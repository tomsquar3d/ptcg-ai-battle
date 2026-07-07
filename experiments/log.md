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

---

## #5 — Belief Engine Layer 2b: opponent model (`src/ptcg/opponent.py`)

**Date:** 2026-07-06
**Code:** `src/ptcg/opponent.py`, `scripts/opponent_smoke.py`; refactor `belief.hyper_at_least`

### Goal
§4 "opponent prediction → a distribution." Three layers, most-exact first.

### What was built
1. **`OpponentTracker` + `opponent_visible_cards`** — the exact, priors-free *evidence*: the
   multiset of opponent cards we've legitimately seen. Identity is harvested only from visible
   zones (their board + attachments, discard, stadium), where physical location proves
   ownership; deduped by serial and accumulated → monotonic knowledge of their decklist.
2. **`OpponentZoneModel`** — given a predicted decklist *template*, `pool = template − visible`
   is their hidden cards; by exchangeability, `prob_in_hand` / `prob_draw_next` / `prob_prized`
   are hypergeometric over the pool (reuses `belief.hyper_at_least`, extracted in a refactor).
   Feeds the threat model and the determinization sampler.
3. **`Archetype` + `classify_archetype`** — Bayesian archetype posterior (naive multinomial
   bag-of-cards, additive smoothing). Framework + tests only; real templates and meta-share
   priors come from replay mining (§5). `build_opponent_belief` composes all three into
   `OpponentBelief` (the `BeliefState.opponent` slot).

### Validation (`scripts/opponent_smoke.py 60`)
```
1. CLASSIFY: posterior favours the signature archetype (Fire 0.998), returns priors under no
            evidence, respects unequal priors
2. TRACKER : revealed() is a sub-multiset of the known opponent decklist + monotonic (60 games)
3. ZONE    : template−visible pool invariant n == opp_deck+hand+prize+facedown held at all
            1869 stable (MAIN) decisions (2695 total)
sample: revealed 6 (5 distinct) | deck39 hand9 prize6 | P(opp holds Maximum Belt in hand)=0.167
```

### Two bugs found & fixed (both in reasoning, not just code)
- **Log `playerIndex` is the actor, not the owner.** Harvesting opponent identity from logs
  mis-attributed *our* cards when the opponent's action targeted them (revealed > decklist).
  Fixed by trusting only visible-zone location for ownership. (Rare reveal-then-rehide cards
  are then missed — an acceptable correctness/completeness trade.)
- **Self-play flips perspective.** `yourIndex` alternates each turn, so one tracker in a
  self-play harness conflates both players. A real agent has a fixed viewpoint; the test now
  keeps one tracker per side, each fed only its own turns.

### Next
- Threat model on top of the zone model: P(our active KO'd next turn), P(gust/switch in their
  hand), expected incoming damage — needs attack/effect reads from the card DB.
- Real archetype templates + meta-share priors from the replay-mining pipeline (§5), then
  measure classifier accuracy vs replay ground truth (§7.3 originality play).

---

## #6 — Threat model: the belief→decision bridge (`src/ptcg/threat.py`)

**Date:** 2026-07-06
**Code:** `src/ptcg/threat.py`, `scripts/threat_smoke.py`

### Goal
Turn board + opponent belief into the danger signals the evaluator/search consume (§4):
incoming damage, KO-now / KO-next-turn, and — via the opponent zone model — P(they hold a
gust/switch or the energy they need).

### What was built
- **Deterministic core (card DB):** `can_pay`/`_cost_shortfall` (energy-cost matcher with
  colorless/Rainbow/Team-Rocket wilds), `attack_damage` (printed damage × weakness, − typed
  resistance), `assess_threat` → `ThreatAssessment(incoming_now, incoming_potential, ko_now,
  ko_potential, best_attack_now, …)`. `incoming_potential` assumes one more energy attach.
- **Probabilistic layer:** with an `OpponentBelief.zone`, fills `p_gust_in_hand` and
  `p_needed_energy_in_hand` via `OpponentZoneModel.prob_in_hand`. Card-id sets from
  `energy_card_ids` and a keyword-heuristic `gust_card_ids`.

### Validation (`scripts/threat_smoke.py 50`)
```
1. UNITS : cost matcher + weakness/resistance math
2. DAMAGE: predicted == actual on 250/254 attack hits (98%) vs the engine's own HP change as
           ground truth; mean abs err 0.6. The 4 misses are text-scaling attacks
           (Frost Barrier conditional; "Beat" printed 10 -> actual 60)
3. THREAT: 2267 decisions, 0 crashes; ko_now 11%, ko_potential 21%, mean incoming 35.2
sample: our HP 150 | incoming 10 (Beat) potential 30 | P(gust)=0.000 P(energy)=0.999
```
The 98% exact-match against engine ground truth is a strong, measurable check on the
base-damage + weakness model (a §7.3 accuracy figure for the writeup).

### v1 limitations (documented, not hidden)
- Uses printed `Attack.damage` × weakness only — text-based conditional/scaling damage,
  abilities, and special-energy multi-type output are unparsed (the 2% of hits, and any
  printed-damage-0 attacks, which are skipped). Effect-text tagging is a later pipeline.
- Weakness ×2, resistance −30 (typed) — historical defaults; validate vs engine.
- Only the opponent's current active is modelled as attacker; bench attackers and
  switch-to-a-charged-attacker lines are future work (the gust probability is the hook).

### Next
- Heuristic evaluator (`src/`): weighted terms over prize-race (belief) + deck-quality (draw
  model) + threat penalties → the week-2 shippable agent, ablation baseline, and clock fallback.
- Effect-text tagging pass to lift damage accuracy above 98% and enable bench/switch threats.

---

## #7 — Heuristic evaluator: the leaf/position score (`src/ptcg/evaluator.py`)

**Date:** 2026-07-06
**Code:** `src/ptcg/evaluator.py`, `scripts/evaluator_smoke.py`; refactor `threat.attack_potential`

### Goal
A single interpretable position score (§3 Layer-5 leaf eval; §4 "weighted terms"): the week-2
shippable agent's brain, the ablation baseline, the search leaf evaluation, and the clock
fallback. Every term is named + weighted so it can be ablated on/off.

### Terms (from the to-move player's view; `EvalWeights` are tunable)
- **prize_diff** (master, ×1000/prize) — prizes we're ahead by.
- **ko_them_now / ko_us_now** (× prize value of the KO'd Pokémon — ex/mega worth more) and
  **offense_next / threat_next** (KO possible with one more energy) — via `threat.attack_potential`
  (extracted so the evaluator scores our offense, not just their threat).
- **active_hp** (HP-fraction edge), **board** (Pokémon in play diff), **hand** (tempo),
  **special_us / special_opp**, **deckout** (risk as our deck runs low). Terminal states → ±WIN.

### Validation (`scripts/evaluator_smoke.py 80`)
```
1. UNITS  : terminal scores, prize-race sign, term breakdown
2. PREDICT: 3779 decisions / 80 games — does a higher eval predict WINNING under random play?
          Pearson(eval, win) = +0.193
          win rate: eval>0 64%  vs  eval<0 41%
          win rate by eval quartile (low→high): 36%  49%  54%  73%
```
Under fully random self-play (huge variance, both sides random), positions the evaluator scores
in the top quartile win **73%** vs **36%** bottom — the score measures genuine advantage. This is
the §4 "prize-path accuracy vs actual game endings" figure; predictiveness is currently
prize-dominated (expected under random continuation), and the threat/board terms should add signal
once paired with a non-random policy.

### Next
- Wire the first real agent (`submissions/v1/`): 1-ply lookahead via the Search API (experiment
  #3: ~8k steps/s) scoring each legal option's resulting state with `evaluate()`, + the §3
  arbitration layer (routine → argmax; critical turn → deeper search; clock → instant fallback) +
  the mandatory try/except watchdog. Then submit to record the first non-random ladder μ.
- Per-term ablation ladder (turn each weight off, measure win-rate delta) — the writeup's spine.

---

## #8 — v1 agent: evaluator-guided 1-ply search (`submissions/v1/`)

**Date:** 2026-07-06
**Code:** `submissions/v1/{main.py, agent.py, ptcg/}`, `scripts/v1_eval.py`, `build_submission.py`

### Design
First real agent. Per decision: determinize the hidden state once (`search_begin`), branch one
`search_step` per legal option (branching from a shared root confirmed — fair comparison), score
each resulting state with the heuristic evaluator, pick the argmax. Perspective handled — a
resulting state on the opponent's turn is scored from their view and negated. Multi-select →
greedy top-`maxCount`. `main.py` wraps it in a watchdog: any exception / clock pressure
(`remainingOverageTime < 15s`) → legal random move. Determinization fills the opponent's hidden
zones with our own decklist (valid filler; our-turn outcomes don't depend on their hidden cards).

### Bug found & fixed — the evaluator needed a setup gradient
First cut **lost to random 7%** (2/28): it never attacked through turn 12. Cause: the evaluator
scored only outcomes (prizes/HP/board/KO), so every setup action (attach energy, develop) scored
identically → 1-ply lookahead had no gradient toward building an attacker, and ties resolved to
END. Fix: added continuous **`our_output`/`their_output`** (per point of best *affordable* attack)
and **`energy_active`** terms, so attaching energy that enables a bigger attack raises the score.

### Result (`scripts/v1_eval.py 40 12`)
```
1. QUALITY (raw SDK, v1 vs random): W/L/D 27/13/0 -> 68% win; move mean 1.1 ms, max 27 ms;
                                     0 brain exceptions
2. INTEGRATION (cabt env, v1 submission vs random): all DONE/DONE (no crash/timeout),
                                     8/12 wins, 0.18 s/game
```
v1 clearly beats the random baseline (experiment #1) and runs cleanly in the real environment,
comfortably inside the clock. This is the week-2 shippable agent / ablation baseline.

### Notes / next
- `submissions/v1/ptcg` is a vendored snapshot of `src/ptcg` (source of truth); re-sync with
  `scripts/build_submission.py v1`. Native SDK libs stay git-ignored (copy from the SDK to run).
- Deck is v0's sample deck for now — deck construction (§5) is a separate pipeline.
- Next: submit v1 to record the first non-random ladder μ; then the §3 arbitration upgrade
  (critical-turn deeper search) and the per-term ablation ladder (writeup spine). 1-ply greedy is
  myopic across a full turn — deeper/turn-aware search is the main quality lever from here.

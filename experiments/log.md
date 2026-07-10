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

---

## #9 — v1 turn-aware search (end-of-turn rollouts)

**Date:** 2026-07-06
**Code:** `submissions/v1/agent.py`

### Motivation
1-ply greedy (#8) evaluates *partial* mid-turn states and compares across the turn boundary
(our-turn vs opponent-turn states aren't comparable under a tempo-blind evaluator). Fix: only
evaluate at *end-of-our-turn* states — complete turns, consistent perspective.

### Method
Turn-aware search per decision: `search_begin` once; rank the root options by immediate score;
for the top `ROOT_BEAM=6`, greedily roll the rest of our turn to its end (each rollout step =
the option maximizing the resulting state's score) and score that end-of-turn state; pick the
root option with the best end-of-turn value. Anytime + bounded (`DEPTH_CAP=24`, `NODE_BUDGET=
7000` search_steps, `MOVE_BUDGET_S=2`), with the same watchdog fallback. Multi-select roots keep
the greedy top-k path.

### Result (`scripts/v1_eval.py 40 12` + head-to-head)
```
vs random (raw SDK): W/L/D 34/6/0 -> 85% (up from 68% at 1-ply); move mean 8.4 ms, max 70 ms
cabt env vs random : all DONE/DONE, 11/12 wins, 0.17 s/game
head-to-head vs the 1-ply agent (#8), 40 games, seats alternated: turn-aware 30/10 -> 75%
```
Turn-aware beats 1-ply decisively (75% h2h) and lifts win-rate vs random 68%→85%, still trivially
inside the clock (max 70 ms/move vs the 600 s bank).

### Next quality levers
- Opponent-response lookahead at the leaf (1-ply for them, belief-determinized) — currently the
  static evaluator's threat terms stand in for "did we leave ourselves open".
- Proper best-first/MCTS with backtracking (rollout is greedy, so it finds *a* good turn, not the
  optimal); gated on the node budget (experiment #3: ~8k steps/s).
- Per-term ablation ladder; submit to record the first non-random ladder μ.

## #10 — Opponent-response lookahead (2-ply), a negative result (`submissions/v2/`)

**Date:** 2026-07-06
**Agent:** `submissions/v2/agent.py` (v1 turn-aware search + opponent-turn rollout)
**Deck:** SDK sample deck (unchanged from v0/v1)

### Hypothesis
v1 (#9) is turn-aware but *reply-blind*: it rolls our turn to its end and statically evaluates
the resulting opponent-to-move state, never simulating the reply. So it can pick a turn that
scores well and then walk into a KO it never searched. Adding a second ply — roll the opponent's
best reply to *its* end, then score that post-reply state — should catch those and beat v1.

### Method
`_greedy_rollout` is perspective-parameterized, so the opponent's turn is the same rollout called
with the opponent's index (they play to maximize their own static score); `_score_state` already
scores from our view and negates for opponent-to-move states. New `_rollout_line` composes our
turn + (toggleable) opponent reply. Budget/clock guards (`NODE_BUDGET`, `MOVE_BUDGET_S`) shared
across both rollouts; added `OPP_DEPTH_CAP=24`. Toggle `opponent_reply` (default on) recovers
exact v1 behaviour for a clean ablation. Determinization unchanged (opponent hidden zones = our
decklist). `agent.STATS` counts rollouts so the harness can assert the 2-ply actually fires.

### Result (`scripts/v2_eval.py`)
```
QUALITY  (reply-on vs random, 40): W/L/D 33/7/0 -> 82% (v1 was 85%; same band)
                                   move mean 40 ms, p95 107 ms, max 249 ms; 0 exceptions
HEAD-TO-HEAD reply-ON vs reply-OFF (== v1), MIRROR (both play the sample deck):
   N=60 : 34/26 -> 57%   (looked positive, but only ~1σ)
   N=200: 95/105 -> 48%
   pooled 260: 129/260 -> 49.6%   (a coin flip)
   opponent rollout fired 10,483× over 200 games (NOT inert), 0 exceptions
INTEGRATION (cabt env vs random, 12): all DONE/DONE, 9/12 wins, max 4.5 s/game
```

### Takeaways
- **No measurable benefit over reply-blind v1.** The 57% at N=60 was noise; N=200 settles at ~50%.
- The 2-ply is provably *working* (rollouts fire, clean, fast) — it just doesn't reorder which of
  our turns looks best. The evaluator's static threat terms (`threat_next`, `ko_us_now`,
  `their_output`) already encode "we're exposed" without simulating the reply.
- Crucially this test is a **mirror**, so the "our-deck determinization is inaccurate" caveat does
  **not** apply here — the opponent's simulated hand is correct, and it still doesn't help. So the
  bottleneck is *not* opponent-hand fidelity; it's that a **greedy** opponent reply + static leaf
  eval adds no signal beyond the eval's threat terms. Real archetype templates alone won't fix this.
- **Do not ship v2 / do not burn a submission on it.** Keep v1 on the ladder. v2 is retained as
  correct, cheap, ready-when-it-pays-off infrastructure (behind the `opponent_reply` toggle).

### Next (redirected by this result)
- Ablate the evaluator's threat terms on/off — quantify how much of "reply awareness" the static
  terms already buy (likely most of it). This is the writeup spine anyway.
- Non-greedy search (best-first/MCTS with backtracking): if our own turn-planning is greedy-myopic,
  a smarter reply has nothing better to punish. Fix the search before re-testing the reply.
- Deck construction (§5) and real opponent templates remain the bigger untapped levers — but note
  the mirror result above: templates are for *cross-deck* play, not this bottleneck.

## #11 — Meta priors ingestion + disruption groundwork (`src/priors/`, `src/deck_lab/`)

**Date:** 2026-07-07
**Code:** `src/priors/{carddb,build_priors}.py`, `src/deck_lab/dcs.py`
**Data:** TrainerHill (`data/external/trainerhill/`) — 8 archetype skeletons + 15-deck matchup CSV

### Goal
Turn human-meta lists (TrainerHill) into machine-usable priors + vulnerability profiles and a
static Disruption Coverage Score — the static half of the disruption programme (strategy §4).
Data artifacts only; no runtime Disruption Planner yet.

### Method / artifacts
- **Card mapping** (`carddb.py`): skeleton `set_code` → EN_Card_Data (Expansion, Collection No.)
  exact join; fallbacks = basic-energy colour alias (EN names energy `Basic {P} Energy`) and
  normalized-name (delete apostrophes — EN uses a curly ’ that ascii-folds away). Pool = engine
  `all_card_data()` (EN "Card ID" == engine `cardId`, both 1267). **96% of 301 rows mapped.**
- `archetype_priors.json` — per card P(copies|archetype) from `pr_` (renormalized; basic energy
  uses `modal_copies` point mass); `meta_share` from matchup game volumes over the full 15-deck
  field (our 8 = 71% of field games).
- `human_matchup_matrix.json` — Beta(1,1)-smoothed P(win|decisive game); mirrors → exactly 0.5.
- `vulnerability_profiles.json` — expected copies: basic energy, **special energy**, tools,
  %Pokémon-with-abilities, draw-supporter reliance.
- `dcs.py` — DCS = Σ meta_share × max-axis effectiveness(our lever strength × their vuln), 4 axes
  (energy/tool/ability/hand). Rubric documented in the module docstring; named constants for ablation.
- `experiments/meta_priors_report.md` — coverage table, constructibility ranking, profile table.

### Findings
- Residual 12 unmapped cards are all the **"CRI" set** — genuinely out of the competition pool,
  incl. **Mega Greninja ex (CRI-022)**, the namesake of `mega-greninja-ex` (so that archetype is
  only ~91% constructible and arguably worse — copy-weighting understates a missing key card).
  Nothing trips the <85% flag, but the report lists every unmapped card per archetype.
- Profiles cleanly separate archetypes: `alakazam-dudunsparce` = ability engine (62% ability
  Pokémon, ~0 draw supporters → Watchtower target); `rockets-honchkrow` = special-energy draw
  engine (8 special energy, 8 draw supporters → Judge/Red-Card + Enhanced-Hammer target).
- **Special energy correction (reviewer note):** decks like `rockets-honchkrow` run special
  energy (which also carry effects). Added `special_energy_count` and fold basic+special into the
  DCS energy axis, so honchkrow/slowking/alakazam correctly read as hammer-exposed instead of
  energy-immune.
- Sample deck (v1/v0) DCS = 0.0 — zero disruption package; a synthetic all-levers deck scores
  0.96 normalized. Baseline established for Deck Lab.

### Next
- Functional DCS (measured Δ opponent first-attack turn / turns-to-6 with disruption in vs out).
- Feed priors into the opponent-model archetype templates (`src/ptcg/opponent.py` placeholder).
- Runtime Disruption Planner (EDV timing) — deferred per task scope.

## #12 — Wire human-meta priors into the opponent model (`src/ptcg/archetypes.py`)

**Date:** 2026-07-07
**Code:** `src/ptcg/archetypes.py`, `src/ptcg/archetype_templates.json`, `scripts/archetypes_smoke.py`

### Goal
Fill the placeholder in the archetype classifier (`opponent.py` shipped the mechanism but no real
templates — §4's "opponent inference" originality play). Turn the #11 priors into `Archetype`
templates the classifier + zone model consume.

### Method
`build_priors.py` now also emits `src/ptcg/archetype_templates.json` (into the agent package so it
bundles): per archetype `template[card_id] = round(expected_copies)` (zeros dropped → the typical
list), `prior = meta_share`. `archetypes.load_archetypes()/default_archetypes()` load them into
`Archetype` objects; `build_opponent_belief` docstring now points there (parameter kept explicit —
no silent import, opponent.py stays pure/non-breaking).

### Result (`scripts/archetypes_smoke.py`; opponent_smoke regression green)
```
8 templates load, sizes 47-59 (mega-greninja 47 — CRI namesake out of pool), priors = meta share
self-consistency : each archetype fed its own template -> itself, 8/8
discrimination   : shared Dragapult core (Dreepy/Drakloak/Dragapult ex) -> 99% on the 3 Dragapult
                   variants, 0 on unrelated decks; all 8 signature cards pin their archetype;
                   no evidence -> normalized priors (MAP = n-zoroark, the largest meta share)
```
The shared-core behaviour is the point: partial evidence yields *calibrated ambiguity* among
variants, not false confidence.

### Next
- Feed `default_archetypes()` into a critical-turn determinization sampler (draw opponent hands
  from the MAP template's zone model) and A/B vs the current our-deck-filler determinization.
- Refine templates from simulator replays once mining exists (human meta will diverge — §6).

## #13 — Round-robin deck harness (`src/deck_lab/round_robin.py`)

**Date:** 2026-07-07
**Code:** `src/deck_lab/round_robin.py`, `scripts/round_robin_smoke.py`

### Goal
First brick of the §5 deck-construction pipeline (step 3): measure deck-vs-deck win rate. Candidate
generation, DCS, and the seven-axis scores are all useless until we can tell whether deck X beats
deck Y — this is that measurement.

### Method
Every unordered deck pair plays N games in the **raw SDK** (`battle_start(deck0, deck1)`), both
sides piloted by the v1 brain (the current best agent). Raw SDK — not the cabt-env subprocess
`selfplay_test.py` uses — so we can inject two *different* decks per game and switch the brain's
determinization deck to the acting player each turn. First-player bias cancelled by alternating
which deck sits in seat 0 across a pairing; results tallied by deck identity; draws count as half.
Outputs a win-rate matrix + ranking (`round_robin()` API + CLI over a dir of deck CSVs).

### Result (`scripts/round_robin_smoke.py`, self-generated fixtures)
```
sample vs sample_mirror (identical decks) : 56.7% / 43.3%  → ~50% within N=30 noise (fair)
basics_only (evolutions+trainers stripped): 10% win rate, ranks last (loses ~90%)
72->60 games/pairing, ~14 s for 3 decks @ 30 games/pairing
```
First cut at N=24 was too noisy to trust (the *same* matchup swung 33%↔54%) and a mild cripple
didn't separate — bumped to a severe cripple + N=30 so the signal dominates. **Coin flips lie at
small N**: §5's ~200 games/pairing is the real target; N is the wall-clock↔confidence knob.

### Next
- Candidate generation: build legal 60-card decks from the archetype templates (`archetypes.py`)
  — template core + basic-energy/flex padding to 60 — so the harness has real decks to compare.
- Then meta-weighted selection (weight each pairing by opponent `meta_share`) and hill-climb.

## #14 — Candidate deck generation (`src/deck_lab/candidates.py`)

**Date:** 2026-07-07
**Code:** `src/deck_lab/candidates.py`, `scripts/candidates_smoke.py`, `experiments/candidate_decks/*.csv`

### Goal
Turn the archetype priors into legal, playable 60-card decks (§5 "template + flex") so the
round-robin (#13) has real meta decks to rank — and so we finally have decks other than the stock
sample.

### Method
Per archetype (from `archetype_priors.json`): seed counts = round(expected_copies) (energy incl.,
capped 4 for non-basic-energy); legalize (≤1 ACE SPEC = keep highest-expected; guarantee ≥1 basic
Pokémon); reconcile to exactly 60 by residual-greedy — add the largest (expected−current) residual
first (restores flex cards that rounded to 0) and only pad with the dominant basic energy once no
positive residual remains; trim the most over-represented card if over 60 (never the last basic).

### Result (`scripts/candidates_smoke.py`)
```
8/8 decks: exactly 60, ≤4 copies (non-basic-energy), exactly 1 ACE SPEC, ≥1 basic Pokémon
ACE SPEC dedup works: e.g. Unfair Stamp / Secret Box / Neo Upper Energy chosen per deck
all 8 play full games to completion in the SDK (no INVALID / crash), piloted by the v1 brain
```
Decks written to `experiments/candidate_decks/`. Honest gaps: mega-greninja-ex is built WITHOUT
its out-of-pool namesake (CRI-022) → legal but weak; and the N=2 legality games hinted the sample
deck may beat several candidates — but that's noise + the v1 brain likely misplaying unfamiliar
archetypes; only a high-N round-robin will give real strength (coin flips lie at small N, #13).

### Next
- High-N round-robin over the 8 candidates (+ sample) → real matchup matrix + ranking.
- Meta-weighted selection (weight opponents by `meta_share`); then hill-climb over flex slots.
- Ship the winner as the first non-sample `deck.csv` (get the agent off the stock deck).

## #15 — Candidate round-robin: the deck eval is pilot-confounded (`experiments/candidate_round_robin.*`)

**Date:** 2026-07-07
**Code:** `scripts/candidate_round_robin.py`; `round_robin()` parallelized (`workers`, progress)

### Setup
9 decks (8 archetype candidates + the SDK `sample`), N=40 games/pairing (1440 games), both sides
piloted by the v1 brain, seats alternated. Parallelized across 9 processes (SDK global battle
state forbids in-process concurrency, so one game/process × 9 workers) — 117 min → ~13 min.
(First serial N=40 attempt was killed at 26 min: real meta games are ~4.9 s each, ~30× the smoke,
because full decks play 20-40 turns — the smoke's crippled deck lost in seconds.)

### Result — the sample deck beats every constructed deck
```
1. sample              raw 80.0%  meta-wt 79.6%   ← the stock SDK deck
2. dragapult-blaziken  raw 59.4%  meta-wt 65.7%
3. rockets-honchkrow   raw 55.9%
...
8. n-zoroark           raw 35.2%
9. mega-greninja-ex    raw 31.2%   (built without its out-of-pool namesake, as predicted)
every candidate vs sample: 5-33% win rate
```

### Diagnosis — a confound, not a verdict
- The candidates track the human meta **among themselves**: Pearson(our candidate-vs-candidate
  matrix, TrainerHill matchup matrix) = **+0.40**. So the builds + harness are realistic, not noise.
- Yet all lose ~uniformly to `sample`. Realistic-among-themselves + uniformly-crushed-by-one-simple
  -deck ⇒ the bottleneck is the **pilot**: the v1 heuristic (greedy turn search, printed-damage eval,
  no combo/sequencing knowledge) plays a simple beatdown deck near-optimally but fumbles evolution/
  ability/energy-accel archetypes. Crude auto-builds (energy padding, out-of-pool gaps) add to it.
- **This is the agent-deck co-evolution problem (§5): you can't rank decks with a pilot too weak to
  play them.** The 80% gap is far beyond N=40 noise, so the qualitative conclusion is safe.

### Takeaways / next
- Do NOT read this as "sample is the best deck." Read it as "under our current pilot, the simplest
  deck wins" — which is itself a real, defensible submission stance for *now* (ship what the agent
  can pilot). Deck sophistication needs pilot sophistication first.
- The construction + round-robin machinery works (correlation proves it). The lever is pilot
  strength (or hand-tuned builds), not more games.
- Fair-eval ideas: pilot each deck with a deck-aware policy, or measure decks by *consistency*
  proxies (turn-2 attack rate, energy-attach curve) rather than only win rate under one weak brain.

## #16 — v3 beam / best-first turn search: fixes the greedy pilot (`submissions/v3/`)

**Date:** 2026-07-07
**Agent:** `submissions/v3/agent.py`
**Deck:** SDK sample (ladder) + candidate decks (test)

### Hypothesis
The pilot's weakness (#15) is *greedy* within-turn planning: v1/v2 commit to the best immediate
option each step, so setup→payoff sequences (evolve→attach→attack, discard-to-draw) are never
found. Within our turn there's no opponent interleave, so it's a single-agent deterministic search
— replace greedy (beam width 1) with a **beam / best-first search** over turn action-sequences.

### Method
v3 keeps `BEAM_WIDTH=8` partial lines per ply, each tagged with its first root option; scores
end-of-turn states for the answer and partial states only to order the frontier; returns the first
move of the best end-of-turn line. Anytime, `NODE_BUDGET=7000`, `DEPTH_CAP=24`, same watchdog.
Reply-blind like v1 (opponent-reply was neutral in #10 — re-test on top later).

### Result (`scripts/v3_eval.py` + complex-deck head-to-head)
```
vs random (sample deck): 82% W/L/D 33/7/0; move mean 15 ms, max 167 ms; 0 exceptions; cabt all DONE
v3 vs v1, SAMPLE deck (simple) : 51/49 -> 51%  = TIE (no sequencing depth; greedy already optimal)
v3 vs v1, DRAGAPULT-EX (complex): 28/12 -> 70% (N=40, ~2.5σ); move mean 72 ms, max 500 ms
```

### Takeaways
- **Beam value scales with deck sequencing complexity** — nil on the simple sample deck, decisive
  (70%) on a complex evolution/combo deck. Directly confirms the #15 pilot-confound diagnosis.
- v3 is a **strict upgrade** over v1: ≥ on simple, ≫ on complex, still trivially inside the clock.
- **Nuance for the ladder:** we currently ship the *simple* sample deck, where v3 ≈ v1 — so v3
  alone won't move the ladder. v3's edge only manifests with a complex deck it can now pilot.

### Next
- Re-run the candidate round-robin with v3 as pilot: does the #15 sample-deck dominance lift once
  the meta decks are piloted competently? If a v3-piloted meta deck beats the field → ship v3 + it.
- Then re-test opponent-response (#10) on top of the stronger search (its premise was "fix search
  first").

## #18 — Prize-mapping leaf (turns-to-6): NEGATIVE — the signal isn't predictive (`src/ptcg/prize.py`)

**Date:** 2026-07-07
**Code:** `src/ptcg/prize.py`, `evaluator.py` (prize_race term, default 0), `scripts/{prize_smoke,pilot_ab}.py`

### Hypothesis
#17 said the meta decks lose because they need multi-turn planning v3 lacks. Give the single-turn
beam multi-turn foresight via a smarter *leaf*: the §4 prize-race differential (their_turns_to_6 −
our_turns_to_6) as an evaluator term, so setups/chip damage that shorten our race score highly.

### Method
`prize.py` = greedy turns-to-6 (KO opponent Pokémon worth our remaining prizes, ceil(hp/output)
turns each, cheapest-per-prize first). Wired into the evaluator as an ablatable term (default 0).
v4 = v3 beam + prize_race=200. Validated the arithmetic (`prize_smoke.py`). Added `pilot_ab.py`
(reusable pilot A/B).

### Result — the term does not predict winning
```
v4 vs v3, dragapult-ex, N=40 : 19/21 -> 48%  (a wash)
Pearson(eval, win) on 16k dragapult random-play decisions:
   default eval          +0.211
   eval + prize_race      +0.057   (adding the term DEGRADES prediction)
   prize_race term alone  +0.027   (essentially noise; fires on 75% of decisions)
```

### Conclusion
- The turns-to-6 term, as implemented, is **noise** — it dilutes the good eval (0.211→0.057). v4
  ties v3 only because the noise perturbs the beam ~randomly; a higher weight would be *worse*.
- The *concept* (prize race = master signal, §4) isn't refuted — my **model** is too crude: uniform
  damage across targets (ignores per-target weakness), no first-mover tempo, fragile fresh-basic
  tail, and 0 output early → no early signal. It doesn't track reality under messy boards.
- **Not shipped.** v4 removed; prize framework kept in `src` but OFF by default (ablatable).

### Methodology lesson (applies going forward)
The cheap **Pearson(eval, win)** test (seconds) diagnosed this far faster than the 12-min head-to-head
— **gate expensive pilot A/Bs behind the predictive test.** Any eval enrichment must beat +0.211 on
that test before we spend head-to-head compute. Pattern across #10/#16/#18: the base eval already
captures a lot; enrichments rarely help unless they add genuinely new predictive signal.

### Next (to rescue prize mapping, if pursued)
Iterate the model against the Pearson test: per-target `attack_potential` (real weakness/HP), a
first-mover/tempo term, and only trust turns-to-6 once an attacker is online. Cheap to iterate now.

## #19 — MCTS-H foundation + benchmark gate (`src/search/`)

**Date:** 2026-07-07
**Code:** `src/search/{determinizer,mcts}.py`, `scripts/mcts_benchmark.py`

### Built (parts 1,2,4 of the MCTS-H task; governor + validation deferred to the gate)
- `determinizer.py`: samples K worlds (our exact hidden pool via `belief`; opponent zone from
  archetype templates, reconciled to exact zone counts + guaranteed Basic). Discard handled by the
  belief layer (subtracts all visible incl. both discards); retrieval-from-discard is resolved by
  the engine during `search_step`, not the determinizer.
- `mcts.py`: determinized PUCT over the native Search API — perspective-aware (our-turn maximise,
  opp-turn minimise), evaluator leaf **squashed via tanh/3000** so Q and the exploration term share
  a scale, forced/multi-select auto-collapsed to the next real branch, injectable priors (uniform
  now), stats aggregated across K worlds. (Chained multi-select + chance resampling = refinements.)

### Benchmark gate (turn-2 dragapult decision, K=1)
```
determinizer valid: search_begin ACCEPTS its output (counts + discard subtraction correct)
determinize        : 4.4 ms / world
search_step        : ~7,000 steps/sec (143 µs/step)   [matches #3]
MCTS sims/sec K=1  : ~3,900/sec ;  K=8 ~4,300/sec (world round-robin ≈ free)
```
Sizing: ~4k sims/sec ⇒ a 5 s critical-turn budget ≈ **~20k sims** → K=8 gives ~2.5k sims/world,
ample for useful PUCT. Caveat: measured early-game (short turns, small eval states); mid/late-game
will be slower per sim — treat 4k/sec as an upper bound.

### Next (post-gate)
Governor (per-move budget from clock + prize-path critical-turn triggers + hard watchdog), tune
K(4-8)/sims to the ~5 s budget, wrap as a submission (next free slot — v2 is taken by #10; use v4),
then validate 200 games vs heuristic + vs v1 with per-matchup win rates, incl. an evolution-line
deck to confirm the multi-turn planning failure (#15/#17) improves.

## #20 — MCTS-H governor + v4 wrap; correctness-validated, big validation HELD (`submissions/v4/`)

**Date:** 2026-07-07
**Code:** `src/search/governor.py`, `submissions/v4/` (agent+main+deck+vendored search), `build_submission.py`

### Built
- `governor.py`: per-move budget from the clock + prize-path critical triggers (KO available, our
  active under KO = race swing, or endgame prizes ≤2) → deep MCTS (5 s, K=8) on critical turns,
  shallow (0.4 s, K=1) routine, instant legal move under 20 s clock. Hard deadline passed to MCTS
  (checked per sim → overshoot bounded to one sim). Fixed a bug: `prize_count==0` at *setup* falsely
  read as endgame-critical → guarded `0 < pc ≤ 2`.
- `submissions/v4/agent.py`: perception → governor → determinize K worlds → PUCT MCTS → most-visited
  root action; multi-select root via one-step heuristic; defensive wall unchanged (main.py watchdog).
- `build_submission.py` now vendors `src/search` alongside `src/ptcg` so v4 ships self-contained.

### Correctness smoke (validation deliberately held per plan)
```
raw SDK, 4 games vs random (fast budgets): 4/4 completed, 0 exceptions, v4 3/4 wins
  move time mean 320 ms, p95 1039 ms, max 1063 ms (=1 s budget + 1-sim overshoot → watchdog holds)
cabt env, 2 games (real bundle, main.py): both DONE/DONE
```
v4 ships with real budgets (CRITICAL_S=5.0, ROUTINE_S=0.4, K=8).

### HELD (awaiting go-ahead)
The 200-game validation vs heuristic + vs v1 (per-matchup win rates) + the evolution-line deck test
to confirm the multi-turn planning gap (#15/#17) actually improves under MCTS. This is the decisive
experiment; it is compute-heavy (MCTS games are slow) and was intentionally deferred.

## #21 — v4 (MCTS-H) validation: MCTS is the strongest pilot yet (`experiments/validation_v4.*`)

**Date:** 2026-07-07
**Code:** `scripts/validate_v4.py` (parallel per-matchup A/B); fixed v4's per-game OpponentTracker reset

### Setup
v4 (MCTS-H, 5 s/K=8 critical budgets) vs the heuristic pilots v3 (beam) and v1 (turn-aware), both
sides piloting the SAME deck, seats alternated, N=40/matchup, on an evolution-line deck
(dragapult-ex — where the #15/#17 planning gap lives) and the simple sample deck. Parallel across
9 workers (~4 min CPU per evolution game — the slow part). N=200 was ~3 h, so N=40 for the signal.

### Result — POSITIVE (the first winning "smarter pilot" of the arc)
```
v4 vs v3, evolution : 62% (25/15)   ~1.6σ
v4 vs v3, sample    : 60% (24/16)   ~1.3σ   ← beats beam even on the simple deck
v4 vs v1, evolution : 78% (31/9)    ~3.5σ   ← decisive, exactly where planning matters
v4 vs v1, sample    : 52% (21/19)   ~even  (simple deck needs no deep search)
```

### Read
- **v4 is the best pilot**: dominates v1 on evolution (78%, rock-solid) and beats v3 on BOTH decks
  (60-62%). Each v4-vs-v3 result is only borderline at N=40, but the *consistency* across two
  independent matchups in the same direction is the real signal — MCTS's multi-turn search is a
  general improvement, largest where sequencing matters.
- Contrast with #10/#16/#17/#18: enriching the *evaluation* (reply term, prize-race leaf) was
  neutral; enriching the *search* (beam #16, now full MCTS) is what pays. Depth > leaf tweaks here.

### Caveats / what this does NOT show
- N=40 → the v4-vs-v3 edges want a higher-N confirmation; v4-vs-v1-evolution (78%) already is solid.
- This tests PILOT strength, not deck strength. Whether meta decks now beat `sample` under the v4
  pilot (the #17 question) is a separate, expensive v4-pilot round-robin — not run here.
- Ladder: we ship the simple sample deck where v4 ≈ v1 (52%), so v4 alone is only a marginal ladder
  gain; its value is unlocking complex decks a good pilot can actually play.

### Next
- (confirm) extend v4 vs v3 to higher N; (unlock) v4-pilot deck round-robin (do meta decks beat
  sample now?); (train) v4 is now a validated strong teacher → self-play RL is de-risked (§3 roadmap).

## #22 — Deck-unlock re-test under v4 pilot: PARTIAL (`experiments/deck_unlock_v4.*`)

**Date:** 2026-07-07
**Code:** `scripts/deck_unlock_v4.py` (v4-vs-v4 cross-deck, two independent instances, reduced 1.5 s budget)

### Setup
The #17 question: do the meta decks beat `sample` when piloted competently? Pit v4(meta deck) vs
v4(`sample`), N=20/deck, both MCTS at a reduced 1.5 s critical budget (tractability — full 5 s over
8 decks was ~hours; still far stronger than the heuristics). Two independent v4 instances so the
per-side opponent trackers don't collide. Mirror sanity-checked at 50% before trusting it.

### Result — a strong pilot narrows the gap but doesn't close it
```
meta-deck win rate vs sample:  v1 pilot ~20% avg  →  v3 ~19%  →  v4 ~29% avg
  rockets-honchkrow  27-32% → 65%   ⭐ the ONLY meta deck that beats sample under v4
  dragapult-blaziken 17% → 35% ; dusknoir 27→30 ; ex 23→25 ; mega-greninja 13→20 ;
  alakazam 10→25 ; n-zoroark 20→30 ; slowking-scr 13→5 (regressed)
sample still beats 7 of 8 meta decks.
```

### Conclusion — BOTH #15/#17 halves were partly right
- **The pilot confound was partly real**: a stronger pilot lifts the meta decks (~19%→~29% avg).
- **The decks are also genuinely weaker**: sample still wins 7/8 even under MCTS.
- **`rockets-honchkrow` is the one real find** — a special-energy Rocket draw/disruption engine that
  *beats* sample (65%) when piloted well; MCTS likely pilots its consistency/disruption lines better
  than the greedy heuristics. First constructed deck to clear sample. (65% at N=20 is ~1.3σ — needs
  higher-N + full-5 s-budget confirmation before shipping.)
- Caveat: reduced 1.5 s budget; full 5 s might lift the decks further. The N=6 "dragapult 50%" earlier
  was noise (25% at N=20).

### Next
- Confirm `rockets-honchkrow` vs sample at higher N + full 5 s budget → if it holds, it's a genuine
  ladder candidate (v4 + rockets-honchkrow beats our current sample ship).
- Otherwise the standing plan holds: v4 is the best pilot, sample the safest deck; begin training
  foundations (encoder → action featurization → net) with v4 as the self-play teacher.

## #23 — rockets-honchkrow confirmation: regression to the mean, sample stays the ship

**Date:** 2026-07-07
**Code:** `scripts/deck_unlock_v4.py` (added deck-name filter); `deck_unlock_v4_rockets-honchkrow_5s.*`

### Result
```
rockets-honchkrow vs sample, v4 pilot, N=60, FULL 5 s budget: 47% (28/32)
  (was 65% at N=20 / 1.5 s — that spike was noise; regressed to ~even.)
```

### Conclusion
- **rockets-honchkrow does NOT beat sample** (47%, essentially even, slightly under) — no shippable
  deck upgrade. **Sample remains the deck we ship.**
- Still a real signal: 47% vs the ~30% under v1/v3 confirms **v4/MCTS pilots this deck far better
  than the heuristics** — it lifts it to near-parity with sample, just not past it.
- **Closes the deck-construction arc: no auto-generated meta deck beats sample, even under a strong
  pilot.** The lever is not the deck — it's the pilot. And on the *simple* sample deck we ship, v4's
  edge over v1 is only marginal (52%, #21). So the near-term ladder is roughly capped by the
  heuristic-pilot-on-simple-deck ceiling.
- **Implication: the real remaining lever is a fundamentally stronger pilot — the ML path** (self-
  play RL with v4 as teacher). Deck construction and search-heuristic tuning are both largely tapped.

### Next
- Begin training foundations (§3 roadmap): state encoder → action featurization → policy/value net →
  self-play recorder (MCTS visits = π, outcome = z) → train → wire net as MCTS prior/value → ONNX-CPU.
- (optional) hand-tuned real Limitless list vs sample under v4 — last cheap check on "crude builds".

## #17 — v3-pilot round-robin: the #15 pilot-confound was WRONG (`candidate_round_robin_v3.*`)

**Date:** 2026-07-07
**Code:** `round_robin.py` (PILOT_DIR env), `candidate_round_robin.py` (pilot-suffixed outputs)

### Test
#15 diagnosed the sample deck's 80% dominance as a pilot confound (heuristic pilot can't play the
complex meta decks). #16 built v3 (beam search) and proved it pilots complex decks 70% better than
v1. So: re-run the 9-deck round-robin (N=30) with **v3 as the pilot on both sides** — if the
confound was real, the meta decks should close the gap on sample.

### Result — they did NOT close the gap
```
sample raw win rate     : 80.0% (v1 pilot) -> 81.2% (v3 pilot)   UNCHANGED
sample avg vs the field : 0.20  (v1 pilot) -> 0.19  (v3 pilot)   UNCHANGED
(per-deck vs-sample deltas are N=30 noise: dusknoir 0.17->0.27, alakazam 0.28->0.10, etc.)
v3-pilot ranking still: 1. sample 81% ... every meta candidate 32-59%.
```

### Conclusion — self-correction
- **The #15 pilot-confound diagnosis was wrong (or at most minor).** Fixing the pilot's *within-turn*
  search — a real, validated improvement (#16) — did not lift the meta decks against sample. So the
  gap is NOT primarily about within-turn piloting.
- The auto-generated meta decks are **genuinely weaker** than the sample deck in the SDK. Two live
  causes I can't yet separate: (a) **crude auto-builds** (round(expected)+energy padding, out-of-pool
  gaps, no ratio/tech tuning), and/or (b) the decks' power needs **multi-turn strategic planning**
  (evolve over 2-3 turns → payoff) that v3 lacks — v3 optimizes single turns, not sequences of turns.
- **v3 is still a strict pilot upgrade** (#16), just not the deck-unlocker I hoped. On the *simple*
  sample deck v3 ≈ v1, so **v3+sample ≈ v1+sample on the ladder — no submission gain from v3 alone.**

### Strategic takeaways
- **Ship a simple, consistent deck** (sample-like): confirmed competitive twice, and the pilot maxes
  it. Deck construction is **not a quick win** — auto-decks are genuinely worse.
- The real ladder levers are elsewhere: a better *simple* deck (hand-tuned aggro), **multi-turn**
  planning in the pilot, evaluator quality, and diagnosing why v1 sits below μ₀=600.
- To separate cause (a) vs (b): pit a **hand-tuned real Limitless list** (pool-filtered) vs sample
  under v3 — if it beats sample, my builds were the problem; if not, it's multi-turn planning / meta.

## #24 — ML foundations: state + action featurization (`src/nn/features.py`)

**Date:** 2026-07-07
**Code:** `src/nn/features.py`, `scripts/features_smoke.py`

First ML-path brick (§3 Layer 3 — the belief-engine → network interface, now that #21 validated v4
as a strong self-play teacher). `encode_state(gs, ci, ai) → StateFeatures`: fixed-shape numpy
arrays — per-Pokémon card-id slots [12] + scalars [12×10] + mask, hand ids [12], stadium id, and a
global vector [36] that folds in the strategy objects (prize race via `prize.py`, offense/threat via
`threat.attack_potential`) as *inputs*, not reward shaping. Pure Python + numpy (no torch), never
raises (perception-style); vocab 1269 (ids 1..1267 + PAD 0 + UNKNOWN 1268 for a face-down opp
Pokémon). Validated: `features_smoke.py` — 1691 states over 8 games, 0 errors, fixed schema, finite
values, ids in vocab, mask/id consistent.

**Action featurization** (§3 Layer 4 policy head, same file): `encode_actions(select, my_index, ai)
→ ActionFeatures` — per legal option: its card id (shared embedding vocab) + `OPTION_SCALAR_DIM=27`
scalars (OptionType one-hot ×17 + attack damage/cost + target-location flags active/bench/hand/
discard/is-mine + count/index/number). Fixed [MAX_OPTIONS=64] shape, mask-padded, truncation-safe.
The full belief→network input contract is now defined. Smoke: 1732 states + 8114 options over 8
games, 0 errors; max 25 options/decision (0 truncated); option-type one-hots verified.

### Next
- Torch encoder + heads (embedding table + per-zone attention pooling → state vector; pointer policy
  head scoring options against it + value head). Needs a **PyTorch install** (training-only; the
  deployed agent uses ONNX-CPU). Then: self-play recorder → AlphaZero train → wire net into MCTS.

## #25 — ML foundations: encoder + policy/value net (`src/nn/encoder.py`)

**Date:** 2026-07-07
**Code:** `src/nn/encoder.py`, `scripts/encoder_smoke.py`, `requirements-train.txt`; torch 2.12 CPU installed

`PTCGNet` (§3 Layers 3-4): shared card embedding → per-Pokémon tokens (embed⊕scalars) → 1 self-
attention layer → masked attention-pool = board; hand attention-pool; globals MLP; state MLP over
[board, hand, stadium, globals] = state vector. Pointer **policy** head scores each option token
(embed⊕option-scalars) against the state (padding → −inf); **value** head → tanh ∈ (−1,1) (matches
the MCTS leaf scale). `collate()` stacks featurized (state, action) pairs → batch tensors.
Validated (`encoder_smoke.py`): forward gives logits [B,64]+value [B] with legal finite / padding
−inf / normalized policy / value∈(−1,1); loss backprops with finite grads; **496,962 params (~2 MB
fp32)** → ONNX-CPU-friendly. torch is dev/training-only (`requirements-train.txt`); never shipped.

### Next
- **Self-play recorder**: run v4 (MCTS-H) games, emit training tuples (state+action features,
  MCTS root visit distribution = policy target π, game outcome = value target z) into a buffer.
- **Trainer**: AlphaZero loss (policy CE vs π + value MSE vs z); acceptance-gate new checkpoints via
  the existing `round_robin` harness. Then wire the net as MCTS `prior_fn` + leaf value (small refactor
  — leaf value currently hardcoded), and ONNX-export for deploy.

---

## #26 — ML foundations: self-play recorder (`src/selfplay/recorder.py`)

**Date:** 2026-07-07
**Code:** `src/selfplay/recorder.py`, `scripts/record_selfplay.py`

The AlphaZero data pump. Plays full **MCTS-H self-play** games (both seats driven by the same
determinize → PUCT-MCTS pipeline) over the native engine and, at every genuine single-select branch,
emits a training tuple: inputs = `encode_state`⊕`encode_actions`; policy target **π** = the MCTS root
visit distribution over legal options + a trailing PASS slot (width `MAX_OPTIONS+1`, stored as raw
counts so the trainer picks normalization / PASS handling); value target **z** = game outcome from the
*deciding player's* perspective (+1/−1/0), back-filled at game end. Multi-select (mulligan/prize/
discard) and truly-forced single-option steps are resolved by a one-world leaf-scored heuristic and
**not** recorded — value still flows through them via the outcome. No change to `mcts.search` was
needed: it already returns `root_visits` in its stats; the recorder calls the pipeline directly (the
frozen v4 `agent.choose` stays untouched). Search runs on the SDK's separate `agent_ptr` sandbox, so
driving `battle_select` and `search_*` from the same loop is safe.

**Validated** (`record_selfplay.py --benchmark`, then a 3-game write + reload):
- Well-formed tuples: `visits` [D,65], every recorded decision has ≥1 visit, most-visited real option
  is always legal (mask-checked), z ∈ {−1,0,+1} with the correct per-mover sign (winner's decisions
  +1, loser's −1).
- **Timing:** a full ~146-turn game = **5.9 s** at `sims=160, k=2` (~**0.05 s/decision**, ~126
  decisions/game). Cheap enough to fan out across processes (one game/process — SDK global battle).
- **End-to-end train step works:** loaded a shard → `PTCGNet.forward` → policy CE (vs visit-π) +
  value MSE (vs z) → finite grads. Confirms the recorded schema is network-compatible — the trainer's
  inner loop is exactly this.

Shards are gitignored (`*.npz`); torch-free recorder (numpy only), so it stays importable in the
agent if ever needed. `sims`/`k` are the quality⇄throughput knobs (160/2 = fast first buffer; raise
for sharper π).

### Next
- **Trainer**: batch shards → AlphaZero loss (policy CE + value MSE), checkpoint, `round_robin`
  acceptance gate. Then wire the net as MCTS `prior_fn` + leaf value (small refactor — leaf value
  currently hardcoded) and ONNX-export for deploy.

---

## #27 — ML foundations: AlphaZero trainer + first PTCGNet fit (`src/nn/dataset.py`, `scripts/train_net.py`)

**Date:** 2026-07-07
**Code:** `src/nn/dataset.py`, `scripts/train_net.py`; bugfix `src/nn/encoder.py` (+ guard in `encoder_smoke.py`)

Trained `PTCGNet` on the first self-play buffer. `dataset.load_buffer` concatenates `.npz` shards and
derives the **policy target π** = visit distribution renormalized over the real option slots (PASS
slot dropped from the 64-wide pointer head; dropped mass **reported**, not silently — 3.7% here).
`train_net.py`: AlphaZero loss (policy CE vs π + value MSE vs z), Adam, grad-clip, **game-level**
train/val split (decision-level would leak), checkpoint on best held-out policy top-1.

**Buffer:** 120 self-play games (8 procs × 15, sims=160/k=2), **2838 decisions**, ~2 min wall.

**Result (30 epochs, val = 18 held-out games / 446 decisions):**
- **Policy top-1 22% → ~70%** — the net predicts the MCTS teacher's most-visited move ~7/10 on
  *unseen games*. A strong learned prior for the search.
- **Value-sign acc 48% → ~64%** — predicts the eventual winner from mid-game states.
- **Value head overfits** (train MSE 1.0→0.37, val MSE 1.0→1.2 diverging after ~ep4) — expected in the
  small-data regime: z is a high-variance single-game Monte-Carlo label, so the scalar value memorizes
  while the denser policy target generalizes. Fix is *more games*, not code; checkpoint is selected on
  policy top-1, so the value overfit doesn't pick a bad net. Early-stopping / more data next.

**BUG FOUND & FIXED (correctness, affects deploy too):** a fully-padded Pokémon board (no active/bench
either side — a degenerate setup state, **153/2838 = 5.4%** of decisions) made the board
`TransformerEncoderLayer` softmax over zero keys → **NaN**, which propagated to the value head and
silently poisoned batch-mean metrics (argmax/sign swallowed it, so it showed as `nan` CE/MSE only).
Fixed in `encode_state`: unmask slot 0 for fully-padded rows (board_pool re-masks by `pokemon_mask`,
so the fabricated token is never pooled in). Added a regression guard to `encoder_smoke.py`. This
matters beyond training — the net is a future MCTS leaf evaluator and must honour the never-NaN
contract on any legal state.

### Next
- Bigger buffer (thousands of games — the value head needs it) + early-stop.
- Wire the net into `search.mcts` as `prior_fn` (hook exists) + leaf value (replace the hardcoded
  `_value` with the net's tanh value; small refactor), then **round_robin acceptance gate** (net-MCTS
  vs plain v4 MCTS — the real test: does the learned prior/value beat the heuristic teacher?).
- ONNX-CPU export for deploy once a net passes the gate.

---

## #28 — Net wired into MCTS as prior + leaf value (`src/nn/guide.py`, `search/mcts.py` refactor)

**Date:** 2026-07-07
**Code:** `src/search/mcts.py` (injectable `value_fn`), `src/nn/guide.py`, `scripts/net_mcts_smoke.py`

The AlphaZero payoff wiring — the learned policy focuses the search and the learned value replaces
the heuristic evaluator at the leaves. `mcts.search` grew an optional `value_fn(state, root_player)
-> (-1,1)` threaded through `_value`/`_Node`/`_step_child`/`_simulate` (the `prior_fn` hook already
existed); **both default to the heuristic**, so plain MCTS-H (v4, recorder) is unchanged and
backward-compatible. `NetGuide` loads a checkpoint and implements both hooks from one net: `priors`
from the pointer head (softmax over legal options; PASS / >64-option get a small exploratory floor
since the head doesn't model them), `value` from the value head with the acting→root perspective flip
(matches how `z` was labelled). Forced/multi picks stay on the cheap heuristic.

**Validated** (`net_mcts_smoke.py`, 5 real branch decisions): net-guided search returns legal picks
with reported root visits, value ∈ (−1,1), and the heuristic default-arg path still works. Correctness
proven — NOT strength. Caveat: at the sampled early decision `max|prior−uniform| = 0.011` (soft
policy) — expected from the 120-game model; the prior will sharpen with a bigger buffer. **This is why
the buffer must grow before the acceptance gate is meaningful.**

Perf note (optimize later, not now): `value` and `priors` each run a net forward, so an expanded node
costs ~2 forwards; batching leaves / caching per-state would cut this. Fine for the gate at low sims.

### Next
- Bigger buffer (few thousand games) + retrain → **round_robin acceptance gate**: net-MCTS vs plain
  v4. The make-or-break test — does the learned prior/value beat the heuristic teacher?
- If it passes: flywheel + ONNX-CPU export. If not: diagnose (more data / sharper policy target / net
  capacity) before any cloud spend.

---

## #29 — 5k-game buffer, retrain, and the ACCEPTANCE GATE = **PASS** (`scripts/gate_net_vs_heuristic.py`)

**Date:** 2026-07-07
**Code:** `scripts/gate_net_vs_heuristic.py`; buffer `experiments/selfplay/buffer` (gitignored); `experiments/checkpoints/net.pt`

**Buffer:** 5,004 self-play games / **118,191 decisions** (9 procs × 556, sims=256/k=2, no errors) —
~42× the #27 smoke buffer. 37 MB on disk.

**Retrain** (`train_net.py`, 15 epochs, batch 512, 10% held-out games):
- **Value head fixed:** train MSE 0.69→0.53 and val MSE 0.61→0.58 now TRACK (no divergence) — the
  #27 value overfit was pure small-data, gone at 42× scale.
- **Value-sign acc 44%→~79%** (was ~64% at 120 games) — the value head calls the winner ~4/5 from
  mid-game. Policy top-1 up to **76%**. Best checkpoint = ep10 (top1 76%, sign 79%).

**THE GATE — net-guided MCTS vs plain heuristic MCTS, identical determinize+MCTS budget (only the
prior+value source differs), seats alternated, 200 games, sims=128/k=2:**
```
  net W/L/D = 141/59/0
  win rate  = 70.5% ± 3.2%   95% CI [64.2%, 76.8%]
  z vs 50%  = +5.80  ->  PASS (net > heuristic)
```
Consistent across all 8 workers (52–80%, no outlier). **The ML path is real: the learned value/prior
substantially out-search the hand-written evaluator.** This is the first agent improvement of the
whole arc that came from a *learned* component rather than search structure.

**Caveats (deploy-relevant, not yet tested):** the gate isolates quality *per simulation* at EQUAL
sims. Under a fixed WALL-CLOCK the net's ~2 forwards/node means fewer sims per move — so the real
deploy question is equal-time, which needs fast inference (**ONNX-CPU**). Also low sims (128) flatter
a strong value head; a higher-sims spot check is worth doing.

**BIG PERF LESSON (cost 1 hung 31-min run):** torch defaults to one intra-op thread *per core, per
process*. 8 worker processes → ~80 threads on 10 cores, load 29, throughput collapsed (0 games in 31
min). Fix: `torch.set_num_threads(1)` + `OMP/MKL/OPENBLAS_NUM_THREADS=1`; parallelism comes from the
process count. Single-threaded was also **~8× faster per game** (3.3s vs 26.6s) — for a 2 MB net,
torch threading is pure overhead. Ship the agent single-threaded too. (Also: never wait on
`while pgrep -f X.py` — the watcher's own command line contains X.py and self-matches forever; gate on
output files instead.)

### Next
- **ONNX-CPU export** of the net → wrap NetGuide's two calls as an ONNX session → re-gate at equal
  WALL-CLOCK (the true deploy test) + a higher-sims spot check.
- If it holds: bundle a submission (`main.py` + `net.onnx` + `deck.csv`, single-threaded, watchdog),
  validate in cabt (all DONE, move-time under clock), submit → own ladder μ.
- Flywheel: this net → stronger self-play → bigger buffer → retrain → re-gate. GPU worth it once CPU
  training >30–60 min (~10k+ games); many-core CPU box helps the self-play bottleneck more.

---

## #30 — ONNX-CPU export + equal-wall-clock re-gate = **PASS (ships)** (`scripts/export_onnx.py`, `src/nn/onnx_guide.py`)

**Date:** 2026-07-08
**Code:** `scripts/export_onnx.py`, `src/nn/onnx_guide.py`, `scripts/gate_net_vs_heuristic.py` (+`--onnx`/`--budget`); `encode_state` pad-fix made branch-free

The deploy path. `PTCGNet` → **two self-contained ONNX graphs**: `value.onnx` (state→value, the hot
leaf path — 1.89 MB) and `policy.onnx` (state+options→raw logits, masked in numpy — 2.01 MB). Only
the batch dim is dynamic (batch=1 confirmed — MCTS calls one state at a time). Validated vs torch at
**fp32 precision** (value Δ 1.2e-7, logit Δ 1.3e-7) on real states + the padded-board NaN guard.
Gotchas handled: torch≥2.x dynamo exporter needs `onnxscript`; it writes weights to a `.onnx.data`
sidecar → re-saved inline so a single file ships; the `all_pad` NaN guard was rewritten branch-free
(tracing bakes out `if`s). `OnnxGuide` = `NetGuide`'s twin over single-threaded `onnxruntime` (no
torch in the deployed agent).

**Equal-SIMS gate (#29 recap):** net 70.5%. **Equal-WALL-CLOCK re-gate (the deploy test), 0.4s/move
(governor routine budget), 200 games, ONNX net vs heuristic, seats alternated:**
```
  avg sims/move: net 8926  heuristic 7846
  net W/L/D = 117/83  ->  58.5% +/- 3.5%   z=+2.40  ->  PASS
```
The edge shrinks 70.5%→58.5% under a clock (the net pays for its forwards) but stays net-positive and
significant. Amortization is why it's not worse: MCTS caches priors/value per node, so forwards are
paid on expansion, not every sim → net gets ~90% of heuristic's sim count (8.9k vs 7.8k), not ~25%.
Consistent across workers (6/8 ≥56%). z=+2.40 is *just* past significance at N=200 — real but modest;
a stronger net (flywheel) should widen it.

**Deps:** onnx, onnxscript, onnxruntime added to `requirements-train.txt`; **onnxruntime is the ONLY
nn dep the shipped agent needs** (torch stays dev-only).

### Next
- **Bundle the submission**: new `submissions/vN/` = `main.py` (ONNX MCTS-H agent, single-threaded,
  watchdog) + `value.onnx` + `policy.onnx` + `deck.csv` + vendored `src/{ptcg,search,nn(features+
  onnx_guide)}`. Validate in cabt (all DONE/DONE, move-time under clock, no torch import), submit.
- Higher-budget spot check (5s critical) + bigger-N re-gate to tighten 58.5%.
- Flywheel: this net → self-play → retrain → re-gate (each gen should widen the edge).

---

## #31 — v5 submission bundle: ONNX net-guided MCTS-H agent (`submissions/v5/`)

**Date:** 2026-07-08
**Code:** `submissions/v5/` (main.py, agent.py, value.onnx, policy.onnx, deck.csv, vendored cg/ptcg/search/nn), `scripts/build_submission.py` (+nn/onnx), `scripts/validate_v5.py`

The shippable trained agent. Same pipeline as v4 (perception → governor → determinize → PUCT MCTS →
most-visited) but MCTS priors + leaf value come from the trained net via `OnnxGuide` (onnxruntime,
single-threaded). **Graceful degradation:** if onnxruntime or the .onnx files are missing at runtime
`_GUIDE` stays None and it falls back to plain heuristic MCTS-H (v4) — the net can never turn into a
crash/timeout loss. `build_submission.py` now vendors a **torch-free** nn subset (features.py +
onnx_guide.py only — no encoder/guide/dataset) and copies value/policy.onnx in, gated on the agent
importing onnx_guide.

**Validation (`validate_v5.py`, torch import poisoned to simulate the Kaggle runtime):**
- **torch blocked ✓** — bundle runs on onnxruntime alone (the only nn dep shipped).
- **net actually loads ✓** (not silently degrading) — caught a validator bug first: the net loads
  lazily on the first `choose()`, not in `set_deck`, so the check must come after real moves.
- **4/4 games DONE, 152 moves, 0 illegal/exceptions**; max move 5.2 s (= 5 s critical governor budget
  + overhead).
- **Bundle 5.3 MB** (`main.py`+`deck.csv` at top level, no torch/.pt) — far under the 197.7 MiB cap.

**Timing caveat (follow-up):** one v5-vs-v5 game ran 157 s wall (both sides net-guided, ~78 s our
share — within the 600 s/game clock). The validator passes `time_left=None`, so the governor's
low-clock throttle (goes instant under `LOW_CLOCK_S`) never engaged; on Kaggle `remainingOverageTime`
is supplied and it throttles as the bank drains. Worth a real-clock timing pass before/after submit.

### Next
- Submit v5 to Kaggle (own ladder μ) — the ground-truth signal vs real opponents.
- Real-clock timing validation (feed a draining `time_left`, confirm never > 600 s/game).
- 5 s-budget spot check + bigger-N re-gate to tighten 58.5%.
- Flywheel gen #2: self-play from v5 → retrain → re-gate (should widen the edge).

**Timing caveat RESOLVED (`scripts/timing_v5.py`).** Drove v5-vs-v5 with a draining per-seat bank
(injected `remainingOverageTime`). Bank=600: worst seat-time **28.2 s / 600 s** (20× margin), min bank
571.8 s, 0 timeouts. Bank=45 stress: the governor throttle engaged — min bank floored at **19.2 s**
(≈ the 20 s LOW_CLOCK instant threshold), 0 timeouts, even counting ungoverned overhead (determinize /
session-init / `_multi_pick`). The proportional throttle (≤10% of remaining, instant below 20 s)
provably prevents clock overrun. **v5 is timeout-safe — ready to submit.**

## #32 — Deck A/B under the net pilot: real meta list vs sample = **sample still wins** (`scripts/gate_deck_ab.py`)

**Hypothesis.** The deck-construction arc (#15-#23) closed under *heuristic* pilots — no constructed
meta deck beat the SDK `sample` deck (sample won ~73-81%). Open question flagged at v5 ship: does the
stronger **trained net pilot** change the verdict? Maybe the net can pilot a complex evolution deck
well enough to finally beat sample. Test: hold the pilot fixed (net-guided MCTS, identical budget/k/
worlds both seats), vary only the DECK. Seat A = `dragapult-dusknoir` (top constructible archetype from
the v3 round-robin, a genuine real-meta list: full Dreepy/Drakloak/Dragapult ex line, Buddy-Buddy
Poffin, Rare Candy, Boss's Orders, Crispin, 8 energy). Seat B = sample. Seats swap every game.

**Method.** New `scripts/gate_deck_ab.py`: same `OnnxGuide` on both seats, each seat determinizes with
its OWN decklist. 6 workers × 12 games = **72 games**, equal-wall-clock `--budget 0.4`s/move.

**Result.** deck-A (dragapult-dusknoir) W/L/D = **14/57/1 = 19.7% ± 4.7%**, z = **−6.41** vs 50%. Sample
wins **80.3%** — *wider* than the v3 heuristic-pilot gap (73.3%). The stronger pilot did NOT reverse it.

**Two confounds, both AGAINST Dragapult** (so 19.7% is a floor, not a clean deck-strength number):
1. **OOD pilot** — the net was trained on sample-deck self-play only, so its policy/value heads are
   out-of-distribution piloting Dragapult.
2. **Complexity tax under equal wall-clock** — the complex board yields more options/decision, so each
   MCTS sim is costlier: deck-A got **796 sims/move vs deck-B's 2765** (3.5×fewer) at the same 0.4 s.

**Verdict.** DECK ARC STAYS CLOSED. No constructed meta deck beats `sample`, now confirmed under the
strongest pilot we have. Sample ships. The lever remains the PILOT, not the deck. A fully clean deck
test would require (a) net self-play trained ON the candidate deck (removes confound 1) and (b) equal
*sims* rather than equal wall-clock (removes confound 2) — not worth it given the −6.41 margin.

## #33 — Multi-deck pilot framework, phase 1: deck-conditioned features (`src/nn/features.py`)

**Date:** 2026-07-08 · **Plan:** docs §3d (appended this session). Motivation = #32's OOD-pilot confound:
the gen-1 net only knows the sample deck, so it can't fairly pilot anything else. Phase 1 gives the
featurizer the hooks a multi-deck pilot needs to *condition on which deck it is playing*.

**Change (additive + versioned).** `encode_state` gains two optional schema-v2 fields:
- `deck_summary` — a **28-d** own-decklist composition vector from new `summarize_deck(deck, ci)`:
  card-type histogram (7) + Pokémon energy-type histogram (12) + 9 aggregates (basic/ex/mega/ability/
  stage1/stage2 fractions, avg Pokémon HP, ACE-SPEC present, distinct-card fraction). Static per game.
- `archetype_id` — a stable embedding id from new `archetype_id_for(name)` (`ARCHETYPE_IDS`, vocab 16;
  0 = unknown/generic, sample=1, dragapult-dusknoir=2, basic-box-m=3, …).
`SCHEMA_VERSION = 2` documents the contract.

**Backward-compat (the requirement): v5 keeps working.** The new fields default to zeros/0, so
`encode_state` stays a single code path. The shipped v5 `OnnxGuide` reads only the seven v1 arrays by
name (`export_onnx._STATE_ARGS`) and calls `encode_state` without the new kwargs → it gets the exact
old tensors. **Verified:** across 120 live states the six v1 arrays (+ stadium_id) are **byte-identical**
between a v1-style call and a deck-conditioned call; a v1 call yields zeros; and `OnnxGuide` loads and
runs unchanged under schema v2. Durable regression added to `scripts/features_smoke.py` (prints
`0. SCHEMA v2 …`). The v2 net architecture (deck_summary MLP + archetype embedding) is built when we
train in phase 5 — phase 1 is just the interface.

## #34 — Phase 6: Kaggle top-episode export recon → **BC dataset is VIABLE now** (recon only, no code)

**Date:** 2026-07-08. Question from §3d step 6: is a behavioral-cloning dataset from top ladder agents
viable yet? Checked via the authed `kaggle` CLI (token in `~/.kaggle/access_token`; comp
`pokemon-tcg-ai-battle`).

**Ladder state (bonus — resolves an open item):** v5 (`submission_v5.tar.gz`, 54454217) scored
**publicScore 553.9**, up from **v1's 509.0** (54411051) — the trained net is a **real ~+45-pt ladder
gain vs the heuristic**, corroborating the #29/#30 gates on live opponents. Leaderboard = **20 teams**,
top score **1178.9** (~2× ours — strong opponents to clone from).

**Episode/replay availability (the deliverable):** `kaggle competitions episodes <submission_id>` lists
a submission's games; **v5 has 48, v1 has 55** (accumulating as the ladder plays, growing daily).
`kaggle competitions replay <episode_id>` downloads the full replay (~1.9 MB JSON). Structure is
directly BC-usable: top-level `rewards` (e.g. `[1,-1]`) gives the **winner**; `steps` = per-turn list of
2 agent-entries, each with `observation` (**includes `select.option` = the exact legal-option list the
agent saw**) + `action` (**chosen option indices**, our agent contract) + `status`(ACTIVE/INACTIVE) +
`reward`. So a replay yields full winner (and loser) trajectories from each player's own perspective —
**our existing `encode_state`/`encode_actions` featurizer consumes them as-is.**

**Volume:** ~50 episodes/submission now → ~100 games across our 2 submissions immediately; ~87 ACTIVE
decisions/game of which **69/87 (~79%) cleanly index their shown option list** → ~2–3k winner-action
samples available *today* from our own episodes alone, growing ~daily. **Scales to the full field:**
the leaderboard CSV carries each `teamId`, and `kaggle competitions team-submissions <teamId>` → their
submissions → `episodes` → `replay` opens the 20-team corpus (potentially 10k+ games).

**One parsing detail to handle when we build the BC loader (phase 5):** the ~21% non-fitting actions
are (a) the initial 60-card **deck-return** step (action = card ids, not option indices — identifiable
by length 60) and (b) a minority of select contexts where the action isn't a plain option index
(value/count selects, or an obs↔action alignment offset within multi-select steps). The loader must
skip/special-case these. Tractable, not a blocker.

**Verdict:** BC pretraining (§3 training-path step 1, so far unrealized) is **viable now** and gets
better daily. Recommended: harvest our own episodes first (cheap, already ours), then expand to the
top-N teams. Slots into phase 5 as the pretrain corpus alongside the scripted-expert games.

## #35 — Phase 2: scripted line-experts framework + dragapult-dusknoir expert (`src/experts/`) — **TIES v1, gate NOT yet cleared**

**Date:** 2026-07-08 · **Plan:** docs §3d step 2. Built the per-archetype scripted-pilot framework and
the first expert; the phase-2 bar is "**beat v1 piloting its own archetype**". Current result: **tie**.

**Built.** `src/experts/base.py` — `ScriptedExpert` with the `set_deck`/`choose(obs_dict, time_left)`
agent contract, a `_Ctx` option-query layer (find PLAY/EVOLVE/ATTACH/ATTACK options by card/attack id,
board/hand queries, opponent-target finder), full crash-safety, and a **v1 fallback** for any decision
no rule handles. `src/experts/dragapult_dusknoir.py` — scripts the skeleton line: Rare Candy Dreepy→
Dragapult ex (when the ex is in hand), evolve up the lines, Crispin before committing, energy onto the
attacker, **Phantom Dive** as the payoff, and its **damage-counter spread** concentrated on an opponent
bench target (KO-setup proxy). `src/experts/__init__.py` — `get_expert(name)` registry. Gate harness
`scripts/gate_expert_vs_v1.py` (expert vs v1 mirror, seats alternated, fan-out `--seed`/`--out`).

**Result (`gate_expert_vs_v1.py`, dragapult-dusknoir mirror, 5×12 = 60 games):** expert W/L/D =
**30/29/1 = 50.8% ± 6.5%, z = +0.13** vs v1 — a statistical **tie**. All 60 games DONE, 0 crashes/
timeouts, 1 draw → the framework is sound, but the expert does **not** out-play v1.

**Why (firing probe): the expert is ~90% v1.** It fires a scripted rule on only ~8 of ~75 decisions/
game (MAIN setup + a couple of Phantom-Dive spread placements) and **defers the other ~67** (Ultra Ball,
Buddy-Buddy Poffin, draw supporters, Boss's, retreat/switch, most sub-selects) to the v1 fallback. A
thin override layer on top of v1 structurally can't beat v1 by much on a mirror — and v1's 1-ply
evaluator already plays this deck's obvious lines competently.

**Interpretation.** To clear the gate the expert must **own more of the turn** with archetype knowledge
that beats v1's myopia — script bench development (Poffin/Ultra Ball incl. their search sub-selects),
draw-supporter timing, Boss's-only-when-it-converts (currently deferred), and smarter Phantom-Dive
targeting off the prize object — using v1 only as a true last resort. That is a fuller rules-based
pilot, the next iteration. Weak side-signal for #32: even a scripted line-pilot can't outplay v1 on
dragapult-dusknoir, mildly consistent with "the deck is genuinely weaker," though #32's pilot was the
net, not v1. **Infrastructure committed; gate open. Phases 3-5 depend on an expert that clears it.**

## #36 — Phase 3: mixed deck-conditioned self-play recorder (`src/selfplay/recorder.py`, `scripts/record_mixed.py`)

**Date:** 2026-07-08 · **Plan:** docs §3d step 3. Extends the recorder to generate across a **deck pool
× pilot pool over all seat pairings**, recording the phase-1 deck-conditioning features — the direct
data fix for #32's OOD confound (chosen path B: train the net on the decks it must pilot, rather than
gating on a scripted expert first).

**Built (additive — old mirror `record_game` untouched):**
- `Pilot` abstraction: `MctsPilot` (heuristic OR net via injected `prior_fn`/`value_fn`; branch
  decisions → root visit distribution, `source=MCTS`) and `ExpertPilot` (a scripted line-expert;
  branch decisions → one-hot BC target, `source=BC`). `build_pilot("heuristic"|"net"|"expert:<arch>")`.
- `record_mixed_game(pilots, decks, names, …)`: a (possibly) different deck AND pilot per seat; each
  seat determinizes with its own deck and its recorded states are encoded with that deck's
  `deck_summary`+`archetype_id`. z still labelled per-mover.
- `_Buffer` now also stacks `deck_summary`/`archetype_id` (pulled straight off `StateFeatures`) and a
  `source` tag per decision.
- CLI `scripts/record_mixed.py`: default pool = {sample, dragapult-dusknoir} × {heuristic, net(with
  `--onnx-dir`), expert:dragapult-dusknoir} → 5 sides / 25 ordered matchups, cycled per `--seed`;
  writes to `experiments/selfplay/mixed/` (gitignored). `--benchmark` = built-in smoke.

**Validated** (inline + `--benchmark`): expert rows are one-hot @ dragapult (arch 2); heuristic/net
rows are soft-π @ their deck (sample arch 1); z = +1 for winner-mover rows; the net-pilot path loads
and records; `_check` passes (shapes, deck_summary 28-d, ≥1 visit/decision, most-visited option legal).
Cost ~0.7–2.2 s/game at sims=96/k=1. **New npz keys are additive and land in a separate dir, so the
existing v5 buffer/trainer path is unaffected** (dataset/trainer consume the new keys in phase 4/5).

**Note:** basic-box-m is not in the pool yet (no deck/expert built) — add its deck under
`experiments/candidate_decks/` + register an expert to include it. Phase 4 (aux targets) next, then
phase 5 generates the real buffer + trains the v2 deck-conditioned net + gates.

## #37 — Phase 4: auxiliary value targets (`recorder.py` aux, `dataset.py` exposure)

**Date:** 2026-07-08 · **Plan:** docs §3d step 4. Two denser, deck-agnostic learning signals recorded
alongside the sparse game outcome z, to reward the setup lines the net must learn on non-sample decks.

**Targets (per recorded decision, mover's perspective), computed from the game trajectory:**
- **`prize_diff_t3`** ∈ [-1,1] — (my prizes taken − opp prizes taken) at `turn+3` (engine turn counter;
  terminal state if the game ends sooner), /6. Rewards building a prize lead. A prize timeline
  `[(turn, rem0, rem1)]` is sampled each decision (prizes only decrease).
- **`turns_to_attack`** ∈ [0,1] — turns until this side's FIRST declared attack (detected via a chosen
  option of type ATTACK), 0 once online, `_ATTACK_CAP=8` if it never attacks, /cap. Rewards getting an
  attacker online fast — exactly the delayed-payoff setup 1-ply eval undervalues (§3c).

**Plumbing.** `record_mixed_game` tracks the timeline + per-seat first-attack turn and, after
`finalize`, attaches both target arrays (+ `turn`) to the shard. `dataset.py` gains `deck_cond_batch`
(deck_summary + archetype_id for the v2 net) and `aux_targets` (prize_diff_t3, turns_to_attack, source,
is_bc), both defaulting to zeros for pre-v2 buffers — `make_batch`'s (batch, π, z) signature is
UNCHANGED so the legacy trainer/net keep working.

**Validated.** Aux targets are meaningful, not noise: on a real game **prize_diff_t3 mean = +0.63 for
winner-mover rows vs −0.66 for loser-mover rows** (tracks the outcome), and **turns_to_attack = 0.33
early (turn≤4) → 0.00 late (turn≥10)** (decays as sides come online). Data path: 4 mixed shards load
with deck_summary(28-d)/archetype_id{1,2}/aux/source; the legacy v5 buffer loads with all v2 fields
defaulted to zero (backward compatible). `record_mixed.py --benchmark` covers it.

**Deferred to phase 5 (with the v2 net):** the encoder's auxiliary heads + the aux MSE loss terms —
they need the new v2 architecture (deck-conditioning inputs + aux heads), which is built at train time.

## #38 — Phase 5(a): v2 net architecture (deck-conditioning + aux heads) + trainer (`encoder.py`, `train_net.py`)

**Date:** 2026-07-08 · **Plan:** docs §3d step 5 (part a). Built + unit-tested the v2 network and
training loop before committing compute to a real buffer (part b generates it; part c trains+gates).

**Architecture (all gated behind `NetConfig` flags → v1 default byte-identical, 496,962 params).**
`deck_cond=True`: fold the own-deck summary (28-d → `deck_mlp`) + `archetype_id` (`arch_embed`,
vocab 16) into the state vector. `aux_heads=True`: two training-only heads off the state vector —
`prize_head`→tanh (prize_diff_t3), `tta_head`→sigmoid (turns_to_attack). New `forward_train` returns
(logits, value, aux); **inference `forward` is unchanged** (aux heads never touched at play time, so
MCTS/guide/ONNX see the same value+policy interface). v2 = **577,540 params** (~2.3 MB).

**Trainer (`train_net.py`) — v1 path unchanged when flags off.** New flags `--deck-cond` / `--aux-heads`
/ `--aux-weight` (0.5) / `--bc-epochs`. Loss = policy CE + value MSE + aux_weight·(prize+tta MSE).
`--bc-epochs` runs a policy-CE-only warmup over the `SOURCE_BC` (scripted-expert) rows first
(behavioral cloning, §3d step 5), then standard AlphaZero training on the full mixed buffer. Batches
merge `deck_cond_batch` keys; `aux_targets` supply the aux MSE targets.

**Unit-tested (small 4-shard mixed buffer + synthetic):** v1 arch unchanged (496,962); v2
forward==forward_train in eval; aux outputs finite + in range; grads flow to `deck_mlp`/`arch_embed`/
aux heads; v2 training loop decreases train CE (1.559→1.549), value MSE (0.95→0.30), aux MSE
(0.46→0.08), no NaN; BC warmup executes over the 34 expert rows (policy CE only); **v1 default
training still runs unchanged**. Durable check added to `encoder_smoke.py` (`5. SCHEMA-v2 …`).
(Val metrics meaningless at 3-game train — this validates loop mechanics; real learning waits on the
part-b buffer.)

**Deferred to 5(c):** wiring the v2 net into MCTS for the gate needs the guide to pass the piloted
deck's `deck_summary`+`archetype_id` into `encode_state` (deck context) — will use the torch `NetGuide`
for the gate (skip a v2 ONNX export until it ships). Part b (real mixed-buffer generation) launching now.

## #39 — Phase 5(c): v2 net trained on the mixed buffer + preliminary gate vs v5 (`gate_v2_vs_v5.py`, `guide.py` deck-cond)

**Date:** 2026-07-08 · **Plan:** docs §3d step 5 (parts b+c). Generated the real mixed buffer, trained
the deck-conditioned v2 net, and gated it against the shipped v5 net.

**Mixed buffer (part b).** 8 workers × 100 games = **800 games / 82,308 decisions** across the pool
(sample+dragapult × heuristic/net/expert, all seat pairings). Composition: 57,903 MCTS-π + **24,405 BC**
(30%) rows; **69,560 dragapult** vs 12,748 sample rows (the OOD deck now heavily represented, the point);
dropped-PASS 1.7%. `experiments/selfplay/mixed/` (gitignored).

**v2 training (part c).** `train_net.py --deck-cond --aux-heads --bc-epochs 3 --epochs 12 --aux-weight
0.5` → `net_v2.pt` (v5's `net.pt` untouched). Val **top1 58.3%**, **sign 71.7%**, aux MSE prize 0.086 /
tta 0.127. (Lower top1 than v5's 76% is expected — a far more diverse target: 3 pilots × 2 decks + BC.)

**Gate wiring.** `NetGuide` extended with `set_deck(deck, archetype)` → injects `deck_summary` +
`archetype_id` into every state/priors batch (no-op for v1 nets, so v5 NetGuide unaffected).
`gate_v2_vs_v5.py`: mirror, same deck both seats, seat A = v2 NetGuide (deck-conditioned, torch),
seat B = v5 OnnxGuide, identical determinize+MCTS budget, seats alternated.

**Preliminary gate result (LOW POWER — see caveat):**
```
  dragapult-dusknoir:  v2 13/11 = 54.2% ±10.2%  z=+0.41   (criterion a: improve → INCONCLUSIVE)
  sample:              v2 13/11 = 54.2% ±10.2%  z=+0.41   (criterion b: no-regress → PASS)
```
So v2 does **not regress on sample** and is **directionally ahead on dragapult** but not materially/
significantly so.

**Caveat — the gate is underpowered, by torch speed.** The v2 NetGuide is torch and rebuilds perception
+ runs a forward on *every sim's leaf* → ~67 s/game at sims=128; 8 concurrent workers thrashed (14 min,
0 completions). Forced down to **sims=16 / N=24** (real gates: sims=128 / N=200). At such shallow search
the two nets' guidance barely diverges (hence the near-coin-flip, identical W/L). **The definitive gate
needs a v2 ONNX export** (fast inference → sims=128, N≥200) — which is also the v2 deploy path. Until
then the verdict is: pipeline works end-to-end, v2 is safe on sample, dragapult improvement unproven.
(Ops lesson repeat: `pkill -f gate_v2_vs_v5` self-matches the launching shell's own command line — it
killed the parent before it spawned workers. Don't -f a pattern that appears in your own command.)

**Next:** export v2 to ONNX (extend `export_onnx.py` for the deck_summary/archetype_id inputs +
`OnnxGuide` deck context) → re-gate at sims=128/N≥200 for the real dragapult verdict; if v2 wins there,
it becomes v6.

## #40 — Phase 5 gate at full power: v2 net vs v5 = **NEGATIVE, v2 does NOT ship** (`export_onnx.py`/`onnx_guide.py` deck-cond)

**Date:** 2026-07-08 · **Plan:** docs §3d step 5, definitive gate. Exported the v2 net to ONNX so the
gate could run at the real budget (sims=128/N=200), not the torch-limited sims=16/N=24 of #39.

**v2 ONNX export.** `export_onnx.py` extended for the deck-conditioning inputs: `_ValueWrapperDC`/
`_PolicyWrapperDC` add `deck_summary`+`archetype_id`; `OnnxGuide` auto-detects deck-cond from the graph's
input names and `set_deck()` injects them. Exported to `experiments/checkpoints/v2/` (**v5's ONNX in
`checkpoints/` untouched**). Numerically exact vs torch (value Δ 1.2e-7, logit Δ 8.3e-7), padded-board
finite. Both guides ONNX now → ~20 s/game at sims=128 (vs ~67 s torch).

**FULL-POWER GATE (mirror, same deck both seats, v2 OnnxGuide vs v5 OnnxGuide, sims=128/k=2, N=200 each):**
```
  dragapult-dusknoir:  v2 94/105/1 = 47.2% ±3.5%  z=-0.78   (criterion a: improve → FAIL, parity)
  sample:              v2 79/121/0 = 39.5% ±3.5%  z=-3.04   (criterion b: no-regress → FAIL, REGRESSES)
```
**v2 does NOT ship. v5 remains the champion.**

**Why (the real finding).** Two things, both instructive:
1. **Sample regression is dilution.** v5 trained on ~118k *sample* decisions; v2's mixed buffer had only
   **12.7k sample rows** (85% of it was dragapult) + 30% one-hot BC from a mediocre expert. The shipped
   deck's signal was starved and its play degraded — a significant −10.5-pt regression.
2. **No dragapult gain refutes the #32 OOD hypothesis.** Training the net *heavily* on dragapult
   (69.6k rows) did **not** make it pilot dragapult better than the sample-only v5 (47%, parity). So v5's
   weak dragapult play was **not** because it was out-of-distribution — it's that **dragapult-dusknoir is
   genuinely harder to pilot well, and none of our teachers play it well** (heuristic, OOD-v5, and an
   expert that only ties v1), so there was no strong dragapult play in the buffer to learn from.
   AlphaZero-style distillation can't exceed its teachers, and on this deck every teacher is weak.

**Strategic takeaways.** (a) **Deck arc stays closed, now with stronger evidence:** even a dragapult-
trained net can't pilot it competitively — the limit is the deck + teacher quality, not pilot
familiarity. (b) **Multi-deck training as done here is net-negative** (dilutes the shipped deck without a
strong-teacher source for the new deck). A useful multi-deck net would need (i) a *dedicated* per-deck
net or per-deck data balancing to avoid dilution, AND (ii) a genuinely strong dragapult teacher first
(the phase-2 wall). Neither is a quick win. **v5 stays shipped; §3d phase 1 closes here as a negative
result.** ONNX deck-cond infra + the v2 pipeline are kept (reusable if a strong teacher ever appears).

## #41 — Ladder-BC: harvest top-tier play, train, gate = **NO v6 (sample tie); meta report is the win**

**Date:** 2026-07-08 · Realizes §3 training-path step 1 (BC on winners) from the actual Kaggle ladder —
the "strong teacher" lever #40 identified.

**1. HARVEST (`scripts/harvest_ladder.py`, in-process Kaggle API).** Leaderboard is **4,603 teams** (the
earlier "20" was one `--show` page); 2,529 ≥ 650 — too many to pull exhaustively, so top-20 teams
(scores 1061-1214) + our v5 → 983 episodes enumerated. **Kaggle rate-limits replays ~200/window (429)**
→ harvested **201 replays** (added retry/backoff/resume). Stored `data/replays/` + `index.json`.

**2. META REPORT (`ladder_meta.py` → `experiments/ladder_meta.md`) — the standout.** 402 deck sightings,
17 archetypes. **Ladder meta: Alakazam Stage-2 dominates 36% (65% win, rating 1161); Kangaskhan 20%;
Garchomp 14%. Complex evolution decks = 72% of top play. Dragapult ex = 1%/25%-win fringe** — independent
confirmation of #40. **Supersedes the TrainerHill priors.**

**3. BC DATASET (`ladder_bc.py`, `build_ladder_bc.py`).** Winner groundable single-selects → one-hot BC
rows, z=+1, deck-conditioned on the winner's recovered decklist, aux, SOURCE_BC. **7,817 rows / 23 decks**,
winner ratings median 1116, 0 unusable.

**4. TRAIN (`net_ladder.pt`).** #40 dilution fix: fresh deck-cond+aux **sample self-play (40,810 rows,
strong MCTS)** merged (`dataset.merge_buffers`, `train_net --bc-buffer`, `record_mixed --sample-only`) with
ladder-BC = 48.6k, **84% sample (dominant → no dilution)**. BC-warmup → AlphaZero. Val top1 67.6%, sign 77%.

**5-6. GATE (equal-wall-clock ONNX; v5+net_v2 untouched) → NO v6:**
```
  (a) sample, must WIN:  49.5% ±3.5% z=-0.14  → TIE (FAIL)
      dragapult (complex): 34.7% ±3.4% z=-4.54 → below v5 (saw ~no dragapult)
```
**Findings:** (1) **Dilution is fixable** — sample-dominant mix keeps the ship deck at PARITY (not #40's
−10pt regression). (2) **Ladder-BC didn't improve sample** — cross-deck transfer from Alakazam/Kangaskhan
to the simple sample deck adds nothing v5 lacked (v5 near sample's ceiling). (3) Net pilots untrained decks
worse (dragapult 34.7%). (4) **The meta report is the durable win** — real simulator meta now known.
Gate (c) line_audit not built; complex-deck win rate is the execution proxy. **v5 stays shipped.** Harvest/
BC/meta infra kept; corpus grows by re-running the resumable harvester across rate-limit windows.

## #42 — Alakazam deck + ladder-BC pilot **BEATS the ship 72.9%** — first non-sample winner; value-mask was the unlock

**Date:** 2026-07-08 · Follow-up to #41: build the ladder's #1 deck (Alakazam) and pilot it with the
ladder-BC net. Result: **the first deck+pilot to beat sample+v5 in the whole project.**

**Deck (`experiments/candidate_decks/alakazam-ladder.csv`).** Extracted the real ladder Alakazam list
from top-agent replays: played 82× by winners at **median rating 1213** (the #1 team's deck). Abra→
Kadabra→Alakazam Stage-2 + Rare Candy + Dunsparce/Dudunsparce draw engine + Poffin/Poké Pad + Hilda/Dawn
+ Boss's + Enhanced Hammer + Telepath Psychic Energy. Legal, ladder-proven.

**The bug + fix (the crux).** First pilot test — `net_ladder` (from #41) vs v5, Alakazam mirror, equal
wall-clock, 96 games: **net_ladder 20.8% (z=-7.04)** — it *mimics* the busy Alakazam engine (~92 moves/
game vs v5's ~17) but LOSES badly. Cause: **BC rows all carry z=+1 (winner-only), so they poison the
value head** toward "+1 everywhere" on Alakazam states → MCTS gets garbage leaf values → bad play.
Fix: `train_net.py --mask-bc-value` trains value ONLY on real-outcome MCTS rows (policy + aux still learn
from BC). Retrained `net_ladder2.pt`.

**Result after the fix (equal wall-clock, ONNX both sides):**
```
  Gate A2  net_ladder2 vs v5, ALAKAZAM mirror        74/22  = 77.1% ±4.3%  z=+6.31
           (net_ladder poisoned = 20.8% → net_ladder2 = 77.1%: a full reversal)
  Gate B   Alakazam+net_ladder2  vs  sample+v5 (SHIP) 70/26  = 72.9% ±4.5%  z=+5.05  ***
```
**Findings.** (1) **The value-mask fix is decisive** — BC-only imitation gives policy but not value;
feeding its fake z into the value head actively breaks MCTS. Masking it rescued the pilot (20.8%→77.1%).
(2) **net_ladder2 pilots the ladder's best deck far better than v5** (77%) — the ladder-BC produced a
genuine strong Alakazam pilot, exactly the strong-teacher payoff #40/#41 were chasing. (3) **Alakazam +
net_ladder2 BEATS our current ship (sample+v5) 72.9%** — the DECK ARC REOPENS. Recall Gate-A(#41) showed
net_ladder≈v5 on *sample* (tie), so this win is the DECK+knowing-pilot, not a generally-stronger net.

**→ v6 CANDIDATE: ship the Alakazam deck piloted by net_ladder2.** Before shipping: (a) build the v6
bundle (alakazam deck.csv + net_ladder2 ONNX + deck-conditioned OnnxGuide that set_deck's Alakazam);
(b) **timeout-validate** — Alakazam turns are long (~92 decisions/game) so re-run the draining-bank clock
test; (c) confirmation controls worth running: net_ladder2(sample) vs v5(sample) should be ~parity (proves
Gate B is the deck), and ideally a real-ladder submission for ground truth. This is the project's first
path past the sample-deck ceiling.

## #43 — v6 submission bundle: deck-conditioned Alakazam agent, built + timeout-validated (`submissions/v6/`)

**Date:** 2026-07-08 · Ships the #42 breakthrough (Alakazam + net_ladder2 beats sample+v5 72.9%).

**Bundle.** `submissions/v6/` = v5 pipeline (perception → governor → determinize → PUCT MCTS → most-visited)
but the shipped net is the **deck-conditioned `net_ladder2`** and the agent `set_deck`s the Alakazam
decklist onto the `OnnxGuide` so the net conditions on it (deck_summary + archetype_id 0, matching
training). Graceful degradation to heuristic MCTS unchanged. `deck.csv` = the ladder Alakazam list.
`build_submission.py` made version-aware (`_ONNX_SUBDIR["v6"]="ladder2"` → vendors net_ladder2's ONNX;
v5's `checkpoints/` ONNX untouched). Tarball **5.4 MB** (main.py+deck.csv+onnx top-level, torch-free).

**Validated (`scripts/timing_v6.py`, torch import poisoned):** torch-free ✓, net loads ✓, **deck-
conditioned on Alakazam ✓** (deck_cond + nonzero deck_summary), all games DONE, 0 illegal.
**Timeout-safe** despite Alakazam's long turns (~90+ decisions/game, vs v5's ~40):
- bank=600 (realistic): 4/4 DONE, worst seat-time **200.6 s / 600 s** (3.3× margin — tighter than v5's
  20× because Alakazam turns are expensive), max single move 5.2 s, **0 timeouts**.
- bank=60 (stress): governor throttle floored min bank at **18.4 s** (≈ 20 s LOW_CLOCK instant threshold),
  **0 timeouts** — the proportional throttle (≤10% of remaining) provably prevents overrun.

**v6 is ready to submit.** Bundle: `submission_v6.tar.gz`. It is the first agent past the sample-deck
ceiling (#42). Post-submit: track its ladder μ vs v5's 553.9 (ground truth vs the real 1200-rated field);
run the control net_ladder2(sample) vs v5(sample) ≈ parity to attribute the gain to the deck.

## #44 — Flywheel gen-2: Alakazam self-play (real value) **beats v6 72.9%** — v7 candidate

**Date:** 2026-07-09 · v6 is live on the ladder at **681.4 (+120 over v5's 560.9)** — biggest gain of the
project, confirming #42 offline. To push further, ran the AlphaZero flywheel on Alakazam (the lever
identified: BC gives policy but not value; self-play from a now-competent pilot gives real value).

**Setup.** (1) **Grew the corpus:** re-ran the resumable harvester (rate-limit window reset) → **201 →
1,834 replays**; rebuilt ladder-BC = **73,001 rows** (winner ratings up to 1305). Fixed a crash on
null-reward (unfinished) replays in `_winner_seat` + hardened `build_ladder_bc`. (2) **Self-play:**
`record_mixed --deck alakazam --net-only --onnx-dir ladder2` (fixed `MctsPilot` to forward `set_deck`
so the net self-plays deck-conditioned) → **52,118 Alakazam rows with REAL value z** (from outcomes, not
BC's fake all-+1). (3) **Train** `net_ladder3.pt` on alakazam_sp (--buffer, real value, value head trains
purely on this) + ladder_bc2 (--bc-buffer, --mask-bc-value, policy only). Deck-conditioning keeps the
non-Alakazam BC from corrupting Alakazam play.

**Gate (equal wall-clock, ONNX both, both deck-conditioned — fixed the gate to `set_deck` BOTH guides):**
```
  net_ladder3 (flywheel) vs net_ladder2 (v6), ALAKAZAM mirror, 96 games:  70/26 = 72.9% ±4.5%  z=+5.05
```
**The flywheel is a decisive win.** Self-play real-value targets improved the Alakazam pilot far beyond
v6 — closing the value gap BC couldn't (v6's value head only generalized to Alakazam from sample; v7's
trained on 52k real Alakazam outcomes). Since v6 already beats sample+v5 72.9%, v7 is transitively well
ahead of the original ship.

**→ v7 CANDIDATE = Alakazam deck + net_ladder3.** Next: build the v7 bundle (as v6, ONNX from `ladder3`)
+ timeout-validate + submit → track ladder μ vs v6's 681.4. The flywheel can iterate again (net_ladder3
self-play → gen-3). Harvest infra now yields ~1.8k replays/window — corpus grows cheaply.

## #45 — v7 submission bundle: flywheel Alakazam agent, built + timeout-validated (`submissions/v7/`)

**Date:** 2026-07-09 · Ships the #44 flywheel result (net_ladder3 beats v6's net 72.9% on Alakazam).

**Bundle.** `submissions/v7/` = identical pipeline to v6 (deck-conditioned agent, Alakazam deck.csv) but
the shipped net is **net_ladder3** (the flywheel net with a value head trained on real Alakazam self-play
outcomes). `build_submission.py` `_ONNX_SUBDIR["v7"]="ladder3"` (v6's ladder2 + v5's checkpoints untouched;
verified v7 value.onnx md5 ≠ v6's). Tarball `submission_v7.tar.gz` **5.5 MB**, torch-free, top-level.

**Validated (`scripts/timing_v7.py`, torch poisoned):** torch-free ✓, net loads ✓, deck-conditioned on
Alakazam ✓, all DONE, 0 illegal. **Timeout-safe:**
- bank=600: 4/4 DONE, worst seat-time **145.5 s / 600 s** (4.1× margin, a touch better than v6's 200 s),
  max move 5.2 s, **0 timeouts**.
- bank=60 stress: throttle floored min bank at **18.2 s**, **0 timeouts**.

**v7 ready to submit** (`submission_v7.tar.gz`). Expected to beat v6's live 681.4 (v7 beats v6 72.9%
offline). Post-submit: track μ; the flywheel can iterate again (net_ladder3 self-play → gen-3).

## #46 — Opponent model: re-mined archetype templates from the real ladder (stale TrainerHill → live meta)

**Date:** 2026-07-09 · v7 is live (COMPLETE) but converging *below* v6 (650.6 vs 678.8) — though on only
44 games vs v6's 66, and with a **higher raw ladder win rate (55.8% vs 49.2%)**, i.e. a still-climbing
provisional rating, not a settled deficit. While it converges, attacked an orthogonal quality lever:
**opponent modeling.**

**The bug.** The opponent model (`opponent.py` naive-Bayes classifier + `determinizer.py` hidden-zone
sampling) runs on `archetype_templates.json`, which was generated offline from a **TrainerHill scrape**
(`build_priors.py`). Its 8 archetypes barely overlap the actual simulator ladder: only *Alakazam* is real;
the rest (three Dragapult variants @ 11%/11%/6% priors, Zoroark, Greninja, Honchkrow, Slowking) are ~1% or
absent on-ladder, while the ladder's #2–#4 decks (Mega Kangaskhan ex, Cynthia's Garchomp ex, Grimmsnarl ex)
had **no template at all**. So for ~64% of opponents the determinizer sampled a *phantom* deck, and every
opponent reply the 2-ply MCTS simulates was played from the wrong cards. (Correction to prior belief: the
determinizer does **not** fill with our own decklist except as a last-resort fallback — it uses the MAP
archetype template. The problem was the *templates*, not the mechanism.)

**Fix — mine templates from the harvest** (`scripts/mine_archetypes.py`). Recover both agents' 60-card
decklists from all **1,834 replays** → **3,668 sightings**, cluster by marquee Pokémon, build each
archetype template = per-card *expected copies* across its sightings, rounded to exactly 60 by
largest-remainder (**no filler-energy padding** — kills that secondary weakness too). `prior` = real ladder
sighting share. Key refinement: **demote generic tech** (Fezandipiti ex, Munkidori, Dudunsparce, Lumineon,
Radiant, Squawkabilly) from the marquee rule — else a Fezandipiti-ex *splash* forms a 35-distinct-list
grab-bag cluster (19.4%) that swallows real engines. Demoting it re-labels those decks by their true engine:
**Alakazam 15.7% → 32.6%** (matching the earlier hand-verified 36%), and the phantom "Fezandipiti archetype"
vanishes.

**Result — 12 real-ladder archetypes covering 93.7% of sightings**, replacing 8 mostly-phantom ones:
Alakazam 32.6%, Mega Kangaskhan ex 15.2%, Marnie's Grimmsnarl ex 12.8%, Mega Lucario 5.2%, Cynthia's
Garchomp ex 5.2%, Archaludon ex 5.0%, Cornerstone Ogerpon 3.7%, Dragapult ex 3.5%, Mega Abomasnow 2.8%,
Togekiss 2.7%, Brambleghast 2.6%, Cubchoo 2.5%. Old TrainerHill file preserved as
`archetype_templates.trainerhill.json`.

**Validated (`scripts/validate_archetypes.py`):** all 12 templates sum to 60, 0 invalid card ids, priors>0;
classifier **top-1 recovers a deck's own archetype at 98.9% full-reveal / 98.7% Pokémon-only** (realistic
in-game evidence) over 1,495 decks — only real semantic overlap (Cornerstone↔Lucario, both Fighting Mega)
misfires; determinize() smoke samples a full 60-card opponent world cleanly. (Separability is in-sample —
a necessary check, not a generalization claim.)

**Not yet shipped** (at time of #46). Offline data/quality change; nothing live affected. `build_submission.py`
re-vendors `src/ptcg` so **v8** picks up the new templates automatically. Gate before ship (→ #47).

## #47 — Gate the re-mined templates → **v8 SHIPS** (Alakazam + net_ladder3 + live-meta opponent templates)

**Date:** 2026-07-09 · Gated #46's templates two ways (heuristic MCTS pilot both sides — no net → no OOD
confound; the change is isolated to which template set the determinizer models the opponent with).

**Mirror gate (`scripts/gate_templates.py`) — NEW vs OLD templates, same deck both seats.** sample (Mega
Abomasnow, which OLD lacks): **54.7% z=+1.15**; Alakazam (both sets have an Alakazam): **53.3% z=+0.73**.
Both lean positive, neither significant. **Lesson: the mirror gate is the wrong instrument** — a mirror never
makes the agent FACE a deck the old set can't represent, so it under-tests the phantom-deck fix.

**Cross-deck gate (`scripts/gate_cross_templates.py`) — the right instrument.** "Us" = Alakazam pilot; the
OPPONENT plays a real ladder deck the OLD set has NO template for (modal lists mined via
`scripts/extract_decks.py`). Vary only OUR determinizer (NEW vs OLD), opponent model fixed, our seat
alternates, conditions interleaved so deck-matchup bias cancels. 200 games/opponent:
```
  vs Cynthia's Garchomp ex : NEW 57.0% / OLD 44.0%  Δ+13.0%  z=+1.84  NEW BETTER
  vs Marnie's Grimmsnarl ex: NEW 30.0% / OLD 26.0%  Δ +4.0%  z=+0.63
  vs Mega Kangaskhan ex    : NEW 52.0% / OLD 50.0%  Δ +2.0%  z=+0.28
  POOLED (300 v 300)       : NEW 46.3% / OLD 40.0%  Δ +6.3%  z=+1.57
```
**All three positive, one significant, ZERO regressions.** Pooled z=+1.57 is a hair under strict 1.64, but
the signal is consistent + directional and the change is **costless** (same net, same deck, only more-accurate
opponent reference decks). This is with the heuristic pilot facing just 3 of the field; the shipped net-MCTS
uses the same determinizer and the live field is more diverse, so this likely **understates** the ladder
effect. **Verdict: PASS → ship v8.**

**v8 bundle.** = v7 byte-for-byte (same `agent.py`, same Alakazam `deck.csv`, **same net_ladder3 ONNX** —
verified value.onnx md5 identical to v7) EXCEPT the vendored `ptcg/archetype_templates.json` is the #46
live-meta set (12 archetypes, verified vendored). `build_submission.py _ONNX_SUBDIR["v8"]="ladder3"`. Tarball
`submission_v8.tar.gz` **5.4 MB**, torch-free, main.py+deck.csv top-level. **Timeout-validated
(`scripts/timing_v8.py`, torch poisoned):** net loaded ✓, deck-conditioned ✓; bank=600 6/6 DONE worst
seat-time **230.8 s/600** (2.6× margin, a bit above v7's 145 s — the 12-archetype classifier costs slightly
more — but 0 timeouts); bank=60 stress floored at 18.6 s, 0 timeouts; 0 illegal across 2,493 moves.

**v8 READY TO SUBMIT.** It's an A/B on v7: identical pilot+net, only sharper opponent modeling. Post-submit,
compare v8 μ to v7's (isolates the template value on the live ladder). Follow-ups still open: per-world
archetype **sampling across the posterior** (currently MAP-argmax collapse); extend `threat.py` beyond the
opponent's active (no bench/switch/ability lines); flywheel gen-3.

## #48 — "net_ladder3 regressed live" — DIAGNOSIS: premise REFUTED, no net regression

**Date:** 2026-07-10 · v6/v7/v8 live: v6 **683.7** (67g, raw WR 50.0%), v7 **638.9** (58g, 49.1%),
v8 **647.1** (48g, **53.2%**, still climbing). The apparent puzzle: v7's net_ladder3 beat v6's net_ladder2
72.9% offline (#44) yet v7's *rating* sits ~45 pts below v6. Hypothesis: net_ladder3 overfit the Alakazam
MIRROR (its only self-play matchup + its only gate) and is worse vs the ~2/3 non-Alakazam field.

**Test (`scripts/gate_net_field.py`):** our Alakazam pilot swaps net_ladder3 ↔ net_ladder2, FACING a real
field deck (heuristic-fixed), templates held constant, seats alternate, conditions interleaved. N=140/opp.
```
  opp Alakazam (control): L3 70.0% / L2 45.7%  Δ+24.3%  z=+2.91
  opp Kangaskhan        : L3 45.7% / L2 30.0%  Δ+15.7%  z=+1.92
  opp Garchomp          : L3 58.6% / L2 45.7%  Δ+12.9%  z=+1.52
  opp Grimmsnarl        : L3 68.6% / L2 68.6%  Δ +0.0%
  POOLED (280 v 280)    : L3 60.7% / L2 47.5%  Δ+13.2%  (sig)
```
**HYPOTHESIS REFUTED.** net_ladder3 ≥ net_ladder2 in ALL 4 configs (mirror AND field) — better by +13–16%
or tied, never worse. It is NOT mirror-overfit; it's the stronger Alakazam pilot, field included.

**So there is no reproducible net regression — the premise was wrong.** The live raw win rates tell the real
story: v6 50.0% vs v7 49.1% are ~identical; the 45-pt RATING gap is matchmaking/convergence path-dependence
(each agent sits near ~50% equilibrium against whatever pool it was matched into; v7 entered lower and, at
49%, hasn't climbed to net_ladder3's true level). Clincher: **v8 (same net_ladder3 + new templates) has the
HIGHEST raw live WR (53.2%) and is climbing** — impossible if ladder3 were broken.

**Conclusion / actions.** (1) Do NOT "fix" net_ladder3 — it's our best net. (2) The v6>v7 live gap is rating
noise, not skill. (3) v8 is the right bet (best net + validated-better templates, #47); let it converge —
winning >50% vs its pool, its μ should rise past v7 toward v6. (4) **Methodology lesson: raw win rate, not
just the rating number, is the skill signal for a young submission** — a fresh agent's rating is path-dependent
and lags its true skill. **Caveat:** field opponents here were heuristic-piloted, not the strong live field;
mirror+heuristic-field consistency (4/4) mitigates but doesn't fully close it — revisit with a strong fixed
opponent only if v8 stalls below v6 after full convergence. New tool: `scripts/gate_net_field.py`.

## #49 — Per-world posterior sampling in the determinizer → PASS (search enrichment; feeds v9)

**Date:** 2026-07-10 · With the #46 templates giving a real 12-archetype posterior, the determinizer was
throwing it away: `build_opponent_belief` took the MAP **argmax** archetype for the zone model, so ALL K
determinized worlds modeled the opponent as *certainly* their single most-likely deck — zero hedging of
opponent-deck uncertainty. Fix (this is a SEARCH enrichment — the lever that pays; not a leaf-eval one):
each world SAMPLES its opponent archetype from the posterior.

**Change (minimal, toggle-guarded).** `build_opponent_belief(..., rng=None)`: when given an rng, draw the
zone-model archetype ~ posterior instead of argmax. `determinize(..., sample_archetype=True)` threads its
per-world rng through (`_split_opponent(sample_arch=)`), default ON, `False` = old MAP-collapse for A/B +
rollback. Backward-compatible: rng=None keeps deterministic MAP (classification/tests/single-belief callers).

**Verified (`scripts/verify_posterior_sampling.py`):** at an early, thin-evidence decision (posterior:
alakazam 0.35 / kangaskhan 0.16 / grimmsnarl 0.14 / …), MAP-collapse → all 24 worlds `alakazam`; sampling →
**8 distinct archetypes over 24 worlds** ≈ the posterior. Collapses to MAP automatically once the posterior
peaks (evidence accrues). Exactly the intended hedge-when-uncertain / commit-when-sure behaviour.

**Gate (`scripts/gate_posterior.py`, heuristic pilot both sides — isolates the search change; our Alakazam
determinizer varies SMP vs MAP, opponent fixed MAP, seats alternate, N=240/opp):**
```
  vs Mega Kangaskhan ex : SMP 60.8% / MAP 42.5%  Δ+18.3%  z=+2.84  SMP BETTER
  vs Alakazam (mirror)  : SMP 53.3% / MAP 46.7%  Δ +6.7%  z=+1.03
  vs Cynthia's Garchomp : SMP 57.5% / MAP 57.5%  Δ +0.0%
  vs Marnie's Grimmsnarl: SMP 38.3% / MAP 40.0%  Δ −1.7%  z=−0.26 (noise)
  POOLED (480 v 480)    : SMP 52.5% / MAP 46.7%  Δ +5.8%  z=+1.81  (significant)
```
**PASS.** Pooled significant, one big significant win (Kangaskhan), no real regression. Kept ON by default.
Costless (same net/deck/templates) → folds into a **v9 = v8 + posterior sampling** (build re-vendors
`src/search`). Recommended sequencing: let v8's rating converge first so v9-vs-v8 cleanly isolates the
sampling increment on the live ladder. Next open lever after this: extend `threat.py` past the opponent's
active (bench/switch/ability) — but that's a leaf-eval enrichment (historically weaker payoff) and matters
less to the net-piloted ship.

## #50 — v9 bundle: v8 + per-world posterior sampling, built + timeout-validated (`submissions/v9/`)

**Date:** 2026-07-10 · Ships #49's determinizer change (posterior sampling PASSED, pooled +5.8% z=+1.81).
v9 = **v8 byte-for-byte** (same net_ladder3 ONNX — value.onnx md5 identical to v8; same Alakazam deck; same
mined templates — templates file identical to v8) EXCEPT the vendored `search/determinizer.py` (+`ptcg/
opponent.py`) now SAMPLE each world's opponent archetype from the posterior (`sample_archetype=True`) instead
of MAP-collapse. `build_submission.py _ONNX_SUBDIR["v9"]="ladder3"`. Tarball `submission_v9.tar.gz` **5.4 MB**,
torch-free, main.py+deck.csv top-level. **Timeout-validated (`scripts/timing_v9.py`, torch poisoned):** net
loaded ✓, deck-conditioned ✓; bank=600 6/6 DONE worst **227.5 s/600** (≈v8's 230.8 — sampling adds no clock
cost), 0 timeouts; bank=60 floored 18.7 s, 0 timeouts; 0 illegal / 2,369 moves.

**v9 READY TO SUBMIT** — a clean A/B on v8 (only the determinizer differs). **SEQUENCING: hold until v8's
rating converges**, then submit v9 so v9-vs-v8 isolates the posterior-sampling increment live (avoids two
provisional submissions muddying the read). Stack so far vs the shipped v7: v8 = +live-meta templates
(#47, live-leading raw WR), v9 = +posterior sampling (#49). Both offline-validated search enrichments.

## #51 — v10 bundle: v6's net_ladder2 + mined templates + posterior sampling, built + validated (`submissions/v10/`)

**Date:** 2026-07-10 · v8 converged to **642.3** (+6.6 over v7's 635.7, raw WR 52.0% vs 48.3% — templates
confirmed a small live positive, matching the offline gate). v9 (v8+sampling) submitted, resolving. But the
whole net_ladder3 line (v7→v9 ≈ 636–642) sits ~40 below **v6's net_ladder2 (683.7)** — a matchmaking/
path-dependence gap (#48), not net skill. **v10 tests the other combination: stack the two GATED search
wins onto the net that is actually winning live.**

**v10 = net_ladder2 (v6's net) + mined templates (#46/#47) + per-world posterior sampling (#49).** Verified:
value.onnx md5 == v6's (net_ladder2); templates == the mined set (== v8/v9); determinizer samples per-world
(sample_archetype default True); net md5 ≠ v9 (ladder2 vs ladder3). Agent/deck/main identical to v6/v9 (the
v6↔v9 agent.py diff is docstring-only). `build_submission.py _ONNX_SUBDIR["v10"]="ladder2"`. Tarball
`submission_v10.tar.gz` **5.4 MB**, torch-free, top-level. **Timeout-validated (`scripts/timing_v10.py`):**
net loaded ✓, deck-conditioned ✓; bank=600 6/6 DONE worst **155.2 s/600** (3.9× margin, better than v8/v9's
~228 — net_ladder2 turns run a touch shorter), 0 timeouts; bank=60 floored 18.1 s, 0 timeouts; 0 illegal.

**v10 READY.** Both component changes are individually gate-passed (templates +6.3% cross-deck; sampling
+5.8% pooled), but on the HEURISTIC pilot — not yet net-specifically A/B'd on net_ladder2. Caveat: the
determinizer change is pilot-agnostic (it changes the searched worlds, not the prior/value source), so
transfer is expected. Submission plan: v10 vs v6's 683.7 isolates whether the two search wins lift the
higher-rated live net past its ceiling. Optional pre-ship: net-specific gate v10 vs v6 (net_ladder2 both,
new vs old determinizer) for offline confirmation.

## #52 — v10 pre-ship gate: determinizer wins attenuate under the net pilot (lean-positive, not sig)

**Date:** 2026-07-10 · Gated v10's determinizer (mined templates #46 + posterior sampling #49) vs v6's
(TrainerHill + MAP), NET held fixed at net_ladder2 both sides.

**Mirror (`gate_v10_vs_v6.py`, Alakazam both, 240 games): 120/120 = 50.0% z=+0.00** — dead even. But a
mirror under-tests the determinizer change (both sets have an Alakazam; opponent IS Alakazam so MAP is
usually right) — the same wrong-instrument lesson as the #47 mirror.

**Field (`gate_cross_v10.py`, net_ladder2 pilots Alakazam vs a heuristic field deck, our determinizer V10 vs
V6, N=160/opp):**
```
  vs Cynthia's Garchomp : V10 50.0% / V6 41.2%  Δ+8.8%  z=+1.11
  vs Mega Kangaskhan    : V10 45.0% / V6 37.5%  Δ+7.5%  z=+0.96
  vs Marnie's Grimmsnarl: V10 68.8% / V6 70.0%  Δ−1.2%  z=−0.17
  POOLED (240 v 240)    : V10 54.6% / V6 49.6%  Δ+5.0%  z=+1.10  (lean +, NOT sig)
```
**KEY FINDING: the determinizer wins ATTENUATE under the net pilot.** Templates +6.3% and sampling +5.8%
each on the HEURISTIC pilot (vs field); combined on the NET pilot they're only +5.0% (vs field) and **0% in
the mirror**. The net leans on its learned priors/values, so it's partly robust to opponent-deck-model
quality — better determinization helps it less than it helps the heuristic. This also explains v8's modest
+6.6 live over v7 (real but small). No regression anywhere; the change stays costless and directionally +.

**Call: v10 is a lean-positive, no-regression, costless variant (same net_ladder2 as v6, timeout-safe #51).**
Its real value is the LIVE experiment: does net_ladder2 (the higher-rated live net, 683.7) + the two search
wins beat v6's ceiling? Offline says +5% vs field but not significant, so the ladder is the arbiter. Ship to
find out (submissions cheap in July), OR hold if unwilling to spend a slot on a sub-significant offline lean.
New tools: `scripts/gate_v10_vs_v6.py`, `scripts/gate_cross_v10.py`.

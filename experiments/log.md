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

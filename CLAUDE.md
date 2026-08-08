# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A solo entry for the **Kaggle "Pokémon TCG AI Battle" competition** — build an agent that
plays the Pokémon Trading Card Game inside the organizers' C++ battle engine (the "cabt"/
`ptcgProgram` simulator). The full strategy and build plan lives in
`docs/ptcg_agent_strategy.md`; the blow-by-blow experiment record is `experiments/log.md`
(entries #1–#32). **Read both if you need history** — they are the source of truth for
architecture, decisions, and what has and hasn't worked.

The repo is well past scaffolding: a full agent stack exists and **v6 (a deck-conditioned,
net-guided MCTS agent) is the current ship, submitted to Kaggle** (best submission, ~683.7 on
the ladder; v5 is the previous sample-deck ship). See "Where we are now" below.

Key dates: simulation entry closes **Aug 9 2026**, agent submission **Aug 16 2026**, strategy
writeup **Sep 13 2026**.

## Where we are now (2026-07)

The agent arc, in submission order (`submissions/vN/`, each a self-contained bundle):
- **v0** — legal-random baseline. **Frozen; do not modify.**
- **v1** — evaluator-guided turn-aware search. **Submitted to Kaggle** (first non-random agent
  on the ladder). Still the baseline the round-robin harness pilots with.
- **v2** — opponent-reply lookahead. **Negative result** (coin-flip vs v1); kept as infra, not shipped.
- **v3** — beam/best-first turn search. Strict pilot upgrade but ~no gain on the simple deck we
  ship, so not submitted.
- **v4** — determinized PUCT **MCTS with a heuristic leaf** (no networks). Strongest *heuristic* pilot.
- **v5** — **previous ship (superseded by v6).** v4's MCTS but priors + leaf value come from a
  trained net served via **ONNX-CPU** (`OnnxGuide`, onnxruntime, single-threaded, torch-free).
  Gracefully degrades to heuristic MCTS if the net/onnxruntime is missing (net can never cause a
  crash/timeout loss).
- **v6** — **CURRENT SHIP.** Same pipeline as v5 but the shipped net is the deck-conditioned
  `net_ladder2`, piloting the ladder Alakazam deck (#43). Best submission to date (~683.7 on the
  ladder). Same graceful degradation to heuristic MCTS as v5.

**The ML pipeline is built and validated end-to-end** (AlphaZero-style): self-play recorder
(`src/selfplay/`) → buffer → trainer (`scripts/train_net.py`) → `PTCGNet` (`src/nn/encoder.py`,
~497k params) → wired into MCTS as prior+value (`src/nn/guide.py`) → ONNX export
(`scripts/export_onnx.py`) → deploy (`src/nn/onnx_guide.py`). **Acceptance gate PASSED:**
net-MCTS beats heuristic-MCTS **70.5% at equal sims / 58.5% at equal wall-clock** (0.4 s/move).

**Two arcs are settled — don't relitigate without new evidence:**
- **Deck construction is CLOSED.** No auto-generated meta deck beats the SDK **`sample` deck**,
  confirmed even under the net pilot (#32: Dragapult 19.7%, sample ~80%). **We ship the sample
  deck** (identical across v1/v4/v5). The lever is the *pilot*, not the deck. Caveat noted in #32:
  the net is trained on sample self-play only, so it is *out-of-distribution* piloting other decks —
  a known open diagnosis, not a proven deck-strength result.
- **Enriching the leaf eval failed** (reply-lookahead #10, prize-race #18); enriching the *search*
  (beam→MCTS→net) is what paid. Gate expensive pilot A/Bs behind the cheap `Pearson(eval,win)` test first.

**What's next:** flywheel gen #2 (self-play from v5 → retrain → re-gate); tighten the 58.5%
equal-wall-clock number at larger N; diagnose the net's OOD play on non-sample decks. GPU only
worth it past ~10k+ games/gen (self-play on the CPU-only C++ engine is the real bottleneck, not training).

## Layout

- `docs/ptcg_agent_strategy.md` — architecture, layer stack, deck plan, judging, milestones.
- `experiments/log.md` — the numbered experiment record (#1–#32); read for *why* things are the way they are.
- `src/` — the agent brain (not shipped directly; vendored into submissions by `build_submission.py`):
  - `ptcg/` — belief→decision stack: `perception`, `belief`, `opponent`, `threat`, `evaluator`,
    `prize`, `archetypes`. Each has a `scripts/*_smoke.py` test.
  - `search/` — `determinizer` (K belief-sampled worlds), `mcts` (determinized PUCT, injectable
    prior_fn/value_fn), `governor` (clock budget + critical-turn triggers).
  - `nn/` — `features` (encode_state/encode_actions), `encoder` (`PTCGNet`), `guide` (`NetGuide`,
    torch), `onnx_guide` (`OnnxGuide`, onnxruntime — the *deployed* path), `dataset`.
  - `selfplay/recorder.py` — self-play → training tuples. `priors/` + `deck_lab/` — offline meta
    priors + deck construction (see the deck arc above).
- `submissions/v0..v5/` — self-contained agent bundles (see "Where we are now"). Build/refresh one
  with `.venv/bin/python scripts/build_submission.py vN` (vendors `cg/`+`ptcg/`+`search/`+nn subset).
- `scripts/` — experiment drivers: `train_net.py`, `record_selfplay.py`, `export_onnx.py`,
  gates (`gate_net_vs_heuristic.py`, `gate_deck_ab.py`), validators (`validate_v5.py`, `timing_v5.py`), smokes.
- `experiments/checkpoints/` — trained net (`net.pt`) + `value.onnx`/`policy.onnx`. **Gitignored**
  except the ONNX files are force-added (`git add -f`) so v5 is reproducible.
- `data/sample_submission/sample_submission/` — the reference submission + Python SDK (`main.py`,
  `deck.csv`, `cg/`). `data/ptcg_engine/ptcgProgram 22/` — vendored **C++ engine source** (reference
  only, not built here). `data/cards/` — card DB CSVs + PDF ID lists. **All of `data/` is gitignored.**
- Memory lives at `<project>/memory/` (see `MEMORY.md`); this is also an Obsidian vault (`.obsidian/`).

## The SDK (`cg/` package)

Thin `ctypes` wrapper over a compiled C library — do not modify the vendored engine/SDK.

- `cg/sim.py` — loads the platform-specific shared lib (`cg.dll` / `libcg*.so` / `libcg.dylib`)
  and declares all `ctypes` signatures. Holds the global `Battle` state.
- `cg/api.py` — the public API. All `Observation` dataclasses + enums, plus:
  - `to_observation_class(dict) -> Observation` — parse the raw obs dict.
  - `all_card_data()`, `all_attack()` — full card/attack DB from the engine.
  - `search_begin` / `search_step` / `search_end` / `search_release` — the **native lookahead
    Search API** for critical-turn planning (belief-sampled determinization).
- `cg/game.py` — driver: `battle_start` / `battle_select` / `battle_finish` / `visualize_data`.
- `cg/utils.py` — recursive dict→dataclass conversion.

## Agent contract (critical)

`agent(obs_dict) -> list[int]` (see `main.py`):
- The engine presents **only legal moves** in `obs.select.option`.
- Return a list of **option indices**, each `0 <= i < len(obs.select.option)`, length between
  `obs.select.minCount` and `obs.select.maxCount` inclusive, **no duplicates**.
- When `obs.select is None` (initial deck selection), return the 60-card deck from `deck.csv`.
- Interface is **multi-select** (mulligans, prize picks, discards select option *sets*).
- **Crash or timeout = instant loss** (10 min/player total clock). The outermost layer of any
  real agent must be try/except + watchdog returning a legal fallback action.

## Environment / running

- **Python** (3.10+ syntax: `list[int]`, `X | None`). Use the repo venv: `.venv/bin/python`.
  The SDK itself is prebuilt native libs (stdlib-only: `ctypes`, `json`), loaded at import.
- **Deps split by phase.** The *shipped* agent needs only **onnxruntime** (ONNX-on-CPU is the
  deploy path — the Kaggle runtime is **2 vCPUs, 12.2 GiB RAM, no network**). *Training-only* deps
  (`torch`, `onnx`, `onnxscript`, numpy) live in `requirements-train.txt` and are **never shipped**;
  `build_submission.py` vendors a torch-free nn subset. **Ship single-threaded**
  (`torch.set_num_threads(1)`; oversubscription tanks throughput and it's ~8× faster/game for this 2MB net).
- Submission bundle: `.tar.gz` ≤ 197.7 MiB with `main.py` + `deck.csv` at top level; files land
  in `/kaggle_simulations/agent/` (note the path fallback in `main.py`/`read_deck_csv`). v5 is ~5.3 MB.
  Build: `.venv/bin/python scripts/build_submission.py v5 && (cd submissions/v5 && tar -czf ../../submission_v5.tar.gz *)`.
- **Never commit native libs** (`cg.dll`, `libcg*.so`, `libcg*.dylib`) — gitignored. GitHub auth is
  SSH, not `gh` (see the `dev-environment-setup` memory).

## Rules & IP constraints (do not violate)

- The engine (`ptcgProgram 22/`) and card data are **licensed for competition use only** — not
  open source. Do not share, republish, or commit card images. The winning repo must be clean of
  card images and Card Data CSVs (deleted at competition end).
- Prefer building agent code under `src/` / `submissions/` rather than editing the vendored
  `data/sample_submission/` reference in place.

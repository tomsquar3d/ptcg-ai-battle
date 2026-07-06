# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A solo entry for the **Kaggle "Pokémon TCG AI Battle" competition** — build an agent that
plays the Pokémon Trading Card Game inside the organizers' C++ battle engine (the "cabt"/
`ptcgProgram` simulator). The full strategy and build plan lives in
`docs/ptcg_agent_strategy.md` — **read it first**; it is the source of truth for architecture,
deadlines, and decisions.

The repo is currently at the **planning + scaffolding stage**: `src/`, `experiments/`, and
`submissions/` are empty. Only the vendored SDK, card data, and the strategy doc exist so far.
This is also an Obsidian vault (`.obsidian/`).

Key dates: simulation entry closes **Aug 9 2026**, agent submission **Aug 16 2026**, strategy
writeup **Sep 13 2026**.

## Layout

- `docs/ptcg_agent_strategy.md` — architecture, layer stack, deck plan, judging, milestones.
- `data/sample_submission/sample_submission/` — the reference submission and Python SDK:
  - `main.py` — the entry point. Implements `agent(obs_dict) -> list[int]`.
  - `deck.csv` — 60 card IDs, one per line (the deck).
  - `cg/` — the SDK package (see below).
- `data/ptcg_engine/ptcgProgram 22/` — vendored **C++ engine source** (header-only, C++20,
  VS2022 solution). Reference only for understanding rules/effects; not built here.
- `data/cards/` — `EN_Card_Data.csv` / `JP_Card_Data.csv` (card DB) + PDF ID lists.
- `src/`, `experiments/`, `submissions/` — empty; where new code/agents/experiments go.

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

- **Python** (3.10+ syntax: `list[int]`, `X | None`). No `package.json`, no build system —
  the SDK is prebuilt native libs loaded at import.
- No dependency manifest yet; the SDK uses only the stdlib (`ctypes`, `json`). If you add
  training/inference deps, note that the Kaggle runtime is **2 vCPUs, 12.2 GiB RAM, no network**;
  ONNX-on-CPU is the intended deployment path.
- Submission bundle: `.tar.gz` ≤ 197.7 MiB with `main.py` + `deck.csv` at top level; files land
  in `/kaggle_simulations/agent/` (note the path fallback in `main.py`/`read_deck_csv`).

## Rules & IP constraints (do not violate)

- The engine (`ptcgProgram 22/`) and card data are **licensed for competition use only** — not
  open source. Do not share, republish, or commit card images. The winning repo must be clean of
  card images and Card Data CSVs (deleted at competition end).
- Prefer building agent code under `src/` / `submissions/` rather than editing the vendored
  `data/sample_submission/` reference in place.

# ptcg-ai-battle
Overview

A game-playing agent for the Pokémon TCG, a hard planning problem because of hidden information (opponent’s hand, deck order) and randomness.

Approach

Search: determinized PUCT Monte Carlo Tree Search. It samples plausible hidden states and searches over them, with a hook for neural-network move priors and leaf-value estimates.
Learning: an AlphaZero-style loop: self-play data collection → replay buffer → policy/value network training in PyTorch.
Deployment: the trained network is exported to ONNX and run with onnxruntime on CPU, so the submission runs without PyTorch under Kaggle’s 2-vCPU, 10-minute-per-player limits.
Robustness: every move goes through a try/except plus a watchdog that falls back to a legal move, since a crash or timeout is an instant loss.

Where to look: src/ (search, belief engine, policy), submissions/vN/ (final agent), experiments/log.md (experiment journal).

Results: [final leaderboard rank, or win rate vs. the v0 random baseline over N self-play games]
Solo entry for the Kaggle **Pokémon TCG AI Battle Challenge** (Simulation + Strategy
categories) — an agent that plays the Pokémon Trading Card Game inside the organizers'
"cabt" battle engine.

The full architecture, layer stack, deck plan, and milestones live in
[`docs/ptcg_agent_strategy.md`](docs/ptcg_agent_strategy.md) — that document is the source of
truth. See [`CLAUDE.md`](CLAUDE.md) for a quick orientation and the agent contract.

## Repository layout

| Path | Purpose |
|---|---|
| `docs/` | Strategy, design notes, meta analysis. |
| `submissions/vN/` | Versioned, submittable agents. Each holds `main.py` + `deck.csv` + `cg/`. |
| `submissions/v0/` | **Frozen** legal-random baseline (do not modify). |
| `src/` | Shared agent code (Belief Engine, policy, heuristics) reused across submissions. |
| `experiments/` | Experiment log + one-off analysis scripts. `log.md` is the running journal. |
| `scripts/` | Dev/test utilities (e.g. `selfplay_test.py`). |
| `data/` | Competition card CSVs, engine source, replays — **git-ignored, never committed** (license requires deletion after the competition). |

## Setup

```bash
python3 -m venv .venv           # if ensurepip is missing, bootstrap pip via get-pip.py
.venv/bin/pip install kaggle-environments kaggle
```

## Running a self-play smoke test

```bash
.venv/bin/python scripts/selfplay_test.py
```

Runs 20 games of `submissions/v0/main.py` against itself in the `cabt` environment, asserts
every game finishes, and prints the win/loss/draw split and average seconds per game.

## Building a submission bundle

The archive must have `main.py` + `deck.csv` at the **top level** (not nested in a folder),
and be ≤ 197.7 MiB.

```bash
cd submissions/v0            # the version you want to ship
tar -czvf ../../submission.tar.gz *
```

`main.py` imports the `cg/` package, which loads the platform-native engine library and reads
`deck.csv`; both must be inside the same folder you archive from.

> **Native engine libraries are git-ignored.** `cg/cg.dll`, `cg/libcg*.so`, and `cg/libcg*.dylib`
> are the compiled competition engine — licensed, must not be republished, and excluded from git
> (see `.gitignore`). They must be present on disk for the agent to run or bundle. On a fresh
> clone, copy them from the official competition SDK into each `submissions/vN/cg/`.

## Hard constraints (never violate)

- **Crash = timeout = instant loss.** Total match clock is 10 min per player; the agent's
  outermost layer must be try/except + watchdog returning a legal fallback move.
- The engine presents **only legal moves**; `agent(obs) -> list[int]` returns option indices
  (`minCount..maxCount`, no duplicates). `select is None` → return the 60-card deck.
- Kaggle runtime: **2 vCPUs, 12.2 GiB RAM, no network** during episodes.
- Competition data and engine are licensed for competition use only — keep them out of git and
  delete at competition end.

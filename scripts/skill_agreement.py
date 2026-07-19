"""Skill-vs-net agreement pre-check (the cheap gate before any A/B).

On every single-select root in N games, compares:
  - net_top   = argmax of the reference policy's prior over the legal options
                (OnnxGuide.priors if the .onnx files + onnxruntime are present, else a heuristic
                 one-ply leaf-value argmax as a stand-in),
  - skill_top = argmax of skill_root_prior(..., lam=1.0)  (pure skill, base weight 0).

Reports the overall agreement rate and the intent histogram, and writes the full DISAGREEMENT set
to a CSV. Interpretation: if agreement ~ 100%, the skill prior is redundant with the net; the
disagreement rows are exactly where the A/B win-rate should be sliced, because that is the only
place the prior changes what MCTS explores first.

Run from repo root:  .venv/bin/python scripts/skill_agreement.py 30
Writes:               experiments/skill_agreement_disagreements.csv
Dev-only; not shipped.
"""
from __future__ import annotations

import csv
import os
import random
import sys
import time
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _cand in ("src", ".", "submissions/v7", "data/sample_submission/sample_submission"):
    # append (don't prepend): keep src/ authoritative so the src copies of search/ptcg/nn win over
    # any vendored bundle that lacks skill_prior; these are only a last-resort cg fallback.
    if os.path.exists(os.path.join(_ROOT, _cand, "cg", "api.py")) and \
            os.path.join(_ROOT, _cand) not in sys.path:
        sys.path.append(os.path.join(_ROOT, _cand))

from kaggle_environments import make                                             # noqa: E402
from cg.api import (to_observation_class, all_card_data, all_attack,             # noqa: E402
                    search_begin, search_step, search_end)
from ptcg.perception import build_game_state                                     # noqa: E402
from ptcg.opponent import OpponentTracker                                        # noqa: E402
from ptcg.archetypes import default_archetypes                                   # noqa: E402
from search.determinizer import determinize                                      # noqa: E402
from search import mcts                                                          # noqa: E402
from search.mcts import _Node, uniform_prior, _value as leaf_value               # noqa: E402
from search.skill_prior import skill_root_prior, turn_plan                       # noqa: E402

_MCTS_MOVE_S = 0.15   # per-move MCTS budget in --mcts play mode (enough to set up; keeps 30g bounded)

_DECK_PATHS = ("submissions/v7/deck.csv", "deck.csv", "submissions/v5/deck.csv")
_MODEL_DIRS = ("submissions/v7", "experiments/checkpoints", "submissions/v5")
_OUT = os.path.join(_ROOT, "experiments", "skill_agreement_disagreements.csv")


def _read_deck():
    # --deck <path> overrides the default (e.g. a flat-damage ex deck that fires the KO tree)
    for i, a in enumerate(sys.argv):
        if a == "--deck" and i + 1 < len(sys.argv):
            fp = sys.argv[i + 1]
            if not os.path.isabs(fp):
                fp = os.path.join(_ROOT, fp)
            return [int(x) for x in open(fp).read().split()[:60]]
    for p in _DECK_PATHS:
        fp = os.path.join(_ROOT, p)
        if os.path.exists(fp):
            with open(fp) as f:
                return [int(x) for x in f.read().split("\n")[:60]]
    raise FileNotFoundError(f"no deck.csv in {_DECK_PATHS}")


def _load_guide(ci, ai, deck):
    """Return (priors_fn, label, guide). Prefer the ONNX net; fall back to a heuristic reference.
    `guide` is the OnnxGuide object (has .value for MCTS play) or None for the heuristic ref."""
    for d in _MODEL_DIRS:
        md = os.path.join(_ROOT, d)
        if os.path.exists(os.path.join(md, "value.onnx")):
            try:
                from nn.onnx_guide import OnnxGuide
                g = OnnxGuide(md, ci, ai)
                try:
                    g.set_deck(deck, None)
                except Exception:
                    pass
                return g.priors, f"net:{d}", g
            except Exception as e:
                print(f"  (OnnxGuide unavailable: {e!r}; using heuristic reference)")
                break

    def heuristic_priors(node):                # one-ply leaf-value argmax as a stand-in policy
        rp = node.state.observation.current.yourIndex
        scores = {}
        for a in node.actions:
            pick = [] if a < 0 else [a]
            try:
                child = search_step(node.state.searchId, pick)
                scores[a] = leaf_value(child, rp, ci, ai)
            except Exception:
                scores[a] = -1.0
        best = max(scores, key=scores.get) if scores else (node.actions[0] if node.actions else 0)
        return {a: (1.0 if a == best else 0.0) for a in node.actions}
    return heuristic_priors, "heuristic", None


def _argmax(p, actions):
    return max(actions, key=lambda a: p.get(a, 0.0)) if actions else None


def main():
    mcts_mode = "--mcts" in sys.argv                    # both sides play net-guided MCTS (realistic)
    greedy = "--greedy" in sys.argv and not mcts_mode   # else one-ply-greedy; else random
    _pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    games = int(_pos[0]) if _pos else 30
    deck = _read_deck()
    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    archs = default_archetypes()
    ref_priors, ref_label, play_guide = _load_guide(ci, ai, deck)
    play = "mcts" if mcts_mode else ("greedy" if greedy else "random")

    stats = Counter(); intents = Counter()
    rows = []

    def _greedy_move(state, sel, rp):
        """One-ply argmax leaf-value legal pick from an already-open search `state`."""
        n = len(sel.option)
        if sel.maxCount == 1:
            opts = list(range(n)) + ([-1] if sel.minCount == 0 else [])   # -1 == pass
            best, bv = None, -1e30
            for a in opts:
                try:
                    v = leaf_value(search_step(state.searchId, [] if a < 0 else [a]), rp, ci, ai)
                except Exception:
                    continue
                if v > bv:
                    bv, best = v, a
            if best is None:
                return [] if sel.minCount == 0 else [0]
            return [] if best < 0 else [best]
        scored = []
        for i in range(n):
            try:
                scored.append((leaf_value(search_step(state.searchId, [i]), rp, ci, ai), i))
            except Exception:
                pass
        scored.sort(reverse=True)
        pick = sorted({i for _v, i in scored[:sel.maxCount]})
        if len(pick) < sel.minCount:
            pick = list(range(max(sel.minCount, 1)))
        return pick[:sel.maxCount]

    def _greedy_pick(o, tracker=None):
        """Legal greedy move for a full observation (own its own search session)."""
        s = o.select; n = len(s.option)
        if n == 0:
            return []
        try:
            worlds = determinize(o, deck, ci, k=1, archetypes=archs, opp_tracker=tracker)
            state = search_begin(o, *worlds[0])
        except Exception:
            return random.sample(range(n), min(s.maxCount, n))
        try:
            return _greedy_move(state, s, o.current.yourIndex)
        finally:
            search_end()

    def _mcts_pick(o, tracker=None):
        """Legal net-guided MCTS move (mirrors the agent's single-select path) — realistic play so
        the game reaches KO/snipe/gust roots. mcts.search owns its own search session."""
        s = o.select; n = len(s.option)
        if n == 0:
            return []
        if s.maxCount > 1:                       # multi-select root → greedy (as the real agent does)
            return _greedy_pick(o, tracker)
        try:
            worlds = determinize(o, deck, ci, k=1, archetypes=archs, opp_tracker=tracker)
        except Exception:
            return random.sample(range(n), min(s.maxCount, n))
        deadline = time.monotonic() + _MCTS_MOVE_S
        pf = play_guide.priors if play_guide is not None else uniform_prior
        vf = play_guide.value if play_guide is not None else None
        try:
            pick, _st = mcts.search(o, worlds, ci, ai, sims=200_000, deadline=deadline,
                                    prior_fn=pf, value_fn=vf)
            return pick if pick is not None else random.sample(range(n), min(s.maxCount, n))
        except Exception:
            return random.sample(range(n), min(s.maxCount, n))

    def probe(obs, tracker):
        """Probe net-vs-skill agreement at a single-select root; return a greedy move (or None)."""
        try:
            worlds = determinize(obs, deck, ci, k=1, archetypes=archs, opp_tracker=tracker)
            state = search_begin(obs, *worlds[0])
        except Exception:
            stats["skip"] += 1; return None
        try:
            root = _Node(state, obs.current.yourIndex, ci, ai)
            if root.terminal or not root.actions:
                stats["skip"] += 1; return None
            base = uniform_prior(root)
            net_top = _argmax(ref_priors(root), root.actions)
            skill_top = _argmax(skill_root_prior(obs, root, ci, ai, base_prior=base, lam=1.0),
                                root.actions)
            try:
                gs = build_game_state(obs, ci)
                intent = getattr(turn_plan(gs, ci, ai), "kind", "?")
                mine = getattr(gs.me, "prize_count", -1); theirs = getattr(gs.opponent, "prize_count", -1)
            except Exception:
                intent, mine, theirs = "?", -1, -1
            intents[intent] += 1
            stats["roots"] += 1
            if net_top == skill_top:
                stats["agree"] += 1
            else:
                stats["disagree"] += 1
                rows.append({"root": stats["roots"], "my_prizes": mine, "opp_prizes": theirs,
                             "n_options": len(root.actions), "intent": intent,
                             "net_top": net_top, "skill_top": skill_top})
            # reuse the same open session to pick the move (avoids a second search_begin)
            return _greedy_move(state, obs.select, obs.current.yourIndex) if greedy else None
        finally:
            search_end()

    def probe_agent():
        tracker = OpponentTracker()
        def agent(obs, *_):
            sel = obs.get("select")
            if sel is None:
                return list(deck)
            o = to_observation_class(obs)
            try:
                tracker.observe(o)
            except Exception:
                pass
            s = o.select; n = len(s.option)
            if n == 0:
                return []
            move = None
            if s.maxCount == 1 and (n > 1 or s.minCount == 0):
                move = probe(o, tracker)         # returns a greedy move in greedy mode, else None
            if play == "mcts":
                return _mcts_pick(o, tracker)
            if play == "greedy":
                return move if move is not None else _greedy_pick(o, tracker)
            return random.sample(range(n), min(s.maxCount, n))
        return agent

    def rnd(obs, *_):
        sel = obs.get("select")
        if sel is None:
            return list(deck)
        n = len(sel["option"])
        return random.sample(range(n), min(sel["maxCount"], n)) if n else []

    def opponent_agent(pick_fn):
        tr = OpponentTracker()
        def agent(obs, *_):
            if obs.get("select") is None:
                return list(deck)
            o = to_observation_class(obs)
            try:
                tr.observe(o)
            except Exception:
                pass
            return pick_fn(o, tr)
        return agent

    env = make("cabt")
    p = probe_agent()
    opp = (opponent_agent(_mcts_pick) if play == "mcts"
           else opponent_agent(_greedy_pick) if play == "greedy" else rnd)
    for _ in range(games):
        try:
            env.reset(); env.run([p, opp])
        except Exception:
            stats["game_err"] += 1

    roots = stats["roots"] or 1
    print(f"reference policy: {ref_label}   play: {play}-vs-{play}")
    print(f"games={games}  roots={stats['roots']}  agree={stats['agree']}  disagree={stats['disagree']}")
    print(f"agreement = {100.0 * stats['agree'] / roots:.1f}%   intents={dict(intents)}")
    print(f"skipped={stats['skip']}  game_err={stats['game_err']}")

    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    with open(_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["root", "my_prizes", "opp_prizes", "n_options",
                                          "intent", "net_top", "skill_top"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} disagreement rows -> {os.path.relpath(_OUT, _ROOT)}")
    for r in rows[:20]:
        print("  ", r)


if __name__ == "__main__":
    main()

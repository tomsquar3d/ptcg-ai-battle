"""Smoke test for the turn-plan / KO-target skill prior.

Runs N games in the `cabt` env, and on every *single-select root* decision it:
  1. rebuilds the MCTS root the way the agent does (determinize -> search_begin -> _Node),
  2. calls `skill_root_prior` at lam in {0.0, 0.5, 1.0} and asserts a valid distribution over
     `node.actions` (non-negative, sums to 1, no NaN); lam=0.0 must equal the base prior,
  3. separately calls `turn_plan` directly and records whether it *evaluated* or raised — a high
     raise rate means the `# VERIFY` GameState fields in skill_prior.py are wrong (the prior would
     be silently falling back to base and doing nothing).

Run from repo root:  .venv/bin/python scripts/skill_prior_smoke.py 20
Exits non-zero if any distribution check fails.  Dev-only; not shipped.
"""
from __future__ import annotations

import os
import random
import sys
from collections import Counter

# --- make src/ (ptcg, search, nn) and cg importable, however this repo lays them out ----
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
from cg.api import to_observation_class, all_card_data, all_attack, search_begin, search_end  # noqa: E402
from ptcg.perception import build_game_state                                     # noqa: E402
from ptcg.opponent import OpponentTracker                                        # noqa: E402
from ptcg.archetypes import default_archetypes                                   # noqa: E402
from search.determinizer import determinize                                      # noqa: E402
from search.mcts import _Node, uniform_prior                                     # noqa: E402
from search.skill_prior import skill_root_prior, turn_plan                       # noqa: E402

_TOL = 1e-5
_DECK_PATHS = ("submissions/v7/deck.csv", "deck.csv", "submissions/v5/deck.csv")


def _read_deck():
    for p in _DECK_PATHS:
        fp = os.path.join(_ROOT, p)
        if os.path.exists(fp):
            with open(fp) as f:
                csv = f.read().split("\n")
            return [int(csv[i]) for i in range(60)]
    raise FileNotFoundError(f"no deck.csv in {_DECK_PATHS}")


def _check_dist(p, actions, stats, lam, base):
    ok = True
    keys = set(p.keys())
    if keys != set(actions):
        ok = False; stats["fail_keys"] += 1
    if any((v < 0 or v != v) for v in p.values()):        # negative or NaN
        ok = False; stats["fail_neg_nan"] += 1
    if abs(sum(p.values()) - 1.0) > _TOL and p:
        ok = False; stats["fail_sum"] += 1
    if lam == 0.0 and any(abs(p.get(a, 0.0) - base.get(a, 0.0)) > _TOL for a in actions):
        ok = False; stats["fail_lam0_not_base"] += 1     # lam=0 must reproduce base exactly
    stats["checks"] += 1
    if not ok:
        stats["fail"] += 1


def _probe(obs, deck, ci, ai, tracker, archs, stats):
    try:
        worlds = determinize(obs, deck, ci, k=1, archetypes=archs, opp_tracker=tracker)
        state = search_begin(obs, *worlds[0])
    except Exception as e:
        stats["skip_determinize"] += 1; stats["last_skip"] = repr(e); return
    try:
        root = _Node(state, obs.current.yourIndex, ci, ai)
        if root.terminal or not root.actions:
            stats["skip_terminal"] += 1; return
        base = uniform_prior(root)
        for lam in (0.0, 0.5, 1.0):
            p = skill_root_prior(obs, root, ci, ai, base_prior=base, lam=lam)
            _check_dist(p, root.actions, stats, lam, base)
        # does the tree itself evaluate, or is the prior silently falling back to base?
        try:
            gs = build_game_state(obs, ci)
            intent = turn_plan(gs, ci, ai)
            stats["intent_ok"] += 1
            stats["intents"][getattr(intent, "kind", "?")] += 1
        except Exception as e:
            stats["intent_err"] += 1; stats["last_intent_err"] = repr(e)
        stats["roots"] += 1
    finally:
        search_end()


def _probe_agent(deck, stats):
    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    archs = default_archetypes()
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
        s = o.select
        n = len(s.option)
        if n == 0:
            return []
        if s.maxCount == 1 and (n > 1 or s.minCount == 0):   # the MCTS-root case
            _probe(o, deck, ci, ai, tracker, archs, stats)
        return random.sample(range(n), min(s.maxCount, n))    # legal move to keep the game going
    return agent


def _rand_agent(deck):
    def agent(obs, *_):
        sel = obs.get("select")
        if sel is None:
            return list(deck)
        n = len(sel["option"])
        return random.sample(range(n), min(sel["maxCount"], n)) if n else []
    return agent


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    deck = _read_deck()
    stats = Counter(); stats["intents"] = Counter()
    env = make("cabt")
    probe = _probe_agent(deck, stats)
    rnd = _rand_agent(deck)
    for _ in range(games):
        try:
            env.reset(); env.run([probe, rnd])
        except Exception as e:
            stats["game_err"] += 1; stats["last_game_err"] = repr(e)

    print(f"games={games}  roots_probed={stats['roots']}  dist_checks={stats['checks']}  "
          f"failures={stats['fail']}")
    print(f"  fail breakdown: keys={stats['fail_keys']} neg/nan={stats['fail_neg_nan']} "
          f"sum={stats['fail_sum']} lam0!=base={stats['fail_lam0_not_base']}")
    print(f"  tree eval: ok={stats['intent_ok']} raised={stats['intent_err']}  intents={dict(stats['intents'])}")
    print(f"  skipped: determinize={stats['skip_determinize']} terminal={stats['skip_terminal']}  "
          f"game_err={stats['game_err']}")
    if stats["intent_err"]:
        print(f"  !! last tree error (check the # VERIFY fields): {stats.get('last_intent_err')}")
    if stats["fail"]:
        print("FAIL: skill prior produced an invalid distribution.")
        sys.exit(1)
    if stats["roots"] == 0:
        print("WARN: no single-select roots were probed (check the harness / deck).")
        sys.exit(2)
    print("PASS")


if __name__ == "__main__":
    main()

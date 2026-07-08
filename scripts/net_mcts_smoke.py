#!/usr/bin/env python3
"""Smoke test: the trained net wired into MCTS as prior + leaf value (src/nn/guide.py + search/mcts.py).

Checks, on real decision states from a live game:
  1. net-guided MCTS returns a legal pick with no error, and reports root visits;
  2. the net actually drives the search — root priors differ from uniform, value ∈ (−1, 1);
  3. the heuristic path (no guide) still works unchanged (default args).

Run:  .venv/bin/python scripts/net_mcts_smoke.py [checkpoint.pt]
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
sys.path.insert(0, os.path.join(REPO, "src"))

from cg.api import to_observation_class, all_card_data, all_attack, search_begin, search_end  # noqa: E402
from cg.game import battle_start, battle_finish, battle_select                               # noqa: E402
from ptcg.opponent import OpponentTracker                                                    # noqa: E402
from search.determinizer import determinize                                                  # noqa: E402
from search import mcts                                                                      # noqa: E402
from nn.guide import NetGuide                                                                # noqa: E402


def _read_deck(path):
    return [int(x) for x in open(path).read().split() if x.strip()]


def _is_branch(sel):
    return sel is not None and sel.maxCount == 1 and (len(sel.option) > 1 or sel.minCount == 0)


def main() -> int:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "experiments", "checkpoints", "net.pt")
    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    deck = _read_deck(os.path.join(REPO, "submissions", "v4", "deck.csv"))
    guide = NetGuide(ckpt, ci, ai)
    print(f"loaded {os.path.relpath(ckpt, REPO)}  (epoch {guide.meta.get('epoch')}, "
          f"val {guide.meta.get('val')})")

    tracker = OpponentTracker()
    obs_dict, _ = battle_start(deck, list(deck))
    checked = 0
    prior_gap = val_seen = None
    SIMS = 48
    try:
        for _ in range(4000):
            obs = to_observation_class(obs_dict)
            cur = getattr(obs, "current", None)
            if cur is not None and getattr(cur, "result", -1) != -1:
                break
            sel = obs.select
            if sel is None:
                break
            try:
                tracker.observe(obs)
            except Exception:
                pass
            if not _is_branch(sel):
                pick = list(range(min(max(sel.minCount, 0), len(sel.option))))
                obs_dict = battle_select(pick or ([0] if sel.option else []))
                continue

            worlds = determinize(obs, deck, ci, k=2, opp_tracker=tracker)

            # (2) net actually used: compare root priors to uniform + value range, on one world
            if checked == 0:
                rs = search_begin(obs, *worlds[0])
                node = mcts._Node(rs, cur.yourIndex, ci, ai)
                pr = guide.priors(node)
                unif = mcts.uniform_prior(node)
                prior_gap = max(abs(pr[a] - unif[a]) for a in node.actions)
                val_seen = guide.value(rs, cur.yourIndex)
                search_end()
                assert abs(sum(pr.values()) - 1.0) < 1e-4, "priors not normalized"
                assert -1.0 <= val_seen <= 1.0, "value out of range"

            # (1) net-guided search returns a legal pick
            pick_net, st_net = mcts.search(obs, worlds, ci, ai, sims=SIMS,
                                           prior_fn=guide.priors, value_fn=guide.value)
            # (3) heuristic path unchanged (default args)
            pick_heur, st_heur = mcts.search(obs, worlds, ci, ai, sims=SIMS)
            for pk in (pick_net, pick_heur):
                assert pk is not None, "no pick"
                assert all(0 <= i < len(sel.option) for i in pk), f"illegal pick {pk}"
            assert st_net.get("root_visits"), "net search reported no visits"

            checked += 1
            obs_dict = battle_select(pick_net if pick_net else [0])
            if checked >= 5:
                break
    finally:
        battle_finish()

    assert checked > 0, "no branch decision encountered"
    assert prior_gap and prior_gap > 1e-3, f"net priors indistinguishable from uniform ({prior_gap})"
    print(f"1. GUIDED   : {checked} branch decisions searched (net + heuristic), all picks legal")
    print(f"2. NET-USED : max |prior−uniform| = {prior_gap:.3f} (net shapes the search); "
          f"leaf value = {val_seen:+.3f} ∈ (−1,1)")
    print(f"3. HEURISTIC: default-arg search still returns legal picks (path unchanged)")
    print("\nOK: trained net is wired into MCTS as prior + leaf value.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

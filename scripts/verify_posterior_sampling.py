#!/usr/bin/env python3
"""Verify per-world posterior sampling in determinize() (#49): K worlds should spread over plausible
opponent archetypes proportional to the classifier posterior when evidence is thin, and collapse to the
MAP once it's peaked — vs the old behaviour where every world used the single MAP archetype.

Drives a real battle to an early single-select decision (thin evidence → diffuse posterior), then
determinizes K worlds both ways and RE-CLASSIFIES each sampled opponent deck to label which archetype it
came from. sample_archetype=True should show a diverse label mix ≈ the posterior; =False should be all-MAP.

    .venv/bin/python scripts/verify_posterior_sampling.py
"""
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
sys.path.insert(0, os.path.join(REPO, "src"))

from cg.api import all_card_data, to_observation_class                 # noqa: E402
from cg.game import battle_start, battle_finish, battle_select         # noqa: E402
from ptcg.archetypes import default_archetypes                         # noqa: E402
from ptcg.opponent import OpponentTracker, classify_archetype          # noqa: E402
from search.determinizer import determinize                           # noqa: E402


def _slug_of(deck, arch):
    """Label a sampled 60-card opponent deck by the archetype the classifier assigns it."""
    post = classify_archetype(Counter(deck), arch)
    return max(post, key=post.get) if post else "?"


def main() -> int:
    ci = {c.cardId: c for c in all_card_data()}
    arch = default_archetypes()
    deck = [int(x) for x in open(os.path.join(REPO, "submissions", "v5", "deck.csv")).read().split() if x.strip()]

    obs_dict, _ = battle_start(list(deck), list(deck))
    tracker = OpponentTracker()
    K = 24
    try:
        for _ in range(120):
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
            # stop at an early single-select decision (turn <= 3) to keep the posterior diffuse
            turn = int(getattr(cur, "turn", 0) or 0) if cur else 0
            if len(sel.option) > 1 and sel.maxCount == 1 and turn <= 3:
                post = classify_archetype(tracker.revealed(), arch)
                top = sorted(post.items(), key=lambda kv: -kv[1])[:5]
                print(f"turn {turn} | posterior (top-5): " +
                      ", ".join(f"{n.split('-')[0]}={p:.2f}" for n, p in top))

                w_map = determinize(obs, deck, ci, k=K, opp_tracker=tracker, seed=1, sample_archetype=False)
                w_smp = determinize(obs, deck, ci, k=K, opp_tracker=tracker, seed=1, sample_archetype=True)
                lab_map = Counter(_slug_of(w[2], arch) for w in w_map)
                lab_smp = Counter(_slug_of(w[2], arch) for w in w_smp)
                print(f"\nMAP-collapse  (sample_archetype=False): {len(lab_map)} distinct archetype(s) over {K} worlds")
                print("   " + ", ".join(f"{n}:{c}" for n, c in lab_map.most_common()))
                print(f"POSTERIOR-sample (sample_archetype=True): {len(lab_smp)} distinct archetype(s) over {K} worlds")
                print("   " + ", ".join(f"{n}:{c}" for n, c in lab_smp.most_common()))
                ok = len(lab_map) == 1 and len(lab_smp) >= 1
                spread = len(lab_smp) > len(lab_map)
                print(f"\nMAP collapses to a single archetype: {'OK' if len(lab_map)==1 else 'UNEXPECTED'}")
                print(f"Sampling spreads over the posterior:  {'YES' if spread else 'no (posterior already peaked here)'}")
                print("\nRESULT:", "PASS (sampling behaves as designed)" if ok else "CHECK ABOVE")
                return 0
            obs_dict = battle_select(list(range(min(sel.minCount, len(sel.option)))) or [0])
    finally:
        battle_finish()
    print("no early single-select decision reached")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Validate the mined archetype templates (companion to scripts/mine_archetypes.py).

Three checks:
  1. STRUCTURE  — templates load via default_archetypes(); each sums to 60; all card ids are real;
                  priors are positive.
  2. SEPARABILITY — over harvested replays, recover each deck, label it by marquee (same rule the
                  miner used), and check the naive-Bayes classifier recovers that label from the
                  revealed cards. Two evidence regimes: FULL deck (upper bound) and POKEMON-ONLY
                  (realistic — what actually hits the board/discard in a game). In-sample by design
                  (templates are built from these decks) — this is a necessary separability check,
                  not a generalization claim.
  3. DETERMINIZE — start a real battle and confirm determinize() samples opponent worlds from a
                  template without error.

    .venv/bin/python scripts/validate_archetypes.py [--limit 800]
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import json                                                            # noqa: E402
from cg.api import all_card_data, to_observation_class                 # noqa: E402
from cg.game import battle_start, battle_finish                        # noqa: E402
from ptcg.archetypes import default_archetypes                         # noqa: E402
from ptcg.opponent import classify_archetype, OpponentTracker          # noqa: E402
from search.determinizer import determinize                           # noqa: E402
from mine_archetypes import _marquee, _recover, _slug, REPLAYS, DECK_LEN  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=800, help="cap replays scanned for separability")
    args = ap.parse_args()

    ci = {c.cardId: c for c in all_card_data()}
    arch = default_archetypes()
    names = {a.name for a in arch}

    # 1. STRUCTURE
    print("=== 1. STRUCTURE ===")
    ok = True
    for a in arch:
        tot = sum(a.template.values())
        bad_ids = [cid for cid in a.template if cid not in ci]
        flag = "" if (tot == DECK_LEN and not bad_ids and a.prior > 0) else "  <-- PROBLEM"
        if flag:
            ok = False
        print(f"  {a.name:30s} sum={tot:3d} cards={len(a.template):2d} prior={a.prior:.4f} "
              f"bad_ids={len(bad_ids)}{flag}")
    print(f"  structure: {'OK' if ok else 'FAILED'} | {len(arch)} archetypes\n")

    # 2. SEPARABILITY
    print("=== 2. SEPARABILITY (classifier recovers the deck's own archetype) ===")
    files = [f for f in os.listdir(REPLAYS) if f.endswith("-replay.json")][:args.limit]
    tot = full_hit = poke_hit = 0
    confus = defaultdict(Counter)
    for fn in files:
        try:
            d = json.load(open(os.path.join(REPLAYS, fn)))
        except Exception:
            continue
        steps = d.get("steps")
        if not steps:
            continue
        for seat in (0, 1):
            deck = _recover(steps, seat)
            if not deck:
                continue
            label = _slug(_marquee(deck, ci))
            if label not in names:          # deck is a dropped rare archetype — skip
                continue
            tot += 1
            full = classify_archetype(Counter(deck), arch)
            if max(full, key=full.get) == label:
                full_hit += 1
            poke = Counter(c for c in deck if ci.get(c) and int(getattr(ci[c], "cardType", -1)) == 0)
            pp = classify_archetype(poke, arch)
            pred = max(pp, key=pp.get)
            if pred == label:
                poke_hit += 1
            else:
                confus[label][pred] += 1
    if tot:
        print(f"  decks tested: {tot}")
        print(f"  FULL-deck reveal   top-1 accuracy: {full_hit/tot:.1%}")
        print(f"  POKEMON-only reveal top-1 accuracy: {poke_hit/tot:.1%}  (realistic in-game evidence)")
        worst = sorted(((l, sum(c.values())) for l, c in confus.items()), key=lambda x: -x[1])[:4]
        for lab, miss in worst:
            top = confus[lab].most_common(2)
            print(f"    {lab}: {miss} pokemon-only misclassifications -> "
                  + ", ".join(f"{p}({n})" for p, n in top))
    else:
        print("  no decks matched kept archetypes (unexpected)")

    # 3. DETERMINIZE smoke
    print("\n=== 3. DETERMINIZE smoke ===")
    deck = [int(x) for x in open(os.path.join(REPO, "submissions", "v5", "deck.csv")).read().split() if x.strip()]
    obs_dict, _ = battle_start(list(deck), list(deck))
    tracker = OpponentTracker()
    smoke_ok = False
    try:
        for _ in range(80):
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
            if len(sel.option) > 1 and sel.maxCount == 1:
                worlds = determinize(obs, deck, ci, k=2, opp_tracker=tracker)
                opp_deck_len = len(worlds[0][2])
                print(f"  determinize returned {len(worlds)} worlds; opp deck sampled = {opp_deck_len} cards")
                smoke_ok = len(worlds) == 2 and opp_deck_len > 0
                break
            from cg.game import battle_select
            obs_dict = battle_select(list(range(min(sel.minCount, len(sel.option)))) or [0])
    finally:
        battle_finish()
    print(f"  determinize smoke: {'OK' if smoke_ok else 'FAILED'}")

    print(f"\nRESULT: {'ALL CHECKS PASSED' if (ok and smoke_ok and tot) else 'SEE PROBLEMS ABOVE'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

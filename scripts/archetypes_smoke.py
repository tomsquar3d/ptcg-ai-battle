#!/usr/bin/env python3
"""Smoke + behaviour test for the human-meta archetype templates (src/ptcg/archetypes.py).

Validates that the templates emitted by scripts/build_priors.py load into the opponent model's
classifier and behave sensibly:

  1. STRUCTURE       — 8 archetypes load; templates are non-empty, ~decklist-sized, priors > 0.
  2. SELF-CONSISTENCY— each archetype fed its own template classifies as itself (MAP).
  3. DISCRIMINATION  — a shared core (Dragapult line) spreads mass across only its variants;
                       a signature card pins the right archetype; no evidence returns the priors.

Run:  .venv/bin/python scripts/archetypes_smoke.py
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ptcg.archetypes import load_archetypes, default_archetypes  # noqa: E402
from ptcg.opponent import classify_archetype                     # noqa: E402

DRAGAPULT_CORE = [121, 119, 120]  # Dragapult ex / Dreepy / Drakloak — shared by all 3 variants


def main() -> int:
    arcs = load_archetypes()
    assert default_archetypes() is default_archetypes(), "default_archetypes not cached"

    # 1. STRUCTURE
    assert len(arcs) == 8, f"expected 8 archetypes, got {len(arcs)}"
    for a in arcs:
        size = sum(a.template.values())
        assert a.template and 40 <= size <= 60, f"{a.name}: implausible template size {size}"
        assert a.prior and a.prior > 0, f"{a.name}: non-positive prior"
    print(f"1. STRUCTURE       : {len(arcs)} archetypes, all templates 40-60 cards, priors > 0  OK")

    # 2. SELF-CONSISTENCY
    hits = sum(max(classify_archetype(a.template, arcs).items(), key=lambda kv: kv[1])[0] == a.name
               for a in arcs)
    assert hits == len(arcs), f"self-classification {hits}/{len(arcs)}"
    print(f"2. SELF-CONSISTENCY: each archetype fed its own template -> itself, {hits}/{len(arcs)}  OK")

    # 3a. shared Dragapult core → mass concentrated on the three Dragapult variants only
    post = classify_archetype(Counter(DRAGAPULT_CORE), arcs)
    drag = sum(p for n, p in post.items() if n.startswith("dragapult"))
    assert drag > 0.95, f"Dragapult core leaked {1 - drag:.2f} mass off the Dragapult variants"

    # 3b. a signature (single-archetype) card pins that archetype as MAP
    templates = {a.name: set(a.template) for a in arcs}
    pinned = 0
    for a in arcs:
        others = set().union(*(t for n, t in templates.items() if n != a.name))
        uniq = templates[a.name] - others
        if not uniq:
            continue
        card = sorted(uniq)[0]
        top = max(classify_archetype(Counter({card: 2}), arcs).items(), key=lambda kv: kv[1])[0]
        assert top == a.name, f"signature card {card} of {a.name} classified as {top}"
        pinned += 1

    # 3c. no evidence → normalized priors (argmax = highest meta share)
    empty = classify_archetype(Counter(), arcs)
    assert abs(sum(empty.values()) - 1.0) < 1e-9, "posterior not normalized"
    top_prior = max(arcs, key=lambda a: a.prior).name
    assert max(empty, key=empty.get) == top_prior, "no-evidence MAP != highest-prior archetype"
    print(f"3. DISCRIMINATION  : Dragapult core -> variants ({drag:.0%}); {pinned} signature cards "
          f"pin their archetype; empty -> priors (MAP={top_prior})  OK")

    print("\nOK: human-meta archetype templates load and classify correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

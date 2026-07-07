"""Real archetype templates for the opponent model (Belief Engine Layer 2b).

`opponent.py` ships the *mechanism* (Bayesian bag-of-cards classifier + hypergeometric zone
model) but leaves the templates to callers. This module is the missing data half: it loads the
human-meta templates mined from TrainerHill (`archetype_templates.json`, emitted by the offline
`scripts/build_priors.py`) into `Archetype` objects the classifier consumes.

The JSON lives next to this module so it bundles with the agent (build_submission vendors
src/ptcg). Each archetype = {card_id -> typical copies} (round(expected_copies) from the priors)
+ `prior` = human meta share. Regenerate the JSON from the CSVs via `build_priors.py`.

Pure module (stdlib only) — safe to import inside the agent.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from .opponent import Archetype

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archetype_templates.json")

_CACHE: Optional[list[Archetype]] = None


def load_archetypes(path: str = _DEFAULT_PATH) -> list[Archetype]:
    """Load archetype templates from the bundled JSON into `Archetype(name, template, prior)`.

    template keys are JSON strings → cast back to int card ids. `prior` is the meta share (the
    classifier normalizes priors internally, so absolute scale is irrelevant); missing/None
    priors fall back to 1.0 (uniform)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: list[Archetype] = []
    for name, e in data.get("archetypes", {}).items():
        template = {int(cid): int(cnt) for cid, cnt in e.get("template", {}).items()}
        if not template:
            continue
        out.append(Archetype(name=name, template=template, prior=e.get("prior") or 1.0))
    return out


def default_archetypes() -> list[Archetype]:
    """Cached load of the bundled human-meta templates — the default for `build_opponent_belief`."""
    global _CACHE
    if _CACHE is None:
        _CACHE = load_archetypes()
    return _CACHE

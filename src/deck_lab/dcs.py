"""Static Disruption Coverage Score (DCS) for a deck.

docs/ptcg_agent_strategy.md §4 defines DCS as the deck-level disruption metric:

    DCS = Σ_archetypes  meta_share(A) × effectiveness(our best answer, A's vulnerability profile)

This is the *static* version — computable pre-round-robin from the human-meta priors
(src/priors/vulnerability_profiles.json + archetype_priors.json for meta_share) and the deck's
card list. The *functional* version (measured Δ in opponents' first-attack turn / turns-to-6
with the disruption package in vs out) comes later and supersedes these effectiveness estimates.

EFFECTIVENESS RUBRIC
--------------------
Four disruption axes, each pairing a lever we might run against a vulnerability the opponent
exposes. For each axis we compute our_strength ∈ [0,1] (how much of that lever our deck runs,
saturating) × their_vuln ∈ [0,1] (how exposed the archetype is, from its profile):

  axis            our lever (cards, saturation)                     their vulnerability signal
  ------------    ---------------------------------------------     ----------------------------------
  energy_denial   Crushing/Enhanced Hammer, Giacomo (sat 4)         (basic+special energy) / ENERGY_MAX
                                                                     (total energy reliance = hammer-
                                                                     exposed; Dragapult via basic,
                                                                     rockets-honchkrow via special —
                                                                     Enhanced Hammer hits special only)
  tool_denial     Jamming Tower, Lost Vacuum (sat 2)                tool_count / TOOL_MAX
  ability_lock    Team Rocket's Watchtower (sat 2)                  pct_pokemon_with_abilities (0..1)
  hand_disruption Judge, Iono, Unfair Stamp, Roxanne (sat 3)        draw_supporter_reliance / DRAW_MAX

effectiveness(A) = MAX over the four axes of (our_strength_axis × their_vuln_axis) — "best
answer" = our single most effective lever against A (not a sum: stacking two mediocre answers
shouldn't read as coverage). DCS is the meta_share-weighted sum over the archetypes we can
profile; we also report it normalized by the covered meta_share so it reads as a [0,1] rate.

CAVEATS (documented, corrected by the functional version):
  * energy_denial counts basic + special energy (special-energy decks like rockets-honchkrow
    ARE hammer-exposed — Crushing Hammer hits any energy, Enhanced Hammer hits special only,
    and special energies often carry effects the removal also strips). It does not yet weight
    Enhanced-Hammer-specific exposure higher for special-energy decks, and real hammer fragility
    is non-monotonic (peaks at low-but-nonzero energy w/o acceleration); the functional version
    measures it directly.
  * ability detection misses Tera passives (engine `skills` only).
  * saturations/maxes are judgment calls, held as named constants for one-line ablation.

Run:  .venv/bin/python src/deck_lab/dcs.py [deck.csv]   (default: submissions/v1/deck.csv)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PRIORS_DIR = os.path.join(REPO, "src", "priors")
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
from cg.api import all_card_data  # noqa: E402

# --- lever taxonomy: names matched (case-insensitive) against the deck's cards, + saturation ---
LEVERS = {
    "energy_denial":   {"names": ["crushing hammer", "enhanced hammer", "giacomo"], "sat": 4},
    "tool_denial":     {"names": ["jamming tower", "lost vacuum"], "sat": 2},
    "ability_lock":    {"names": ["team rocket's watchtower"], "sat": 2},
    "hand_disruption": {"names": ["judge", "iono", "unfair stamp", "roxanne"], "sat": 3},
}
# vulnerability normalization constants (named for ablation)
ENERGY_MAX = 8.0       # basic-energy count that saturates energy_denial vulnerability
TOOL_MAX = 4.0         # tool_count that saturates tool_denial vulnerability
DRAW_MAX = 8.0         # draw_supporter_reliance that saturates hand_disruption vulnerability


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _vulnerabilities(profile: dict) -> dict[str, float]:
    return {
        "energy_denial": _clamp(
            (profile["basic_energy_count"] + profile.get("special_energy_count", 0.0)) / ENERGY_MAX),
        "tool_denial": _clamp(profile["tool_count"] / TOOL_MAX),
        "ability_lock": _clamp(profile["pct_pokemon_with_abilities"]),
        "hand_disruption": _clamp(profile["draw_supporter_reliance"] / DRAW_MAX),
    }


def deck_lever_strengths(deck_ids: list[int]) -> dict[str, float]:
    """Our_strength ∈ [0,1] per axis = (copies of that lever in the deck) / saturation, clamped."""
    names = {c.cardId: c.name.lower() for c in all_card_data()}
    counts = Counter(deck_ids)
    strengths, present = {}, {}
    for axis, spec in LEVERS.items():
        copies = sum(n for cid, n in counts.items()
                     if any(lv in names.get(cid, "") for lv in spec["names"]))
        strengths[axis] = _clamp(copies / spec["sat"])
        present[axis] = copies
    return strengths, present


def read_deck_csv(path: str) -> list[int]:
    with open(path, newline="", encoding="utf-8") as f:
        return [int(x) for x in f.read().split("\n") if x.strip()]


def score_deck(deck_ids: list[int], *, priors_dir: str = PRIORS_DIR) -> dict:
    vuln = json.load(open(os.path.join(priors_dir, "vulnerability_profiles.json")))["archetypes"]
    priors = json.load(open(os.path.join(priors_dir, "archetype_priors.json")))["archetypes"]
    strengths, present = deck_lever_strengths(deck_ids)

    per_arch, dcs, covered = {}, 0.0, 0.0
    for arch, prof in vuln.items():
        share = priors.get(arch, {}).get("meta_share") or 0.0
        vulns = _vulnerabilities(prof)
        axis_eff = {ax: round(strengths[ax] * vulns[ax], 4) for ax in LEVERS}
        best_axis = max(axis_eff, key=axis_eff.get)
        eff = axis_eff[best_axis]
        per_arch[arch] = {"meta_share": share, "effectiveness": eff, "best_axis": best_axis,
                          "axis_effectiveness": axis_eff, "vulnerabilities": {k: round(v, 3) for k, v in vulns.items()}}
        dcs += share * eff
        covered += share
    return {
        "dcs": round(dcs, 4),
        "dcs_normalized": round(dcs / covered, 4) if covered else 0.0,
        "covered_meta_share": round(covered, 4),
        "lever_strengths": {k: round(v, 3) for k, v in strengths.items()},
        "lever_copies": present,
        "per_archetype": per_arch,
    }


def _print(res: dict, deck_path: str):
    print(f"=== Static DCS for {os.path.relpath(deck_path, REPO)} ===")
    print(f"deck disruption levers (copies): "
          + ", ".join(f"{k}={res['lever_copies'][k]}" for k in LEVERS))
    print(f"lever strengths [0..1]         : "
          + ", ".join(f"{k}={v}" for k, v in res["lever_strengths"].items()))
    print(f"\nDCS (field-weighted)  : {res['dcs']}   "
          f"(covered meta_share {res['covered_meta_share']:.0%})")
    print(f"DCS (normalized rate) : {res['dcs_normalized']}")
    print("\nper-archetype best answer:")
    print(f"  {'archetype':24} {'share':>6} {'eff':>6}  best axis")
    for arch, a in sorted(res["per_archetype"].items(), key=lambda kv: -kv[1]["meta_share"]):
        print(f"  {arch:24} {a['meta_share']:6.1%} {a['effectiveness']:6.3f}  {a['best_axis']}")


def main():
    deck_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "submissions", "v1", "deck.csv")
    res = score_deck(read_deck_csv(deck_path))
    _print(res, deck_path)


if __name__ == "__main__":
    main()

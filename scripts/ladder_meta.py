#!/usr/bin/env python3
"""Ladder meta report from harvested replays (docs §3d follow-up; step 2).

Recovers BOTH agents' decklists from every replay (the 60-id deck-selection action), clusters them by
marquee Pokémon, and reports the archetype mix, how it tracks rating, and how often complex evolution
decks appear in the top tier. This supersedes the offline TrainerHill priors as our primary meta source
— it is the deck distribution of the actual simulator ladder.

    .venv/bin/python scripts/ladder_meta.py   # writes experiments/ladder_meta.md
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
sys.path.insert(0, os.path.join(REPO, "src"))

from cg.api import all_card_data, CardType                          # noqa: E402

REPLAYS = os.path.join(REPO, "data", "replays")
OUT = os.path.join(REPO, "experiments", "ladder_meta.md")
_EID = re.compile(r"episode-(\d+)-replay\.json")
_CT = {c.value: c.name for c in CardType}


def _marquee(deck, ci):
    """Archetype label = the deck's headline Pokémon: highest (ex, stage, hp)."""
    best, key = None, (-1, -1, -1)
    for cid in set(deck):
        c = ci.get(cid)
        if not c or int(c.cardType) != 0:
            continue
        k = (int(bool(getattr(c, "ex", False)) or bool(getattr(c, "megaEx", False))),
             2 * int(bool(getattr(c, "stage2", False))) + int(bool(getattr(c, "stage1", False))),
             getattr(c, "hp", 0) or 0)
        if k > key:
            key, best = k, c.name
    return best or "?"


def _is_complex(deck, ci) -> bool:
    """Complex evolution deck: runs a Stage-2 line and/or Rare Candy."""
    has_stage2 = any(ci.get(c) and getattr(ci[c], "stage2", False) for c in set(deck))
    has_rare_candy = any(ci.get(c) and ci[c].name.lower().startswith("rare candy") for c in deck)
    return has_stage2 or has_rare_candy


def _recover(steps, seat):
    for st in steps[:6]:
        a = st[seat].get("action")
        if isinstance(a, list) and len(a) == 60 and all(isinstance(x, int) for x in a):
            return list(a)
    return None


def main() -> int:
    ci = {c.cardId: c for c in all_card_data()}
    idx = json.load(open(os.path.join(REPLAYS, "index.json")))["index"] if os.path.exists(os.path.join(REPLAYS, "index.json")) else {}
    files = [f for f in os.listdir(REPLAYS) if f.endswith("-replay.json")]

    # one record per (deck) sighting: (deck_tuple, rating, won)
    seen = []                       # list of dicts
    deck_names = {}                 # deck_tuple -> marquee
    for fn in files:
        try:
            d = json.load(open(os.path.join(REPLAYS, fn)))
        except Exception:
            continue
        steps = d.get("steps"); rew = d.get("rewards")
        if not steps or not rew:
            continue
        eid = (_EID.search(fn) or [None, None])[1]
        agents = (idx.get(eid, {}) or {}).get("agents", [])
        rating_by_reward = {a.get("reward"): a.get("rating") for a in agents}
        for seat in (0, 1):
            deck = _recover(steps, seat)
            if not deck:
                continue
            key = tuple(sorted(deck))
            deck_names.setdefault(key, _marquee(deck, ci))
            seen.append({"key": key, "rating": rating_by_reward.get(rew[seat]),
                         "won": rew[seat] == max(rew), "complex": _is_complex(deck, ci)})

    n = len(seen)
    ratings = [s["rating"] for s in seen if s["rating"]]
    hi_cut = sorted(ratings)[int(len(ratings) * 0.5)] if ratings else 0    # top-half rating band

    # archetype aggregation
    arch = defaultdict(lambda: {"n": 0, "wins": 0, "ratings": [], "complex": False, "keys": set()})
    for s in seen:
        name = deck_names[s["key"]]
        a = arch[name]
        a["n"] += 1; a["wins"] += int(s["won"]); a["complex"] = a["complex"] or s["complex"]
        a["keys"].add(s["key"])
        if s["rating"]:
            a["ratings"].append(s["rating"])
    ranked = sorted(arch.items(), key=lambda kv: -kv[1]["n"])

    complex_share = sum(1 for s in seen if s["complex"]) / n if n else 0
    hi = [s for s in seen if s["rating"] and s["rating"] >= hi_cut]
    lo = [s for s in seen if s["rating"] and s["rating"] < hi_cut]
    complex_hi = sum(1 for s in hi if s["complex"]) / len(hi) if hi else 0
    complex_lo = sum(1 for s in lo if s["complex"]) / len(lo) if lo else 0

    lines = []
    lines.append("# Ladder Meta Report\n")
    lines.append(f"Source: {len(files)} harvested replays (`data/replays/`, top-20 ladder teams + our v5) "
                 f"→ {n} deck sightings, {len(deck_names)} distinct decklists, {len(arch)} archetypes. "
                 f"**This supersedes the TrainerHill priors as our primary meta source** — it is the actual "
                 f"simulator-ladder deck distribution.\n")
    lines.append(f"**Complex evolution decks (Stage-2 line and/or Rare Candy): {complex_share:.0%} of all "
                 f"sightings.** By rating band (split at {hi_cut:.0f}): top-half **{complex_hi:.0%}** vs "
                 f"bottom-half {complex_lo:.0%}.\n")
    lines.append("## Archetypes (by frequency)\n")
    lines.append("| archetype (marquee Pokémon) | sightings | share | win% | median rating | complex | distinct lists |")
    lines.append("|---|--:|--:|--:|--:|:-:|--:|")
    for name, a in ranked[:25]:
        med = sorted(a["ratings"])[len(a["ratings"]) // 2] if a["ratings"] else None
        med_s = f"{med:.0f}" if med else "—"
        lines.append(f"| {name} | {a['n']} | {a['n']/n:.0%} | {a['wins']/a['n']:.0%} | {med_s} | "
                     f"{'yes' if a['complex'] else 'no'} | {len(a['keys'])} |")
    lines.append("\n## Takeaway\n")
    top_arch = ranked[0][0] if ranked else "?"
    lines.append(f"- The ladder's dominant archetype is **{top_arch}** ({ranked[0][1]['n']/n:.0%} of sightings).")
    lines.append(f"- Complex evolution decks are **{'more' if complex_hi >= complex_lo else 'less'} common in the "
                 f"top tier** ({complex_hi:.0%} vs {complex_lo:.0%}) — {'confirming' if complex_hi>0.5 else 'nuancing'} "
                 f"that strong ladder play favours evolution decks.")
    lines.append(f"- Our shipped `sample` deck's marquee would slot as one of these archetypes; the gap between it "
                 f"and the top archetypes is the deck-strength question the pilot work has been circling.")
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"wrote {os.path.relpath(OUT, REPO)} | {n} sightings, {len(arch)} archetypes, "
          f"complex-deck share {complex_share:.0%} (top {complex_hi:.0%} / bottom {complex_lo:.0%})")
    print("top archetypes:", ", ".join(f"{nm}({a['n']})" for nm, a in ranked[:8]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

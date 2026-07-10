#!/usr/bin/env python3
"""Mine opponent archetype templates from harvested ladder replays (fixes the stale-templates gap).

The opponent model (`src/ptcg/opponent.py` classifier + `src/search/determinizer.py` zone sampling)
runs on `archetype_templates.json`. That file was generated offline from a TrainerHill scrape
(`build_priors.py`) whose 8 archetypes barely overlap the actual simulator ladder — only Alakazam
is a real ladder deck; the rest (three Dragapult variants, Zoroark, Greninja, ...) are ~1% or absent,
while the ladder's #2-#4 decks (Mega Kangaskhan ex 20%, Cynthia's Garchomp ex 14%, Grimmsnarl ex 8%)
have no template at all. See `experiments/ladder_meta.md`.

This regenerates the templates from the REAL ladder: recover both agents' 60-card decklists from every
replay, cluster by marquee Pokémon (same rule as `ladder_meta.py`), and for each archetype build a
representative 60-card template = the per-card *expected copies* across that archetype's sightings,
rounded to exactly 60 by the largest-remainder method (so no filler-energy padding is needed downstream).
`prior` = the archetype's real ladder sighting share.

    .venv/bin/python scripts/mine_archetypes.py                 # writes src/ptcg/archetype_templates.json
    .venv/bin/python scripts/mine_archetypes.py --dry-run       # print, don't write
    .venv/bin/python scripts/mine_archetypes.py --min-sightings 20 --max-archetypes 12
"""
import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
sys.path.insert(0, os.path.join(REPO, "src"))

from cg.api import all_card_data                                    # noqa: E402

REPLAYS = os.path.join(REPO, "data", "replays")
OUT = os.path.join(REPO, "src", "ptcg", "archetype_templates.json")
_EID = re.compile(r"episode-(\d+)-replay\.json")
DECK_LEN = 60


# generic tech/support Pokémon that get splashed across many decks — they should NOT define an
# archetype label unless a deck has nothing bigger (else they form a blurry grab-bag cluster).
_DEMOTE = ("fezandipiti", "munkidori", "dudunsparce", "lumineon", "radiant", "squawkabilly")


def _marquee(deck, ci, demote=_DEMOTE) -> str:
    """Archetype label = the deck's headline Pokémon: highest (ex/mega, stage, hp). Mirrors ladder_meta,
    but generic tech Pokémon (`demote`) are considered only if the deck has no other Pokémon — so a
    Fezandipiti-ex splash doesn't swallow a deck's real engine into one grab-bag cluster."""
    def _score(c, is_demoted):
        return (0 if is_demoted else 1,
                int(bool(getattr(c, "ex", False)) or bool(getattr(c, "megaEx", False))),
                2 * int(bool(getattr(c, "stage2", False))) + int(bool(getattr(c, "stage1", False))),
                getattr(c, "hp", 0) or 0)
    best, key = None, (-1, -1, -1, -1)
    for cid in set(deck):
        c = ci.get(cid)
        if not c or int(getattr(c, "cardType", -1)) != 0:
            continue
        is_demoted = any(d in c.name.lower() for d in demote)
        k = _score(c, is_demoted)
        if k > key:
            key, best = k, c.name
    return best or "?"


def _recover(steps, seat):
    for st in steps[:6]:
        a = st[seat].get("action")
        if isinstance(a, list) and len(a) == DECK_LEN and all(isinstance(x, int) for x in a):
            return list(a)
    return None


def _slug(name: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", name.lower())).strip("-") or "unknown"


def _largest_remainder(expected: dict[int, float], total: int = DECK_LEN) -> dict[int, int]:
    """Round fractional expected-copy counts to non-negative integers summing to exactly `total`,
    giving the leftover slots to the largest fractional remainders (Hamilton's method)."""
    floors = {cid: int(v) for cid, v in expected.items()}
    used = sum(floors.values())
    rem = total - used
    if rem > 0:
        # hand out remaining slots to the biggest fractional parts
        order = sorted(expected, key=lambda c: (expected[c] - floors[c], expected[c]), reverse=True)
        for cid in order[:rem]:
            floors[cid] += 1
    elif rem < 0:
        # over-allocated (rare) — trim from smallest fractional parts first
        order = sorted((c for c in floors if floors[c] > 0),
                       key=lambda c: (expected[c] - int(expected[c]), expected[c]))
        i = 0
        while rem < 0 and order:
            cid = order[i % len(order)]
            if floors[cid] > 0:
                floors[cid] -= 1
                rem += 1
            i += 1
    return {cid: n for cid, n in floors.items() if n > 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default=REPLAYS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--min-sightings", type=int, default=20,
                    help="drop archetypes seen fewer than this many times (template too noisy)")
    ap.add_argument("--max-archetypes", type=int, default=12, help="cap kept archetypes (by frequency)")
    ap.add_argument("--dry-run", action="store_true", help="print the mined templates but do not write")
    args = ap.parse_args()

    ci = {c.cardId: c for c in all_card_data()}
    idx_path = os.path.join(args.replays, "index.json")
    idx = json.load(open(idx_path))["index"] if os.path.exists(idx_path) else {}
    files = [f for f in os.listdir(args.replays) if f.endswith("-replay.json")]

    # aggregate per archetype: sighting count, summed per-card copies, wins, ratings
    agg = defaultdict(lambda: {"n": 0, "cards": Counter(), "wins": 0, "ratings": [], "lists": set()})
    n_replays = n_sightings = 0
    for fn in files:
        try:
            d = json.load(open(os.path.join(args.replays, fn)))
        except Exception:
            continue
        steps, rew = d.get("steps"), d.get("rewards")
        if not steps:
            continue
        n_replays += 1
        eid = (_EID.search(fn) or [None, None])[1]
        agents = (idx.get(eid, {}) or {}).get("agents", [])
        rating_by_reward = {a.get("reward"): a.get("rating") for a in agents}
        best = max(rew) if rew and all(r is not None for r in rew) else None
        for seat in (0, 1):
            deck = _recover(steps, seat)
            if not deck:
                continue
            name = _marquee(deck, ci)
            a = agg[name]
            a["n"] += 1
            a["cards"].update(deck)
            a["lists"].add(tuple(sorted(deck)))
            if rew and rew[seat] is not None:
                if best is not None and rew[seat] == best:
                    a["wins"] += 1
                r = rating_by_reward.get(rew[seat])
                if r:
                    a["ratings"].append(r)
            n_sightings += 1

    ranked = sorted(agg.items(), key=lambda kv: -kv[1]["n"])
    kept = [(nm, a) for nm, a in ranked if a["n"] >= args.min_sightings][:args.max_archetypes]
    kept_sightings = sum(a["n"] for _, a in kept)

    archetypes = {}
    print(f"mined {n_replays} replays → {n_sightings} deck sightings, {len(agg)} raw archetypes\n")
    print(f"{'archetype':30s} {'sight':>6s} {'share':>6s} {'win%':>5s} {'medR':>5s} {'lists':>5s} {'tmpl':>5s}")
    for nm, a in kept:
        n = a["n"]
        expected = {cid: cnt / n for cid, cnt in a["cards"].items()}   # sums to 60 (every deck is 60)
        template = _largest_remainder(expected, DECK_LEN)
        share = n / n_sightings if n_sightings else 0.0
        ratings = sorted(a["ratings"])
        med = ratings[len(ratings) // 2] if ratings else None
        archetypes[_slug(nm)] = {
            "template": {str(cid): int(cnt) for cid, cnt in sorted(template.items())},
            "prior": round(share, 6),
            # provenance (ignored by the loader; handy for auditing)
            "marquee": nm,
            "sightings": n,
            "distinct_lists": len(a["lists"]),
            "win_rate": round(a["wins"] / n, 3) if n else None,
            "median_rating": med,
        }
        print(f"{_slug(nm):30s} {n:6d} {share:5.1%} {a['wins']/n:4.0%} "
              f"{(str(med) if med else '—'):>5s} {len(a['lists']):5d} {sum(template.values()):5d}")

    dropped = [(nm, a["n"]) for nm, a in ranked if (nm, a) not in kept]
    drop_sightings = n_sightings - kept_sightings
    print(f"\nkept {len(kept)} archetypes = {kept_sightings/n_sightings:.1%} of sightings; "
          f"dropped {len(dropped)} rare/small archetypes ({drop_sightings} sightings, "
          f"{drop_sightings/n_sightings:.1%}) below --min-sightings={args.min_sightings}")
    if dropped[:8]:
        print("  dropped (top by sightings):", ", ".join(f"{nm}({n})" for nm, n in dropped[:8]))

    if args.dry_run:
        print("\n[dry-run] not writing.")
        return 0

    payload = {
        "_meta": {
            "description": ("Opponent-model archetype templates MINED FROM THE ACTUAL LADDER "
                            "(scripts/mine_archetypes.py over data/replays). template[card_id] = "
                            "expected copies across that archetype's ladder sightings, rounded to 60 by "
                            "largest-remainder; prior = real ladder sighting share. Supersedes the "
                            "TrainerHill-derived templates (archetype_templates.trainerhill.json)."),
            "source_replays": n_replays,
            "source_sightings": n_sightings,
            "min_sightings": args.min_sightings,
        },
        "archetypes": archetypes,
    }
    # back up the old TrainerHill file once
    bak = os.path.join(os.path.dirname(args.out), "archetype_templates.trainerhill.json")
    if os.path.exists(args.out) and not os.path.exists(bak):
        shutil.copy2(args.out, bak)
        print(f"\nbacked up old templates → {os.path.relpath(bak, REPO)}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {os.path.relpath(args.out, REPO)} | {len(archetypes)} archetypes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

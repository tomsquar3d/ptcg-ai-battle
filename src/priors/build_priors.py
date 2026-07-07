"""Build meta-priors artifacts from the TrainerHill human-meta CSVs + the competition card pool.

Emits (all under src/priors/, except the report):
  * archetype_priors.json      — per archetype: meta_share + per-card P(copies | archetype)
  * human_matchup_matrix.json  — Beta-smoothed win rates + n for every deck pair
  * vulnerability_profiles.json — per archetype: hammer/tool/ability/draw-supporter exposure
  * experiments/meta_priors_report.md — card-mapping coverage + constructibility ranking

Reads:
  * data/external/trainerhill/*skeleton*.csv  — one archetype skeleton each (schema in the task)
  * data/external/trainerhill/*matchup*.csv   — deck1,deck2,wins,losses,ties,total,win_rate
  * the card pool via carddb (EN CSV + engine DB)

This is the *static* half of the disruption programme in docs/ptcg_agent_strategy.md §4: priors
and vulnerability profiles from human lists, feeding the static Disruption Coverage Score
(src/deck_lab/dcs.py). The functional (simulated-game) half comes later.

Run:  .venv/bin/python src/priors/build_priors.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics
from collections import defaultdict

from carddb import load_card_db, CardDB  # noqa: E402  (run as a script from this dir)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TH_DIR = os.path.join(REPO, "data", "external", "trainerhill")
OUT_DIR = HERE
REPORT = os.path.join(REPO, "experiments", "meta_priors_report.md")
# Runtime-ready templates for the opponent model, emitted INTO the agent package so they ship
# with the submission (build_submission vendors src/ptcg). Consumed by src/ptcg/archetypes.py.
PTCG_TEMPLATES = os.path.join(REPO, "src", "ptcg", "archetype_templates.json")

BETA_ALPHA = 1.0   # Beta(1,1) Laplace prior on the (win | decisive game) rate
CONSTRUCTIBILITY_FLAG = 0.85


# --------------------------------------------------------------------------- skeletons + priors

def _f(x: str) -> float:
    x = (x or "").strip()
    return float(x) if x else 0.0


def _copy_distribution(row: dict, is_basic_energy: bool) -> tuple[dict[str, float], float]:
    """P(copies | archetype) over {0,1,2,3,4}, renormalized. Basic energy uses `modal_copies`
    as a point mass (the pr_ columns for basic energy are image-histogram estimates — see the
    `note` field — so the modal count is the trustworthy signal)."""
    modal = (row.get("modal_copies") or "").strip()
    if is_basic_energy and modal:
        m = int(modal)
        return {str(m): 1.0}, float(m)
    p = {str(k): _f(row.get(f"pr_{k}")) / 100.0 for k in (1, 2, 3, 4)}
    p["0"] = max(0.0, 1.0 - sum(p.values()))
    total = sum(p.values()) or 1.0
    p = {k: v / total for k, v in p.items()}
    exp = sum(int(k) * v for k, v in p.items())
    return p, exp


def load_skeletons(db: CardDB):
    """Return {archetype: [card_entry...]} plus a mapping report. Each entry carries the resolved
    pool card (or None), the section, and the copy distribution."""
    archetypes: dict[str, list[dict]] = {}
    for fn in sorted(glob.glob(os.path.join(TH_DIR, "*skeleton*.csv"))):
        with open(fn, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                arch = r["archetype"].strip()
                res = db.resolve(r["card"], r["set_code"])
                pc = db.pool.get(res.card_id) if res.mapped else None
                dist, exp = _copy_distribution(r, pc.is_basic_energy if pc else False)
                archetypes.setdefault(arch, []).append({
                    "name": r["card"], "set_code": r["set_code"],
                    "section": r["section"].strip(), "overall_play_rate": _f(r["overall_play_rate"]),
                    "card_id": res.card_id, "method": res.method, "pool_card": pc,
                    "copy_prob": dist, "expected_copies": exp,
                })
    return archetypes


def build_archetype_priors(archetypes, meta_share) -> dict:
    out = {"_meta": {
        "description": "Per-archetype human-meta priors from TrainerHill skeletons, filtered to "
                       "the competition pool. copy_prob = P(copies of card | archetype) over "
                       "{0..4}, renormalized; basic energy uses modal_copies as a point mass.",
        "meta_share_source": "matchup CSV game volumes (share of the full 15-deck field)",
        "unmapped_cards_excluded": True,
    }, "archetypes": {}}
    for arch, cards in archetypes.items():
        entry = {"meta_share": meta_share.get(arch, {}).get("share"),
                 "games": meta_share.get(arch, {}).get("games"), "cards": {}}
        for c in cards:
            if not c["card_id"]:
                continue  # pool-filtered: drop unmapped
            pc = c["pool_card"]
            entry["cards"][str(c["card_id"])] = {
                "name": pc.name, "section": c["section"], "card_type": pc.card_type,
                "overall_play_rate": c["overall_play_rate"],
                "copy_prob": {k: round(v, 4) for k, v in c["copy_prob"].items() if v > 0},
                "expected_copies": round(c["expected_copies"], 3),
                "basic_energy": pc.is_basic_energy,
            }
        out["archetypes"][arch] = entry
    return out


# ------------------------------------------------------------------------------- matchup matrix

def load_matchups():
    """Parse the matchup CSV → (rows, meta_share). meta_share is over the FULL field (every deck
    appearing in the CSV), so our 8 skeleton archetypes get realistic shares; DCS later weights
    only the archetypes we can profile and reports the covered fraction."""
    path = next(iter(glob.glob(os.path.join(TH_DIR, "*matchup*.csv"))), None)
    if not path:
        raise SystemExit("no matchup CSV found in data/external/trainerhill/")
    rows = []
    games = defaultdict(float)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d1, d2 = r["deck1"].strip(), r["deck2"].strip()
            w, l, t = int(r["wins"]), int(r["losses"]), int(r["ties"])
            total = int(r["total"])
            rows.append((d1, d2, w, l, t, total, _f(r.get("win_rate"))))
            games[d1] += total
    grand = sum(games.values()) or 1.0
    meta_share = {d: {"games": g, "share": g / grand} for d, g in games.items()}
    return rows, meta_share, os.path.basename(path)


def build_matchup_matrix(rows) -> dict:
    matrix: dict[str, dict[str, dict]] = defaultdict(dict)
    for d1, d2, w, l, t, total, src_wr in rows:
        decisive = w + l
        # Beta(1,1) on the (win | decisive game) rate; ties excluded from the rate, reported
        # separately. Handles the mirror rows (wins==losses==decisive) → smooths to ~0.5.
        smoothed = (w + BETA_ALPHA) / (decisive + 2 * BETA_ALPHA) if decisive >= 0 else None
        matrix[d1][d2] = {
            "wins": w, "losses": l, "ties": t, "total": total,
            "win_rate_source": src_wr,
            "win_rate_smoothed": round(smoothed, 4),
            "n_decisive": decisive,
        }
    return {"_meta": {
        "description": "Human matchup matrix (row deck1 vs column deck2). win_rate_smoothed = "
                       "Beta(1,1)-smoothed P(win | decisive game); ties excluded from the rate. "
                       "n_decisive = wins+losses (smoothing confidence).",
        "beta_alpha": BETA_ALPHA,
    }, "decks": sorted({d for d in matrix}), "matrix": {k: dict(v) for k, v in matrix.items()}}


def matchup_spread(matrix, arch) -> float:
    """Std-dev of an archetype's smoothed win rates across non-mirror opponents. High spread =
    polarized matchup profile (strong/weak matchups → exploitable or exploitable-against)."""
    row = matrix["matrix"].get(arch, {})
    wrs = [c["win_rate_smoothed"] for opp, c in row.items() if opp != arch]
    return statistics.pstdev(wrs) if len(wrs) > 1 else 0.0


# ------------------------------------------------------------------------- vulnerability profiles

def build_vulnerability_profiles(archetypes) -> dict:
    """Per archetype, expected deck-slot exposure on each disruption axis, summing expected_copies
    over the mapped+in-pool cards (skeleton + 'other', so play-rate-weighted flex is included)."""
    out = {"_meta": {
        "description": "Static vulnerability profiles from priors + pool card types/abilities. "
                       "Counts are expected copies (Σ expected_copies over the class).",
        "axes": {
            "basic_energy_count": "expected basic-energy copies — Crushing Hammer (any energy) exposure",
            "special_energy_count": "expected special-energy copies — Enhanced/Crushing Hammer "
                                    "exposure; special energies often carry effects/abilities that "
                                    "the removal also strips (e.g. rockets-honchkrow's Rocket energy)",
            "tool_count": "expected Pokémon Tool copies — Jamming Tower exposure",
            "pct_pokemon_with_abilities": "ability-bearing share of expected Pokémon copies — "
                                          "Team Rocket's Watchtower exposure (engine `skills`; "
                                          "Tera passives are NOT counted)",
            "draw_supporter_reliance": "expected draw-Supporter copies — Special Red Card / Judge exposure",
        },
    }, "archetypes": {}}
    for arch, cards in archetypes.items():
        basic_e = special_e = tool = draw_sup = pk_total = pk_abil = 0.0
        for c in cards:
            pc = c["pool_card"]
            if not pc:
                continue
            e = c["expected_copies"]
            if pc.is_basic_energy:
                basic_e += e
            if pc.is_special_energy:
                special_e += e
            if pc.is_tool:
                tool += e
            if pc.is_draw_supporter:
                draw_sup += e
            if pc.is_pokemon:
                pk_total += e
                if pc.has_ability:
                    pk_abil += e
        out["archetypes"][arch] = {
            "basic_energy_count": round(basic_e, 2),
            "special_energy_count": round(special_e, 2),
            "tool_count": round(tool, 2),
            "pct_pokemon_with_abilities": round(pk_abil / pk_total, 3) if pk_total else 0.0,
            "draw_supporter_reliance": round(draw_sup, 2),
            "expected_pokemon_copies": round(pk_total, 2),
        }
    return out


# -------------------------------------------------------------- runtime opponent-model templates

def build_runtime_templates(priors) -> dict:
    """Compact archetype templates for src/ptcg/opponent.py's classifier + zone model, derived
    from the priors: template count = round(expected_copies) (zeros dropped → the *typical* list;
    flex cards played <½ the time fall out), prior = meta_share. Kept small and native-free so it
    bundles with the agent."""
    out = {"_meta": {
        "description": "Runtime archetype templates for the opponent model (src/ptcg/archetypes.py). "
                       "template[card_id] = round(expected_copies) from archetype_priors.json; "
                       "prior = meta_share. Regenerate via scripts/build_priors.py.",
    }, "archetypes": {}}
    for arch, entry in priors["archetypes"].items():
        template = {}
        for cid, c in entry["cards"].items():
            cnt = round(c["expected_copies"])
            if cnt > 0:
                template[cid] = cnt
        out["archetypes"][arch] = {
            "prior": entry["meta_share"],
            "template_size": sum(template.values()),
            "template": template,
        }
    return out


# -------------------------------------------------------------------------------- constructibility

def constructibility(cards) -> dict:
    """Fraction of the skeleton (section=='skeleton') that maps to pool cards, by card count and
    by expected deck slots (copy-weighted — missing a 4-of key card hurts more than a 1-of)."""
    skel = [c for c in cards if c["section"] == "skeleton"]
    mapped = [c for c in skel if c["card_id"]]
    slots_all = sum(c["expected_copies"] for c in skel) or 1.0
    slots_mapped = sum(c["expected_copies"] for c in mapped)
    return {
        "skeleton_cards": len(skel), "mapped_cards": len(mapped),
        "by_cards": len(mapped) / len(skel) if skel else 0.0,
        "by_copies": slots_mapped / slots_all,
        "unmapped": [(c["name"], c["set_code"]) for c in skel if not c["card_id"]],
    }


# -------------------------------------------------------------------------------------- report

def write_report(archetypes, meta_share, matrix, vuln, con, matchup_src):
    lines = ["# Meta-Priors Report", "",
             f"Source: TrainerHill skeletons (`data/external/trainerhill/`) + `{matchup_src}`, "
             "joined onto the competition pool (EN_Card_Data + engine DB).", ""]

    # Mapping coverage
    lines += ["## 1. Card mapping coverage", "",
              "| archetype | skeleton cards | mapped | by-cards | by-copies | flag |",
              "|---|--:|--:|--:|--:|:--:|"]
    ranking = []
    for arch in archetypes:
        c = con[arch]
        share = meta_share.get(arch, {}).get("share", 0.0)
        spread = matchup_spread(matrix, arch)
        composite = c["by_copies"] * share * spread
        ranking.append((arch, composite, c, share, spread))
        flag = "⚠️ <85%" if c["by_copies"] < CONSTRUCTIBILITY_FLAG else "ok"
        lines.append(f"| {arch} | {c['skeleton_cards']} | {c['mapped_cards']} | "
                     f"{c['by_cards']:.0%} | {c['by_copies']:.0%} | {flag} |")

    # Unmapped detail
    lines += ["", "### Unmapped skeleton cards (out of pool)", ""]
    for arch in archetypes:
        u = con[arch]["unmapped"]
        if u:
            lines.append(f"- **{arch}**: " + ", ".join(f"{n} ({sc})" for n, sc in u))
    lines.append("")

    # Ranking
    lines += ["## 2. Archetype ranking — constructibility × meta_share × matchup_spread", "",
              "| rank | archetype | constructibility (copies) | meta_share | matchup_spread | composite |",
              "|--:|---|--:|--:|--:|--:|"]
    for i, (arch, comp, c, share, spread) in enumerate(sorted(ranking, key=lambda x: -x[1]), 1):
        lines.append(f"| {i} | {arch} | {c['by_copies']:.0%} | {share:.1%} | "
                     f"{spread:.3f} | {comp:.4f} |")

    # Vulnerability profiles
    lines += ["", "## 3. Vulnerability profiles (expected copies)", "",
              "| archetype | basic energy | special energy | tools (jamming) | %Pokémon w/ ability (watchtower) | draw supporters (red card/judge) |",
              "|---|--:|--:|--:|--:|--:|"]
    for arch in archetypes:
        v = vuln["archetypes"][arch]
        lines.append(f"| {arch} | {v['basic_energy_count']} | {v['special_energy_count']} | "
                     f"{v['tool_count']} | {v['pct_pokemon_with_abilities']:.0%} | "
                     f"{v['draw_supporter_reliance']} |")

    lines += ["", "## Notes", "",
              "- meta_share is over the full 15-deck field in the matchup CSV; the 8 skeleton "
              "archetypes cover "
              f"{sum(meta_share.get(a, {}).get('share', 0) for a in archetypes):.0%} of field games.",
              "- Ability detection uses the engine `skills` list; Tera passive abilities are not "
              "modelled there and are undercounted.",
              "- Unmapped cards are genuinely out of the competition pool (chiefly the 'CRI' set), "
              "not mapping failures — they lower constructibility as they should."]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------------------ main

def main():
    db = load_card_db()
    archetypes = load_skeletons(db)
    rows, meta_share, matchup_src = load_matchups()

    priors = build_archetype_priors(archetypes, meta_share)
    matrix = build_matchup_matrix(rows)
    vuln = build_vulnerability_profiles(archetypes)
    con = {a: constructibility(cs) for a, cs in archetypes.items()}

    for name, obj in [("archetype_priors.json", priors),
                      ("human_matchup_matrix.json", matrix),
                      ("vulnerability_profiles.json", vuln)]:
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    with open(PTCG_TEMPLATES, "w", encoding="utf-8") as f:
        json.dump(build_runtime_templates(priors), f, indent=2, ensure_ascii=False)

    write_report(archetypes, meta_share, matrix, vuln, con, matchup_src)

    # Console summary (mapping coverage per archetype).
    print("=== card mapping (per archetype) ===")
    tot_rows = tot_mapped = 0
    for arch, cs in archetypes.items():
        m = sum(1 for c in cs if c["card_id"])
        tot_rows += len(cs); tot_mapped += m
        c = con[arch]
        flag = "  <85% FLAG" if c["by_copies"] < CONSTRUCTIBILITY_FLAG else ""
        print(f"  {arch:24} {m:2}/{len(cs):2} rows mapped | skeleton constructibility "
              f"{c['by_copies']:.0%} (copies){flag}")
    print(f"  TOTAL {tot_mapped}/{tot_rows} rows mapped ({tot_mapped/tot_rows:.1%})")
    print(f"\nwrote archetype_priors.json, human_matchup_matrix.json, vulnerability_profiles.json "
          f"to {OUT_DIR}\nwrote runtime templates to {PTCG_TEMPLATES}\nwrote report to {REPORT}")


if __name__ == "__main__":
    main()

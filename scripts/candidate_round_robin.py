#!/usr/bin/env python3
"""Round-robin over the 8 archetype candidate decks + the sample deck, ranked raw and
meta-share-weighted (§5 step 3→4). Writes experiments/candidate_round_robin.{json,md}.

Meta-weighted win rate = Σ_opp winrate(deck, opp) × meta_share(opp) / Σ meta_share(opp), over the
archetype opponents only (the sample deck has no meta share). This is the selection signal: how a
deck fares against the field *as it actually appears*, not a flat average.

Run:  .venv/bin/python scripts/candidate_round_robin.py [games_per_pairing]
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))

from deck_lab.round_robin import round_robin, read_deck  # noqa: E402

PILOT = os.environ.get("PILOT_DIR", "v1")
CAND_DIR = os.path.join(REPO, "experiments", "candidate_decks")
PRIORS = os.path.join(REPO, "src", "priors", "archetype_priors.json")
_suffix = "" if PILOT == "v1" else f"_{PILOT}"  # keep the v1 (#15) record at the bare name
OUT_JSON = os.path.join(REPO, "experiments", f"candidate_round_robin{_suffix}.json")
OUT_MD = os.path.join(REPO, "experiments", f"candidate_round_robin{_suffix}.md")


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    decks = {os.path.splitext(f)[0]: read_deck(os.path.join(CAND_DIR, f))
             for f in sorted(os.listdir(CAND_DIR)) if f.endswith(".csv")}
    decks["sample"] = read_deck(os.path.join(REPO, "submissions", "v1", "deck.csv"))

    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"round-robin: {len(decks)} decks, N={n}/pairing, {workers} workers, pilot={PILOT}", flush=True)
    res = round_robin(decks, games_per_pairing=n, workers=workers, progress_every=50)

    meta = {a: e["meta_share"] for a, e in json.load(open(PRIORS))["archetypes"].items()}
    for r in res["ranking"]:
        d = r["deck"]
        num = den = 0.0
        for opp, share in meta.items():
            if opp == d or opp not in res["matrix"].get(d, {}):
                continue
            num += res["matrix"][d][opp] * share
            den += share
        r["meta_weighted_win_rate"] = round(num / den, 3) if den else None

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    names = [r["deck"] for r in res["ranking"]]
    lines = [f"# Candidate Round-Robin ({n} games/pairing, pilot={PILOT})", "",
             "Decks: 8 archetype candidates (`experiments/candidate_decks/`) + the SDK `sample` "
             f"deck, all piloted by the `{PILOT}` brain. Draws count as half.", "",
             "## Ranking", "",
             "| rank | deck | raw win rate | meta-weighted | W/L/D |", "|--:|---|--:|--:|--:|"]
    for i, r in enumerate(res["ranking"], 1):
        mw = f"{r['meta_weighted_win_rate']:.1%}" if r["meta_weighted_win_rate"] is not None else "—"
        lines.append(f"| {i} | {r['deck']} | {r['win_rate']:.1%} | {mw} | "
                     f"{r['wins']}/{r['losses']}/{r['draws']} |")
    lines += ["", "## Win-rate matrix (row vs column, draws as half)", "",
              "| vs | " + " | ".join(nm[:10] for nm in names) + " |",
              "|---" * (len(names) + 1) + "|"]
    for a in names:
        row = " | ".join(f"{res['matrix'][a].get(b, '—')}" if a != b else "—" for b in names)
        lines.append(f"| {a} | {row} |")
    lines += ["", f"_N={n} per pairing — provisional; coin flips lie at small N (§5 wants ~200). "
              "Confirm the top decks at high N before selecting._"]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"=== candidate round-robin (N={n}) — ranking ===")
    for i, r in enumerate(res["ranking"], 1):
        mw = f"{r['meta_weighted_win_rate']:.1%}" if r["meta_weighted_win_rate"] is not None else "  — "
        print(f"  {i}. {r['deck']:24} raw {r['win_rate']:.1%}  meta-wt {mw}  "
              f"(W/L/D {r['wins']}/{r['losses']}/{r['draws']})")
    print(f"\nwrote {OUT_MD} and {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

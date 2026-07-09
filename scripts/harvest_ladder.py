#!/usr/bin/env python3
"""Harvest Kaggle ladder replays for behavioral cloning (docs §3d follow-up; #34 harvestability).

Pulls episode replays for the TOP teams of the `pokemon-tcg-ai-battle` leaderboard (the strongest
teachers — #40 showed weak teachers can't be distilled) plus all of our own v5 episodes, dedupes by
episode id, and stores them under `data/replays/` (gitignored). Writes an `index.json` mapping each
episode → its two agents (submission id, team name, rating, win reward) so the meta-report + BC-dataset
steps know the winner + opponents without re-deriving them.

Scope note: the leaderboard has thousands of teams ≥650, far too many to pull exhaustively — so we take
the top `--top-teams` by rating (all ≫650) and cap `--max-eps-per-sub` per submission. Reports the
threshold/coverage tradeoff. Resumable: existing replay files are skipped.

    .venv/bin/python scripts/harvest_ladder.py --list-only              # enumerate + report, no download
    .venv/bin/python scripts/harvest_ladder.py --top-teams 20 --max-eps-per-sub 60
"""
import argparse
import csv
import io
import json
import os
import sys
import time
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP = "pokemon-tcg-ai-battle"
OUR_SUB = 54454217          # our v5 submission
OUR_TEAM = "TomTom Taylor"


def _api():
    from kaggle import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def load_leaderboard(api, tmp_dir: str) -> list[dict]:
    """Download + parse the full public leaderboard → rows sorted by score desc."""
    os.makedirs(tmp_dir, exist_ok=True)
    api.competition_leaderboard_download(COMP, tmp_dir, quiet=True)
    zpath = os.path.join(tmp_dir, f"{COMP}.zip")
    with zipfile.ZipFile(zpath) as z:
        data = z.read(z.namelist()[0]).decode("utf-8-sig", "replace")
    rows = list(csv.DictReader(io.StringIO(data)))
    for r in rows:
        try:
            r["_score"] = float(r["Score"])
        except Exception:
            r["_score"] = 0.0
    rows.sort(key=lambda r: -r["_score"])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-rating", type=float, default=650.0)
    ap.add_argument("--top-teams", type=int, default=20, help="harvest this many top teams (all ≥ min-rating)")
    ap.add_argument("--max-eps-per-sub", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(REPO, "data", "replays"))
    ap.add_argument("--sleep", type=float, default=0.25, help="seconds between replay downloads (be polite)")
    ap.add_argument("--list-only", action="store_true", help="enumerate + report volume, download nothing")
    args = ap.parse_args()

    api = _api()
    tmp = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp")
    lb = load_leaderboard(api, tmp)
    n_ge = sum(1 for r in lb if r["_score"] >= args.min_rating)
    print(f"leaderboard: {len(lb)} teams | {n_ge} ≥ {args.min_rating:.0f} | "
          f"taking top {args.top_teams} (scores {lb[0]['_score']:.0f}..{lb[min(args.top_teams, len(lb))-1]['_score']:.0f})")
    if lb[min(args.top_teams, len(lb)) - 1]["_score"] < args.min_rating:
        print(f"  NOTE: top-{args.top_teams} dips below --min-rating; all taken teams are still the strongest.")

    teams = lb[:args.top_teams]
    # rating lookup: submission_id -> team score, via each team's submissions
    sub_rating: dict[int, float] = {}
    sub_team: dict[int, str] = {}
    target_subs: list[tuple[int, str, float]] = []
    for t in teams:
        tid = int(t["TeamId"])
        try:
            subs = api.competition_team_submissions(tid)
        except Exception as e:
            print(f"  team {tid} submissions failed: {type(e).__name__}")
            continue
        best = None
        for s in subs:
            try:
                sc = float(s.public_score)
            except Exception:
                sc = 0.0
            sub_rating[int(s.id)] = sc
            sub_team[int(s.id)] = t["TeamName"]
            if best is None or sc > best[2]:
                best = (int(s.id), t["TeamName"], sc)
        if best:
            target_subs.append(best)
    # always include our v5
    sub_rating.setdefault(OUR_SUB, 553.9)
    sub_team.setdefault(OUR_SUB, OUR_TEAM)
    target_subs.append((OUR_SUB, OUR_TEAM, 553.9))

    # enumerate episodes across target submissions, dedupe by episode id
    index: dict[str, dict] = {}
    for sub_id, tname, sc in target_subs:
        try:
            eps = api.competition_list_episodes(sub_id)
        except Exception as e:
            print(f"  episodes for sub {sub_id} failed: {type(e).__name__}")
            continue
        for e in eps[:args.max_eps_per_sub]:
            eid = str(e.id)
            if eid in index:
                continue
            agents = []
            for a in (e.agents or []):
                asid = int(getattr(a, "submission_id", 0) or 0)
                agents.append({
                    "submission_id": asid,
                    "team_name": getattr(a, "team_name", None) or sub_team.get(asid),
                    "rating": sub_rating.get(asid),
                    "reward": getattr(a, "reward", None),
                })
            index[eid] = {"episode_id": eid, "state": str(getattr(e, "state", "")),
                          "from_sub": sub_id, "agents": agents}

    opponents = {a["team_name"] for m in index.values() for a in m["agents"] if a["team_name"]}
    print(f"enumerated: {len(target_subs)} submissions → {len(index)} unique episodes "
          f"| {len(opponents)} distinct agents/opponents")

    if args.list_only:
        print("(--list-only: nothing downloaded)")
        os.makedirs(args.out, exist_ok=True)
        json.dump({"index": index, "min_rating": args.min_rating, "top_teams": args.top_teams},
                  open(os.path.join(args.out, "index.json"), "w"))
        print(f"wrote index → {os.path.relpath(os.path.join(args.out, 'index.json'), REPO)}")
        return 0

    os.makedirs(args.out, exist_ok=True)
    json.dump({"index": index, "min_rating": args.min_rating, "top_teams": args.top_teams},
              open(os.path.join(args.out, "index.json"), "w"))
    dl = skip = fail = 0
    t0 = time.time()
    for i, eid in enumerate(index, 1):
        dest = os.path.join(args.out, f"episode-{eid}-replay.json")
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            skip += 1
            continue
        try:
            api.competition_episode_replay(int(eid), path=args.out, quiet=True)
            dl += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  replay {eid} failed: {type(e).__name__} {str(e)[:80]}")
        if args.sleep:
            time.sleep(args.sleep)
        if i % 50 == 0:
            print(f"  [{i}/{len(index)}] downloaded {dl}, skipped {skip}, failed {fail} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"\ndone: {dl} downloaded, {skip} already present, {fail} failed → {os.path.relpath(args.out, REPO)}")
    print(f"total replays on disk: {len([f for f in os.listdir(args.out) if f.endswith('-replay.json')])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

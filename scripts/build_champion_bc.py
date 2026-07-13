#!/usr/bin/env python3
"""Champion-BC dataset: clone the TOP-TIER agents piloting our ship deck (Alakazam) — the new framework.

We rank 2790/4904; the leaders are ~1100–1272 and play ALAKAZAM, the SAME deck we ship (recon 2026-07-12).
So the gap is PILOTING. This clones every groundable (observation → action) decision made by a high-rated
(>= --min-rating) Alakazam agent in our replays — WIN OR LOSE — with z = that game's REAL outcome for the
champion seat. Because champions play both won and lost games, z is balanced (NOT the winner-only all-+1 that
poisoned #42's value head), so the value head trains cleanly on strong-player outcomes.

    .venv/bin/python scripts/build_champion_bc.py --min-rating 1000 --archetype alakazam \
        --out experiments/selfplay/champion_bc
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from cg.api import all_card_data, all_attack, to_observation_class          # noqa: E402
from ptcg.perception import build_game_state                                # noqa: E402
from nn import features as F                                                # noqa: E402
from selfplay.recorder import _Buffer, _aux_targets, save_shard, SOURCE_MCTS, _OT_ATTACK  # noqa: E402
from selfplay.ladder_bc import _recover_deck                                # noqa: E402
from mine_archetypes import _marquee, _slug                                 # noqa: E402

_EID = re.compile(r"episode-(\d+)-replay\.json")


def _load_leaderboard():
    """team_name -> current leaderboard score (robust rating source: the harvest clobbers index.json, but
    each replay's info.TeamNames maps straight to the leaderboard). Returns {} if unavailable."""
    try:
        import io as _io, csv as _csv, zipfile as _zip
        os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
        from kaggle import KaggleApi
        api = KaggleApi(); api.authenticate()
        d = os.path.join(REPO, "experiments", "_lb_cache"); os.makedirs(d, exist_ok=True)
        api.competition_leaderboard_download("pokemon-tcg-ai-battle", d)
        zf = [f for f in os.listdir(d) if f.endswith(".zip")]
        raw = (_zip.ZipFile(os.path.join(d, zf[0])).read(_zip.ZipFile(os.path.join(d, zf[0])).namelist()[0]).decode()
               if zf else open(os.path.join(d, [f for f in os.listdir(d) if f.endswith(".csv")][0])).read())
        out = {}
        for r in _csv.DictReader(_io.StringIO(raw)):
            try:
                out[r["TeamName"]] = float(r["Score"])
            except Exception:
                pass
        return out
    except Exception as e:
        print(f"  (leaderboard load failed: {e}; falling back to index.json ratings)")
        return {}


def clone_seat(steps, seat, ci, ai, deck, result):
    """Clone `seat`'s groundable single-select decisions; z from the game `result` (real champion outcome)."""
    ds = F.summarize_deck(deck, ci)
    buf = _Buffer()
    timeline = []
    first_attack = {0: None, 1: None}
    for st in steps:
        for s2 in (0, 1):                       # prize timeline from whichever seat is showing current
            od = st[s2].get("observation") or {}
            cur = od.get("current")
            if cur and cur.get("players"):
                turn = int(cur.get("turn") or 0)
                try:
                    rem0 = len(cur["players"][0].get("prize") or [])
                    rem1 = len(cur["players"][1].get("prize") or [])
                    if not timeline or timeline[-1][0] != turn:
                        timeline.append((turn, rem0, rem1))
                    else:
                        timeline[-1] = (turn, rem0, rem1)
                except Exception:
                    pass
                break
        ae = st[seat]
        if ae.get("status") != "ACTIVE":
            continue
        obs_d = ae.get("observation") or {}
        sel = obs_d.get("select")
        act = ae.get("action")
        if not sel or act is None:
            continue
        opts = sel.get("option") or []
        n = len(opts)
        turn = int((obs_d.get("current") or {}).get("turn") or 0)
        if (isinstance(act, list) and len(act) == 1 and 0 <= act[0] < n and first_attack[seat] is None
                and int((opts[act[0]].get("type") if isinstance(opts[act[0]], dict) else -1) or -1) == _OT_ATTACK):
            first_attack[seat] = turn
        if not (sel.get("minCount") == 1 and sel.get("maxCount") == 1
                and isinstance(act, list) and len(act) == 1 and 0 <= act[0] < n):
            continue
        try:
            obs = to_observation_class(obs_d)
            gs = build_game_state(obs, ci)
            sf = F.encode_state(gs, ci, ai, deck_summary=ds, archetype_id=0)
            af = F.encode_actions(obs.select, gs.my_index, ai)
            buf.add(sf, af, {int(act[0]): 1}, seat, SOURCE_MCTS, turn=turn)
        except Exception:
            continue
    if len(buf) == 0:
        return None
    tup = buf.finalize(result)                  # z = +1 where mover(seat)==result → real outcome
    if tup is None:
        return None
    tup.update(_aux_targets(tup["turn"], tup["mover"], timeline, first_attack))
    return tup


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default=os.path.join(REPO, "data", "replays"))
    ap.add_argument("--out", default=os.path.join(REPO, "experiments", "selfplay", "champion_bc"))
    ap.add_argument("--min-rating", type=float, default=1000.0)
    ap.add_argument("--archetype", default="alakazam")
    ap.add_argument("--team", default=None, help="clone ONLY this team's seat (narrow, consistent teacher)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    lb = _load_leaderboard()
    print(f"  leaderboard ratings loaded: {len(lb)} teams")
    idx = json.load(open(os.path.join(args.replays, "index.json")))["index"] \
        if os.path.exists(os.path.join(args.replays, "index.json")) else {}
    files = sorted(f for f in os.listdir(args.replays) if f.endswith("-replay.json"))
    if args.limit:
        files = files[:args.limit]
    os.makedirs(args.out, exist_ok=True)

    n_seats = total_rows = wins = losses = 0
    ratings = []
    zbal = Counter()
    team_rows = Counter()
    for i, fn in enumerate(files, 1):
        try:
            d = json.load(open(os.path.join(args.replays, fn)))
        except Exception:
            continue
        steps, rew = d.get("steps"), d.get("rewards")
        if not steps or not rew or len(rew) != 2 or rew[0] is None or rew[1] is None or rew[0] == rew[1]:
            continue
        result = 0 if rew[0] > rew[1] else 1
        eid = (_EID.search(fn) or [None, None])[1]
        # rating source: replay info.TeamNames -> leaderboard (robust); fallback index.json by reward
        team_names = (d.get("info") or {}).get("TeamNames") or [None, None]
        agents = (idx.get(eid, {}) or {}).get("agents", [])
        r_by_reward = {a.get("reward"): a.get("rating") for a in agents}
        for seat in (0, 1):
            tname = team_names[seat] if seat < len(team_names) else None
            r = lb.get(tname) or r_by_reward.get(rew[seat])
            if not r or r < args.min_rating:
                continue
            if args.team and tname != args.team:
                continue
            deck = _recover_deck(steps, seat)
            if not deck or _slug(_marquee(deck, ci)) != args.archetype:
                continue
            team_rows[tname] += 1
            try:
                tup = clone_seat(steps, seat, ci, ai, deck, result)
            except Exception:
                tup = None
            if tup is None:
                continue
            save_shard(os.path.join(args.out, f"champ_{eid}_s{seat}.npz"), tup)
            n_seats += 1
            nr = len(tup["z"])
            total_rows += nr
            wins += int(seat == result); losses += int(seat != result)
            zbal["+1"] += int((tup["z"] > 0).sum()); zbal["-1"] += int((tup["z"] < 0).sum())
            ratings.append(r)
        if i % 300 == 0:
            print(f"  [{i}/{len(files)}] {n_seats} champ-seats, {total_rows} rows", flush=True)

    r = np.array(ratings) if ratings else np.array([0.0])
    print(f"\nCHAMPION-BC (>= {args.min_rating:.0f}, {args.archetype}): {n_seats} champion-seat games → "
          f"{total_rows} cloned decisions in {os.path.relpath(args.out, REPO)}")
    print(f"  champion outcomes: {wins}W/{losses}L = {wins/max(wins+losses,1):.0%} "
          f"(→ z balance +1:{zbal['+1']} / -1:{zbal['-1']} = {zbal['+1']/max(total_rows,1):.0%} positive)")
    print(f"  teacher ratings: n={len(ratings)} min={r.min():.0f} median={np.median(r):.0f} "
          f"p90={np.percentile(r,90):.0f} max={r.max():.0f}")
    print(f"  avg decisions/game: {total_rows/max(n_seats,1):.0f}")
    print("  top teacher-teams by champion-game count (for a narrow --team clone):")
    for tn, c in team_rows.most_common(8):
        print(f"    {c:4d}  {tn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Qualitative study: what measurably separates TOP-tier Alakazam play from mid-tier? (#60)

Behavioral cloning is capped (#59) — the top agents' edge is strategy/tempo, not a copyable move distribution.
So MEASURE the difference along engineerable axes, binned by the Alakazam agent's rating:
  (1) DECKLIST  — modal list per band; card-count diffs vs our shipped alakazam-ladder.csv
  (2) SPEED     — first-attack turn, and turn Alakazam (the Stage-2) first hits the board (setup speed)
  (3) TEMPO     — game length (turns), win rate, prizes taken/turn
If the top band attacks a turn earlier / sets up more reliably / runs specific tech, that's an engineerable
edge for the PILOT or the DECK.

    .venv/bin/python scripts/study_top_agents.py
"""
import io
import csv
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from cg.api import all_card_data                                   # noqa: E402
from mine_archetypes import _marquee, _recover, _slug             # noqa: E402
from selfplay.recorder import _OT_ATTACK                          # noqa: E402

REPLAYS = os.path.join(REPO, "data", "replays")
_EID = re.compile(r"episode-(\d+)-replay\.json")
BANDS = [("1150+", 1150, 9999), ("1075-1150", 1075, 1150), ("1000-1075", 1000, 1075), ("<1000", 0, 1000)]


def _load_lb():
    os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
    from kaggle import KaggleApi
    api = KaggleApi(); api.authenticate()
    d = os.path.join(REPO, "experiments", "_lb_cache"); os.makedirs(d, exist_ok=True)
    api.competition_leaderboard_download("pokemon-tcg-ai-battle", d)
    zf = [f for f in os.listdir(d) if f.endswith(".zip")][0]
    raw = zipfile.ZipFile(os.path.join(d, zf)).read(zipfile.ZipFile(os.path.join(d, zf)).namelist()[0]).decode()
    return {r["TeamName"]: float(r["Score"]) for r in csv.DictReader(io.StringIO(raw)) if r.get("Score")}


def _band(r):
    for name, lo, hi in BANDS:
        if lo <= r < hi:
            return name
    return None


def _first_attack_and_setup(steps, seat, ci, alk_ids):
    """Return (first_attack_turn, alakazam_online_turn) for `seat` — None if not observed."""
    fa = ao = None
    for st in steps:
        ae = st[seat]
        obs = ae.get("observation") or {}
        cur = obs.get("current") or {}
        turn = int(cur.get("turn") or 0)
        # Alakazam online: our board (players[yourIndex]) has an Alakazam card id
        yi = cur.get("yourIndex", 0) or 0
        pl = (cur.get("players") or [None, None])
        me = pl[yi] if yi < len(pl) else None
        if ao is None and me:
            board = []
            for slot in (me.get("active") or []):
                if slot: board.append(slot)
            for slot in (me.get("bench") or []):
                if slot: board.append(slot)
            for p in board:
                cid = p.get("cardId") if isinstance(p, dict) else None
                if cid in alk_ids:
                    ao = turn; break
        # first attack: ACTIVE + chose a single ATTACK option
        if fa is None and ae.get("status") == "ACTIVE":
            sel = obs.get("select"); act = ae.get("action")
            if sel and isinstance(act, list) and len(act) == 1:
                opts = sel.get("option") or []
                if 0 <= act[0] < len(opts):
                    o = opts[act[0]]
                    if int((o.get("type") if isinstance(o, dict) else -1) or -1) == _OT_ATTACK:
                        fa = turn
        if fa is not None and ao is not None:
            break
    return fa, ao


def main() -> int:
    ci = {c.cardId: c for c in all_card_data()}
    alk_ids = {cid for cid, c in ci.items() if getattr(c, "name", "").lower().startswith("alakazam")}
    lb = _load_lb()
    print(f"leaderboard: {len(lb)} teams | Alakazam card ids: {sorted(alk_ids)}")

    agg = defaultdict(lambda: {"games": 0, "wins": 0, "len": [], "fa": [], "ao": [], "prizes": [],
                               "decks": Counter()})
    files = [f for f in os.listdir(REPLAYS) if f.endswith("-replay.json")]
    for fn in files:
        try:
            d = json.load(open(os.path.join(REPLAYS, fn)))
        except Exception:
            continue
        steps, rew = d.get("steps"), d.get("rewards")
        if not steps or not rew or len(rew) != 2 or rew[0] is None or rew[1] is None or rew[0] == rew[1]:
            continue
        result = 0 if rew[0] > rew[1] else 1
        tnames = (d.get("info") or {}).get("TeamNames") or [None, None]
        for seat in (0, 1):
            r = lb.get(tnames[seat]) if seat < len(tnames) else None
            band = _band(r) if r else None
            if band is None:
                continue
            deck = _recover(steps, seat)
            if not deck or _slug(_marquee(deck, ci)) != "alakazam":
                continue
            a = agg[band]
            a["games"] += 1
            a["wins"] += int(seat == result)
            # game length + prizes taken (6 - opp final prize)
            last = steps[-1]
            gl = 0; opp_prz = 6
            for st in (steps[-1], steps[-2] if len(steps) > 1 else steps[-1]):
                for s2 in (0, 1):
                    cur = (st[s2].get("observation") or {}).get("current") or {}
                    if cur:
                        gl = max(gl, int(cur.get("turn") or 0))
                        pl = cur.get("players") or []
                        if 1 - seat < len(pl) and pl[1 - seat]:
                            opp_prz = min(opp_prz, len(pl[1 - seat].get("prize") or [6] * 6))
            a["len"].append(gl)
            a["prizes"].append(6 - opp_prz)
            fa, ao = _first_attack_and_setup(steps, seat, ci, alk_ids)
            if fa: a["fa"].append(fa)
            if ao: a["ao"].append(ao)
            a["decks"][tuple(sorted(deck))] += 1

    def med(x): return float(np.median(x)) if x else float("nan")
    print(f"\n{'band':12} {'games':>6} {'win%':>5} {'gLen':>5} {'1stAtk':>6} {'AlkOnline':>9} {'prizes':>6}")
    for name, _lo, _hi in BANDS:
        a = agg[name]
        if not a["games"]:
            continue
        print(f"{name:12} {a['games']:6d} {a['wins']/a['games']:4.0%} {med(a['len']):5.1f} "
              f"{med(a['fa']):6.1f} {med(a['ao']):9.1f} {med(a['prizes']):6.1f}")

    # decklist diff: modal 1150+ list vs our shipped alakazam-ladder.csv
    ours = [int(x) for x in open(os.path.join(REPO, "experiments", "candidate_decks",
                                              "alakazam-ladder.csv")).read().split() if x.strip()]
    top = agg["1150+"]["decks"] or agg["1075-1150"]["decks"]
    if top:
        modal = list(top.most_common(1)[0][0])
        print(f"\ntop-band modal deck (x{top.most_common(1)[0][1]}) vs our alakazam-ladder.csv:")
        co, cm = Counter(ours), Counter(modal)
        diffs = [(ci.get(cid).name if ci.get(cid) else cid, cm.get(cid, 0), co.get(cid, 0))
                 for cid in set(co) | set(cm) if cm.get(cid, 0) != co.get(cid, 0)]
        if not diffs:
            print("  IDENTICAL to our shipped list.")
        for nm, t, o in sorted(diffs, key=lambda x: -(x[1] - x[2])):
            print(f"    {nm:34} top×{t}  ours×{o}  ({'+' if t>o else ''}{t-o})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

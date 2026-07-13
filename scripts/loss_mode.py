#!/usr/bin/env python3
"""#2 loss-mode: WHEN are Alakazam games decided? + extract the top-tier decklist for gating. (#60)

The study (#60) showed top agents win MORE in the SAME ~14 turns — so the edge is in-game, not speed. This
locates WHEN the lead is built: the median prize differential (my prizes taken − opp prizes taken) at turn
milestones, by rating band and by eventual win/loss. If the top band is already ahead by mid-game, the edge
is early/mid decisions; if it converts late, it's closing. Also writes the 1150+ modal deck to
candidate_decks/alakazam-toptier.csv (the engineerable deck-tech from #60).

    .venv/bin/python scripts/loss_mode.py
"""
import io, csv, json, os, re, sys, zipfile
from collections import Counter, defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from cg.api import all_card_data                                   # noqa: E402
from mine_archetypes import _marquee, _recover, _slug             # noqa: E402

REPLAYS = os.path.join(REPO, "data", "replays")
MILE = [4, 6, 8, 10, 12]
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


def _prize_diff_at(steps, seat):
    """Return dict {turn_milestone: my_prizes_taken - opp_prizes_taken} sampling the prize timeline.
    prizes taken = 6 - remaining. Positive = seat is ahead on prizes."""
    # build (turn -> (my_rem, opp_rem)) from whichever obs shows current
    tl = {}
    for st in steps:
        for s2 in (0, 1):
            cur = (st[s2].get("observation") or {}).get("current") or {}
            pl = cur.get("players") or []
            if cur and len(pl) == 2 and pl[0] and pl[1]:
                turn = int(cur.get("turn") or 0)
                myr = len(pl[seat].get("prize") or [])
                opr = len(pl[1 - seat].get("prize") or [])
                tl[turn] = (myr, opr)
                break
    if not tl:
        return {}
    out = {}
    for m in MILE:
        # nearest recorded turn <= m
        ts = [t for t in tl if t <= m]
        if ts:
            myr, opr = tl[max(ts)]
            out[m] = (6 - myr) - (6 - opr)      # my taken - opp taken
    return out


def main() -> int:
    ci = {c.cardId: c for c in all_card_data()}
    lb = _load_lb()
    files = [f for f in os.listdir(REPLAYS) if f.endswith("-replay.json")]

    traj = defaultdict(lambda: defaultdict(list))   # band -> milestone -> [prize_diff]
    traj_wl = defaultdict(lambda: defaultdict(list))  # (band,W/L) -> milestone -> [diff]
    topdecks = Counter()
    for fn in files:
        try:
            d = json.load(open(os.path.join(REPLAYS, fn)))
        except Exception:
            continue
        steps, rew = d.get("steps"), d.get("rewards")
        if not steps or not rew or len(rew) != 2 or rew[0] is None or rew[1] is None or rew[0] == rew[1]:
            continue
        result = 0 if rew[0] > rew[1] else 1
        tn = (d.get("info") or {}).get("TeamNames") or [None, None]
        for seat in (0, 1):
            r = lb.get(tn[seat]) if seat < len(tn) else None
            band = _band(r) if r else None
            if band is None:
                continue
            deck = _recover(steps, seat)
            if not deck or _slug(_marquee(deck, ci)) != "alakazam":
                continue
            if band == "1150+":
                topdecks[tuple(sorted(deck))] += 1
            pdf = _prize_diff_at(steps, seat)
            won = "W" if seat == result else "L"
            for m, v in pdf.items():
                traj[band][m].append(v)
                traj_wl[(band, won)][m].append(v)

    def med(x): return float(np.median(x)) if x else float("nan")
    print("median prize differential (my prizes − opp prizes) at turn milestones, by band:")
    print(f"  {'band':12} " + " ".join(f"t{m:>2}" for m in MILE))
    for name, _lo, _hi in BANDS:
        row = " ".join(f"{med(traj[name][m]):+4.1f}" for m in MILE)
        if traj[name]:
            print(f"  {name:12} {row}")
    print("\nsame, split by eventual Win/Loss (does the top band pull ahead earlier even in wins?):")
    for name, _lo, _hi in BANDS:
        for wl in ("W", "L"):
            if traj_wl[(name, wl)]:
                row = " ".join(f"{med(traj_wl[(name,wl)][m]):+4.1f}" for m in MILE)
                print(f"  {name:12} [{wl}] {row}")

    # write the 1150+ modal deck
    if topdecks:
        modal, cnt = topdecks.most_common(1)[0]
        out = os.path.join(REPO, "experiments", "candidate_decks", "alakazam-toptier.csv")
        with open(out, "w") as f:
            f.write("\n".join(str(x) for x in modal) + "\n")
        print(f"\nwrote top-tier (1150+) modal deck (x{cnt}, {len(modal)} cards) → {os.path.relpath(out, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

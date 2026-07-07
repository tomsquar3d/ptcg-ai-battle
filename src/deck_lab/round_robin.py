"""Round-robin deck tournament harness — Deck Lab, strategy §5 step 3.

Pits candidate decks against each other in the raw SDK, both sides piloted by the current best
agent (the v1 brain), and reports a win-rate matrix + ranking. This is the measurement every
downstream piece (candidate generation, DCS, the seven-axis scores) ultimately has to answer:
does deck X actually beat deck Y? Coin flips lie at small N (§5 wants ~200 games/pairing), so
`games_per_pairing` is the knob that trades wall-clock for confidence.

Raw SDK (not the cabt-env subprocess that selfplay_test.py uses) because we must (a) inject two
*different* decks into one game via `battle_start(deck0, deck1)` and (b) point the brain's
determinization filler at the acting player's deck each turn. First-player advantage is balanced
by alternating which deck sits in seat 0 across a pairing's games; results are tallied by deck
identity, not seat. Draws count as half (ladder scores draws as 0 reward for both — §1).

Run:  .venv/bin/python src/deck_lab/round_robin.py <decks_dir> [games_per_pairing]
      decks_dir holds one <name>.csv per candidate (60 card ids, one per line).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# The piloting brain. Defaults to v1; set PILOT_DIR=v3 (etc.) to compare pilots on the same decks.
PILOT = os.environ.get("PILOT_DIR", "v1")
AGENT_DIR = os.path.join(REPO, "submissions", PILOT)
sys.path.insert(0, AGENT_DIR)

from cg.api import to_observation_class                       # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
import agent as brain                                          # noqa: E402

_LOG_RESULT = 23
_MOVE_CAP = 4000  # hard loop guard per game


def _winner_from_obs(obs):
    """Seat index (0/1) of the winner, 2 for a draw, or None if undetermined."""
    if obs.current is not None and obs.current.result in (0, 1, 2):
        return obs.current.result
    for lg in obs.logs or []:
        if lg.type == _LOG_RESULT and lg.result is not None:
            return lg.result
    return None


def _rand_pick(sel_dict):
    n = len(sel_dict["option"])
    return random.sample(range(n), min(sel_dict["maxCount"], n)) if n else []


def read_deck(path: str) -> list[int]:
    with open(path, newline="", encoding="utf-8") as f:
        return [int(x) for x in f.read().split("\n") if x.strip()]


def play_game(deck0: list[int], deck1: list[int], time_left: int = 600) -> int | None:
    """One game, both seats piloted by the brain. Returns the winning SEAT (0/1), 2=draw, or None.
    The brain's determinization deck is switched to the acting player each turn."""
    decks = (list(deck0), list(deck1))
    obs_dict, _ = battle_start(decks[0], list(decks[1]))
    winner = None
    try:
        for _ in range(_MOVE_CAP):
            obs = to_observation_class(obs_dict)
            if obs.select is None or (obs.current is not None and obs.current.result != -1):
                winner = _winner_from_obs(obs)
                break
            me = obs.current.yourIndex
            brain.set_deck(decks[me])
            try:
                pick = brain.choose(obs_dict, time_left=time_left)
            except Exception:
                pick = _rand_pick(obs_dict["select"])
            obs_dict = battle_select(pick)
    finally:
        battle_finish()
    return winner


def play_pairing(deck_a: list[int], deck_b: list[int], games: int) -> tuple[int, int, int]:
    """(a_wins, b_wins, draws) over `games`, alternating who starts in seat 0 to cancel any
    first-player bias. Tallied by deck identity."""
    aw = bw = dr = 0
    for g in range(games):
        a_seat = g % 2  # 0 → A in seat 0, 1 → B in seat 0
        d0, d1 = (deck_a, deck_b) if a_seat == 0 else (deck_b, deck_a)
        w = play_game(d0, d1)
        if w == 2:
            dr += 1
        elif w is not None:
            a_won = (w == a_seat)
            aw += a_won
            bw += (not a_won)
    return aw, bw, dr


def _play_spec(spec):
    """Worker entry point (module-level so it pickles): play one game, return who won. spec =
    (deck0, deck1, a_name, b_name, a_is_seat0). Real games are ~5 s each, so a round-robin is
    parallelized across processes — the SDK's global battle state forbids concurrency *within* a
    process, but each worker runs its games serially, exactly like the single-process path."""
    d0, d1, a, b, a_seat0 = spec
    w = play_game(list(d0), list(d1))
    return a, b, a_seat0, w


def round_robin(decks: dict[str, list[int]], games_per_pairing: int = 50,
                include_mirror: bool = False, workers: int = 1, progress_every: int = 0) -> dict:
    """Every unordered pair of decks plays `games_per_pairing` games. Returns a structured result
    with a win-rate matrix (draws as half) and a ranking by aggregate win rate. `workers` > 1
    fans games out across processes; `progress_every` prints an ETA line every N games."""
    names = list(decks)
    pairs = list(combinations(names, 2))
    if include_mirror:
        pairs += [(n, n) for n in names]

    specs = []
    for a, b in pairs:
        da, db = tuple(decks[a]), tuple(decks[b])
        for g in range(games_per_pairing):
            a_seat0 = (g % 2 == 0)
            d0, d1 = (da, db) if a_seat0 else (db, da)
            specs.append((d0, d1, a, b, a_seat0))

    tally = {p: [0, 0, 0] for p in pairs}  # (a,b) -> [a_wins, b_wins, draws]
    total = len(specs)
    t0 = time.perf_counter()

    def _record(res, i):
        a, b, a_seat0, w = res
        rec = tally[(a, b)]
        if w == 2:
            rec[2] += 1
        elif w is not None:
            a_won = (w == (0 if a_seat0 else 1))
            rec[0 if a_won else 1] += 1
        if progress_every and (i % progress_every == 0 or i == total):
            el = time.perf_counter() - t0
            eta = el / i * (total - i) if i else 0.0
            print(f"  [{i}/{total} games] {el/60:.1f} min elapsed, ~{eta/60:.1f} min left", flush=True)

    if workers and workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_play_spec, specs, chunksize=1), 1):
                _record(res, i)
    else:
        for i, spec in enumerate(specs, 1):
            _record(_play_spec(spec), i)

    agg = {n: {"wins": 0, "losses": 0, "draws": 0, "games": 0} for n in names}
    matrix = {a: {} for a in names}
    for (a, b), (aw, bw, dr) in tally.items():
        tot = aw + bw + dr or 1
        matrix[a][b] = round((aw + 0.5 * dr) / tot, 3)
        matrix[b][a] = round((bw + 0.5 * dr) / tot, 3)
        for name, w, l in ((a, aw, bw), (b, bw, aw)):
            agg[name]["wins"] += w
            agg[name]["losses"] += l
            agg[name]["draws"] += dr
            agg[name]["games"] += tot

    ranking = []
    for n in names:
        s = agg[n]
        wr = (s["wins"] + 0.5 * s["draws"]) / s["games"] if s["games"] else 0.0
        ranking.append({"deck": n, "win_rate": round(wr, 3), **s})
    ranking.sort(key=lambda r: -r["win_rate"])
    return {"games_per_pairing": games_per_pairing, "matrix": matrix, "ranking": ranking}


def _print(res: dict):
    names = [r["deck"] for r in res["ranking"]]
    print(f"=== round-robin ({res['games_per_pairing']} games/pairing) ===")
    print("\nwin-rate matrix (row vs column, draws as half):")
    w = max(len(n) for n in names)
    print(" " * (w + 2) + " ".join(f"{n[:6]:>6}" for n in names))
    for a in names:
        cells = " ".join(f"{res['matrix'][a].get(b, '  -  '):>6}" if a != b else f"{'  -  ':>6}"
                          for b in names)
        print(f"  {a:<{w}} {cells}")
    print("\nranking:")
    for i, r in enumerate(res["ranking"], 1):
        print(f"  {i}. {r['deck']:<{w}}  win_rate {r['win_rate']:.1%}  "
              f"(W/L/D {r['wins']}/{r['losses']}/{r['draws']} over {r['games']} games)")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: round_robin.py <decks_dir> [games_per_pairing]")
    decks_dir = sys.argv[1]
    gpp = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    decks = {os.path.splitext(f)[0]: read_deck(os.path.join(decks_dir, f))
             for f in sorted(os.listdir(decks_dir)) if f.endswith(".csv")}
    if len(decks) < 2:
        raise SystemExit(f"need >=2 deck CSVs in {decks_dir}, found {len(decks)}")
    workers = max(1, (os.cpu_count() or 2) - 1)
    res = round_robin(decks, games_per_pairing=gpp, workers=workers, progress_every=50)
    _print(res)
    out = os.path.join(decks_dir, "round_robin_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

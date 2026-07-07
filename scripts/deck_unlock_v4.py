"""#17 re-test under a strong pilot: does each meta deck beat `sample` when BOTH sides are piloted
by v4 (MCTS-H)?

#15/#17 found `sample` crushes every constructed meta deck (~80%) under the v1 and v3 heuristics,
and diagnosed it as (maybe) a pilot confound. v4 is now a validated strong pilot (#21). So pit
v4(meta deck) vs v4(sample) for each candidate and compare each deck's win-rate-vs-sample to the
v1/v3 baselines. Gap closes → the decks were mispiloted (MCTS unlocks them); gap holds → the decks
are genuinely weak.

Each side pilots a DIFFERENT deck, and v4 keeps a per-game opponent tracker, so we load TWO
independent v4 instances (one per side) — their trackers must not collide. Reduced MCTS budget for
tractability (still far stronger than the heuristics); parallel across processes.

Run:  .venv/bin/python scripts/deck_unlock_v4.py [N_per_deck] [critical_seconds] [deck_name_filter]
      deck_name_filter (optional): only run candidate decks whose name contains this substring.
"""
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))  # shared cg/ptcg

from cg.api import to_observation_class                       # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

CRIT_S = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "submissions", "v4", "agent.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


V4A = _load("v4_side_a")   # pilots the meta deck
V4B = _load("v4_side_b")   # pilots sample
from search import governor                                    # noqa: E402  (shared by both)
governor.CRITICAL_S = CRIT_S
governor.ROUTINE_S = min(governor.ROUTINE_S, 0.3)
for _b in (V4A, V4B):
    try:
        _b._lazy_init()
    except Exception:
        pass

CAND_DIR = os.path.join(REPO, "experiments", "candidate_decks")
SAMPLE = tuple(int(x) for x in open(os.path.join(REPO, "submissions", "v1", "deck.csv")).read().split() if x.strip())


def _rand(sel):
    n = len(sel["option"])
    return random.sample(range(n), min(sel["maxCount"], n)) if n else []


def _winner(obs):
    if obs.current is not None and obs.current.result in (0, 1, 2):
        return obs.current.result
    for lg in obs.logs or []:
        if lg.type == 23 and lg.result is not None:
            return lg.result
    return None


def play_spec(spec):
    """(deck_name, meta_tuple, meta_seat) → (deck_name, 'meta'|'sample'|'draw'). meta deck in
    meta_seat piloted by V4A; sample in the other seat piloted by V4B."""
    deck_name, meta_t, meta_seat = spec
    meta = list(meta_t)
    V4A.set_deck(meta)
    V4B.set_deck(list(SAMPLE))
    seat0 = meta if meta_seat == 0 else list(SAMPLE)
    seat1 = list(SAMPLE) if meta_seat == 0 else meta
    obs_dict, _ = battle_start(seat0, seat1)
    winner = None
    try:
        for _ in range(4000):
            obs = to_observation_class(obs_dict)
            if obs.select is None or (obs.current is not None and obs.current.result != -1):
                winner = _winner(obs)
                break
            brain = V4A if obs.current.yourIndex == meta_seat else V4B
            try:
                pick = brain.choose(obs_dict, time_left=600)
            except Exception:
                pick = _rand(obs_dict["select"])
            obs_dict = battle_select(pick)
    finally:
        battle_finish()
    if winner == meta_seat:
        return deck_name, "meta"
    if winner in (0, 1):
        return deck_name, "sample"
    return deck_name, "draw"


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    filt = sys.argv[3] if len(sys.argv) > 3 else None
    decks = {os.path.splitext(f)[0]: tuple(int(x) for x in open(os.path.join(CAND_DIR, f)).read().split() if x.strip())
             for f in sorted(os.listdir(CAND_DIR)) if f.endswith(".csv")
             and (filt is None or filt in f)}
    if not decks:
        raise SystemExit(f"no candidate decks match filter {filt!r}")
    specs = [(name, t, g % 2) for name, t in decks.items() for g in range(n)]
    random.shuffle(specs)

    import multiprocessing as mp
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"deck-unlock: {len(specs)} games ({n}/deck vs sample), v4 pilot @ {CRIT_S}s crit, {workers} workers", flush=True)

    tally = {name: Counter() for name in decks}
    t0 = time.perf_counter()
    with mp.Pool(workers) as pool:
        for i, (name, r) in enumerate(pool.imap_unordered(play_spec, specs, chunksize=1), 1):
            tally[name][r] += 1
            if i % 20 == 0 or i == len(specs):
                el = time.perf_counter() - t0
                print(f"  [{i}/{len(specs)}] {el/60:.1f} min, ~{el/i*(len(specs)-i)/60:.1f} min left", flush=True)

    # baselines from #15 (v1 pilot) / #17 (v3 pilot): each deck's win rate vs sample
    base = {}
    for tag, fn in (("v1", "candidate_round_robin.json"), ("v3", "candidate_round_robin_v3.json")):
        try:
            m = json.load(open(os.path.join(REPO, "experiments", fn)))["matrix"]
            base[tag] = {d: m[d].get("sample") for d in m if d != "sample"}
        except Exception:
            base[tag] = {}

    lines = [f"# Deck-unlock re-test — v4 (MCTS-H @ {CRIT_S}s) pilot, {n} games/deck vs sample", "",
             "Each meta deck's win rate vs `sample`, both piloted by v4, vs the v1(#15)/v3(#17) baselines.", "",
             "| deck | vs sample: v1 pilot | v3 pilot | **v4 pilot** | v4 W/L/D |", "|---|--:|--:|--:|--:|"]
    print("\n=== meta deck win rate vs sample (v4 pilot) ===")
    result = {}
    for name, c in tally.items():
        tot = c["meta"] + c["sample"] + c["draw"] or 1
        wr = (c["meta"] + 0.5 * c["draw"]) / tot
        b1 = base.get("v1", {}).get(name)
        b3 = base.get("v3", {}).get(name)
        result[name] = {"v4": round(wr, 3), "v1": b1, "v3": b3, **dict(c)}
        f1 = f"{b1:.0%}" if b1 is not None else "—"
        f3 = f"{b3:.0%}" if b3 is not None else "—"
        lines.append(f"| {name} | {f1} | {f3} | **{wr:.0%}** | {c['meta']}/{c['sample']}/{c['draw']} |")
        print(f"  {name:24} v1 {f1:>4}  v3 {f3:>4}  ->  v4 {wr:.0%}  (W/L/D {c['meta']}/{c['sample']}/{c['draw']})")

    suffix = f"_{filt}_{int(CRIT_S)}s" if filt else ""
    out = os.path.join(REPO, "experiments", f"deck_unlock_v4{suffix}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(REPO, "experiments", f"deck_unlock_v4{suffix}.json"), "w") as f:
        json.dump({"critical_s": CRIT_S, "n": n, "decks": result}, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

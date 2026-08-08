#!/usr/bin/env python3
"""v9 training-corpus generator — the SHIP-PILOT self-play recorder (NOT record_selfplay.py, which
runs heuristic MCTS-H on the sample deck).

Pilots Alakazam with the ship-candidate-policyonly config — net POLICY prior from the policyonly
`ptcgnet.onnx` (56-wide merged graph) + a prize-difference leaf, governed exactly like the ship:
routine ROUTINE_S=0.4 / 563-sim cap (k=1), criticals 5s / K=8 (v6 governor, ph_critical=False) —
against the full field pool (mirror + mega-kangaskhan-ex + cynthia-s-garchomp-ex + marnie-s-grimmsnarl-ex),
both seats. Only the Alakazam side is recorded (Alakazam-native data for BOTH heads: MCTS visit π +
outcome z); in the mirror both sides are the ship so both are kept.

Uses `record_mixed_game` → full schema per decision: features, MCTS visit counts, turn index, mover,
z, deck_summary/archetype_id, aux targets. Shards written continuously (one .npz per game) to a NEW
dir — MUST NOT be merged with the 64-wide sample_dc/alakazam_sp buffers (incompatible width).

Parallelize with MULTIPLE PROCESSES (spawn) — the SDK holds one global battle per process.

  .venv/bin/python scripts/record_v9_corpus.py --workers 2 --games 2      # smoke
  .venv/bin/python scripts/record_v9_corpus.py --workers 9 --games 0      # overnight (0 = unlimited)
"""
import argparse
import os
import sys
import time
import multiprocessing as mp

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))    # cg (native SDK)
sys.path.insert(0, os.path.join(REPO, "src"))                   # ptcg / search / nn / selfplay win

from cg.api import all_card_data, all_attack                                    # noqa: E402
from ptcg.perception import build_game_state                                    # noqa: E402
from ptcg.archetypes import default_archetypes                                  # noqa: E402
from search.determinizer import determinize                                     # noqa: E402
from search import mcts, governor                                              # noqa: E402
from nn import features as F                                                    # noqa: E402
from nn.onnx_guide import OnnxGuide                                            # noqa: E402
from selfplay.recorder import (record_mixed_game, save_shard, MctsPilot,        # noqa: E402
                               _heuristic_multi, _legal_min, SOURCE_MCTS)

# ship config (v6 governor defaults already match; set explicit for clarity)
governor.ROUTINE_S = 0.4
governor.CRITICAL_S = 5.0
governor.CRITICAL_K = 8
ROUTINE_SIMS = 563
MAX_SIMS = 200_000
PH_CRITICAL = False          # ship uses v6's blind governor (no Powerful-Hand escalation)

FIELD = ["alakazam", "mega-kangaskhan-ex", "cynthia-s-garchomp-ex", "marnie-s-grimmsnarl-ex"]
CAND = os.path.join(REPO, "experiments", "candidate_decks")
ONNX_DIR = os.path.join(REPO, "submissions", "policyonly")     # has the 56-wide ptcgnet.onnx (tracked)


def read_deck(slug):
    f = os.path.join(CAND, "alakazam-ladder.csv" if slug == "alakazam" else f"{slug}.csv")
    return [int(x) for x in open(f).read().split() if x.strip()][:60]


class ShipPilot(MctsPilot):
    """Alakazam ship pilot: net policy prior + prize-diff leaf, governed (routine 0.4s/563, crit 5s/K8)."""
    def __init__(self, guide, ci, ai):
        super().__init__(name="ship", prior_fn=guide.priors, value_fn=None, guide=guide)
        self.ci, self.ai = ci, ai

    def _prize_value(self, state, root_player):
        obs = state.observation
        gs = build_game_state(obs, self.ci)
        v = max(-1.0, min(1.0, (gs.opponent.prize_count - gs.me.prize_count) / 6.0))
        cur = getattr(obs, "current", None)
        if cur is not None and getattr(cur, "yourIndex", root_player) != root_player:
            v = -v
        return v

    def act(self, obs, obs_dict, sel, mi, deck, ci, ai, archetypes, tracker,
            *, kind, k, sims, move_time_cap, seed):
        if kind == "multi":
            return _heuristic_multi(obs, mi, deck, ci, ai, archetypes, tracker), None, SOURCE_MCTS
        if kind == "forced":
            return [0], None, SOURCE_MCTS
        try:
            gs = build_game_state(obs, ci)
            budget = governor.allocate(gs, None, ci, ai, ph_critical=PH_CRITICAL)   # time_left None -> 600s
            b_sims = ROUTINE_SIMS if budget.k == 1 else MAX_SIMS                     # routine 563 | critical MAX
            worlds = determinize(obs, deck, ci, k=budget.k, archetypes=archetypes,
                                 opp_tracker=tracker, seed=seed)
            deadline = time.monotonic() + budget.seconds
            pick, stats = mcts.search(obs, worlds, ci, ai, sims=b_sims, deadline=deadline,
                                      prior_fn=self.guide.priors, value_fn=self._prize_value)
            rv = stats.get("root_visits")
            if pick is not None and rv:
                return pick, rv, SOURCE_MCTS
            return (pick if pick is not None else _legal_min(sel)), None, SOURCE_MCTS
        except Exception:
            return (_legal_min(sel) or [0]), None, SOURCE_MCTS


def worker(wid, out_dir, total_games, opp_sims, gcount, rcount, lock, start_ts):
    import random
    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    archs = default_archetypes()
    guide = OnnxGuide(ONNX_DIR, ci, ai)
    assert getattr(guide, "merged", False), "ptcgnet.onnx (merged 56-wide) not loaded"
    assert F.MAX_OPTIONS == 56, f"MAX_OPTIONS={F.MAX_OPTIONS}, expected 56 (are we on v9-dev with #71?)"
    ship = ShipPilot(guide, ci, ai)
    opp = MctsPilot("heuristic")
    alak = read_deck("alakazam")
    field = {m: read_deck(m) for m in FIELD}
    random.seed(1000 + wid)
    c = 0
    while True:
        with lock:
            if total_games > 0 and gcount.value >= total_games:
                break
        matchup = FIELD[(wid + c) % len(FIELD)]
        alak_seat = (wid + c) % 2
        if matchup == "alakazam":                        # mirror: both seats are the ship -> keep both
            pilots = {0: ship, 1: ship}
            decks = {0: alak, 1: alak}
            names = {0: None, 1: None}                   # None -> archetype_id 0 (matches ship set_deck)
        else:
            other = 1 - alak_seat
            pilots = {alak_seat: ship, other: opp}
            decks = {alak_seat: alak, other: field[matchup]}
            names = {alak_seat: None, other: matchup}
        try:
            tup, _st = record_mixed_game(pilots, decks, names, ci, ai, archetypes=archs,
                                         k=1, sims=opp_sims, move_time_cap=6.0, seed=None)
        except Exception:
            tup = None
        rows = 0
        if tup is not None:
            if matchup != "alakazam":                    # keep only the Alakazam-native (ship) rows
                m = tup["mover"] == alak_seat
                tup = {k: v[m] for k, v in tup.items()} if bool(m.any()) else None
            if tup is not None and len(tup["z"]) > 0:
                rows = int(len(tup["z"]))
                # explicit game id per row (unique across workers) — for TD(lambda) trajectory grouping
                gid = os.getpid() * 100_000 + c            # unique across runs AND workers (no collision)
                tup["game_id"] = np.full(rows, gid, dtype=np.int64)
                save_shard(os.path.join(out_dir, f"w{wid}_p{os.getpid()}_{c:06d}.npz"), tup)
        c += 1
        with lock:
            gcount.value += 1
            rcount.value += rows
            g, r = gcount.value, rcount.value
        if g % 25 == 0:
            el = max(time.time() - start_ts, 1e-9)
            print(f"[v9] games={g} rows={r} {g/el*3600:.0f} games/hr rows/hr={r/el*3600:.0f} "
                  f"({el:.0f}s elapsed)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--games", type=int, default=0, help="total games across workers; 0 = unlimited")
    ap.add_argument("--opp-sims", type=int, default=300, help="field opponent heuristic MCTS-H sims")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "experiments", "selfplay", "v9_corpus"))
    args = ap.parse_args()
    assert F.MAX_OPTIONS == 56, f"MAX_OPTIONS={F.MAX_OPTIONS}, expected 56 (v9-dev/#71 not checked out?)"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"v9 corpus: {args.workers} workers, games={'unlimited' if args.games==0 else args.games}, "
          f"MAX_OPTIONS={F.MAX_OPTIONS}, out={os.path.relpath(args.out_dir, REPO)}, "
          f"ship pilot=net policy+prize leaf (routine 0.4s/563, crit 5s/K8), opp=heuristic@{args.opp_sims}",
          flush=True)
    ctx = mp.get_context("spawn")               # each worker gets its OWN cg battle state
    gcount = ctx.Value("l", 0)
    rcount = ctx.Value("l", 0)
    lock = ctx.Lock()
    start = time.time()
    procs = [ctx.Process(target=worker, args=(w, args.out_dir, args.games, args.opp_sims,
                                              gcount, rcount, lock, start))
             for w in range(args.workers)]
    for p in procs:
        p.start()
        time.sleep(4)     # stagger: each worker loads onnxruntime + card DB; simultaneous spawn spikes RAM
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
    el = max(time.time() - start, 1e-9)
    print(f"\nDONE: games={gcount.value} rows={rcount.value} in {el:.0f}s "
          f"({gcount.value/el*3600:.0f} games/hr, {rcount.value/el*3600:.0f} rows/hr)", flush=True)


if __name__ == "__main__":
    main()

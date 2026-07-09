"""Load self-play `.npz` shards into training batches for `PTCGNet` (docs §3).

A shard (`selfplay.recorder.save_shard`) holds one game's recorded decisions: the featurized state
and action arrays, the MCTS visit counts (`visits`, width `MAX_OPTIONS+1` — last slot = PASS), and
the per-decision outcome `z`. This module concatenates shards into one in-memory buffer (a few games
× ~100 decisions fits trivially), derives the **policy target π** from the visit counts, and slices
batches whose dict keys match `PTCGNet.forward`.

Policy target: π is the visit distribution renormalized over the *real* option slots (0..63). The
trailing PASS slot is dropped from the 64-wide pointer head — its mass is reported by `policy_targets`
so we know whether it's negligible (it was ~a few % in early buffers); if it ever grows, the head
gets an explicit PASS logit. Rows whose visits are *entirely* PASS become an all-zero target and
contribute no policy gradient (value still trains on them).

Split by GAME, not by decision — decisions within a game are correlated, so a decision-level split
leaks. Training-only (imports torch); the deployed agent never imports this.
"""
from __future__ import annotations

import glob
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from . import features as F

# state/action arrays consumed by PTCGNet.forward (stadium_id handled separately — it's 1-D)
_STATE_STACK = ("pokemon_ids", "pokemon_scalars", "pokemon_mask", "hand_ids", "hand_mask", "globals")
_ACTION_STACK = ("option_card_ids", "option_scalars", "option_mask")


@dataclass
class Buffer:
    data: dict            # key -> np.ndarray [N, ...]
    pi: np.ndarray        # float32 [N, MAX_OPTIONS]  policy target (real-option visit distribution)
    game_id: np.ndarray   # int32 [N]  shard index each decision came from
    n_games: int
    dropped_pass_mass: float   # fraction of all visit mass that sat in the PASS slot (dropped)


def policy_targets(visits: np.ndarray) -> tuple[np.ndarray, float]:
    """visits [N, MAX_OPTIONS+1] → (π [N, MAX_OPTIONS] over real options, dropped PASS-mass fraction)."""
    real = visits[:, :F.MAX_OPTIONS].astype(np.float32)
    pass_v = visits[:, F.MAX_OPTIONS].astype(np.float32)
    denom = real.sum(1, keepdims=True)
    pi = np.divide(real, denom, out=np.zeros_like(real), where=denom > 0)
    total = float(real.sum() + pass_v.sum())
    dropped = float(pass_v.sum()) / total if total > 0 else 0.0
    return pi, dropped


def load_buffer(buffer_dir: str) -> Buffer:
    """Concatenate every `.npz` shard in `buffer_dir` into one training buffer."""
    paths = sorted(glob.glob(os.path.join(buffer_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"no .npz shards in {buffer_dir}")
    cols: dict[str, list] = defaultdict(list)
    gid: list[np.ndarray] = []
    for i, p in enumerate(paths):
        d = np.load(p)
        m = int(d["z"].shape[0])
        if m == 0:
            continue
        for k in d.files:
            cols[k].append(d[k])
        gid.append(np.full(m, i, dtype=np.int32))
    data = {k: np.concatenate(v) for k, v in cols.items()}
    pi, dropped = policy_targets(data["visits"])
    return Buffer(data=data, pi=pi, game_id=np.concatenate(gid),
                  n_games=len(paths), dropped_pass_mass=dropped)


def merge_buffers(*bufs: Buffer) -> Buffer:
    """Concatenate buffers on their COMMON keys (drops loader-only extras like ladder `rating`/
    `deck_key`), offsetting game ids so games stay distinct for the GAME-level split. π is recomputed
    from the merged visit counts. Used to mix sample self-play + ladder-BC (docs §3d follow-up)."""
    bufs = [b for b in bufs if b is not None and len(b.game_id)]
    if len(bufs) == 1:
        return bufs[0]
    common = set(bufs[0].data)
    for b in bufs[1:]:
        common &= set(b.data)
    data = {k: np.concatenate([b.data[k] for b in bufs]) for k in common}
    gids, off = [], 0
    for b in bufs:
        gids.append(b.game_id + off)
        off += b.n_games
    game_id = np.concatenate(gids)
    pi, dropped = policy_targets(data["visits"])
    return Buffer(data=data, pi=pi, game_id=game_id,
                  n_games=sum(b.n_games for b in bufs), dropped_pass_mass=dropped)


def split_by_game(buf: Buffer, val_frac: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) row indices, holding out whole games for validation."""
    rng = np.random.default_rng(seed)
    games = np.unique(buf.game_id)
    rng.shuffle(games)
    n_val = max(1, int(round(len(games) * val_frac))) if len(games) > 1 else 0
    val_games = set(games[:n_val].tolist())
    is_val = np.array([g in val_games for g in buf.game_id])
    all_idx = np.arange(len(buf.game_id))
    return all_idx[~is_val], all_idx[is_val]


def make_batch(buf: Buffer, idx: np.ndarray, device: str = "cpu") -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Slice rows `idx` into (batch dict for PTCGNet.forward, π target, z target). v1-schema keys only
    (deck-conditioning + aux targets are fetched separately via `deck_cond_batch`/`aux_targets` so the
    legacy trainer/net keep working unchanged)."""
    d = buf.data
    batch = {
        "pokemon_ids": torch.from_numpy(d["pokemon_ids"][idx]).long(),
        "pokemon_scalars": torch.from_numpy(d["pokemon_scalars"][idx]).float(),
        "pokemon_mask": torch.from_numpy(d["pokemon_mask"][idx]).bool(),
        "hand_ids": torch.from_numpy(d["hand_ids"][idx]).long(),
        "hand_mask": torch.from_numpy(d["hand_mask"][idx]).bool(),
        "stadium_id": torch.from_numpy(d["stadium_id"][idx]).long(),
        "globals": torch.from_numpy(d["globals"][idx]).float(),
        "option_card_ids": torch.from_numpy(d["option_card_ids"][idx]).long(),
        "option_scalars": torch.from_numpy(d["option_scalars"][idx]).float(),
        "option_mask": torch.from_numpy(d["option_mask"][idx]).bool(),
    }
    batch = {k: v.to(device) for k, v in batch.items()}
    pi = torch.from_numpy(buf.pi[idx]).float().to(device)
    z = torch.from_numpy(d["z"][idx]).float().to(device)
    return batch, pi, z


def has_deck_cond(buf: Buffer) -> bool:
    return "deck_summary" in buf.data


def has_aux(buf: Buffer) -> bool:
    return "prize_diff_t3" in buf.data


def deck_cond_batch(buf: Buffer, idx: np.ndarray, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Schema-v2 deck-conditioning inputs for the v2 net: (deck_summary [B, DECK_SUMMARY_DIM],
    archetype_id [B]). Zeros/0 for a buffer recorded before the v2 schema (docs §3d)."""
    d = buf.data
    n = len(idx)
    if "deck_summary" in d:
        ds = torch.from_numpy(d["deck_summary"][idx]).float()
        aid = torch.from_numpy(d["archetype_id"][idx]).long()
    else:
        ds = torch.zeros(n, F.DECK_SUMMARY_DIM, dtype=torch.float32)
        aid = torch.zeros(n, dtype=torch.long)
    return ds.to(device), aid.to(device)


def aux_targets(buf: Buffer, idx: np.ndarray, device: str = "cpu") -> dict:
    """Auxiliary value targets + provenance for a batch (docs §3d step 4):
      prize_diff_t3 [B] ∈ [-1,1], turns_to_attack [B] ∈ [0,1], source [B] (0=MCTS,1=BC),
      is_bc [B] bool. Missing keys (legacy buffer) default to zeros so callers can `.get`-style guard
      on `has_aux(buf)`."""
    d = buf.data
    n = len(idx)

    def col(k):
        arr = d[k][idx] if k in d else np.zeros(n, dtype=np.float32)
        return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device)

    src = torch.from_numpy(np.asarray(d["source"][idx] if "source" in d else np.zeros(n),
                                      dtype=np.int64)).to(device)
    return {
        "prize_diff_t3": col("prize_diff_t3"),
        "turns_to_attack": col("turns_to_attack"),
        "source": src,
        "is_bc": src.eq(1),
    }

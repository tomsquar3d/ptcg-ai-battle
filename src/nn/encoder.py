"""Encoder + policy/value heads — network Layers 3-4 (docs §3).

Consumes `features.StateFeatures` / `features.ActionFeatures` (numpy) and produces a per-option
policy logit vector + a scalar value. Architecture (kept small — single-digit-M params so ONNX-CPU
inference is <10 ms):

  card embedding table (shared by board, hand, stadium, and options)
    → per-Pokémon tokens (embed ⊕ scalars) → 1 self-attention layer → masked attention-pool = board
    → hand tokens → masked attention-pool = hand
    → globals MLP
    → state MLP over [board, hand, stadium-embed, globals] = state vector
  policy (pointer): per-option token (embed ⊕ option scalars) scored against the state vector,
    padded options masked to −inf.
  value: MLP(state) → tanh ∈ (−1, 1) — matches the MCTS leaf-value scale.

Priors/value plug into `search.mcts` later (its prior_fn hook already exists); trained by
self-play (MCTS visit counts = policy target, game outcome = value target).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from . import features as F


@dataclass
class NetConfig:
    embed_dim: int = 64
    d_model: int = 128
    g_dim: int = 64
    state_dim: int = 256
    n_heads: int = 4
    dropout: float = 0.1


class _AttnPool(nn.Module):
    """Pool a masked set of tokens with a learned query."""
    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * d ** -0.5)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # tokens [B,N,d], mask [B,N] (True = valid)
        scores = (tokens @ self.q).masked_fill(~mask, -1e9)     # [B,N]
        w = scores.softmax(-1).unsqueeze(-1)                    # [B,N,1]
        return (w * tokens).sum(1)                              # [B,d]


class PTCGNet(nn.Module):
    def __init__(self, cfg: NetConfig = NetConfig()):
        super().__init__()
        self.cfg = cfg
        self.card_embed = nn.Embedding(F.VOCAB_SIZE, cfg.embed_dim, padding_idx=F.PAD_ID)

        self.pk_in = nn.Linear(cfg.embed_dim + F.POKEMON_SCALAR_DIM, cfg.d_model)
        self.pk_attn = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, dim_feedforward=cfg.d_model * 2,
            dropout=cfg.dropout, batch_first=True)
        self.board_pool = _AttnPool(cfg.d_model)

        self.hand_in = nn.Linear(cfg.embed_dim, cfg.d_model)
        self.hand_pool = _AttnPool(cfg.d_model)

        self.global_mlp = nn.Sequential(
            nn.Linear(F.GLOBAL_DIM, cfg.g_dim), nn.ReLU(), nn.Linear(cfg.g_dim, cfg.g_dim), nn.ReLU())

        self.state_mlp = nn.Sequential(
            nn.Linear(cfg.d_model * 2 + cfg.embed_dim + cfg.g_dim, cfg.state_dim), nn.ReLU(),
            nn.Linear(cfg.state_dim, cfg.state_dim), nn.ReLU())

        self.opt_in = nn.Linear(cfg.embed_dim + F.OPTION_SCALAR_DIM, cfg.d_model)
        self.policy_scorer = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.state_dim, cfg.d_model), nn.ReLU(),
            nn.Linear(cfg.d_model, 1))
        self.value_head = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.d_model), nn.ReLU(), nn.Linear(cfg.d_model, 1))

    def encode_state(self, b: dict) -> torch.Tensor:
        pk = torch.cat([self.card_embed(b["pokemon_ids"]), b["pokemon_scalars"]], dim=-1)
        pk = self.pk_in(pk)                                          # [B,12,d]
        # A fully-padded board (no Pokémon either side — a degenerate setup state) would make the
        # transformer softmax over zero keys → NaN. Unmask slot 0 for those rows so attention is
        # defined; board_pool re-masks by pokemon_mask, so the fabricated token is never pooled in.
        pad = ~b["pokemon_mask"]
        all_pad = pad.all(dim=1)
        if all_pad.any():
            pad = pad.clone()
            pad[all_pad, 0] = False
        pk = self.pk_attn(pk, src_key_padding_mask=pad)
        board = self.board_pool(pk, b["pokemon_mask"])              # [B,d]

        hand = self.hand_in(self.card_embed(b["hand_ids"]))
        hand = self.hand_pool(hand, b["hand_mask"])                 # [B,d]

        stad = self.card_embed(b["stadium_id"])                     # [B,embed]
        glob = self.global_mlp(b["globals"])                        # [B,g]
        return self.state_mlp(torch.cat([board, hand, stad, glob], dim=-1))  # [B,state]

    def forward(self, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.encode_state(b)                               # [B,state]
        opt = self.opt_in(torch.cat([self.card_embed(b["option_card_ids"]), b["option_scalars"]], -1))
        st = state.unsqueeze(1).expand(-1, opt.shape[1], -1)       # [B,O,state]
        logits = self.policy_scorer(torch.cat([opt, st], -1)).squeeze(-1)   # [B,O]
        logits = logits.masked_fill(~b["option_mask"], float("-inf"))
        value = torch.tanh(self.value_head(state)).squeeze(-1)     # [B]
        return logits, value

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def collate(states: Sequence[F.StateFeatures], actions: Sequence[F.ActionFeatures],
            device: str = "cpu") -> dict:
    """Stack featurized (state, action) pairs into a batch of tensors for `PTCGNet.forward`."""
    def s(attr):
        return torch.from_numpy(np.stack([getattr(x, attr) for x in states])).to(device)

    def a(attr):
        return torch.from_numpy(np.stack([getattr(x, attr) for x in actions])).to(device)

    return {
        "pokemon_ids": s("pokemon_ids").long(),
        "pokemon_scalars": s("pokemon_scalars").float(),
        "pokemon_mask": s("pokemon_mask").bool(),
        "hand_ids": s("hand_ids").long(),
        "hand_mask": s("hand_mask").bool(),
        "stadium_id": torch.tensor([int(x.stadium_id) for x in states], device=device).long(),
        "globals": s("globals").float(),
        "option_card_ids": a("option_card_ids").long(),
        "option_scalars": a("option_scalars").float(),
        "option_mask": a("option_mask").bool(),
    }

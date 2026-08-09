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
    # --- schema-v2 extensions (docs §3d); all default OFF → byte-identical to the shipped v5 arch ---
    deck_cond: bool = False      # fold own-deck summary + archetype embedding into the state vector
    aux_heads: bool = False      # add prize-diff@t+3 + turns-to-first-attack auxiliary value heads
    deck_dim: int = 32           # deck_summary MLP width
    arch_dim: int = 16           # archetype embedding width
    # --- v9 extensions; defaults keep the v5/v6 arch and state_dict keys byte-identical -------------
    n_attn_layers: int = 1       # depth of the board self-attention stack
    activation: str = "relu"     # "relu" (v5/v6) | "gelu" (v9) — used by every MLP and the attn FFN
    norm_first: bool = False     # pre-norm transformer blocks (+ a final LayerNorm on the stack)


def v9_config(**overrides) -> NetConfig:
    """The v9 preset — sized for a ~300k-row corpus, not the 1M originally assumed.

    Wider tokens (embed 64→128) and a deeper, pre-norm board encoder (1→2 layers, d_model 128→192,
    GELU) with a wider state trunk (256→384). Deck conditioning + aux heads on.
    """
    cfg = dict(embed_dim=128, d_model=192, state_dim=384, n_heads=4, n_attn_layers=2,
               activation="gelu", norm_first=True, deck_cond=True, aux_heads=True)
    cfg.update(overrides)
    return NetConfig(**cfg)


def _act(cfg: NetConfig) -> nn.Module:
    return nn.GELU() if cfg.activation == "gelu" else nn.ReLU()


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

        def _attn_layer():
            return nn.TransformerEncoderLayer(
                cfg.d_model, cfg.n_heads, dim_feedforward=cfg.d_model * 2,
                dropout=cfg.dropout, batch_first=True,
                activation=cfg.activation, norm_first=cfg.norm_first)

        # Layer 1 stays `pk_attn` (not a ModuleList) so v5/v6 checkpoints keep loading by key;
        # v9's extra depth lives in `pk_attn_rest`, which those checkpoints simply don't have.
        self.pk_attn = _attn_layer()
        self.pk_attn_rest = nn.ModuleList(_attn_layer() for _ in range(max(0, cfg.n_attn_layers - 1)))
        # pre-norm stacks need a final LayerNorm; post-norm (v5/v6) already ends normalized
        self.pk_norm = nn.LayerNorm(cfg.d_model) if cfg.norm_first else nn.Identity()
        self.board_pool = _AttnPool(cfg.d_model)

        self.hand_in = nn.Linear(cfg.embed_dim, cfg.d_model)
        self.hand_pool = _AttnPool(cfg.d_model)

        self.global_mlp = nn.Sequential(
            nn.Linear(F.GLOBAL_DIM, cfg.g_dim), _act(cfg), nn.Linear(cfg.g_dim, cfg.g_dim), _act(cfg))

        # schema-v2 deck conditioning: own-deck summary MLP + archetype embedding → widen state input
        state_in = cfg.d_model * 2 + cfg.embed_dim + cfg.g_dim
        if cfg.deck_cond:
            self.deck_mlp = nn.Sequential(
                nn.Linear(F.DECK_SUMMARY_DIM, cfg.deck_dim), _act(cfg),
                nn.Linear(cfg.deck_dim, cfg.deck_dim), _act(cfg))
            self.arch_embed = nn.Embedding(F.ARCHETYPE_VOCAB, cfg.arch_dim)
            state_in += cfg.deck_dim + cfg.arch_dim

        self.state_mlp = nn.Sequential(
            nn.Linear(state_in, cfg.state_dim), _act(cfg),
            nn.Linear(cfg.state_dim, cfg.state_dim), _act(cfg))

        self.opt_in = nn.Linear(cfg.embed_dim + F.OPTION_SCALAR_DIM, cfg.d_model)
        self.policy_scorer = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.state_dim, cfg.d_model), _act(cfg),
            nn.Linear(cfg.d_model, 1))
        self.value_head = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.d_model), _act(cfg), nn.Linear(cfg.d_model, 1))
        # auxiliary value heads (training-only; inference reads only value_head/policy_scorer)
        if cfg.aux_heads:
            self.prize_head = nn.Sequential(
                nn.Linear(cfg.state_dim, cfg.d_model), _act(cfg), nn.Linear(cfg.d_model, 1))
            self.tta_head = nn.Sequential(
                nn.Linear(cfg.state_dim, cfg.d_model), _act(cfg), nn.Linear(cfg.d_model, 1))

    def encode_state(self, b: dict) -> torch.Tensor:
        pk = torch.cat([self.card_embed(b["pokemon_ids"]), b["pokemon_scalars"]], dim=-1)
        pk = self.pk_in(pk)                                          # [B,12,d]
        # A fully-padded board (no Pokémon either side — a degenerate setup state) would make the
        # transformer softmax over zero keys → NaN. Unmask slot 0 for those rows so attention is
        # defined; board_pool re-masks by pokemon_mask, so the fabricated token is never pooled in.
        # Branch-free (no data-dependent `if`) so the guard survives ONNX tracing/export.
        pad = ~b["pokemon_mask"]                               # [B,N] True = padded
        all_pad = pad.all(dim=1, keepdim=True)                 # [B,1] row entirely padded
        is_col0 = torch.arange(pad.shape[1], device=pad.device) == 0   # [N] True only at slot 0
        pad = pad & ~(all_pad & is_col0.unsqueeze(0))          # unmask slot 0 on all-padded rows
        pk = self.pk_attn(pk, src_key_padding_mask=pad)
        for layer in self.pk_attn_rest:                             # v9 depth (empty for v5/v6)
            pk = layer(pk, src_key_padding_mask=pad)
        pk = self.pk_norm(pk)
        board = self.board_pool(pk, b["pokemon_mask"])              # [B,d]

        hand = self.hand_in(self.card_embed(b["hand_ids"]))
        hand = self.hand_pool(hand, b["hand_mask"])                 # [B,d]

        stad = self.card_embed(b["stadium_id"])                     # [B,embed]
        glob = self.global_mlp(b["globals"])                        # [B,g]
        parts = [board, hand, stad, glob]
        if self.cfg.deck_cond:                                      # schema-v2 deck conditioning
            parts.append(self.deck_mlp(b["deck_summary"]))          # [B,deck_dim]
            parts.append(self.arch_embed(b["archetype_id"]))        # [B,arch_dim]
        return self.state_mlp(torch.cat(parts, dim=-1))            # [B,state]

    def _policy(self, state: torch.Tensor, b: dict) -> torch.Tensor:
        opt = self.opt_in(torch.cat([self.card_embed(b["option_card_ids"]), b["option_scalars"]], -1))
        st = state.unsqueeze(1).expand(-1, opt.shape[1], -1)       # [B,O,state]
        logits = self.policy_scorer(torch.cat([opt, st], -1)).squeeze(-1)   # [B,O]
        return logits.masked_fill(~b["option_mask"], float("-inf"))

    def forward(self, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.encode_state(b)                               # [B,state]
        logits = self._policy(state, b)
        value = torch.tanh(self.value_head(state)).squeeze(-1)     # [B]
        return logits, value

    def forward_train(self, b: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Training forward: (policy logits, value, aux). `aux` carries the schema-v2 auxiliary heads
        (empty dict when `cfg.aux_heads` is off). Inference uses `forward` and never touches aux."""
        state = self.encode_state(b)
        logits = self._policy(state, b)
        value = torch.tanh(self.value_head(state)).squeeze(-1)
        aux = {}
        if self.cfg.aux_heads:
            aux["prize_diff_t3"] = torch.tanh(self.prize_head(state)).squeeze(-1)      # [B] ∈ (-1,1)
            aux["turns_to_attack"] = torch.sigmoid(self.tta_head(state)).squeeze(-1)   # [B] ∈ (0,1)
        return logits, value, aux

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

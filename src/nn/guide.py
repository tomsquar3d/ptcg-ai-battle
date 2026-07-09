"""Neural guide for `search.mcts` — wraps a trained `PTCGNet` as the MCTS prior + leaf value (docs §3).

`search.mcts.search` exposes two hooks: `prior_fn(node) -> {action: prob}` (which moves to explore)
and `value_fn(state, root_player) -> (-1, 1)` (how good is this leaf). `NetGuide` implements both
from one network — the AlphaZero payoff: the learned policy focuses the search and the learned value
replaces the hand-written evaluator at the leaves.

Perspective: `encode_state` is always built from the acting player's view (`build_game_state` sets
`my_index = yourIndex`), and the value head was trained on `z` from that same acting-player view, so
`value()` flips the sign when the acting player isn't the search root — matching the heuristic
`_value` it stands in for.

PASS and any options beyond `MAX_OPTIONS` (rare — the pointer head is 64 wide) carry no network
logit, so they get a small exploratory floor rather than zero prior. Dev-time (torch); deploy will
swap this for an ONNX session with the same two methods.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import torch

from ptcg.perception import build_game_state
from . import features as F
from .encoder import PTCGNet, NetConfig, collate

# the PASS pseudo-action id used by search.mcts (import lazily to avoid any import-order surprises)
try:
    from search.mcts import _PASS
except Exception:  # pragma: no cover - fallback if import path differs
    _PASS = -1

_FLOOR_FRAC = 0.25   # exploratory prior for unmodeled actions (PASS / >MAX_OPTIONS), as a fraction
                     # of the mean modeled-option prior


class NetGuide:
    def __init__(self, ckpt_path: str, card_index: Mapping, attack_index: Mapping,
                 device: str = "cpu"):
        self.ci = card_index
        self.ai = attack_index
        self.device = device
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = NetConfig(**ck.get("config", {})) if isinstance(ck.get("config"), dict) else NetConfig()
        self.net = PTCGNet(cfg).to(device).eval()
        self.net.load_state_dict(ck["model"])
        self.meta = {k: ck.get(k) for k in ("epoch", "val")}
        # schema-v2 deck conditioning: the net wants the piloted deck's summary + archetype id.
        # Set via set_deck(); until then it degrades to the "unknown deck" (zeros/0) row.
        self.deck_cond = bool(getattr(cfg, "deck_cond", False))
        self._ds = np.zeros(F.DECK_SUMMARY_DIM, dtype=np.float32)
        self._aid = 0

    def set_deck(self, deck, archetype_name: Optional[str] = None) -> None:
        """Provide the deck the net is piloting (deck-conditioned v2 nets). No-op for v1 nets."""
        if self.deck_cond:
            self._ds = F.summarize_deck(list(deck), self.ci)
            self._aid = F.archetype_id_for(archetype_name)

    def _deck_keys(self) -> dict:
        if not self.deck_cond:
            return {}
        return {
            "deck_summary": torch.from_numpy(self._ds).float().unsqueeze(0).to(self.device),
            "archetype_id": torch.tensor([int(self._aid)], device=self.device).long(),
        }

    # --- state featurization ---------------------------------------------------------
    def _state_batch(self, sf: F.StateFeatures) -> dict:
        t = lambda a, f: f(torch.from_numpy(a)).unsqueeze(0).to(self.device)
        b = {
            "pokemon_ids": t(sf.pokemon_ids, lambda x: x.long()),
            "pokemon_scalars": t(sf.pokemon_scalars, lambda x: x.float()),
            "pokemon_mask": t(sf.pokemon_mask, lambda x: x.bool()),
            "hand_ids": t(sf.hand_ids, lambda x: x.long()),
            "hand_mask": t(sf.hand_mask, lambda x: x.bool()),
            "stadium_id": torch.tensor([int(sf.stadium_id)], device=self.device).long(),
            "globals": t(sf.globals, lambda x: x.float()),
        }
        b.update(self._deck_keys())
        return b

    # --- MCTS hooks ------------------------------------------------------------------
    @torch.no_grad()
    def value(self, state: Any, root_player: int) -> float:
        """Leaf value in (-1, 1) from the ROOT player's perspective (value head, sign-flipped)."""
        obs = state.observation
        gs = build_game_state(obs, self.ci)
        sf = F.encode_state(gs, self.ci, self.ai)
        sv = self.net.encode_state(self._state_batch(sf))
        v = float(torch.tanh(self.net.value_head(sv)).item())
        cur = getattr(obs, "current", None)
        if cur is not None and getattr(cur, "yourIndex", root_player) != root_player:
            v = -v
        return v

    @torch.no_grad()
    def priors(self, node: Any) -> dict:
        """Policy prior over `node.actions` from the pointer head (softmax over legal options)."""
        obs = node.state.observation
        gs = build_game_state(obs, self.ci)
        sf = F.encode_state(gs, self.ci, self.ai)
        af = F.encode_actions(obs.select, gs.my_index, self.ai)
        batch = collate([sf], [af], self.device)
        batch.update(self._deck_keys())                   # v2 deck conditioning (no-op for v1 nets)
        logits, _ = self.net(batch)                       # [1, MAX_OPTIONS], padded → -inf
        probs = torch.softmax(logits[0], dim=0).cpu().numpy()

        actions = list(node.actions)
        modeled: dict = {}
        unmodeled: list = []
        for a in actions:
            if a != _PASS and 0 <= a < F.MAX_OPTIONS:
                modeled[a] = float(probs[a])
            else:                                          # PASS or truncated option → no net logit
                unmodeled.append(a)
        if unmodeled:
            floor = _FLOOR_FRAC * (sum(modeled.values()) / len(modeled) if modeled else 1.0)
            for a in unmodeled:
                modeled[a] = floor
        total = sum(modeled.values())
        if total <= 0:                                     # degenerate → uniform
            u = 1.0 / len(actions) if actions else 0.0
            return {a: u for a in actions}
        return {a: modeled[a] / total for a in actions}

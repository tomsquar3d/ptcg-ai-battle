"""ONNX-CPU neural guide for `search.mcts` — the deploy-path twin of `nn.guide.NetGuide` (docs §3).

Same two MCTS hooks (`priors(node)`, `value(state, root_player)`), but inference runs through
single-threaded `onnxruntime` sessions instead of torch — no torch in the shipped agent. Two graphs
(`export_onnx.py`): `value.onnx` (state → value, the hot path) and `policy.onnx` (state+options → raw
unmasked logits; this guide applies the legal-option mask + softmax in numpy).

Single-threaded on purpose: the agent runs on 2 vCPU and fans nothing out, and for a net this small
onnxruntime's intra-op threads are pure overhead (mirrors the training-time lesson).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ptcg.perception import build_game_state
from . import features as F

try:
    from search.mcts import _PASS
except Exception:  # pragma: no cover
    _PASS = -1

_FLOOR_FRAC = 0.25   # exploratory prior for unmodeled actions (PASS / >MAX_OPTIONS)
_STATE_ARGS = ("pokemon_ids", "pokemon_scalars", "pokemon_mask", "hand_ids", "hand_mask",
               "stadium_id", "globals")


class OnnxGuide:
    def __init__(self, model_dir: str, card_index: Mapping, attack_index: Mapping):
        import os
        import onnxruntime as ort
        self.ci = card_index
        self.ai = attack_index
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        prov = ["CPUExecutionProvider"]
        self.vs = ort.InferenceSession(os.path.join(model_dir, "value.onnx"), so, providers=prov)
        self.ps = ort.InferenceSession(os.path.join(model_dir, "policy.onnx"), so, providers=prov)

    # --- featurization → onnx input dicts --------------------------------------------
    def _state_inputs(self, sf: F.StateFeatures) -> dict:
        return {
            "pokemon_ids": sf.pokemon_ids[None].astype(np.int64),
            "pokemon_scalars": sf.pokemon_scalars[None].astype(np.float32),
            "pokemon_mask": sf.pokemon_mask[None],
            "hand_ids": sf.hand_ids[None].astype(np.int64),
            "hand_mask": sf.hand_mask[None],
            "stadium_id": np.asarray([int(sf.stadium_id)], dtype=np.int64),
            "globals": sf.globals[None].astype(np.float32),
        }

    # --- MCTS hooks ------------------------------------------------------------------
    def value(self, state: Any, root_player: int) -> float:
        obs = state.observation
        gs = build_game_state(obs, self.ci)
        sf = F.encode_state(gs, self.ci, self.ai)
        v = float(self.vs.run(["value"], self._state_inputs(sf))[0][0])
        cur = getattr(obs, "current", None)
        if cur is not None and getattr(cur, "yourIndex", root_player) != root_player:
            v = -v
        return v

    def priors(self, node: Any) -> dict:
        obs = node.state.observation
        gs = build_game_state(obs, self.ci)
        sf = F.encode_state(gs, self.ci, self.ai)
        af = F.encode_actions(obs.select, gs.my_index, self.ai)
        inp = self._state_inputs(sf)
        inp["option_card_ids"] = af.option_card_ids[None].astype(np.int64)
        inp["option_scalars"] = af.option_scalars[None].astype(np.float32)
        logits = self.ps.run(["logits"], inp)[0][0]             # [MAX_OPTIONS], raw/unmasked

        mask = af.option_mask                                    # True on real option slots
        m = np.where(mask, logits, -np.inf)
        if np.isfinite(m).any():
            m = m - np.nanmax(m[np.isfinite(m)])
            e = np.where(mask, np.exp(m), 0.0)
            probs = e / e.sum() if e.sum() > 0 else np.zeros_like(e)
        else:
            probs = np.zeros_like(logits)

        actions = list(node.actions)
        modeled, unmodeled = {}, []
        for a in actions:
            if a != _PASS and 0 <= a < F.MAX_OPTIONS:
                modeled[a] = float(probs[a])
            else:
                unmodeled.append(a)
        if unmodeled:
            floor = _FLOOR_FRAC * (sum(modeled.values()) / len(modeled) if modeled else 1.0)
            for a in unmodeled:
                modeled[a] = floor
        total = sum(modeled.values())
        if total <= 0:
            u = 1.0 / len(actions) if actions else 0.0
            return {a: u for a in actions}
        return {a: modeled[a] / total for a in actions}

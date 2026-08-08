"""ONNX-CPU neural guide for `search.mcts` — the deploy-path twin of `nn.guide.NetGuide` (docs §3).

Same two MCTS hooks (`priors(node)`, `value(state, root_player)`), but inference runs through
single-threaded `onnxruntime` sessions instead of torch — no torch in the shipped agent.

Two graph layouts are supported, auto-detected from the model dir:
  - **merged** (`ptcgnet.onnx`, preferred): one fused graph, state+options -> (value, raw logits) from a
    SINGLE encode_state (trunk) pass. `value()` runs it once at node creation and caches the result on the
    `SearchState`; `priors()` reuses that cache — so the shared trunk runs ONCE per node, not twice.
  - **two-file** (`value.onnx` + `policy.onnx`, fallback): the original layout — `value()` runs the trunk
    for the value, `priors()` runs the trunk again for the logits. Used verbatim when `ptcgnet.onnx` is
    absent, so a bundle without the merged graph behaves exactly as before.

`policy`/merged logits are raw and unmasked; this guide applies the legal-option mask + softmax in numpy.

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
_CACHE_ATTR = "_ptcg_merged"   # per-node fused-eval cache, stashed on the SearchState object


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
        merged_path = os.path.join(model_dir, "ptcgnet.onnx")
        self.merged = os.path.exists(merged_path)
        if self.merged:                    # one fused graph → share a single run per node
            self.ms = ort.InferenceSession(merged_path, so, providers=prov)
            self.vs = self.ps = None
            probe = self.ms
        else:                              # fallback: original two-file layout, unchanged behaviour
            self.ms = None
            self.vs = ort.InferenceSession(os.path.join(model_dir, "value.onnx"), so, providers=prov)
            self.ps = ort.InferenceSession(os.path.join(model_dir, "policy.onnx"), so, providers=prov)
            probe = self.vs
        # schema-v2 deck conditioning: detected from the graph's inputs (auto for v1 or v2 models).
        self.deck_cond = any(i.name == "deck_summary" for i in probe.get_inputs())
        self._ds = np.zeros(F.DECK_SUMMARY_DIM, dtype=np.float32)
        self._aid = 0

    def set_deck(self, deck, archetype_name=None) -> None:
        """Provide the piloted deck for a deck-conditioned (v2) model. No-op for v1 models."""
        if self.deck_cond:
            self._ds = F.summarize_deck(list(deck), self.ci)
            self._aid = F.archetype_id_for(archetype_name)

    # --- featurization → onnx input dicts --------------------------------------------
    def _state_inputs(self, sf: F.StateFeatures) -> dict:
        d = {
            "pokemon_ids": sf.pokemon_ids[None].astype(np.int64),
            "pokemon_scalars": sf.pokemon_scalars[None].astype(np.float32),
            "pokemon_mask": sf.pokemon_mask[None],
            "hand_ids": sf.hand_ids[None].astype(np.int64),
            "hand_mask": sf.hand_mask[None],
            "stadium_id": np.asarray([int(sf.stadium_id)], dtype=np.int64),
            "globals": sf.globals[None].astype(np.float32),
        }
        if self.deck_cond:
            d["deck_summary"] = self._ds[None].astype(np.float32)
            d["archetype_id"] = np.asarray([int(self._aid)], dtype=np.int64)
        return d

    def _option_inputs(self, af: F.ActionFeatures) -> dict:
        return {
            "option_card_ids": af.option_card_ids[None].astype(np.int64),
            "option_scalars": af.option_scalars[None].astype(np.float32),
        }

    def _merged_eval(self, state: Any):
        """Run the fused graph ONCE for this node and cache (value, raw_logits, option_mask) on the
        SearchState, so value() and priors() share a single session.run per node. Idempotent: a second
        call (either hook, in any order) returns the cached tuple instead of re-running the trunk."""
        cache = getattr(state, _CACHE_ATTR, None)
        if cache is not None:
            return cache
        obs = state.observation
        gs = build_game_state(obs, self.ci)
        sf = F.encode_state(gs, self.ci, self.ai)
        af = F.encode_actions(obs.select, gs.my_index, self.ai)
        inp = self._state_inputs(sf)
        inp.update(self._option_inputs(af))
        value, logits = self.ms.run(["value", "logits"], inp)
        cache = (float(value[0]), logits[0], af.option_mask)
        try:
            setattr(state, _CACHE_ATTR, cache)   # SearchState is a plain dataclass → attachable
        except Exception:
            pass
        return cache

    # --- MCTS hooks ------------------------------------------------------------------
    def value(self, state: Any, root_player: int) -> float:
        obs = state.observation
        if self.merged:
            v = self._merged_eval(state)[0]
        else:
            gs = build_game_state(obs, self.ci)
            sf = F.encode_state(gs, self.ci, self.ai)
            v = float(self.vs.run(["value"], self._state_inputs(sf))[0][0])
        cur = getattr(obs, "current", None)
        if cur is not None and getattr(cur, "yourIndex", root_player) != root_player:
            v = -v
        return v

    def priors(self, node: Any) -> dict:
        obs = node.state.observation
        if self.merged:
            _v, logits, mask = self._merged_eval(node.state)   # reuse value()'s single run for this node
        else:
            gs = build_game_state(obs, self.ci)
            sf = F.encode_state(gs, self.ci, self.ai)
            af = F.encode_actions(obs.select, gs.my_index, self.ai)
            inp = self._state_inputs(sf)
            inp.update(self._option_inputs(af))
            logits = self.ps.run(["logits"], inp)[0][0]        # [MAX_OPTIONS], raw/unmasked
            mask = af.option_mask                              # True on real option slots

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

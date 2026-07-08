#!/usr/bin/env python3
"""Export the trained PTCGNet to ONNX for CPU deployment (docs §3, ONNX-CPU path).

Two graphs from the same weights, mirroring how the agent calls the net inside MCTS:
  value.onnx  : state tensors            -> value  [B]        (the hot path — every leaf)
  policy.onnx : state + option tensors   -> raw logits [B,64] (unmasked; the guide masks in numpy)

Only the batch dim is dynamic; the option dim is fixed at MAX_OPTIONS. Validates that ONNX outputs
match torch within 1e-4 on real decision states (including a fully-padded board — the NaN-guard path).

Run:  .venv/bin/python scripts/export_onnx.py [checkpoint.pt] [out_dir]
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
sys.path.insert(0, os.path.join(REPO, "src"))

from cg.api import to_observation_class, all_card_data, all_attack           # noqa: E402
from cg.game import battle_start, battle_finish, battle_select               # noqa: E402
from ptcg.perception import build_game_state                                 # noqa: E402
from nn import features as F                                                 # noqa: E402
from nn.encoder import PTCGNet, NetConfig, collate                           # noqa: E402

_STATE_ARGS = ("pokemon_ids", "pokemon_scalars", "pokemon_mask", "hand_ids", "hand_mask",
               "stadium_id", "globals")


class _ValueWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, pokemon_ids, pokemon_scalars, pokemon_mask, hand_ids, hand_mask, stadium_id, globals):
        b = dict(pokemon_ids=pokemon_ids, pokemon_scalars=pokemon_scalars, pokemon_mask=pokemon_mask,
                 hand_ids=hand_ids, hand_mask=hand_mask, stadium_id=stadium_id, globals=globals)
        sv = self.net.encode_state(b)
        return torch.tanh(self.net.value_head(sv)).squeeze(-1)


class _PolicyWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, pokemon_ids, pokemon_scalars, pokemon_mask, hand_ids, hand_mask, stadium_id,
                globals, option_card_ids, option_scalars):
        b = dict(pokemon_ids=pokemon_ids, pokemon_scalars=pokemon_scalars, pokemon_mask=pokemon_mask,
                 hand_ids=hand_ids, hand_mask=hand_mask, stadium_id=stadium_id, globals=globals)
        state = self.net.encode_state(b)
        opt = self.net.opt_in(torch.cat([self.net.card_embed(option_card_ids), option_scalars], -1))
        st = state.unsqueeze(1).expand(-1, opt.shape[1], -1)
        return self.net.policy_scorer(torch.cat([opt, st], -1)).squeeze(-1)   # raw, unmasked


def _collect_states(n_want, ci, ai):
    """Grab a few real (state, action) featurizations to trace/validate with."""
    deck = [int(x) for x in open(os.path.join(REPO, "submissions", "v4", "deck.csv")).read().split() if x.strip()]
    states, actions = [], []
    obs_dict, _ = battle_start(deck, list(deck))
    try:
        for _ in range(3000):
            obs = to_observation_class(obs_dict)
            cur = getattr(obs, "current", None)
            if cur is not None and getattr(cur, "result", -1) != -1:
                break
            sel = obs.select
            if sel is None:
                break
            gs = build_game_state(obs, ci)
            states.append(F.encode_state(gs, ci, ai))
            actions.append(F.encode_actions(sel, gs.my_index, ai))
            n = len(sel.option)
            obs_dict = battle_select(list(range(min(max(sel.minCount, 1), n))) if n else [])
            if len(states) >= n_want:
                break
    finally:
        battle_finish()
    return states, actions


def main() -> int:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "experiments", "checkpoints", "net.pt")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "experiments", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = NetConfig(**ck["config"]) if isinstance(ck.get("config"), dict) else NetConfig()
    net = PTCGNet(cfg).eval()
    net.load_state_dict(ck["model"])

    states, actions = _collect_states(8, ci, ai)
    batch = collate(states, actions)
    state_inputs = tuple(batch[k] for k in _STATE_ARGS)
    policy_inputs = state_inputs + (batch["option_card_ids"], batch["option_scalars"])

    dyn = {"batch": {0: "batch"}}
    v_path = os.path.join(out_dir, "value.onnx")
    p_path = os.path.join(out_dir, "policy.onnx")

    torch.onnx.export(
        _ValueWrapper(net), state_inputs, v_path, input_names=list(_STATE_ARGS),
        output_names=["value"], opset_version=18,
        dynamic_axes={k: {0: "batch"} for k in list(_STATE_ARGS) + ["value"]})
    torch.onnx.export(
        _PolicyWrapper(net), policy_inputs, p_path,
        input_names=list(_STATE_ARGS) + ["option_card_ids", "option_scalars"],
        output_names=["logits"], opset_version=18,
        dynamic_axes={k: {0: "batch"} for k in list(_STATE_ARGS) +
                      ["option_card_ids", "option_scalars", "logits"]})

    # The exporter writes weights to a sidecar `.onnx.data` file — fragile in a submission bundle.
    # Re-save each graph self-contained (weights inline) and drop the sidecar, so a single .onnx ships.
    import onnx
    for p in (v_path, p_path):
        onnx.save_model(onnx.load(p), p, save_as_external_data=False)
        if os.path.exists(p + ".data"):
            os.remove(p + ".data")

    # --- validate ONNX == torch on the real states (+ a fully-padded board) ---------------
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    vs = ort.InferenceSession(v_path, so, providers=["CPUExecutionProvider"])
    ps = ort.InferenceSession(p_path, so, providers=["CPUExecutionProvider"])

    def np_state():
        return {k: batch[k].numpy() for k in _STATE_ARGS}

    with torch.no_grad():
        t_logits, t_value = net(batch)
    v_out = vs.run(["value"], np_state())[0]
    p_in = np_state(); p_in["option_card_ids"] = batch["option_card_ids"].numpy(); p_in["option_scalars"] = batch["option_scalars"].numpy()
    p_out = ps.run(["logits"], p_in)[0]

    om = batch["option_mask"].numpy()
    v_diff = float(np.abs(v_out - t_value.numpy()).max())
    l_diff = float(np.abs(p_out[om] - t_logits.numpy()[om]).max())   # compare on real (unmasked) slots

    # fully-padded board must stay finite through ONNX (the NaN guard)
    zpad = {k: v.copy() for k, v in np_state().items()}
    zpad["pokemon_mask"][:] = False
    v_pad = vs.run(["value"], zpad)[0]

    assert v_diff < 1e-4, f"value mismatch {v_diff}"
    assert l_diff < 1e-4, f"logit mismatch {l_diff}"
    assert np.isfinite(v_pad).all(), "fully-padded board -> non-finite value in ONNX"

    print(f"exported:\n  {os.path.relpath(v_path, REPO)}  ({os.path.getsize(v_path)/1e6:.2f} MB)")
    print(f"  {os.path.relpath(p_path, REPO)}  ({os.path.getsize(p_path)/1e6:.2f} MB)")
    print(f"validate (vs torch, {len(states)} real states): "
          f"value maxΔ {v_diff:.2e}, logit maxΔ {l_diff:.2e}; padded-board finite ✓")
    print("\nOK: PTCGNet exported to ONNX and numerically matches torch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

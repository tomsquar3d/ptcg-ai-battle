#!/usr/bin/env python3
"""Export the trained PTCGNet to ONNX for CPU deployment (docs §3, ONNX-CPU path).

Two graphs from the same weights, mirroring how the agent calls the net inside MCTS:
  value.onnx  : state tensors            -> value  [B]        (the hot path — every leaf)
  policy.onnx : state + option tensors   -> raw logits [B,64] (unmasked; the guide masks in numpy)

With `--merged` it ALSO emits a single fused graph (the two-file path stays intact):
  ptcgnet.onnx: state + option tensors   -> (value [B], raw logits [B,64]) from ONE encode_state pass,
                so the guide runs the shared trunk once per node instead of twice (value + policy).

Only the batch dim is dynamic; the option dim is fixed at MAX_OPTIONS. Validates that ONNX outputs
match torch within 1e-4 on real decision states (including a fully-padded board — the NaN-guard path);
with `--merged`, also validates the fused graph matches the two-file graphs within 1e-4.

Run:  .venv/bin/python scripts/export_onnx.py [checkpoint.pt] [out_dir] [--merged]
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


class _ValueWrapperDC(nn.Module):
    """Deck-conditioned (schema-v2) value graph: adds deck_summary + archetype_id state inputs."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, pokemon_ids, pokemon_scalars, pokemon_mask, hand_ids, hand_mask, stadium_id,
                globals, deck_summary, archetype_id):
        b = dict(pokemon_ids=pokemon_ids, pokemon_scalars=pokemon_scalars, pokemon_mask=pokemon_mask,
                 hand_ids=hand_ids, hand_mask=hand_mask, stadium_id=stadium_id, globals=globals,
                 deck_summary=deck_summary, archetype_id=archetype_id)
        return torch.tanh(self.net.value_head(self.net.encode_state(b))).squeeze(-1)


class _PolicyWrapperDC(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, pokemon_ids, pokemon_scalars, pokemon_mask, hand_ids, hand_mask, stadium_id,
                globals, deck_summary, archetype_id, option_card_ids, option_scalars):
        b = dict(pokemon_ids=pokemon_ids, pokemon_scalars=pokemon_scalars, pokemon_mask=pokemon_mask,
                 hand_ids=hand_ids, hand_mask=hand_mask, stadium_id=stadium_id, globals=globals,
                 deck_summary=deck_summary, archetype_id=archetype_id)
        state = self.net.encode_state(b)
        opt = self.net.opt_in(torch.cat([self.net.card_embed(option_card_ids), option_scalars], -1))
        st = state.unsqueeze(1).expand(-1, opt.shape[1], -1)
        return self.net.policy_scorer(torch.cat([opt, st], -1)).squeeze(-1)


class _MergedWrapper(nn.Module):
    """Fused graph: one encode_state pass -> (value, raw unmasked logits). Same weights, same math as
    _ValueWrapper + _PolicyWrapper, but the shared trunk runs once."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, pokemon_ids, pokemon_scalars, pokemon_mask, hand_ids, hand_mask, stadium_id,
                globals, option_card_ids, option_scalars):
        b = dict(pokemon_ids=pokemon_ids, pokemon_scalars=pokemon_scalars, pokemon_mask=pokemon_mask,
                 hand_ids=hand_ids, hand_mask=hand_mask, stadium_id=stadium_id, globals=globals)
        state = self.net.encode_state(b)
        value = torch.tanh(self.net.value_head(state)).squeeze(-1)
        opt = self.net.opt_in(torch.cat([self.net.card_embed(option_card_ids), option_scalars], -1))
        st = state.unsqueeze(1).expand(-1, opt.shape[1], -1)
        logits = self.net.policy_scorer(torch.cat([opt, st], -1)).squeeze(-1)   # raw, unmasked
        return value, logits


class _MergedWrapperDC(nn.Module):
    """Deck-conditioned (schema-v2) fused graph: adds deck_summary + archetype_id state inputs."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, pokemon_ids, pokemon_scalars, pokemon_mask, hand_ids, hand_mask, stadium_id,
                globals, deck_summary, archetype_id, option_card_ids, option_scalars):
        b = dict(pokemon_ids=pokemon_ids, pokemon_scalars=pokemon_scalars, pokemon_mask=pokemon_mask,
                 hand_ids=hand_ids, hand_mask=hand_mask, stadium_id=stadium_id, globals=globals,
                 deck_summary=deck_summary, archetype_id=archetype_id)
        state = self.net.encode_state(b)
        value = torch.tanh(self.net.value_head(state)).squeeze(-1)
        opt = self.net.opt_in(torch.cat([self.net.card_embed(option_card_ids), option_scalars], -1))
        st = state.unsqueeze(1).expand(-1, opt.shape[1], -1)
        logits = self.net.policy_scorer(torch.cat([opt, st], -1)).squeeze(-1)   # raw, unmasked
        return value, logits


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
    merged = "--merged" in sys.argv
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    ckpt = pos[0] if len(pos) > 0 else os.path.join(REPO, "experiments", "checkpoints", "net.pt")
    out_dir = pos[1] if len(pos) > 1 else os.path.join(REPO, "experiments", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)

    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = NetConfig(**ck["config"]) if isinstance(ck.get("config"), dict) else NetConfig()
    net = PTCGNet(cfg).eval()
    net.load_state_dict(ck["model"])
    # v9: collapse e_c = W @ attr(c) + r_c into a plain [VOCAB, embed] table. Mathematically
    # identical, keeps the graph's single Gather, and leaves the card attributes out of the bundle.
    if getattr(cfg, "compositional_embed", False):
        net.materialize_card_embed()
        print("materialized compositional card embedding → plain table (no card data in the graph)")
    deck_cond = bool(getattr(cfg, "deck_cond", False))
    print(f"exporting {'deck-conditioned (v2)' if deck_cond else 'v1'} net → {out_dir}")

    states, actions = _collect_states(8, ci, ai)
    batch = collate(states, actions)
    if deck_cond:                      # attach the schema-v2 deck-conditioning state inputs
        deck = [int(x) for x in open(os.path.join(REPO, "submissions", "v4", "deck.csv")).read().split() if x.strip()]
        B = batch["pokemon_ids"].shape[0]
        ds = F.summarize_deck(deck, ci)
        batch["deck_summary"] = torch.from_numpy(np.repeat(ds[None], B, 0)).float()
        batch["archetype_id"] = torch.full((B,), F.archetype_id_for("sample"), dtype=torch.long)

    state_args = list(_STATE_ARGS) + (["deck_summary", "archetype_id"] if deck_cond else [])
    ValueW = _ValueWrapperDC if deck_cond else _ValueWrapper
    PolicyW = _PolicyWrapperDC if deck_cond else _PolicyWrapper
    state_inputs = tuple(batch[k] for k in state_args)
    policy_inputs = state_inputs + (batch["option_card_ids"], batch["option_scalars"])

    v_path = os.path.join(out_dir, "value.onnx")
    p_path = os.path.join(out_dir, "policy.onnx")

    torch.onnx.export(
        ValueW(net), state_inputs, v_path, input_names=state_args,
        output_names=["value"], opset_version=18,
        dynamic_axes={k: {0: "batch"} for k in state_args + ["value"]})
    torch.onnx.export(
        PolicyW(net), policy_inputs, p_path,
        input_names=state_args + ["option_card_ids", "option_scalars"],
        output_names=["logits"], opset_version=18,
        dynamic_axes={k: {0: "batch"} for k in state_args +
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
        return {k: batch[k].numpy() for k in state_args}

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

    if merged:
        # --- fused single-trunk graph (the two files above are left intact) ------------------
        MergedW = _MergedWrapperDC if deck_cond else _MergedWrapper
        m_path = os.path.join(out_dir, "ptcgnet.onnx")
        torch.onnx.export(
            MergedW(net), policy_inputs, m_path,
            input_names=state_args + ["option_card_ids", "option_scalars"],
            output_names=["value", "logits"], opset_version=18,
            dynamic_axes={k: {0: "batch"} for k in state_args +
                          ["option_card_ids", "option_scalars", "value", "logits"]})
        onnx.save_model(onnx.load(m_path), m_path, save_as_external_data=False)
        if os.path.exists(m_path + ".data"):
            os.remove(m_path + ".data")

        ms = ort.InferenceSession(m_path, so, providers=["CPUExecutionProvider"])
        mv, ml = ms.run(["value", "logits"], p_in)          # same inputs as the policy graph
        # fused vs the two-file graphs (the parity that matters for a behaviour-identical swap)
        mv_vs_v = float(np.abs(mv - v_out).max())
        ml_vs_p = float(np.abs(ml[om] - p_out[om]).max())
        # and fused vs torch
        mv_vs_t = float(np.abs(mv - t_value.numpy()).max())
        ml_vs_t = float(np.abs(ml[om] - t_logits.numpy()[om]).max())
        m_zpad = dict(zpad)
        m_zpad["option_card_ids"] = batch["option_card_ids"].numpy()
        m_zpad["option_scalars"] = batch["option_scalars"].numpy()
        mv_pad, ml_pad = ms.run(["value", "logits"], m_zpad)

        assert mv_vs_v < 1e-4, f"merged value vs value.onnx mismatch {mv_vs_v}"
        assert ml_vs_p < 1e-4, f"merged logits vs policy.onnx mismatch {ml_vs_p}"
        assert mv_vs_t < 1e-4 and ml_vs_t < 1e-4, f"merged vs torch mismatch {mv_vs_t}/{ml_vs_t}"
        assert np.isfinite(mv_pad).all() and np.isfinite(ml_pad).all(), \
            "fully-padded board -> non-finite in merged ONNX"
        print(f"  {os.path.relpath(m_path, REPO)}  ({os.path.getsize(m_path)/1e6:.2f} MB)")
        print(f"validate merged (vs two-file, {len(states)} real states): "
              f"value maxΔ {mv_vs_v:.2e}, logit maxΔ {ml_vs_p:.2e}; vs torch "
              f"{mv_vs_t:.2e}/{ml_vs_t:.2e}; padded-board finite ✓")

    print("\nOK: PTCGNet exported to ONNX and numerically matches torch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

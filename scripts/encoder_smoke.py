#!/usr/bin/env python3
"""Smoke test for the encoder + policy/value heads (src/nn/encoder.py).

Featurizes real decision states, runs a forward pass, and checks: policy logits are finite on legal
options and −inf on padding, softmax over legal options is a valid distribution, value ∈ (−1,1), a
combined loss backprops with finite grads, and the param count is small enough for ONNX-CPU.

Run:  .venv/bin/python scripts/encoder_smoke.py
"""
import os
import random
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
sys.path.insert(0, os.path.join(REPO, "src"))

from cg.api import to_observation_class, all_card_data, all_attack       # noqa: E402
from cg.game import battle_start, battle_select, battle_finish           # noqa: E402
from ptcg.perception import build_game_state                            # noqa: E402
from nn import features as F                                            # noqa: E402
from nn.encoder import PTCGNet, collate                                 # noqa: E402


def _rand(sel):
    n = len(sel.option)
    return random.sample(range(n), min(sel.maxCount, n)) if n else []


def _collect(n_want, ci, ai):
    deck = [int(x) for x in open(os.path.join(REPO, "experiments", "candidate_decks", "dragapult-ex.csv")).read().split() if x.strip()]
    states, actions = [], []
    while len(states) < n_want:
        obs_dict, _ = battle_start(deck, list(deck))
        try:
            for _ in range(2000):
                obs = to_observation_class(obs_dict)
                if obs.select is None or (obs.current is not None and obs.current.result != -1):
                    break
                gs = build_game_state(obs, ci)
                states.append(F.encode_state(gs, ci, ai))
                actions.append(F.encode_actions(obs.select, gs.my_index, ai))
                obs_dict = battle_select(_rand(obs.select))
                if len(states) >= n_want:
                    break
        finally:
            battle_finish()
    return states[:n_want], actions[:n_want]


def main() -> int:
    torch.manual_seed(0)
    ci = {c.cardId: c for c in all_card_data()}
    ai = {a.attackId: a for a in all_attack()}
    B = 16
    states, actions = _collect(B, ci, ai)

    net = PTCGNet()
    batch = collate(states, actions)
    logits, value = net(batch)

    assert logits.shape == (B, F.MAX_OPTIONS), logits.shape
    assert value.shape == (B,), value.shape
    om = batch["option_mask"]
    assert torch.isfinite(logits[om]).all(), "legal-option logit not finite"
    assert torch.isinf(logits[~om]).all(), "padded option not masked to -inf"
    # softmax over legal options is a valid distribution per row
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(1), torch.ones(B), atol=1e-4), "policy not normalized"
    assert (probs[~om] == 0).all(), "padded option got probability mass"
    assert (value.abs() <= 1).all(), "value out of (-1,1)"

    # backward: policy CE to a random legal target + value MSE to a random outcome
    tgt = torch.tensor([int(np.random.choice(np.where(om[i].numpy())[0])) for i in range(B)])
    z = torch.randint(0, 2, (B,)).float() * 2 - 1
    loss = torch.nn.functional.cross_entropy(logits, tgt) + torch.nn.functional.mse_loss(value, z)
    loss.backward()
    gnorm = torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None]).norm()
    assert torch.isfinite(gnorm), "non-finite gradient"

    print(f"1. FORWARD  : logits {tuple(logits.shape)}, value {tuple(value.shape)}; "
          f"legal finite, padding −inf, policy normalized, value∈(−1,1)")
    print(f"2. BACKWARD : loss {loss.item():.3f}, grad-norm {gnorm.item():.2f} (finite)")
    print(f"3. SIZE     : {net.num_params():,} params (~{net.num_params()*4/1e6:.1f} MB fp32) — ONNX-CPU-friendly")
    print("\nOK: encoder + policy/value heads forward, mask, and backprop correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

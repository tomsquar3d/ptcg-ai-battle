#!/usr/bin/env python3
"""v9 encoder smoke: build with every flag on, forward, export to ONNX, check torch/ONNX parity.

Covers the three v9 commits together — the sized preset (wider tokens, 2-layer pre-norm GELU board
attention), the Option C compositional card embedding, and the aux heads — then hands the exported
graphs to the real `export_onnx.py` path so parity is checked by the same code that ships nets.

    .venv/bin/python scripts/v9_smoke.py

No training: weights are random (the residual table is deliberately perturbed off its zero init so
materialization is a non-trivial check).
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
torch.set_num_threads(1)

from nn.encoder import PTCGNet, NetConfig, v9_config, _CompositionalCardEmbedding   # noqa: E402
from nn import features as F                                                        # noqa: E402

CARD_FEATURES = os.path.join(REPO, "experiments", "checkpoints", "card_features.npz")


def synth_batch(B=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, F.VOCAB_SIZE, (B, F.POKEMON_SLOTS), generator=g)
    hand = torch.randint(0, F.VOCAB_SIZE, (B, F.MAX_HAND), generator=g)
    pm = torch.rand(B, F.POKEMON_SLOTS, generator=g) > 0.3
    pm[:, 0] = True
    hm = torch.rand(B, F.MAX_HAND, generator=g) > 0.4
    om = torch.rand(B, F.MAX_OPTIONS, generator=g) > 0.5
    om[:, 0] = True
    return {
        "pokemon_ids": ids, "pokemon_scalars": torch.randn(B, F.POKEMON_SLOTS, F.POKEMON_SCALAR_DIM, generator=g),
        "pokemon_mask": pm, "hand_ids": hand, "hand_mask": hm,
        "stadium_id": torch.randint(0, F.VOCAB_SIZE, (B,), generator=g),
        "globals": torch.randn(B, F.GLOBAL_DIM, generator=g),
        "deck_summary": torch.randn(B, F.DECK_SUMMARY_DIM, generator=g),
        "archetype_id": torch.randint(0, F.ARCHETYPE_VOCAB, (B,), generator=g),
        "option_card_ids": torch.randint(0, F.VOCAB_SIZE, (B, F.MAX_OPTIONS), generator=g),
        "option_scalars": torch.randn(B, F.MAX_OPTIONS, F.OPTION_SCALAR_DIM, generator=g),
        "option_mask": om,
    }


def main() -> int:
    if not os.path.exists(CARD_FEATURES):
        print(f"missing {os.path.relpath(CARD_FEATURES, REPO)} — run scripts/build_card_features.py first")
        return 1

    torch.manual_seed(0)
    n_res = int(np.load(CARD_FEATURES)["n_residual_cards"])
    cfg = v9_config(n_residual_cards=n_res)
    net = PTCGNet(cfg).eval()
    info = net.load_card_features(CARD_FEATURES)
    assert isinstance(net.card_embed, _CompositionalCardEmbedding)

    print("1. BUILD    : " + " ".join(f"{k}={getattr(cfg, k)}" for k in
          ("embed_dim", "d_model", "n_attn_layers", "state_dim", "activation", "norm_first")))
    print(f"             deck_cond={cfg.deck_cond} aux_heads={cfg.aux_heads} "
          f"compositional={cfg.compositional_embed} n_residual_cards={cfg.n_residual_cards}")
    aux_p = sum(p.numel() for n, p in net.named_parameters() if n.startswith(("prize_head", "tta_head")))
    print(f"2. SIZE     : {net.num_params():,} params total | {net.num_params() - aux_p:,} at inference "
          f"(aux heads are training-only) | card features {info['attrs']}")

    # perturb the residual off its zero init so materialization has something real to fold in
    with torch.no_grad():
        net.card_embed.residual.weight.normal_(0, 0.05)
        net.card_embed.residual.weight[0].zero_()            # row 0 stays the frozen zero row

    b = synth_batch()
    with torch.no_grad():
        logits, value, aux = net.forward_train(b)
    assert logits.shape == (8, F.MAX_OPTIONS) and value.shape == (8,)
    assert torch.isfinite(logits[b["option_mask"]]).all() and (logits[~b["option_mask"]] == float("-inf")).all()
    assert torch.isfinite(value).all() and (value.abs() < 1).all()
    assert set(aux) == {"prize_diff_t3", "turns_to_attack"}
    assert (aux["prize_diff_t3"].abs() < 1).all() and ((aux["turns_to_attack"] > 0) & (aux["turns_to_attack"] < 1)).all()
    print(f"3. FORWARD  : logits {tuple(logits.shape)}, value {tuple(value.shape)}, aux {sorted(aux)}; "
          f"padded options −inf, value∈(−1,1), aux in range")

    pad = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    pad["pokemon_mask"][:] = False
    with torch.no_grad():
        lg_pad, v_pad = net(pad)
    assert torch.isfinite(v_pad).all() and torch.isfinite(lg_pad[pad["option_mask"]]).all()
    print("4. NAN-GUARD: fully-padded board → finite value + logits (torch)")

    # materialize must reproduce W @ attr + r exactly — it is what the ONNX graph will carry
    all_ids = torch.arange(F.VOCAB_SIZE)
    with torch.no_grad():
        comp = net.card_embed(all_ids)
        table = net.card_embed.materialize()
        d_mat = float((table(all_ids) - comp).abs().max())
    assert d_mat == 0.0, f"materialize differs from compositional by {d_mat}"
    assert float(comp[F.PAD_ID].abs().max()) == 0.0, "e_PAD must be exactly zero"
    unseen = int((net.card_embed.residual_index == 0).sum()) - 1     # minus PAD
    print(f"5. EMBED    : materialize == W@attr+r exactly (maxΔ {d_mat:.0e}); e_PAD == 0; "
          f"{unseen} cards with no residual row still get a compositional vector")

    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "v9_smoke.pt")
        torch.save({"model": net.state_dict(), "config": vars(cfg), "epoch": 0}, ckpt)
        print("6. EXPORT   : running scripts/export_onnx.py --merged (torch/ONNX parity @ 1e-4)")
        r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "export_onnx.py"),
                            ckpt, td, "--merged"], capture_output=True, text=True, cwd=REPO)
        out = (r.stdout or "") + (r.stderr or "")
        for line in out.splitlines():
            keep = line.startswith(("materialized", "validate", "OK")) or ".onnx  (" in line
            if keep:
                print("             " + line.strip())
        if r.returncode != 0:
            print(out[-4000:])
            return 1

    print("\nOK: v9 net builds, forwards, exports, and matches torch through ONNX within 1e-4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

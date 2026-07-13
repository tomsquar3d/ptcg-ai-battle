#!/usr/bin/env python3
"""Top-k champion action-match on held-out games — measures how good the champion-clone is as a SEARCH PRIOR.

top-1 says "does the net pick the champion's exact move"; top-3 says "is the champion's move among the net's
best 3" — which is what matters when the net is an MCTS prior (search explores the top few and the value head
sorts them out). Masks logits to legal options first (a fair prior read).

    .venv/bin/python scripts/eval_topk.py [--ckpt experiments/checkpoints/champion.pt] [--buffer .../champion_bc]
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from nn import dataset as D                                        # noqa: E402
from nn.encoder import PTCGNet, NetConfig                          # noqa: E402
import train_net as T                                             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(REPO, "experiments", "checkpoints", "champion.pt"))
    ap.add_argument("--buffer", default=os.path.join(REPO, "experiments", "selfplay", "champion_bc"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    buf = D.load_buffer(args.buffer)
    _tr, va = D.split_by_game(buf, args.val_frac, seed=args.seed)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = NetConfig(**ck["config"]) if isinstance(ck.get("config"), dict) else NetConfig()
    net = PTCGNet(cfg).eval()
    net.load_state_dict(ck["model"])

    hits = {1: 0, 2: 0, 3: 0}
    n = 0
    n_opts_sum = 0
    with torch.no_grad():
        for s in range(0, len(va), 256):
            b = va[s:s + 256]
            bd, pi, _z = T.make_batch(buf, b, "cpu", cfg.deck_cond)
            logits, _v, _a = net.forward_train(bd)
            om = bd.get("option_mask")
            if om is not None:
                logits = logits.masked_fill(~om.bool(), float("-inf"))
            has = pi.sum(1) > 0
            if not bool(has.any()):
                continue
            tgt = pi.argmax(1)[has]
            lg = logits[has]
            k = min(3, lg.shape[1])
            topk = lg.topk(k, dim=1).indices
            for j in (1, 2, 3):
                jj = min(j, k)
                hits[j] += int((topk[:, :jj] == tgt[:, None]).any(1).sum())
            n += int(has.sum())
            if om is not None:
                n_opts_sum += float(om[has].sum())

    if n == 0:
        print("no gradable val rows"); return 0
    print(f"champion-clone as a search prior | {n} held-out decisions | avg legal options {n_opts_sum/n:.1f}")
    print(f"  top-1 match: {hits[1]/n:.1%}")
    print(f"  top-2 match: {hits[2]/n:.1%}")
    print(f"  top-3 match: {hits[3]/n:.1%}")
    print(f"  (random top-1 ≈ {n/ n_opts_sum if n_opts_sum else 0:.1%}; top-3 ≈ {min(1.0, 3*n/n_opts_sum) if n_opts_sum else 0:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

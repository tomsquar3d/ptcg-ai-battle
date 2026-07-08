#!/usr/bin/env python3
"""Train PTCGNet on self-play shards — AlphaZero loss (docs §3): policy CE vs MCTS visit-π +
value MSE vs game outcome z.

    .venv/bin/python scripts/train_net.py --buffer experiments/selfplay/buffer \
        --epochs 30 --batch 256 --out experiments/checkpoints/net.pt

Reports, per epoch, train/val policy CE + value MSE and two interpretable held-out metrics:
  - policy top-1 : does the net's most-likely option match MCTS's most-visited option
                   (over decisions that have a real-option target)?
  - value sign   : does sign(value) match sign(z) (who's winning)?
Saves the best-val-policy checkpoint (+ a meta json). Checkpoints are gitignored (`*.pt`).

This trains the net; it does NOT yet gate on the round_robin harness — that happens once the net is
wired into MCTS as prior_fn + leaf value (the acceptance gate needs the net *playing*, not just its
held-out loss). Held-out policy/value accuracy here is the cheap pre-gate: a net that can't predict
the teacher's moves won't help the search.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))   # cg/ (unused here but keeps paths uniform)
sys.path.insert(0, os.path.join(REPO, "src"))

from nn.encoder import PTCGNet, NetConfig                      # noqa: E402
from nn import dataset as D                                    # noqa: E402


def policy_ce(logits: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the visit-distribution target against masked logits (padded → −inf)."""
    logp = torch.log_softmax(logits, dim=1)
    # only real-option slots carry target mass; avoid (−inf)·0 by zeroing where pi==0
    return -(torch.where(pi > 0, pi * logp, torch.zeros_like(logp))).sum(1).mean()


@torch.no_grad()
def evaluate(net: PTCGNet, buf: D.Buffer, idx: np.ndarray, batch: int, device: str) -> dict:
    net.eval()
    tot_ce = tot_mse = 0.0
    n = top1_hits = top1_n = sign_hits = 0
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        bd, pi, z = D.make_batch(buf, b, device)
        logits, value = net(bd)
        tot_ce += policy_ce(logits, pi).item() * len(b)
        tot_mse += torch.nn.functional.mse_loss(value, z).item() * len(b)
        n += len(b)
        has_tgt = pi.sum(1) > 0
        if has_tgt.any():
            pred = logits.argmax(1)
            tgt = pi.argmax(1)
            top1_hits += int((pred[has_tgt] == tgt[has_tgt]).sum())
            top1_n += int(has_tgt.sum())
        nz = z != 0
        if nz.any():
            sign_hits += int((torch.sign(value[nz]) == torch.sign(z[nz])).sum())
    return {
        "policy_ce": tot_ce / max(n, 1),
        "value_mse": tot_mse / max(n, 1),
        "policy_top1": top1_hits / max(top1_n, 1),
        "value_sign_acc": sign_hits / max(n, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", default=os.path.join(REPO, "experiments", "selfplay", "buffer"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "experiments", "checkpoints", "net.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    buf = D.load_buffer(args.buffer)
    tr, va = D.split_by_game(buf, args.val_frac, seed=args.seed)
    print(f"buffer: {len(buf.game_id)} decisions from {buf.n_games} games "
          f"| train {len(tr)} / val {len(va)} | dropped PASS mass {buf.dropped_pass_mass:.2%}")
    if buf.dropped_pass_mass > 0.05:
        print(f"  NOTE: {buf.dropped_pass_mass:.1%} of visit mass was PASS (dropped) — consider a PASS logit")

    net = PTCGNet(NetConfig()).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)

    base = evaluate(net, buf, va, args.batch, device)
    print(f"init  val: CE {base['policy_ce']:.3f}  MSE {base['value_mse']:.3f}  "
          f"top1 {base['policy_top1']:.1%}  sign {base['value_sign_acc']:.1%}")

    best = -1.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ep in range(1, args.epochs + 1):
        net.train()
        order = tr.copy()
        rng.shuffle(order)
        run_ce = run_mse = 0.0
        for s in range(0, len(order), args.batch):
            b = order[s:s + args.batch]
            bd, pi, z = D.make_batch(buf, b, device)
            logits, value = net(bd)
            ce = policy_ce(logits, pi)
            mse = torch.nn.functional.mse_loss(value, z)
            loss = ce + args.value_weight * mse
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            run_ce += ce.item() * len(b)
            run_mse += mse.item() * len(b)
        m = evaluate(net, buf, va, args.batch, device)
        star = ""
        if m["policy_top1"] > best:
            best = m["policy_top1"]
            torch.save({"model": net.state_dict(), "config": vars(NetConfig()),
                        "val": m, "epoch": ep}, args.out)
            star = " *"
        print(f"ep {ep:3d}  train CE {run_ce/len(tr):.3f} MSE {run_mse/len(tr):.3f}  |  "
              f"val CE {m['policy_ce']:.3f} MSE {m['value_mse']:.3f} "
              f"top1 {m['policy_top1']:.1%} sign {m['value_sign_acc']:.1%}{star}")

    meta = {"buffer": os.path.relpath(args.buffer, REPO), "games": buf.n_games,
            "decisions": int(len(buf.game_id)), "epochs": args.epochs,
            "best_val_top1": best, "dropped_pass_mass": buf.dropped_pass_mass}
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nsaved best checkpoint (val top1 {best:.1%}) -> {os.path.relpath(args.out, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

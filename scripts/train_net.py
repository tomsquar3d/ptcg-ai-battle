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

from nn.encoder import PTCGNet, NetConfig, v9_config           # noqa: E402
from nn import dataset as D                                    # noqa: E402


def policy_ce(logits: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the visit-distribution target against masked logits (padded → −inf)."""
    logp = torch.log_softmax(logits, dim=1)
    # only real-option slots carry target mass; avoid (−inf)·0 by zeroing where pi==0
    return -(torch.where(pi > 0, pi * logp, torch.zeros_like(logp))).sum(1).mean()


def make_batch(buf: D.Buffer, idx: np.ndarray, device: str, deck_cond: bool):
    """v1 batch, plus schema-v2 deck-conditioning keys merged in when the net wants them."""
    bd, pi, z = D.make_batch(buf, idx, device)
    if deck_cond:
        ds, aid = D.deck_cond_batch(buf, idx, device)
        bd["deck_summary"] = ds
        bd["archetype_id"] = aid
    return bd, pi, z


def aux_loss(aux: dict, tgt: dict, include_bc: bool = False) -> torch.Tensor:
    """MSE of the auxiliary heads against their targets (docs §3d step 4).

    BC rows carry fabricated outcomes, so their prize-diff@t+3 / turns-to-attack targets are not
    real supervision — #56 rooted a whole failed generation in exactly this kind of poisoning, and
    `--mask-bc-value` already guards the main value head against it. `aux_targets` has always
    returned `is_bc` for this purpose; nothing consumed it until now.
    """
    mse = torch.nn.functional.mse_loss
    keep = torch.ones_like(tgt["is_bc"]) if include_bc else ~tgt["is_bc"]
    if not bool(keep.any()):
        return aux["prize_diff_t3"].sum() * 0.0        # keep the graph, contribute no gradient
    return (mse(aux["prize_diff_t3"][keep], tgt["prize_diff_t3"][keep])
            + mse(aux["turns_to_attack"][keep], tgt["turns_to_attack"][keep]))


@torch.no_grad()
def evaluate(net: PTCGNet, buf: D.Buffer, idx: np.ndarray, batch: int, device: str,
             deck_cond: bool, aux_heads: bool, aux_include_bc: bool = False) -> dict:
    net.eval()
    src_all = buf.data.get("source")            # value metrics must EXCLUDE fake-z BC rows (source==1)
    tot_ce = tot_mse = tot_aux = 0.0
    n = top1_hits = top1_n = sign_hits = val_n = 0
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        bd, pi, z = make_batch(buf, b, device, deck_cond)
        logits, value, aux = net.forward_train(bd)
        tot_ce += policy_ce(logits, pi).item() * len(b)
        if aux_heads:
            tot_aux += aux_loss(aux, D.aux_targets(buf, b, device), aux_include_bc).item() * len(b)
        n += len(b)
        has_tgt = pi.sum(1) > 0
        if has_tgt.any():
            pred = logits.argmax(1)
            tgt = pi.argmax(1)
            top1_hits += int((pred[has_tgt] == tgt[has_tgt]).sum())
            top1_n += int(has_tgt.sum())
        # value metrics on REAL-outcome rows only (source==0 if present) with nonzero z
        real = (torch.from_numpy((src_all[b] == 0)).to(device) if src_all is not None
                else torch.ones(len(b), dtype=torch.bool, device=device))
        vm = real & (z != 0)
        if vm.any():
            tot_mse += torch.nn.functional.mse_loss(value[vm], z[vm]).item() * int(vm.sum())
            sign_hits += int((torch.sign(value[vm]) == torch.sign(z[vm])).sum())
            val_n += int(vm.sum())
    return {
        "policy_ce": tot_ce / max(n, 1),
        "value_mse": tot_mse / max(val_n, 1),
        "aux_mse": tot_aux / max(n, 1),
        "policy_top1": top1_hits / max(top1_n, 1),
        "value_sign_acc": sign_hits / max(val_n, 1),
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
    # schema-v2 (docs §3d): deck-conditioned inputs, aux heads, and a BC-pretrain warmup
    ap.add_argument("--deck-cond", action="store_true", help="v2: feed deck_summary + archetype_id")
    ap.add_argument("--aux-heads", action="store_true", help="v2: train prize/turns-to-attack aux heads")
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--aux-include-bc", action="store_true",
                    help="train the aux heads on BC rows too (their aux targets are fabricated — off by default)")
    ap.add_argument("--bc-epochs", type=int, default=0,
                    help="v2: policy-only BC warmup epochs over SOURCE_BC (expert) rows before main training")
    ap.add_argument("--bc-buffer", default=None,
                    help="v2: extra buffer (e.g. ladder-BC) merged with --buffer on common keys")
    ap.add_argument("--mask-bc-value", action="store_true",
                    help="train value ONLY on MCTS rows (BC rows have fake all-+1 z → poison the value head)")
    ap.add_argument("--select", choices=["policy", "value", "sum"], default="policy",
                    help="checkpoint criterion: policy=top1 (default), value=value_sign_acc, sum=top1+sign")
    # v9: the sized-for-our-corpus preset + the Option C compositional card embedding
    ap.add_argument("--preset", choices=["v6", "v9"], default="v6",
                    help="v6 = the shipped arch (flags below apply); v9 = the sized preset (deck-cond + aux on)")
    ap.add_argument("--card-features", default=os.path.join(REPO, "experiments", "checkpoints",
                                                            "card_features.npz"),
                    help="frozen card-attribute matrix from scripts/build_card_features.py (v9 preset)")
    ap.add_argument("--residual-l2", type=float, default=1e-4,
                    help="explicit L2 pull toward zero on the per-card residual table r_c")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    buf = D.load_buffer(args.buffer)
    if args.bc_buffer:
        bc = D.load_buffer(args.bc_buffer)
        n_main, n_bc = len(buf.game_id), len(bc.game_id)
        buf = D.merge_buffers(buf, bc)
        src = buf.data.get("source")
        print(f"merged: {n_main} main + {n_bc} BC = {len(buf.game_id)} rows "
              f"({int((src==1).sum()) if src is not None else '?'} BC-source)")
    tr, va = D.split_by_game(buf, args.val_frac, seed=args.seed)
    print(f"buffer: {len(buf.game_id)} decisions from {buf.n_games} games "
          f"| train {len(tr)} / val {len(va)} | dropped PASS mass {buf.dropped_pass_mass:.2%}")
    if buf.dropped_pass_mass > 0.05:
        print(f"  NOTE: {buf.dropped_pass_mass:.1%} of visit mass was PASS (dropped) — consider a PASS logit")
    if args.preset == "v9":
        cfg = v9_config()
        if args.card_features and os.path.exists(args.card_features):
            n_res = int(np.load(args.card_features)["n_residual_cards"])
            cfg = v9_config(n_residual_cards=n_res)     # size r_c from the corpus frequency curve
    else:
        cfg = NetConfig(deck_cond=args.deck_cond, aux_heads=args.aux_heads)
    net = PTCGNet(cfg).to(device)
    if cfg.compositional_embed:
        if not os.path.exists(args.card_features):
            raise SystemExit(f"--preset v9 needs the card-attribute matrix; run "
                             f"scripts/build_card_features.py (missing: {args.card_features})")
        info = net.load_card_features(args.card_features)
        print(f"card features: attrs {info['attrs']}, residual rows {info['n_residual_cards']} "
              f"(L2 {args.residual_l2:g})")
    print(f"net: {net.num_params():,} params | preset={args.preset} deck_cond={cfg.deck_cond} "
          f"aux_heads={cfg.aux_heads} compositional={cfg.compositional_embed}")
    # check the BUILT config, not the flags — a preset turns these on without the flag being set
    if cfg.deck_cond and not D.has_deck_cond(buf):
        print("  WARN: deck_cond is on but buffer has no deck_summary (defaults to zeros)")
    if cfg.aux_heads and not D.has_aux(buf):
        print("  WARN: aux_heads is on but buffer has no aux targets (defaults to zeros)")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)

    base = evaluate(net, buf, va, args.batch, device, cfg.deck_cond, cfg.aux_heads)
    print(f"init  val: CE {base['policy_ce']:.3f}  MSE {base['value_mse']:.3f}  "
          f"top1 {base['policy_top1']:.1%}  sign {base['value_sign_acc']:.1%}")

    # --- BC pretrain: policy-only CE over expert (SOURCE_BC) rows (docs §3d step 5) -------------
    if args.bc_epochs > 0:
        src = buf.data.get("source")
        bc = tr[src[tr] == 1] if src is not None else np.array([], dtype=int)
        if len(bc) == 0:
            print("BC pretrain: no SOURCE_BC rows in train split — skipped")
        else:
            print(f"BC pretrain: {len(bc)} expert rows, {args.bc_epochs} epochs (policy CE only)")
            for ep in range(1, args.bc_epochs + 1):
                net.train()
                order = bc.copy(); rng.shuffle(order); run = 0.0
                for s in range(0, len(order), args.batch):
                    b = order[s:s + args.batch]
                    bd, pi, _z = make_batch(buf, b, device, cfg.deck_cond)
                    logits, _v, _a = net.forward_train(bd)
                    ce = policy_ce(logits, pi)
                    opt.zero_grad(); ce.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
                    run += ce.item() * len(b)
                m = evaluate(net, buf, va, args.batch, device, cfg.deck_cond, cfg.aux_heads)
                print(f"  bc ep {ep:2d}  train CE {run/len(bc):.3f}  |  val top1 {m['policy_top1']:.1%}")

    best = -1.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ep in range(1, args.epochs + 1):
        net.train()
        order = tr.copy()
        rng.shuffle(order)
        run_ce = run_mse = run_aux = 0.0
        for s in range(0, len(order), args.batch):
            b = order[s:s + args.batch]
            bd, pi, z = make_batch(buf, b, device, cfg.deck_cond)
            logits, value, aux = net.forward_train(bd)
            ce = policy_ce(logits, pi)
            if args.mask_bc_value:                        # value only on real-outcome (MCTS) rows
                src = torch.from_numpy(buf.data["source"][b]).to(device)
                vm = src == 0
                mse = torch.nn.functional.mse_loss(value[vm], z[vm]) if bool(vm.any()) else value.sum() * 0.0
            else:
                mse = torch.nn.functional.mse_loss(value, z)
            loss = ce + args.value_weight * mse
            if cfg.aux_heads:
                al = aux_loss(aux, D.aux_targets(buf, b, device), args.aux_include_bc)
                loss = loss + args.aux_weight * al
                run_aux += al.item() * len(b)
            if cfg.compositional_embed and args.residual_l2 > 0:
                loss = loss + args.residual_l2 * net.residual_l2()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            run_ce += ce.item() * len(b)
            run_mse += mse.item() * len(b)
        m = evaluate(net, buf, va, args.batch, device, cfg.deck_cond, cfg.aux_heads)
        crit = ({"policy": m["policy_top1"], "value": m["value_sign_acc"],
                 "sum": m["policy_top1"] + m["value_sign_acc"]})[args.select]
        star = ""
        if crit > best:
            best = crit
            torch.save({"model": net.state_dict(), "config": vars(cfg),
                        "val": m, "epoch": ep}, args.out)
            star = " *"
        aux_str = f" aux {run_aux/len(tr):.3f}/{m['aux_mse']:.3f}" if cfg.aux_heads else ""
        print(f"ep {ep:3d}  train CE {run_ce/len(tr):.3f} MSE {run_mse/len(tr):.3f}{aux_str}  |  "
              f"val CE {m['policy_ce']:.3f} MSE {m['value_mse']:.3f} "
              f"top1 {m['policy_top1']:.1%} sign {m['value_sign_acc']:.1%}{star}")

    meta = {"buffer": os.path.relpath(args.buffer, REPO), "games": buf.n_games,
            "decisions": int(len(buf.game_id)), "epochs": args.epochs,
            "deck_cond": cfg.deck_cond, "aux_heads": cfg.aux_heads,
            "select": args.select, "best_val_criterion": best, "dropped_pass_mass": buf.dropped_pass_mass}
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nsaved best checkpoint (select={args.select}, crit {best:.3f}) -> {os.path.relpath(args.out, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

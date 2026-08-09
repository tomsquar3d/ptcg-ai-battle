#!/usr/bin/env python3
"""Train the v9 preset on Kaggle (Step C). Runs locally too, on the same paths, for a dry run.

Designed around two facts about the Kaggle runtime:
  - a session can be killed at any time (12h cap, idle timeouts, preemption), so EVERY epoch writes
    a resumable checkpoint to /kaggle/working and `--resume` picks it back up mid-run;
  - the training path never imports `cg` (verified: the native SDK is nowhere in the chain), so no
    engine is needed — only numpy + torch over recorded shards.

METRICS DISCIPLINE (#72/#73): every metric is reported as a train/held-out PAIR, never alone. #72's
value head looked excellent at 0.871 AUC on the training buffer and was 0.660 on held-out; #73 saw
the same collapse (0.85 -> 0.59). A train number on its own is not evidence of anything here.

Gates, both on HELD-OUT:
  policy top-1  > 0.720   (v6 / net_ladder2 baseline, #73 table)
  value AUC     > 0.662   (prize-diff on #73's eval set)

The 0.662 bar comes from a different position set than this corpus, so the script also computes
prize-diff AUC on the *same* held-out rows as the net (globals[4] is the prize difference, the
feature #72 benchmarked). That second number is the internally-valid comparison; the 0.662 is the
cross-run reference. A value head only earns the leaf if it beats BOTH.

    # Kaggle: attach the corpus dataset, then
    !python train_kaggle_v9.py --epochs 40
    # local dry run
    .venv/bin/python scripts/train_kaggle_v9.py --epochs 1 --limit-games 200 --out-dir /tmp/v9run
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ON_KAGGLE = os.path.isdir("/kaggle/input")
KAGGLE_IN = "/kaggle/input"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POLICY_GATE = 0.720      # v6 / net_ladder2 held-out top-1 (#73)
VALUE_GATE = 0.662       # prize-diff held-out AUC on #73's eval set
PRIZE_DIFF_GLOBAL_IX = 4  # globals[4] = (opp_prizes - my_prizes)/6 — the #72 baseline feature


def _find(*, contains: str = None, name: str = None, kind: str = "dir") -> str | None:
    """Locate a corpus dir / file under /kaggle/input without hard-coding the dataset slug."""
    if not ON_KAGGLE:
        return None
    for root, dirs, files in os.walk(KAGGLE_IN):
        if kind == "file" and name in files:
            return os.path.join(root, name)
        if kind == "dir" and contains and glob.glob(os.path.join(root, contains)):
            return root
    return None


def rankdata(x: np.ndarray) -> np.ndarray:
    """Tie-averaged ranks (vectorized), so AUC is exact when scores collide — prize-diff is ~0 on
    most positions (#72), which makes tie handling the difference between a real number and noise."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    uniq, first, counts = np.unique(xs, return_index=True, return_counts=True)
    avg = first + (counts - 1) / 2.0 + 1.0
    r = np.empty(len(x), dtype=np.float64)
    r[order] = avg[np.searchsorted(uniq, xs)]
    return r


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score(win) > score(loss))."""
    pos = int(positive.sum())
    neg = int(len(positive) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(np.asarray(scores, dtype=np.float64))
    return float((r[positive].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None, help="dir of .npz shards (auto-discovered on Kaggle)")
    ap.add_argument("--card-features", default=None, help="card_features.npz (auto-discovered)")
    ap.add_argument("--src", default=None, help="dir containing the `nn` package (auto-discovered)")
    ap.add_argument("--out-dir", default="/kaggle/working" if ON_KAGGLE else "/tmp/v9run")
    ap.add_argument("--epochs", type=int, default=40, help="ceiling; early stopping usually ends it sooner")
    ap.add_argument("--patience", type=int, default=5,
                    help="stop after this many epochs with no held-out top-1 improvement (0 = off)")
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="held-out top-1 must improve by more than this to reset patience")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--residual-l2", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-metric-rows", type=int, default=20000,
                    help="rows sampled from train for the train half of each metric pair (0 = all)")
    ap.add_argument("--limit-games", type=int, default=0, help="dry-run: cap games loaded")
    ap.add_argument("--resume", action="store_true", help="continue from out-dir/v9_last.pt if present")
    args = ap.parse_args()

    src = args.src or _find(contains="nn") or os.path.join(REPO, "src")
    sys.path.insert(0, src)
    import torch                                                    # noqa: E402
    from nn.encoder import PTCGNet, v9_config                       # noqa: E402
    from nn import dataset as D                                     # noqa: E402
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from train_net import policy_ce, aux_loss                       # noqa: E402

    corpus = args.corpus or _find(contains="*.npz") or os.path.join(REPO, "experiments", "selfplay", "v9_corpus")
    feats = (args.card_features or _find(name="card_features.npz", kind="file")
             or os.path.join(REPO, "experiments", "checkpoints", "card_features.npz"))
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"src={src}\ncorpus={corpus}\ncard_features={feats}\nout={args.out_dir}  device={device}")
    if not os.path.exists(feats):
        raise SystemExit(f"card_features.npz not found ({feats}) — upload it with the corpus, or "
                         f"rebuild it with scripts/build_card_features.py")

    t0 = time.time()
    buf = D.load_buffer(corpus)
    if args.limit_games:
        games = np.unique(buf.game_id)[:args.limit_games]
        keep = np.isin(buf.game_id, games)
        buf = D.Buffer(data={k: v[keep] for k, v in buf.data.items()}, pi=buf.pi[keep],
                       game_id=buf.game_id[keep], n_games=len(games),
                       dropped_pass_mass=buf.dropped_pass_mass)
    tr, va = D.split_by_game(buf, args.val_frac, seed=args.seed)
    assert not (set(buf.game_id[tr].tolist()) & set(buf.game_id[va].tolist())), "train/val share a game"
    print(f"buffer: {len(buf.game_id):,} decisions / {buf.n_games:,} games in {time.time()-t0:.0f}s | "
          f"train {len(tr):,} / val {len(va):,} rows | dropped PASS mass {buf.dropped_pass_mass:.2%}")

    n_res = int(np.load(feats)["n_residual_cards"])
    cfg = v9_config(n_residual_cards=n_res)
    net = PTCGNet(cfg).to(device)
    net.load_card_features(feats)
    aux_p = sum(p.numel() for n, p in net.named_parameters() if n.startswith(("prize_head", "tta_head")))
    print(f"net: {net.num_params():,} params ({net.num_params()-aux_p:,} at inference) | "
          f"n_residual_cards={n_res} | deck_cond={cfg.deck_cond} aux={cfg.aux_heads}")
    if not D.has_deck_cond(buf):
        print("  WARN: buffer has no deck_summary — deck conditioning will train on zeros")
    if not D.has_aux(buf):
        print("  WARN: buffer has no aux targets — aux heads will train on zeros")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_ep, best, stale, stopped_early = 1, -1.0, 0, False
    last_path = os.path.join(args.out_dir, "v9_last.pt")
    best_path = os.path.join(args.out_dir, "v9_best.pt")
    hist_path = os.path.join(args.out_dir, "v9_history.json")
    history = []
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        start_ep, best = ck["epoch"] + 1, ck.get("best", -1.0)
        stale = ck.get("stale", 0)          # patience must survive a killed session, not reset to 0
        history = ck.get("history", [])
        print(f"resumed from {last_path} at epoch {start_ep} (best held-out top1 {best:.4f}, "
              f"{stale}/{args.patience} epochs stale)")

    def batches(idx, shuffle, rng=None):
        order = idx.copy()
        if shuffle:
            rng.shuffle(order)
        for s in range(0, len(order), args.batch):
            yield order[s:s + args.batch]

    def make(b):
        bd, pi, z = D.make_batch(buf, b, device)
        ds, aid = D.deck_cond_batch(buf, b, device)
        bd["deck_summary"], bd["archetype_id"] = ds, aid
        return bd, pi, z

    src_col = buf.data.get("source")
    globals_col = buf.data["globals"]

    @torch.no_grad()
    def metrics(idx: np.ndarray) -> dict:
        """CE / MSE / aux + policy top-1 + value AUC on one index set. Used for BOTH halves of every
        reported pair, so train and held-out numbers are computed identically and are comparable."""
        net.eval()
        tot_ce = tot_mse = tot_aux = 0.0
        n = top1_hits = top1_n = 0
        vals, zs = [], []
        for b in batches(idx, shuffle=False):
            bd, pi, z = make(b)
            logits, value, aux = net.forward_train(bd)
            tot_ce += policy_ce(logits, pi).item() * len(b)
            tot_aux += aux_loss(aux, D.aux_targets(buf, b, device)).item() * len(b)
            n += len(b)
            has = pi.sum(1) > 0
            if bool(has.any()):
                top1_hits += int((logits.argmax(1)[has] == pi.argmax(1)[has]).sum())
                top1_n += int(has.sum())
            real = ((torch.from_numpy((src_col[b] == 0)).to(device) if src_col is not None
                     else torch.ones(len(b), dtype=torch.bool, device=device)) & (z != 0))
            if bool(real.any()):
                tot_mse += torch.nn.functional.mse_loss(value[real], z[real]).item() * int(real.sum())
                vals.append(value[real].detach().cpu().numpy())
                zs.append(z[real].detach().cpu().numpy())
        v = np.concatenate(vals) if vals else np.zeros(0)
        zz = np.concatenate(zs) if zs else np.zeros(0)
        real_idx = idx[(src_col[idx] == 0) if src_col is not None else np.ones(len(idx), bool)]
        real_z = buf.data["z"][real_idx]
        pd_mask = real_z != 0
        return {
            "policy_ce": tot_ce / max(n, 1),
            "value_mse": tot_mse / max(len(v), 1),
            "aux_mse": tot_aux / max(n, 1),
            "policy_top1": top1_hits / max(top1_n, 1),
            "value_auc": auc(v, zz > 0) if len(v) else float("nan"),
            "prize_auc": (auc(globals_col[real_idx][pd_mask, PRIZE_DIFF_GLOBAL_IX], real_z[pd_mask] > 0)
                          if pd_mask.any() else float("nan")),
            "n_rows": int(len(idx)), "n_value_rows": int(len(v)),
        }

    rng = np.random.default_rng(args.seed)
    tr_metric_idx = tr if (args.train_metric_rows <= 0 or len(tr) <= args.train_metric_rows) else \
        rng.choice(tr, args.train_metric_rows, replace=False)

    def report(ep, tm, vm, extra=""):
        # GAP = train - held-out, its own column on both headline metrics. #72 and #73 each turned
        # on this number (0.871 -> 0.660 and 0.85 -> 0.59); it should be impossible to miss here.
        gap_top1 = tm["policy_top1"] - vm["policy_top1"]
        gap_auc = tm["value_auc"] - vm["value_auc"]
        print(f"ep {ep:3d} | CE  train {tm['policy_ce']:.4f}  held-out {vm['policy_ce']:.4f}"
              f"   | MSE train {tm['value_mse']:.4f}  held-out {vm['value_mse']:.4f}"
              f"   | aux train {tm['aux_mse']:.4f}  held-out {vm['aux_mse']:.4f}")
        p_ok = "PASS" if vm["policy_top1"] > POLICY_GATE else "fail"
        v_ok = "PASS" if vm["value_auc"] > VALUE_GATE else "fail"
        v_ok2 = "PASS" if vm["value_auc"] > vm["prize_auc"] else "fail"
        print(f"        | top1 train {tm['policy_top1']:.4f}  held-out {vm['policy_top1']:.4f}"
              f"  GAP {gap_top1:+.4f}  vs gate {POLICY_GATE:.3f} [{p_ok}]")
        print(f"        | AUC  train {tm['value_auc']:.4f}  held-out {vm['value_auc']:.4f}"
              f"  GAP {gap_auc:+.4f}  vs gate {VALUE_GATE:.3f} [{v_ok}]"
              f"  vs prize-diff here {vm['prize_auc']:.4f} [{v_ok2}]{extra}")
        return {"gap_policy_top1": gap_top1, "gap_value_auc": gap_auc}

    print(f"\ntrain metrics on {len(tr_metric_idx):,} sampled train rows; held-out on all {len(va):,}. "
          f"No metric is reported without its pair (#72/#73).")
    base_tr, base_va = metrics(tr_metric_idx), metrics(va)
    report(0, base_tr, base_va, "  (init)")

    for ep in range(start_ep, args.epochs + 1):
        net.train()
        t1 = time.time()
        run_ce = run_mse = run_aux = 0.0
        seen = 0
        for b in batches(tr, shuffle=True, rng=rng):
            bd, pi, z = make(b)
            logits, value, aux = net.forward_train(bd)
            ce = policy_ce(logits, pi)
            mse = torch.nn.functional.mse_loss(value, z)
            loss = ce + args.value_weight * mse + args.aux_weight * aux_loss(
                aux, D.aux_targets(buf, b, device))
            if args.residual_l2 > 0:
                loss = loss + args.residual_l2 * net.residual_l2()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            run_ce += ce.item() * len(b)
            run_mse += mse.item() * len(b)
            seen += len(b)
        tm, vm = metrics(tr_metric_idx), metrics(va)
        gaps = report(ep, tm, vm, f"   [{time.time()-t1:.0f}s]")

        row = {"epoch": ep, "train": tm, "held_out": vm, **gaps,
               "train_loss_ce": run_ce / max(seen, 1), "train_loss_mse": run_mse / max(seen, 1)}
        history.append(row)
        # update `best` BEFORE writing v9_last: a resume reads it back, and a stale value would let
        # a later, worse epoch clear the bar and overwrite a better v9_best.
        improved = vm["policy_top1"] > best + args.min_delta
        if improved:
            best = vm["policy_top1"]
            stale = 0
        else:
            stale += 1
        state = {"model": net.state_dict(), "optim": opt.state_dict(), "config": vars(cfg),
                 "epoch": ep, "best": best, "stale": stale, "history": history}
        torch.save(state, last_path)                 # EVERY epoch — a killed session resumes here
        if improved:
            torch.save(state, best_path)
            print(f"        * new best held-out top1 {best:.4f} -> {os.path.basename(best_path)}")
        with open(hist_path, "w") as fh:
            json.dump(history, fh, indent=2)

        if args.patience > 0 and stale >= args.patience:
            stopped_early = True
            print(f"\nEARLY STOP: held-out top-1 has not improved for {stale} epochs "
                  f"(patience {args.patience}); best {best:.4f} at epoch "
                  f"{max(history, key=lambda r: r['held_out']['policy_top1'])['epoch']}. "
                  f"Ceiling was {args.epochs}.")
            break
        if not improved:
            print(f"        . no improvement ({stale}/{args.patience} toward early stop)")

    if history:
        top = max(history, key=lambda r: r["held_out"]["policy_top1"])
        fin = top["held_out"]
        print(f"\n=== GATES (held-out, best epoch {top['epoch']} of {history[-1]['epoch']} run"
              f"{', early-stopped' if stopped_early else ''}) ===")
        print(f"GAP at that epoch: top-1 {top['gap_policy_top1']:+.4f}, AUC {top['gap_value_auc']:+.4f} "
              f"(train − held-out)")
        print(f"policy top-1 {fin['policy_top1']:.4f} vs {POLICY_GATE:.3f} "
              f"-> {'PASS' if fin['policy_top1'] > POLICY_GATE else 'FAIL'}")
        print(f"value AUC    {fin['value_auc']:.4f} vs {VALUE_GATE:.3f} (#73 cross-run) "
              f"-> {'PASS' if fin['value_auc'] > VALUE_GATE else 'FAIL'}")
        print(f"value AUC    {fin['value_auc']:.4f} vs {fin['prize_auc']:.4f} (prize-diff, same rows) "
              f"-> {'PASS' if fin['value_auc'] > fin['prize_auc'] else 'FAIL'}")
        print(f"\ncheckpoints: {best_path} (best), {last_path} (resume), {hist_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Train + evaluate the NHITSx baseline on Beijing PRSA Aotizhongxin PM2.5 with
known-future covariates.

Comparability contract -- identical to train_tide.py / train_tft.py /
train_deepar.py / etc:
  * Same 70/10/20 chronological split via PRSA_EncDec_Dataset.
  * Same StandardScaler-on-train-target -- z-scored MAE/MSE land in the same
    z-space as every other baseline; raw µg/m³ = z * target_scaler.scale_[0].
  * Known-future covariates (meteorology + other pollutants + calendar) are
    concatenated to each NHITS block's MLP input (the "x" of NHITSx).
    PM2.5 over the forecast window is NEVER an input.
  * MSE loss + point-estimate output (B, pred_len, 1) for direct MAE
    comparability.
  * Early stopping on val MAE.

Run
---
    PYTHONPATH=. python -u comparison_model/train_nhits.py \\
        --dataset ./dataset/PRSA_data/Aotizhongxin.csv \\
        --lookback_len 96 --pred_len 24 \\
        --epoch 50 --batch_size 128 --lr 1e-3 --patience 8 \\
        --pool_kernels 8 4 1 --mlp_hidden 512 --mlp_depth 2 --dropout 0.1 \\
        --checkpoint_dir ./checkpoint/nhits
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comparison_model.nhits_model import NHITSx
from comparison_model.encdec_dataset import PRSA_EncDec_Dataset


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--lookback_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--epoch", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    # NHITSx hyperparameters
    p.add_argument("--pool_kernels", type=int, nargs="+", default=[24, 12, 1],
                   help="One per stack. Decreasing recommended. For "
                        "pred_len=96 on hourly data we default to (24, 12, 1) "
                        "= daily / half-day / hourly basis — the first block "
                        "captures the multi-day trend, the middle block "
                        "captures intra-day pattern, the last block captures "
                        "hourly detail. For pred_len=24 a tighter (8, 4, 1) "
                        "is more appropriate.")
    p.add_argument("--mlp_hidden", type=int, default=512)
    p.add_argument("--mlp_depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoint/nhits")
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def set_seed(seed=0):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)


def _move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (mse_z, mae_z) averaged over every forecast element."""
    model.eval()
    mae_sum, mse_sum, denom = 0.0, 0.0, 0
    for batch in loader:
        batch = _move_batch(batch, device)
        y_hat = model(batch)
        y_true = batch["y_true"]
        mae_sum += (y_hat - y_true).abs().sum().item()
        mse_sum += ((y_hat - y_true) ** 2).sum().item()
        denom += y_true.numel()
    return mse_sum / max(1, denom), mae_sum / max(1, denom)


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    set_seed(args.seed)

    print("\n── Loading data ──")
    train_set = PRSA_EncDec_Dataset(args.dataset, "train", args.lookback_len, args.pred_len)
    val_set = PRSA_EncDec_Dataset(args.dataset, "vali", args.lookback_len, args.pred_len,
                                   target_scaler=train_set.target_scaler,
                                   real_scaler=train_set.real_scaler)
    test_set = PRSA_EncDec_Dataset(args.dataset, "test", args.lookback_len, args.pred_len,
                                    target_scaler=train_set.target_scaler,
                                    real_scaler=train_set.real_scaler)
    print(f"  real covariates ({train_set.n_real}): {train_set.real_cols}")
    print(f"  calendar features = {train_set.n_mark}")
    print(f"  train / vali / test windows = "
          f"{len(train_set)} / {len(val_set)} / {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print("\n── Building NHITSx ──")
    model = NHITSx(
        n_real=train_set.n_real, n_mark=train_set.n_mark,
        lookback_len=args.lookback_len, pred_len=args.pred_len,
        pool_kernels=tuple(args.pool_kernels),
        mlp_hidden=args.mlp_hidden, mlp_depth=args.mlp_depth,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  pool_kernels = {args.pool_kernels}  ({len(args.pool_kernels)} blocks)")
    print(f"  total trainable params: {n_params/1e6:.2f} M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "nhits.pth")

    if not args.eval_only:
        best_val_mae = float("inf"); best_epoch = -1; no_improve = 0
        print("\n── Training ──")
        for epoch in range(args.epoch):
            model.train()
            cum, n = 0.0, 0
            for batch in train_loader:
                batch = _move_batch(batch, device)
                y_hat = model(batch)
                loss = criterion(y_hat, batch["y_true"])
                optimizer.zero_grad()
                loss.backward()
                if args.clip_grad_norm and args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                cum += loss.item(); n += 1
            val_mse, val_mae = evaluate(model, val_loader, device)
            improved = val_mae < best_val_mae
            marker = " *" if improved else ""
            print(f"  [epoch {epoch+1:>3d}/{args.epoch}]  "
                  f"train_MSE={cum/max(1,n):.4f}  val MAE={val_mae:.4f}{marker}")
            if improved:
                best_val_mae = val_mae; best_epoch = epoch + 1; no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"  Early stop: no val-MAE improvement for {args.patience} "
                      f"epochs. Best val MAE = {best_val_mae:.4f} @ epoch {best_epoch}.")
                break
        print(f"\n  Best val MAE = {best_val_mae:.4f} @ epoch {best_epoch}.")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}; train first.")
    print(f"\n── Loading best checkpoint ({ckpt_path}) ──")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print("\n── Test forecast results (NHITSx) ──")
    test_mse, test_mae = evaluate(model, test_loader, device)
    scale = test_set.target_scale()
    test_mse_raw = test_mse * (scale ** 2); test_mae_raw = test_mae * scale
    print(f"  test MSE      = {test_mse:.6f}  (z-scored)")
    print(f"  test MAE      = {test_mae:.6f}  (z-scored)")
    print(f"  test MSE      = {test_mse_raw:.4f}  (RAW µg/m³, RMSE = {test_mse_raw**0.5:.4f})")
    print(f"  test MAE      = {test_mae_raw:.4f}  (RAW µg/m³)")
    print("\n  (Directly comparable to CBM/train_cbm.py and every other "
          "comparison_model/* baseline.)")


if __name__ == "__main__":
    main()

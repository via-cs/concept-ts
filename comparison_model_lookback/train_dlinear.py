"""
Train + evaluate DLinear as a STRICT LOOKBACK-ONLY baseline on Beijing PRSA
Aotizhongxin PM2.5. Directly comparable to ProtoLLM's CBM run in
``experiment_mode='lookback_only'`` (same 70/10/20 split, same
train-only target StandardScaler, same z-space MAE + raw µg/m³ reporting).

Run
---
    PYTHONPATH=. python -u comparison_model_lookback/train_dlinear.py \\
        --dataset ./dataset/PRSA_data/Aotizhongxin.csv \\
        --lookback_len 96 --pred_len 96 \\
        --moving_avg 25 --individual 0 \\
        --epoch 30 --batch_size 32 --lr 1e-3 --patience 5 \\
        --checkpoint_dir ./checkpoint/dlinear_lookback
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
from comparison_model_lookback.DLinear import DLinear
from comparison_model_lookback.lookback_dataset import PRSA_LookbackOnly_Dataset


def get_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--lookback_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--epoch", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed. If omitted, a fresh random seed is drawn each run and printed (so runs differ but stay traceable).")
    # DLinear hyperparameters
    p.add_argument("--moving_avg", type=int, default=25,
                   help="Kernel size for series_decomp trend extraction.")
    p.add_argument("--individual", type=int, default=0,
                   help="1 = separate seasonal/trend linears per channel; "
                        "0 = shared across channels (paper's default).")
    p.add_argument("--clip_grad_norm", type=float, default=0.0)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoint/dlinear_lookback")
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _move_tuple(batch, device):
    return tuple(t.to(device, non_blocking=True) for t in batch)


@torch.no_grad()
def evaluate(model, loader, device):
    """MAE / MSE on the PM2.5 channel only (last channel of the multi-channel
    forecast), z-scored space. Same aggregation as CBM/train_cbm.py."""
    model.eval()
    mae_sum, mse_sum, denom = 0.0, 0.0, 0
    for batch in loader:
        batch = _move_tuple(batch, device)
        inputs, y_true = model.prepare_batch(batch)                             # y_true: (B, H, 1)
        y_hat = model(*inputs)                                                  # (B, H, C)
        y_hat = y_hat[:, :, -1:]                                                # (B, H, 1) target = last channel
        mae_sum += (y_hat - y_true).abs().sum().item()
        mse_sum += ((y_hat - y_true) ** 2).sum().item()
        denom   += y_true.numel()
    return mse_sum / max(1, denom), mae_sum / max(1, denom)


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if args.seed is None:
        args.seed = random.randrange(2**31)
    print(f"[seed] run seed = {args.seed}")
    set_seed(args.seed)

    print("\n── Loading data (STRICT lookback-only) ──")
    train_set = PRSA_LookbackOnly_Dataset(
        args.dataset, "train", args.lookback_len, args.pred_len,
    )
    val_set = PRSA_LookbackOnly_Dataset(
        args.dataset, "vali", args.lookback_len, args.pred_len,
        target_scaler=train_set.target_scaler,
        feature_scaler=train_set.feature_scaler,
    )
    test_set = PRSA_LookbackOnly_Dataset(
        args.dataset, "test", args.lookback_len, args.pred_len,
        target_scaler=train_set.target_scaler,
        feature_scaler=train_set.feature_scaler,
    )
    C = train_set.C
    print(f"  input channels (lookback only, target LAST): {C}")
    print(f"  feature cols: {train_set.feature_cols} + [PM2.5]")
    print(f"  train / vali / test windows = "
          f"{len(train_set)} / {len(val_set)} / {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)

    print("\n── Building DLinear ──")
    model = DLinear(
        seq_len=args.lookback_len, pred_len=args.pred_len,
        moving_avg=args.moving_avg, enc_in=C, individual=bool(args.individual),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total trainable params: {n_params/1e6:.4f} M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "dlinear.pth")

    if not args.eval_only:
        best_val_mae = float("inf"); best_epoch = -1; no_improve = 0
        print("\n── Training ──")
        for epoch in range(args.epoch):
            model.train()
            cum, n = 0.0, 0
            for batch in train_loader:
                batch = _move_tuple(batch, device)
                inputs, y_true = model.prepare_batch(batch)
                y_hat = model(*inputs)[:, :, -1:]                               # (B, H, 1)
                loss = criterion(y_hat, y_true)
                optimizer.zero_grad()
                loss.backward()
                if args.clip_grad_norm > 0:
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
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}; train first (drop --eval_only).")
    print(f"\n── Loading best checkpoint ({ckpt_path}) ──")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print("\n── Test forecast results (DLinear, lookback-only) ──")
    test_mse, test_mae = evaluate(model, test_loader, device)
    scale = test_set.target_scale()
    print(f"  test MSE = {test_mse:.6f}  (z-scored)")
    print(f"  test MAE = {test_mae:.6f}  (z-scored)")
    print(f"  test MSE = {test_mse * scale**2:.4f}  (RAW µg/m³, RMSE = {(test_mse * scale**2)**0.5:.4f})")
    print(f"  test MAE = {test_mae * scale:.4f}  (RAW µg/m³)")
    print("\n  (Directly comparable to CBM/train_cbm.py --experiment_mode lookback_only.)")


if __name__ == "__main__":
    main()

"""
Train + evaluate the Informer baseline on Beijing PRSA Aotizhongxin PM2.5,
reporting metrics in the SAME units as ProtoLLM's CBM / TiDE / TFT.

Comparability contract -- identical to train_tide.py / train_tft.py:
  * Same 70/10/20 chronological split via PRSA_EncDec_Dataset.
  * Same StandardScaler-on-train-target so z-scored MAE/MSE land in the
    same z-space as TiDE/TFT/CBM; raw µg/m³ = z * target_scaler.scale_[0].
  * MSE training loss, MAE early-stopping criterion.
  * KNOWN-FUTURE exogenous (other pollutants + meteorology) in the forecast
    horizon -- the future half of x_dec carries the real future values
    (PM2.5 column zeroed), unlike the cap original which fed zeros.

Run
---
    PYTHONPATH=. python -u comparison_model/train_informer.py \\
        --dataset ./dataset/PRSA_data/Aotizhongxin.csv \\
        --lookback_len 96 --pred_len 24 --label_len 48 \\
        --epoch 30 --batch_size 32 --lr 1e-4 --patience 5 \\
        --d_model 256 --n_heads 4 --e_layers 2 --d_layers 1 --d_ff 512 \\
        --dropout 0.05 \\
        --checkpoint_dir ./checkpoint/informer
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
from comparison_model.Informer import Informer
from comparison_model.encdec_dataset import PRSA_EncDec_Dataset


def get_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--lookback_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--epoch", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed. If omitted, a fresh random seed is drawn each run and printed (so runs differ but stay traceable).")
    # Informer hyperparameters
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_layers", type=int, default=1)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--distil", action="store_true",
                   help="Use distillation conv layers between encoder layers.")
    p.add_argument("--embed", type=str, default="timeF",
                   choices=["fixed", "timeF"],
                   help="'timeF' (default) consumes the real normalized "
                        "calendar features emitted by PRSA_EncDec_Dataset.")
    p.add_argument("--freq", type=str, default="h")
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoint/informer")
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (mse_z, mae_z) averaged over every forecast element -- same
    aggregation as train_tide / train_tft / CBM/train_cbm."""
    model.eval()
    mae_sum, mse_sum, denom = 0.0, 0.0, 0
    for batch in loader:
        batch = _move_batch(batch, device)
        inputs, y_true = model.prepare_batch(batch)
        y_hat = model(*inputs)
        mae_sum += (y_hat - y_true).abs().sum().item()
        mse_sum += ((y_hat - y_true) ** 2).sum().item()
        denom += y_true.numel()
    return mse_sum / max(1, denom), mae_sum / max(1, denom)


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if args.seed is None:
        args.seed = random.randrange(2**31)
    print(f"[seed] run seed = {args.seed}")
    set_seed(args.seed)

    print("\n── Loading data ──")
    train_set = PRSA_EncDec_Dataset(args.dataset, "train", args.lookback_len, args.pred_len)
    val_set = PRSA_EncDec_Dataset(
        args.dataset, "vali", args.lookback_len, args.pred_len,
        target_scaler=train_set.target_scaler, real_scaler=train_set.real_scaler,
    )
    test_set = PRSA_EncDec_Dataset(
        args.dataset, "test", args.lookback_len, args.pred_len,
        target_scaler=train_set.target_scaler, real_scaler=train_set.real_scaler,
    )
    print(f"  real covariates ({train_set.n_real}): {train_set.real_cols}")
    print(f"  calendar features = {train_set.n_mark} (normalized in [-0.5, 0.5])")
    print(f"  train / vali / test windows = "
          f"{len(train_set)} / {len(val_set)} / {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # ----- Model: enc_in = dec_in = 1 + n_real (target + reals) -----
    in_dim = 1 + train_set.n_real
    print("\n── Building Informer ──")
    model = Informer(
        enc_in=in_dim, dec_in=in_dim,
        pred_len=args.pred_len, label_len=args.label_len,
        d_model=args.d_model, embed=args.embed, freq=args.freq,
        dropout=args.dropout, factor=args.factor, n_heads=args.n_heads,
        e_layers=args.e_layers, d_layers=args.d_layers, d_ff=args.d_ff,
        distil=args.distil,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total trainable params: {n_params/1e6:.2f} M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "informer.pth")

    if not args.eval_only:
        best_val_mae = float("inf"); best_epoch = -1; no_improve = 0
        print("\n── Training ──")
        for epoch in range(args.epoch):
            model.train()
            cum, n = 0.0, 0
            for batch in train_loader:
                batch = _move_batch(batch, device)
                inputs, y_true = model.prepare_batch(batch)
                y_hat = model(*inputs)
                loss = criterion(y_hat, y_true)
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
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}; train first (drop --eval_only).")
    print(f"\n── Loading best checkpoint ({ckpt_path}) ──")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print("\n── Test forecast results (Informer) ──")
    test_mse, test_mae = evaluate(model, test_loader, device)
    scale = test_set.target_scale()
    test_mse_raw = test_mse * (scale ** 2); test_mae_raw = test_mae * scale
    print(f"  test MSE      = {test_mse:.6f}  (z-scored)")
    print(f"  test MAE      = {test_mae:.6f}  (z-scored)")
    print(f"  test MSE      = {test_mse_raw:.4f}  (RAW µg/m³, RMSE = {test_mse_raw**0.5:.4f})")
    print(f"  test MAE      = {test_mae_raw:.4f}  (RAW µg/m³)")
    print("\n  (Directly comparable to CBM/train_cbm.py, train_tide.py, train_tft.py: "
          "same split, same target scaler, same known-future-covariate regime.)")


if __name__ == "__main__":
    main()

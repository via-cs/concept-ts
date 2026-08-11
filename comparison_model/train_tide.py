"""
Train + evaluate the TiDE baseline on Beijing PRSA Aotizhongxin PM2.5,
reporting metrics in the SAME units as ProtoLLM's CBM (CBM/train_cbm.py) so the
two are directly comparable.

Comparability guarantees
-------------------------
  * Same split + windows as dataloader/dataloader.py (legacy 70/10/20).
  * Same target normalization space (StandardScaler on train PM2.5), so:
      - z-scored MSE/MAE match ProtoLLM's "z-scored" numbers;
      - raw µg/m³ MAE = z_MAE * scale ; raw MSE = z_MSE * scale^2 ; RMSE = sqrt.
  * Known-future covariates (meteorology + other pollutants + calendar) over
    the whole forecast horizon -- the same information ProtoLLM's CBM uses --
    while PM2.5 is never an input over the forecast window.
  * Early stopping on validation MAE (same selection criterion as ProtoLLM).

Run
---
    PYTHONPATH=. python -u comparison_model/train_tide.py \
        --dataset ./dataset/PRSA_data/Aotizhongxin.csv \
        --lookback_len 96 --pred_len 96 \
        --epoch 30 --batch_size 16 --lr 3e-4 --patience 5 \
        --hidden_size 256 --num_encoder_layers 2 --num_decoder_layers 2 \
        --decoder_output_dim 8 --temporal_decoder_hidden 128 \
        --feat_size 4 --drop_prob 0.3 \
        --checkpoint_dir ./checkpoint/tide

To only evaluate an existing checkpoint, add --eval_only.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tide_model import TiDEModel
from tide_dataset import PRSA_TiDE_Dataset


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, required=True,
                   help="Path to Aotizhongxin.csv.")
    p.add_argument("--lookback_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--epoch", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5,
                   help="Early-stopping patience on val MAE. 0 disables.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed. If omitted, a fresh random seed is drawn each run and printed (so runs differ but stay traceable).")
    # TiDE hyperparameters
    p.add_argument("--feat_size", type=int, default=4)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_encoder_layers", type=int, default=2)
    p.add_argument("--num_decoder_layers", type=int, default=2)
    p.add_argument("--decoder_output_dim", type=int, default=8)
    p.add_argument("--temporal_decoder_hidden", type=int, default=128)
    p.add_argument("--drop_prob", type=float, default=0.3)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoint/tide")
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (mse_z, mae_z) averaged over every forecast element (same
    aggregation as CBM/train_cbm.evaluate: sum over all elements / numel)."""
    model.eval()
    mae_sum, mse_sum, denom = 0.0, 0.0, 0
    for y_past, x_cov, y_true in loader:
        y_past = y_past.to(device)
        x_cov = x_cov.to(device)
        y_true = y_true.to(device)
        y_hat = model(y_past, x_cov)
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

    # ----- Data (val/test share the TRAIN scalers) -----
    print("\n── Loading data ──")
    train_set = PRSA_TiDE_Dataset(args.dataset, "train", args.lookback_len, args.pred_len)
    val_set = PRSA_TiDE_Dataset(args.dataset, "vali", args.lookback_len, args.pred_len,
                                target_scaler=train_set.target_scaler,
                                cov_scaler=train_set.cov_scaler)
    test_set = PRSA_TiDE_Dataset(args.dataset, "test", args.lookback_len, args.pred_len,
                                 target_scaler=train_set.target_scaler,
                                 cov_scaler=train_set.cov_scaler)
    print(f"  covariate channels = {train_set.n_cov}  ({train_set.cov_cols})")
    print(f"  train / vali / test windows = "
          f"{len(train_set)} / {len(val_set)} / {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # ----- Model -----
    print("\n── Building TiDE ──")
    model = TiDEModel(
        n_covariates=train_set.n_cov,
        lookback_len=args.lookback_len,
        pred_len=args.pred_len,
        feat_size=args.feat_size,
        hidden_size=args.hidden_size,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        decoder_output_dim=args.decoder_output_dim,
        temporal_decoder_hidden=args.temporal_decoder_hidden,
        drop_prob=args.drop_prob,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total trainable params: {n_params/1e6:.2f} M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "tide.pth")

    # ----- Train -----
    if not args.eval_only:
        best_val_mae = float("inf")
        best_epoch = -1
        no_improve = 0
        print("\n── Training ──")
        for epoch in range(args.epoch):
            model.train()
            cum, n = 0.0, 0
            for y_past, x_cov, y_true in train_loader:
                y_past = y_past.to(device)
                x_cov = x_cov.to(device)
                y_true = y_true.to(device)
                y_hat = model(y_past, x_cov)
                loss = criterion(y_hat, y_true)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cum += loss.item()
                n += 1
            val_mse, val_mae = evaluate(model, val_loader, device)
            improved = val_mae < best_val_mae
            marker = " *" if improved else ""
            print(f"  [epoch {epoch+1:>3d}/{args.epoch}]  "
                  f"train_MSE={cum/max(1,n):.4f}  val MAE={val_mae:.4f}{marker}")
            if improved:
                best_val_mae = val_mae
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"  Early stop: no val-MAE improvement for {args.patience} "
                      f"epochs. Best val MAE = {best_val_mae:.4f} @ epoch {best_epoch}.")
                break
        print(f"\n  Best val MAE = {best_val_mae:.4f} @ epoch {best_epoch}.")

    # ----- Test -----
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}; train first (drop --eval_only).")
    print(f"\n── Loading best checkpoint ({ckpt_path}) ──")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print("\n── Test forecast results (TiDE) ──")
    test_mse, test_mae = evaluate(model, test_loader, device)
    scale = test_set.target_scale()
    test_mse_raw = test_mse * (scale ** 2)
    test_mae_raw = test_mae * scale
    print(f"  test MSE      = {test_mse:.6f}  (z-scored)")
    print(f"  test MAE      = {test_mae:.6f}  (z-scored)")
    print(f"  test MSE      = {test_mse_raw:.4f}  (RAW µg/m³, RMSE = {test_mse_raw**0.5:.4f})")
    print(f"  test MAE      = {test_mae_raw:.4f}  (RAW µg/m³)")
    print("\n  (Directly comparable to CBM/train_cbm.py's test MSE/MAE lines: "
          "same split, same target scaler, same known-future-covariate regime.)")


if __name__ == "__main__":
    main()

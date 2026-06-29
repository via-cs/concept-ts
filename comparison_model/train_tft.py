"""
Train + evaluate the Temporal Fusion Transformer (TFT) baseline on Beijing
PRSA Aotizhongxin PM2.5, reporting metrics in the SAME units as ProtoLLM's
CBM (CBM/train_cbm.py) and the TiDE baseline (comparison_model/train_tide.py)
so all three are directly comparable.

Comparability guarantees
-------------------------
  * Same chronological 70/10/20 split as dataloader/dataloader.py (when
    USE_LAST_YEAR_SPLIT = False), via PRSA_TFT_Dataset.
  * Same target normalization space: a single StandardScaler is fit on the
    train PM2.5 slice and reused for val/test. z-scored MSE/MAE therefore
    match ProtoLLM's "z-scored" numbers; raw µg/m³ MAE = z_MAE * scale.
  * Known-future covariates over the whole forecast horizon -- the same
    information ProtoLLM's CBM uses. PM2.5 is NEVER fed as input over the
    forecast window.
  * Early stopping on validation MAE (same selection criterion).
  * MSE loss + single-point output (not quantile) -- ensures the comparison
    is about ARCHITECTURE, not loss function (quantile loss optimises the
    median, which is not the MAE optimum for non-symmetric distributions).

Run
---
    PYTHONPATH=. python -u comparison_model/train_tft.py \
        --dataset ./dataset/PRSA_data/Aotizhongxin.csv \
        --lookback_len 96 --pred_len 24 \
        --epoch 30 --batch_size 64 --lr 5e-4 --patience 5 \
        --hidden_size 64 --n_heads 4 --num_lstm_layers 1 \
        --dropout 0.1 --embedding_dim 8 \
        --checkpoint_dir ./checkpoint/tft

Add --eval_only to skip training and load the existing checkpoint.
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
from tft_model import TFTModel
from tft_dataset import PRSA_TFT_Dataset


def get_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", type=str, required=True,
                   help="Path to Aotizhongxin.csv.")
    p.add_argument("--lookback_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=24)
    p.add_argument("--epoch", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5,
                   help="Early-stopping patience on val MAE. 0 disables.")
    p.add_argument("--seed", type=int, default=0)
    # TFT hyperparameters
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--num_lstm_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--embedding_dim", type=int, default=8)
    p.add_argument("--clip_grad_norm", type=float, default=1.0,
                   help="Gradient-norm clipping (TFT trains stably with ~1.0).")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoint/tft")
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _forward(model, batch):
    return model(
        batch["y_past_tgt"], batch["x_real_past"], batch["x_real_future"],
        batch["x_cat_past"], batch["x_cat_future"],
    )


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (mse_z, mae_z) averaged over every forecast element -- same
    aggregation as train_tide.evaluate and CBM/train_cbm.evaluate."""
    model.eval()
    mae_sum, mse_sum, denom = 0.0, 0.0, 0
    for batch in loader:
        batch = _move_batch(batch, device)
        y_hat, _ = _forward(model, batch)
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

    # ----- Data (val/test share the TRAIN scalers) -----
    print("\n── Loading data ──")
    train_set = PRSA_TFT_Dataset(args.dataset, "train", args.lookback_len, args.pred_len)
    val_set = PRSA_TFT_Dataset(
        args.dataset, "vali", args.lookback_len, args.pred_len,
        target_scaler=train_set.target_scaler, real_scaler=train_set.real_scaler,
    )
    test_set = PRSA_TFT_Dataset(
        args.dataset, "test", args.lookback_len, args.pred_len,
        target_scaler=train_set.target_scaler, real_scaler=train_set.real_scaler,
    )
    print(f"  real covariates  ({train_set.n_real}): {train_set.real_cols}")
    print(f"  cat  covariates  ({train_set.n_cat}): {train_set.cat_cols}  "
          f"cardinalities={train_set.cat_cardinalities}")
    print(f"  train / vali / test windows = "
          f"{len(train_set)} / {len(val_set)} / {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # ----- Model -----
    print("\n── Building TFT ──")
    model = TFTModel(
        n_real_known=train_set.n_real,
        cat_cardinalities=train_set.cat_cardinalities,
        lookback_len=args.lookback_len,
        pred_len=args.pred_len,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout,
        embedding_dim=args.embedding_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total trainable params: {n_params/1e6:.2f} M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "tft.pth")

    # ----- Train -----
    if not args.eval_only:
        best_val_mae = float("inf")
        best_epoch = -1
        no_improve = 0
        print("\n── Training ──")
        for epoch in range(args.epoch):
            model.train()
            cum, n = 0.0, 0
            for batch in train_loader:
                batch = _move_batch(batch, device)
                y_hat, _ = _forward(model, batch)
                loss = criterion(y_hat, batch["y_true"])
                optimizer.zero_grad()
                loss.backward()
                if args.clip_grad_norm and args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.clip_grad_norm)
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
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}; train first (drop --eval_only)."
        )
    print(f"\n── Loading best checkpoint ({ckpt_path}) ──")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print("\n── Test forecast results (TFT) ──")
    test_mse, test_mae = evaluate(model, test_loader, device)
    scale = test_set.target_scale()
    test_mse_raw = test_mse * (scale ** 2)
    test_mae_raw = test_mae * scale
    print(f"  test MSE      = {test_mse:.6f}  (z-scored)")
    print(f"  test MAE      = {test_mae:.6f}  (z-scored)")
    print(f"  test MSE      = {test_mse_raw:.4f}  (RAW µg/m³, RMSE = {test_mse_raw**0.5:.4f})")
    print(f"  test MAE      = {test_mae_raw:.4f}  (RAW µg/m³)")
    print("\n  (Directly comparable to CBM/train_cbm.py and "
          "comparison_model/train_tide.py: same split, same target scaler, "
          "same known-future-covariate regime.)")


if __name__ == "__main__":
    main()

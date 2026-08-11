"""
Train + evaluate the Autoformer baseline on Beijing PRSA Aotizhongxin PM2.5,
reporting metrics in the SAME z-space as ProtoLLM's CBM / TiDE / TFT / Informer.

Caveat: Autoformer architecturally does NOT consume real future-window
exogenous covariates -- only future CALENDAR features via x_mark_dec. See
comparison_model/Autoformer.py docstring. We keep the original architecture
unchanged so the comparison reflects each model's true capability.

Run
---
    PYTHONPATH=. python -u comparison_model/train_autoformer.py \\
        --dataset ./dataset/PRSA_data/Aotizhongxin.csv \\
        --lookback_len 96 --pred_len 24 \\
        --epoch 30 --batch_size 32 --lr 1e-4 --patience 5 \\
        --d_model 256 --n_heads 4 --num_layers 2 --d_ff 512 \\
        --dropout 0.05 \\
        --checkpoint_dir ./checkpoint/autoformer
"""

import random
import argparse, os, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comparison_model.Autoformer import Autoformer
from comparison_model.encdec_dataset import PRSA_EncDec_Dataset


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--lookback_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--epoch", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed. If omitted, a fresh random seed is drawn each run and printed (so runs differ but stay traceable).")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--moving_avg", type=int, default=25)
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--embed", type=str, default="timeF", choices=["fixed", "timeF"])
    p.add_argument("--freq", type=str, default="h")
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoint/autoformer")
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def set_seed(seed=0):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)


def _move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
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
    val_set = PRSA_EncDec_Dataset(args.dataset, "vali", args.lookback_len, args.pred_len,
                                   target_scaler=train_set.target_scaler,
                                   real_scaler=train_set.real_scaler)
    test_set = PRSA_EncDec_Dataset(args.dataset, "test", args.lookback_len, args.pred_len,
                                    target_scaler=train_set.target_scaler,
                                    real_scaler=train_set.real_scaler)
    print(f"  real covariates ({train_set.n_real}): {train_set.real_cols}")
    print(f"  train / vali / test windows = "
          f"{len(train_set)} / {len(val_set)} / {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    in_dim = 1 + train_set.n_real
    print("\n── Building Autoformer ──")
    model = Autoformer(
        input_dim=in_dim, output_dim=1,
        seq_len=args.lookback_len, pred_len=args.pred_len,
        d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
        num_layers=args.num_layers, dropout=args.dropout,
        moving_avg=args.moving_avg, factor=args.factor,
        embed=args.embed, freq=args.freq,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total trainable params: {n_params/1e6:.2f} M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "autoformer.pth")

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

    print("\n── Test forecast results (Autoformer) ──")
    test_mse, test_mae = evaluate(model, test_loader, device)
    scale = test_set.target_scale()
    test_mse_raw = test_mse * (scale ** 2); test_mae_raw = test_mae * scale
    print(f"  test MSE      = {test_mse:.6f}  (z-scored)")
    print(f"  test MAE      = {test_mae:.6f}  (z-scored)")
    print(f"  test MSE      = {test_mse_raw:.4f}  (RAW µg/m³, RMSE = {test_mse_raw**0.5:.4f})")
    print(f"  test MAE      = {test_mae_raw:.4f}  (RAW µg/m³)")
    print("\n  NOTE: Autoformer's forward() ignores future-real exogenous values "
          "in x_dec by design (decoder input is derived from x_enc via series "
          "decomposition). Only future CALENDAR features are consumed (via "
          "x_mark_dec). For an apples-to-apples future-real-exogenous "
          "comparison, see train_informer.py, train_tft.py, train_tide.py.")


if __name__ == "__main__":
    main()

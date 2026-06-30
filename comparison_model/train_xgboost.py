"""
XGBoost baseline for Beijing PRSA Aotizhongxin PM2.5 forecasting with
known-future covariates.

Same flattened-feature contract as train_random_forest.py / train_lightgbm.py
(see those files' docstrings for the feature layout).

Why this trainer is built sequentially (and NOT with joblib parallelism)
------------------------------------------------------------------------
Two XGBoost 2.x/3.x failure modes were causing mysterious crashes that
appeared as a phantom "^C" in the terminal even when the user never pressed
Ctrl+C:

  1. ``multi_strategy="multi_output_tree"`` with hist trees and large
     pred_len allocates per-feature, per-output histograms whose memory
     scales linearly with pred_len. For pred_len=96, 2784 features, 256
     hist bins, this is multi-GB per worker. The process OOM-crashes or
     XGBoost internally raises SIGINT, which the terminal echoes as "^C".

  2. ``sklearn.MultiOutputRegressor(n_jobs=-1)`` wrapping XGBoost spawns
     N joblib worker processes (loky backend). When any child process
     crashes (segfault, OOM, native C++ exception not propagated cleanly),
     joblib forwards the failure to the parent AS SIGINT, which the
     terminal again echoes as "^C". This is a long-standing joblib + native-
     code interaction issue.

Both bugs are SIGINT-pretending-to-be-user-interrupt. The fix is to do
NEITHER. We train ``pred_len`` XGBoost regressors SEQUENTIALLY in one Python
process, each using XGBoost's internal CPU threading (``n_jobs=-1``) for
parallelism within a single tree. No joblib, no worker spawning, no
multi-output trees -- so neither failure mode is reachable.

Trade-off: a sequential loop is slower in total wall-clock than an ideal
joblib parallel run would be, but on an 8-core machine the per-tree
parallelism inside XGBoost still gives us ~5-15 min for pred_len=96 with
the defaults. And -- importantly -- it actually finishes.

Run
---
    PYTHONPATH=. python -u comparison_model/train_xgboost.py \\
        --dataset ./dataset/PRSA_data/Aotizhongxin.csv \\
        --lookback_len 96 --pred_len 96 \\
        --n_estimators 100 --max_depth 6 --learning_rate 0.1 \\
        --inner_n_jobs -1 \\
        --checkpoint_dir ./checkpoint/xgboost
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comparison_model.encdec_dataset import (
    PRSA_EncDec_Dataset, flatten_dataset_for_tabular,
)


# ---------------------------------------------------------------------------
# Multi-output wrapper that is just a list of single-output regressors.
# Defined at module scope so joblib can pickle it.
# ---------------------------------------------------------------------------
class XGBSequentialMultiOutput:
    """Holds ``pred_len`` independent XGBoost single-output regressors and
    exposes a sklearn-style ``predict``. No multiprocessing involved."""

    def __init__(self, regressors):
        self.regressors = list(regressors)

    def predict(self, X):
        # Each regressor returns (N,); stack along the new axis to get (N, pred_len).
        return np.stack([r.predict(X) for r in self.regressors], axis=1)


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--lookback_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    # XGBoost hyperparameters
    p.add_argument("--n_estimators", type=int, default=100,
                   help="Boosting rounds per output regressor.")
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--subsample", type=float, default=0.8,
                   help="Row subsample per boosting iter.")
    p.add_argument("--colsample_bytree", type=float, default=0.8,
                   help="Feature subsample per tree.")
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument("--tree_method", type=str, default="hist",
                   choices=["hist", "exact", "approx"])
    p.add_argument("--inner_n_jobs", type=int, default=-1,
                   help="XGBoost internal thread count. -1 = all cores. "
                        "Used PER REGRESSOR; since we train sequentially "
                        "(one regressor at a time), oversubscription is "
                        "impossible.")
    p.add_argument("--progress_every", type=int, default=8,
                   help="Print a progress line every N regressors trained.")
    p.add_argument("--subsample_train", type=int, default=1,
                   help="Keep every Nth training window. 1 = all windows.")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoint/xgboost")
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = get_args()
    set_seed(args.seed)

    try:
        import xgboost as xgb
        import joblib
    except ImportError as e:
        raise ImportError(
            f"XGBoost baseline needs xgboost + joblib: {e}. "
            "Install with: pip install xgboost joblib"
        )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "xgb.joblib")

    # ----- Data -----
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
    print(f"  train / vali / test windows = "
          f"{len(train_set)} / {len(val_set)} / {len(test_set)}")

    print("\n── Flattening features ──")
    t0 = time.time()
    X_train, Y_train = flatten_dataset_for_tabular(train_set)
    X_val,   Y_val   = flatten_dataset_for_tabular(val_set)
    X_test,  Y_test  = flatten_dataset_for_tabular(test_set)
    print(f"  flattened in {time.time()-t0:.1f}s")

    if args.subsample_train > 1:
        X_train = X_train[::args.subsample_train]
        Y_train = Y_train[::args.subsample_train]
        print(f"  subsampled train to {len(X_train)} windows")

    print(f"  train X: {X_train.shape}   Y: {Y_train.shape}")
    print(f"  vali  X: {X_val.shape}     Y: {Y_val.shape}")
    print(f"  test  X: {X_test.shape}    Y: {Y_test.shape}")
    print(f"  XGBoost version: {xgb.__version__}")

    # ----- Train -----
    if not args.eval_only:
        print(f"\n── Training XGBoost (sequential, {args.pred_len} regressors) ──")
        print(f"  Each regressor: n_estimators={args.n_estimators}  "
              f"max_depth={args.max_depth}  lr={args.learning_rate}")
        print(f"  Internal threading: n_jobs={args.inner_n_jobs}  "
              f"tree_method={args.tree_method}")
        print(f"  No multiprocessing/joblib: avoids both the "
              f"multi_output_tree OOM and the joblib-SIGINT-cascade.")
        print(f"  Progress logged every {args.progress_every} regressors.\n")

        common_kwargs = dict(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            reg_alpha=args.reg_alpha,
            n_jobs=args.inner_n_jobs,
            random_state=args.seed,
            tree_method=args.tree_method,
            objective="reg:squarederror",
            verbosity=0,
        )

        regressors = []
        total_start = time.time()
        for step in range(args.pred_len):
            step_start = time.time()
            reg = xgb.XGBRegressor(**common_kwargs)
            reg.fit(X_train, Y_train[:, step])
            regressors.append(reg)

            done = step + 1
            if done % args.progress_every == 0 or done == args.pred_len:
                elapsed = time.time() - total_start
                rate = done / elapsed                      # regressors/sec
                eta_s = (args.pred_len - done) / max(rate, 1e-9)
                step_s = time.time() - step_start
                print(f"    [{done:>3d}/{args.pred_len}]  "
                      f"step={step_s:.1f}s  total={elapsed:.0f}s  "
                      f"ETA={eta_s:.0f}s  rate={rate*60:.1f}/min")

        fit_time = time.time() - total_start
        print(f"\n  total fit time: {fit_time:.1f}s ({fit_time/60:.1f} min)")

        model = XGBSequentialMultiOutput(regressors)

        # Val MAE
        print("  evaluating on val...")
        Y_val_hat = model.predict(X_val)
        val_mae = float(np.mean(np.abs(Y_val_hat - Y_val)))
        val_mse = float(np.mean((Y_val_hat - Y_val) ** 2))
        print(f"  val MAE = {val_mae:.4f} (z) / "
              f"{val_mae * val_set.target_scale():.4f} µg/m³")
        print(f"  val MSE = {val_mse:.4f} (z)")

        joblib.dump(model, ckpt_path)
        print(f"  saved to {ckpt_path}")

    # ----- Test -----
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}; train first.")
    print(f"\n── Loading checkpoint ({ckpt_path}) ──")
    model = joblib.load(ckpt_path)

    print("\n── Test forecast results (XGBoost) ──")
    Y_test_hat = model.predict(X_test)
    test_mae = float(np.mean(np.abs(Y_test_hat - Y_test)))
    test_mse = float(np.mean((Y_test_hat - Y_test) ** 2))
    scale = test_set.target_scale()
    test_mae_raw = test_mae * scale
    test_mse_raw = test_mse * (scale ** 2)
    print(f"  test MSE      = {test_mse:.6f}  (z-scored)")
    print(f"  test MAE      = {test_mae:.6f}  (z-scored)")
    print(f"  test MSE      = {test_mse_raw:.4f}  (RAW µg/m³, RMSE = {test_mse_raw**0.5:.4f})")
    print(f"  test MAE      = {test_mae_raw:.4f}  (RAW µg/m³)")
    print("\n  (Directly comparable to CBM/train_cbm.py and every other "
          "comparison_model/* baseline.)")


if __name__ == "__main__":
    main()

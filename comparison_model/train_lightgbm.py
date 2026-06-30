"""
LightGBM baseline for Beijing PRSA Aotizhongxin PM2.5 forecasting with
known-future covariates.

Same flattened-feature contract as train_random_forest.py / train_xgboost.py
(see those files' docstrings for the feature layout).

Multi-output strategy
---------------------
Unlike XGBoost 2.0+, LightGBM does NOT have a native vector-leaf multi-output
tree mode -- so we wrap ``lightgbm.LGBMRegressor`` in
``sklearn.multioutput.MultiOutputRegressor``. This trains ``pred_len``
independent boosters, one per forecast step. LightGBM is fast enough that
even 96 separate boosters fit in 5-15 min on a 24k-window training split
with -1 cores.

The independent-per-step approach has a subtle property: each forecast step
gets its OWN feature importance / split structure. This often gives stronger
results than tying everything to one tree, because near-term and far-term
forecasts depend on different features (recent lookback dominates step+1,
future weather dominates step+96).

Comparability: same 70/10/20 split + same StandardScaler-on-train-target as
all other baselines so z-scored MAE/MSE numbers are directly comparable.
Raw µg/m³ = z_MAE * target_scaler.scale_[0].

Run
---
    PYTHONPATH=. python -u comparison_model/train_lightgbm.py \\
        --dataset ./dataset/PRSA_data/Aotizhongxin.csv \\
        --lookback_len 96 --pred_len 96 \\
        --n_estimators 300 --learning_rate 0.1 --num_leaves 31 \\
        --n_jobs -1 \\
        --checkpoint_dir ./checkpoint/lightgbm
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


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--lookback_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    # LightGBM hyperparameters
    p.add_argument("--n_estimators", type=int, default=300,
                   help="Boosting rounds per output regressor.")
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--num_leaves", type=int, default=31,
                   help="Maximum number of leaves per tree (LightGBM's main "
                        "complexity knob -- prefer this over max_depth).")
    p.add_argument("--max_depth", type=int, default=-1,
                   help="Max tree depth. -1 disables the limit (let "
                        "num_leaves govern complexity).")
    p.add_argument("--min_data_in_leaf", type=int, default=20,
                   help="Minimum samples per leaf (LightGBM's name for "
                        "sklearn's min_samples_leaf).")
    p.add_argument("--feature_fraction", type=float, default=0.8,
                   help="Fraction of features sampled per tree.")
    p.add_argument("--bagging_fraction", type=float, default=0.8,
                   help="Fraction of rows sampled per boosting iter.")
    p.add_argument("--bagging_freq", type=int, default=5,
                   help="Perform bagging every k iterations. 0 disables.")
    p.add_argument("--reg_lambda", type=float, default=0.0)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--inner_n_jobs", type=int, default=1,
                   help="Threads INSIDE each LightGBM regressor. Total "
                        "parallelism = outer * inner. With outer=-1 and "
                        "inner=1, sklearn parallelises the 96 regressors "
                        "across all cores, one LightGBM core per regressor.")
    p.add_argument("--subsample_train", type=int, default=1,
                   help="Keep every Nth training window. 1 = all windows.")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoint/lightgbm")
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = get_args()
    set_seed(args.seed)

    try:
        import lightgbm as lgb
        from sklearn.multioutput import MultiOutputRegressor
        import joblib
    except ImportError as e:
        raise ImportError(
            f"LightGBM baseline needs lightgbm + scikit-learn + joblib: {e}. "
            "Install with: pip install lightgbm scikit-learn joblib"
        )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "lgbm.joblib")

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
    print(f"  LightGBM version: {lgb.__version__}")

    # ----- Train -----
    if not args.eval_only:
        print("\n── Training LightGBM (MultiOutputRegressor wrapper) ──")
        print(f"  Training pred_len={args.pred_len} independent boosters "
              f"({args.n_estimators} rounds each).")
        base = lgb.LGBMRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            min_data_in_leaf=args.min_data_in_leaf,
            feature_fraction=args.feature_fraction,
            bagging_fraction=args.bagging_fraction,
            bagging_freq=args.bagging_freq,
            reg_lambda=args.reg_lambda,
            reg_alpha=args.reg_alpha,
            n_jobs=args.inner_n_jobs,
            random_state=args.seed,
            objective="regression",
            verbosity=-1,
        )
        model = MultiOutputRegressor(base, n_jobs=args.n_jobs)
        t0 = time.time()
        model.fit(X_train, Y_train)
        print(f"  fit took {time.time()-t0:.1f}s")

        # Val MAE
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

    print("\n── Test forecast results (LightGBM) ──")
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

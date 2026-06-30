"""
XGBoost baseline for Beijing PRSA Aotizhongxin PM2.5 forecasting with
known-future covariates.

Same flattened-feature contract as train_random_forest.py (see that file's
docstring for the feature layout). XGBoost is one of the strongest tabular
baselines for forecasting tasks and routinely outperforms deep models in
published air-quality forecasting work, so it's a credibility check worth
including in any deep-learning comparison.

Multi-output strategy
---------------------
XGBoost 2.0+ supports native multi-output trees via
``multi_strategy="multi_output_tree"`` (one tree, one leaf, vector output).
Older XGBoost versions are wrapped via ``sklearn.multioutput.MultiOutputRegressor``
which trains pred_len independent regressors -- slower but functionally
equivalent. We auto-detect which path is available and fall back gracefully.

Early stopping
--------------
We pass the validation set as ``eval_set`` and use ``early_stopping_rounds``
to halt boosting when val MAE plateaus -- the same selection criterion as
every other baseline in this folder.

Comparability: same 70/10/20 split + same StandardScaler-on-train-target as
all other baselines so z-scored MAE/MSE numbers are directly comparable.
Raw µg/m³ = z_MAE * target_scaler.scale_[0].

Run
---
    PYTHONPATH=. python -u comparison_model/train_xgboost.py \\
        --dataset ./dataset/PRSA_data/Aotizhongxin.csv \\
        --lookback_len 96 --pred_len 24 \\
        --n_estimators 500 --max_depth 6 --learning_rate 0.05 \\
        --early_stopping_rounds 30 --n_jobs -1 \\
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


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--lookback_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    # XGBoost hyperparameters
    p.add_argument("--n_estimators", type=int, default=500)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8,
                   help="Fraction of training rows sampled per boosting iter.")
    p.add_argument("--colsample_bytree", type=float, default=0.8,
                   help="Fraction of features sampled per tree.")
    p.add_argument("--reg_lambda", type=float, default=1.0,
                   help="L2 regularization on leaf weights.")
    p.add_argument("--reg_alpha", type=float, default=0.0,
                   help="L1 regularization on leaf weights.")
    p.add_argument("--early_stopping_rounds", type=int, default=30,
                   help="Stop boosting after this many rounds with no val "
                        "improvement. 0 disables early stopping.")
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--tree_method", type=str, default="hist",
                   choices=["hist", "exact", "approx"],
                   help="hist (default) is the fastest reliable method.")
    p.add_argument("--force_multioutput_wrapper", action="store_true",
                   help="Force the sklearn MultiOutputRegressor wrapper even "
                        "on XGBoost 2.0+ (one regressor per forecast step).")
    p.add_argument("--subsample_train", type=int, default=1,
                   help="Keep every Nth training window. 1 = all windows.")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoint/xgboost")
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def _native_multioutput_supported(xgb_version: str) -> bool:
    """xgboost 2.0+ supports multi_strategy='multi_output_tree'."""
    try:
        major = int(xgb_version.split(".")[0])
        return major >= 2
    except Exception:
        return False


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
        common_kwargs = dict(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            reg_alpha=args.reg_alpha,
            n_jobs=args.n_jobs,
            random_state=args.seed,
            tree_method=args.tree_method,
            objective="reg:squarederror",
            eval_metric="mae",
        )

        use_native = (
            _native_multioutput_supported(xgb.__version__)
            and not args.force_multioutput_wrapper
        )
        if use_native:
            print("\n── Training XGBoost (native multi-output) ──")
            print(f"  Using multi_strategy='multi_output_tree' (xgboost 2.0+).")
            model = xgb.XGBRegressor(multi_strategy="multi_output_tree", **common_kwargs)
            t0 = time.time()
            try:
                model.fit(
                    X_train, Y_train,
                    eval_set=[(X_val, Y_val)],
                    early_stopping_rounds=(args.early_stopping_rounds
                                            if args.early_stopping_rounds > 0 else None),
                    verbose=10,
                )
            except TypeError:
                # Some 2.x patch versions changed the early_stopping kwarg into
                # a constructor-only param. Retry without it.
                model = xgb.XGBRegressor(
                    multi_strategy="multi_output_tree",
                    early_stopping_rounds=(args.early_stopping_rounds
                                            if args.early_stopping_rounds > 0 else None),
                    **common_kwargs,
                )
                model.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=10)
            print(f"  fit took {time.time()-t0:.1f}s")
        else:
            print("\n── Training XGBoost (MultiOutputRegressor wrapper) ──")
            print("  XGBoost < 2.0 or wrapper forced: training pred_len "
                  "independent regressors.")
            from sklearn.multioutput import MultiOutputRegressor
            base = xgb.XGBRegressor(**common_kwargs)
            # The wrapper doesn't natively pipe eval_set per-step, so we
            # skip early stopping in this fallback path. Use more conservative
            # n_estimators to avoid overfit.
            model = MultiOutputRegressor(base, n_jobs=1)  # parallelism is in xgb
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
          "comparison_model/* baseline: same split, same target scaler, same "
          "known-future-covariate regime.)")


if __name__ == "__main__":
    main()

"""
XGBoost baseline for Beijing PRSA Aotizhongxin PM2.5 forecasting with
known-future covariates.

Same flattened-feature contract as train_random_forest.py (see that file's
docstring for the feature layout).

Multi-output strategy
---------------------
XGBoost 2.0+ supports native multi-output trees via
``multi_strategy="multi_output_tree"`` (one tree per boost iteration with
vector-valued leaves). This is FAST and what we use by default.

Older XGBoost falls back to ``sklearn.multioutput.MultiOutputRegressor``,
which trains ``pred_len`` independent regressors -- slower but functionally
equivalent.

No early stopping with multi-output
-----------------------------------
xgboost 2.x has a known incompatibility between ``eval_metric``,
``early_stopping_rounds`` (which both moved into the CONSTRUCTOR in 2.x),
and ``multi_strategy="multi_output_tree"`` -- the single-output metrics
("mae"/"rmse") raise a shape error when combined with vector-valued targets.
To avoid the bug we just train for a fixed ``--n_estimators`` and evaluate
on val/test separately. If you need early stopping, drop pred_len to 1 (one
horizon step at a time) or use ``--force_multioutput_wrapper`` and rely on
each sub-regressor's own early stopping.

Run
---
    PYTHONPATH=. python -u comparison_model/train_xgboost.py \\
        --dataset ./dataset/PRSA_data/Aotizhongxin.csv \\
        --lookback_len 96 --pred_len 96 \\
        --n_estimators 300 --max_depth 6 --learning_rate 0.1 \\
        --n_jobs -1 --checkpoint_dir ./checkpoint/xgboost
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
    p.add_argument("--n_estimators", type=int, default=300,
                   help="Number of boosting rounds. 300 with default depth=6 "
                        "is a sane starting point; increase for stronger fit.")
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--subsample", type=float, default=0.8,
                   help="Row subsample per boosting iter.")
    p.add_argument("--colsample_bytree", type=float, default=0.8,
                   help="Feature subsample per tree.")
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--tree_method", type=str, default="hist",
                   choices=["hist", "exact", "approx"])
    p.add_argument("--force_multioutput_wrapper", action="store_true",
                   help="Force the sklearn MultiOutputRegressor wrapper "
                        "even on XGBoost 2.0+ (one regressor per forecast "
                        "step). Much slower but lets you use early stopping.")
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
        # CONSTRUCTOR ONLY: no eval_set, no early_stopping_rounds, no
        # eval_metric. xgboost 2.x raises shape errors when combining those
        # with multi_strategy="multi_output_tree" -- see module docstring.
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
        )

        use_native = (
            _native_multioutput_supported(xgb.__version__)
            and not args.force_multioutput_wrapper
        )
        if use_native:
            print("\n── Training XGBoost (native multi-output) ──")
            print(f"  Using multi_strategy='multi_output_tree' (xgboost 2.0+).")
            print(f"  Training for {args.n_estimators} rounds without early "
                  f"stopping (see module docstring for why).")
            model = xgb.XGBRegressor(
                multi_strategy="multi_output_tree", **common_kwargs,
            )
            t0 = time.time()
            model.fit(X_train, Y_train, verbose=False)
            print(f"  fit took {time.time()-t0:.1f}s")
        else:
            print("\n── Training XGBoost (MultiOutputRegressor wrapper) ──")
            print("  Training pred_len independent regressors. Slower but "
                  "compatible with XGBoost < 2.0.")
            from sklearn.multioutput import MultiOutputRegressor
            base = xgb.XGBRegressor(**common_kwargs)
            model = MultiOutputRegressor(base, n_jobs=1)
            t0 = time.time()
            model.fit(X_train, Y_train)
            print(f"  fit took {time.time()-t0:.1f}s")

        # Val MAE for reporting (no early stopping; this is informational only).
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

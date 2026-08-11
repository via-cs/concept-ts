"""
LightGBM baseline for Beijing PRSA Aotizhongxin PM2.5 forecasting with
known-future covariates.

Same flattened-feature contract as train_random_forest.py / train_xgboost.py
(see those files' docstrings for the feature layout).

Multi-output strategy
---------------------
LightGBM doesn't have a native vector-leaf multi-output tree mode, so we
train ``pred_len`` independent boosters (one per forecast step) SEQUENTIALLY
in one Python process. LightGBM uses its internal multi-threading
(``n_jobs=-1``) for parallelism within each booster.

We deliberately AVOID ``sklearn.multioutput.MultiOutputRegressor(n_jobs=-1)``
even though it's the obvious choice: joblib workers wrapping native LightGBM
have a long-standing problem where any child crash (OOM, segfault, C++
exception) gets propagated to the parent as SIGINT, which the terminal
echoes as a phantom "^C" -- looking like the user pressed Ctrl+C when
they didn't. This bites XGBoost too (see train_xgboost.py docstring).
Sequential single-process avoids the issue entirely.

The independent-per-step approach has a useful property: each forecast step
gets its own feature importance / split structure. Near-term steps (step+1)
discover that recent lookback dominates; far-term steps (step+96) discover
that future weather dominates. This per-step specialization often beats
forced-tied multi-output trees in practice.

Comparability: same 70/10/20 split + same StandardScaler-on-train-target as
all other baselines so z-scored MAE/MSE numbers are directly comparable.
Raw µg/m³ = z_MAE * target_scaler.scale_[0].

Run
---
    PYTHONPATH=. python -u comparison_model/train_lightgbm.py \\
        --dataset ./dataset/PRSA_data/Aotizhongxin.csv \\
        --lookback_len 96 --pred_len 96 \\
        --n_estimators 300 --learning_rate 0.1 --num_leaves 31 \\
        --inner_n_jobs -1 \\
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


# Defined at module scope so joblib can pickle it.
class LGBMSequentialMultiOutput:
    """Holds ``pred_len`` independent LightGBM single-output regressors."""

    def __init__(self, regressors):
        self.regressors = list(regressors)

    def predict(self, X):
        return np.stack([r.predict(X) for r in self.regressors], axis=1)


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--lookback_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed. If omitted, a fresh random seed is drawn each run and printed (so runs differ but stay traceable).")
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
    p.add_argument("--inner_n_jobs", type=int, default=-1,
                   help="LightGBM internal threads. -1 = all cores. Used "
                        "PER REGRESSOR; since we train sequentially (one "
                        "regressor at a time), oversubscription is impossible.")
    p.add_argument("--progress_every", type=int, default=8,
                   help="Print a progress line every N regressors trained.")
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
    if args.seed is None:
        args.seed = random.randrange(2**31)
    print(f"[seed] run seed = {args.seed}")
    set_seed(args.seed)

    try:
        import lightgbm as lgb
        import joblib
    except ImportError as e:
        raise ImportError(
            f"LightGBM baseline needs lightgbm + joblib: {e}. "
            "Install with: pip install lightgbm joblib"
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
        print(f"\n── Training LightGBM (sequential, {args.pred_len} regressors) ──")
        print(f"  Each regressor: n_estimators={args.n_estimators}  "
              f"num_leaves={args.num_leaves}  lr={args.learning_rate}")
        print(f"  Internal threading: n_jobs={args.inner_n_jobs}")
        print(f"  No multiprocessing/joblib: avoids the joblib-SIGINT-cascade "
              f"(see module docstring).")
        print(f"  Progress logged every {args.progress_every} regressors.\n")

        common_kwargs = dict(
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

        regressors = []
        total_start = time.time()
        for step in range(args.pred_len):
            step_start = time.time()
            reg = lgb.LGBMRegressor(**common_kwargs)
            reg.fit(X_train, Y_train[:, step])
            regressors.append(reg)

            done = step + 1
            if done % args.progress_every == 0 or done == args.pred_len:
                elapsed = time.time() - total_start
                rate = done / elapsed
                eta_s = (args.pred_len - done) / max(rate, 1e-9)
                step_s = time.time() - step_start
                print(f"    [{done:>3d}/{args.pred_len}]  "
                      f"step={step_s:.1f}s  total={elapsed:.0f}s  "
                      f"ETA={eta_s:.0f}s  rate={rate*60:.1f}/min")

        fit_time = time.time() - total_start
        print(f"\n  total fit time: {fit_time:.1f}s ({fit_time/60:.1f} min)")

        model = LGBMSequentialMultiOutput(regressors)

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

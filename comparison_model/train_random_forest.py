"""
Random Forest baseline for Beijing PRSA Aotizhongxin PM2.5 forecasting with
known-future covariates.

Comparability contract -- IDENTICAL to train_tide.py / train_tft.py /
train_informer.py / etc:
  * Same 70/10/20 chronological split via PRSA_EncDec_Dataset.
  * Same StandardScaler-on-train-target so z-scored MAE/MSE land in the same
    z-space as every other baseline; raw µg/m³ = z * target_scaler.scale_[0].
  * Known-future covariates (other pollutants + meteorology + calendar) are
    flattened into the feature vector alongside the lookback target and
    lookback covariates. PM2.5 over the forecast horizon is NEVER an input
    (no leakage); it's the target only.

Feature vector per window (in this order, flattened):
  - y_past_tgt (96 vals)                  -- lookback target
  - x_real_past flattened (96*10 = 960)   -- lookback covariates
  - x_real_future flattened (24*10 = 240) -- KNOWN-FUTURE covariates
  - x_mark_past flattened (96*4 = 384)    -- lookback calendar
  - x_mark_future flattened (24*4 = 96)   -- KNOWN-FUTURE calendar
  Total: 1776 features per window.

Target: y_true (24 z-scored PM2.5 values), predicted jointly by sklearn's
RandomForestRegressor which natively supports multi-output (one tree per
estimator, leaves carry a (pred_len,)-vector).

Run
---
    PYTHONPATH=. python -u comparison_model/train_random_forest.py \\
        --dataset ./dataset/PRSA_data/Aotizhongxin.csv \\
        --lookback_len 96 --pred_len 96 \\
        --n_estimators 100 --max_depth 15 --min_samples_leaf 20 \\
        --max_features sqrt --n_jobs -1 \\
        --checkpoint_dir ./checkpoint/random_forest

Tips
----
  * sklearn's multi-output RF cost is roughly O(n_estimators * n_samples *
    log(n_samples) * max_features * n_outputs). For pred_len=96 the n_outputs
    term is ~4x what it was at pred_len=24, so default trees are smaller and
    max_features='sqrt' (not 0.3) to keep fit time in the 10-20 min range.
  * For a faster sanity-check run, add ``--subsample_train 4`` to use every
    4th training window (~5 min).
  * For a STRONGER fit (and more patience): ``--n_estimators 300
    --max_depth 25 --min_samples_leaf 5 --max_features 0.3``.
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
    p.add_argument("--dataset", type=str, required=True,
                   help="Path to Aotizhongxin.csv.")
    p.add_argument("--lookback_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed. If omitted, a fresh random seed is drawn each run and printed (so runs differ but stay traceable).")
    # Random Forest hyperparameters
    #
    # Defaults tuned for pred_len=96. sklearn's multi-output RF computes split
    # impurity averaged over ALL outputs per split, so cost scales roughly
    # linearly with pred_len (96 outputs → ~4x slower than pred_len=24 with
    # the same trees). The defaults below trade a small bit of accuracy for
    # ~10-20 min fit time on a 24k-window training split with -1 cores:
    #
    #   n_estimators=100   (was 200)  -- halves tree count
    #   max_depth=15       (was 25)   -- prevents unbounded leaf growth
    #   min_samples_leaf=20 (was 5)   -- prevents the tail of one-sample
    #                                    leaves which dominate fit time
    #   max_features='sqrt' (was 0.3) -- considers ~53 of 2784 features per
    #                                    split (was ~835), the single biggest
    #                                    speedup
    #
    # Crank these up if you want a stronger fit and have time to spare.
    p.add_argument("--n_estimators", type=int, default=100,
                   help="Number of trees.")
    p.add_argument("--max_depth", type=int, default=15,
                   help="Max tree depth. 0 disables the limit (slower, "
                        "stronger).")
    p.add_argument("--min_samples_leaf", type=int, default=20,
                   help="Minimum samples per leaf. Higher = more regularised "
                        "and much faster fit.")
    p.add_argument("--max_features", type=str, default="sqrt",
                   help="Number of features considered per split. 'sqrt' or "
                        "'log2', or a float fraction (e.g. 0.3).")
    p.add_argument("--n_jobs", type=int, default=-1,
                   help="Parallel workers (-1 = all cores).")
    p.add_argument("--subsample_train", type=int, default=1,
                   help="Keep every Nth training window. 1 = all windows.")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoint/random_forest")
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def _parse_max_features(raw):
    """sklearn accepts float or 'sqrt' / 'log2'. We expose it as a string CLI
    arg and parse to the right type here."""
    if raw in ("sqrt", "log2"):
        return raw
    try:
        return float(raw)
    except ValueError:
        return raw


def main():
    args = get_args()
    if args.seed is None:
        args.seed = random.randrange(2**31)
    print(f"[seed] run seed = {args.seed}")
    set_seed(args.seed)

    try:
        from sklearn.ensemble import RandomForestRegressor
        import joblib
    except ImportError as e:
        raise ImportError(
            f"RandomForest baseline needs scikit-learn + joblib: {e}. "
            "Install with: pip install scikit-learn joblib"
        )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "rf.joblib")

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
    print(f"  feature vector dim = {X_train.shape[1]}")

    if args.subsample_train > 1:
        X_train = X_train[::args.subsample_train]
        Y_train = Y_train[::args.subsample_train]
        print(f"  subsampled train to {len(X_train)} windows")

    print(f"  train X: {X_train.shape}   Y: {Y_train.shape}")
    print(f"  vali  X: {X_val.shape}     Y: {Y_val.shape}")
    print(f"  test  X: {X_test.shape}    Y: {Y_test.shape}")

    # ----- Train -----
    if not args.eval_only:
        max_depth = None if args.max_depth == 0 else args.max_depth
        max_features = _parse_max_features(args.max_features)
        print("\n── Training RandomForest ──")
        print(f"  n_estimators={args.n_estimators}  max_depth={max_depth}  "
              f"min_samples_leaf={args.min_samples_leaf}  max_features={max_features}")
        rf = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=max_features,
            n_jobs=args.n_jobs,
            random_state=args.seed,
            verbose=1,
        )
        t0 = time.time()
        rf.fit(X_train, Y_train)
        fit_time = time.time() - t0
        print(f"  fit took {fit_time:.1f}s")

        # Val MAE for reporting (no early stopping in sklearn RF).
        Y_val_hat = rf.predict(X_val)
        val_mae = float(np.mean(np.abs(Y_val_hat - Y_val)))
        val_mse = float(np.mean((Y_val_hat - Y_val) ** 2))
        print(f"  val MAE = {val_mae:.4f} (z) / "
              f"{val_mae * val_set.target_scale():.4f} µg/m³")
        print(f"  val MSE = {val_mse:.4f} (z)")

        joblib.dump(rf, ckpt_path)
        print(f"  saved to {ckpt_path}")

    # ----- Test -----
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}; train first.")
    print(f"\n── Loading checkpoint ({ckpt_path}) ──")
    rf = joblib.load(ckpt_path)

    print("\n── Test forecast results (Random Forest) ──")
    Y_test_hat = rf.predict(X_test)
    test_mae = float(np.mean(np.abs(Y_test_hat - Y_test)))
    test_mse = float(np.mean((Y_test_hat - Y_test) ** 2))
    scale = test_set.target_scale()
    test_mae_raw = test_mae * scale
    test_mse_raw = test_mse * (scale ** 2)
    print(f"  test MSE      = {test_mse:.6f}  (z-scored)")
    print(f"  test MAE      = {test_mae:.6f}  (z-scored)")
    print(f"  test MSE      = {test_mse_raw:.4f}  (RAW µg/m³, RMSE = {test_mse_raw**0.5:.4f})")
    print(f"  test MAE      = {test_mae_raw:.4f}  (RAW µg/m³)")
    print("\n  (Directly comparable to CBM/train_cbm.py, train_tide.py, "
          "train_tft.py, train_informer.py, train_autoformer.py, "
          "train_fedformer.py: same split, same target scaler, same "
          "known-future-covariate regime.)")


if __name__ == "__main__":
    main()

"""
PRSA Aotizhongxin dataset for the TiDE baseline.

Comparability contract (MUST match dataloader/dataloader.py's legacy split so
TiDE's test MAE/MSE are directly comparable to ProtoLLM's):

  * Split: chronological 70 / 10 / 20 on the raw CSV row order, with
    train_end = int(0.7*N), val_end = int(0.8*N). Each split only emits
    windows that lie ENTIRELY inside its slice (no cross-split windows) --
    identical to LOF_Dataset when USE_LAST_YEAR_SPLIT = False.
  * Window = lookback_len (L) + pred_len (H); dense start indices within the
    slice (stride 1), exactly like LOF_Dataset.__getitem__.
  * TARGET (PM2.5) normalization: a single sklearn StandardScaler fit on the
    TRAIN target only -- the SAME object/contract ProtoLLM uses
    (LOF_Dataset.ans_type). So z-scored metrics live in the same space and
    raw µg/m³ = z * scaler.scale_[0] (+ mean for values).
  * Covariates (the 14 exogenous channels = every column except PM2.5) are
    z-scored with their own StandardScaler fit on TRAIN only. This is internal
    to TiDE and does not affect the target metric space.
  * KNOWN-FUTURE covariates: each item returns covariates over the WHOLE
    window (L + H). PM2.5 over the forecast window is NEVER returned as input
    -- only y_past (lookback target) and y_true (held out for loss/metrics).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - sklearn should be present in the env
    StandardScaler = None


# Target column from the single source of truth (dataset_config), selected by
# PROTO_DATASET (default "PRSA" → unchanged).
try:
    from dataset_config import D as _DSCFG
except ModuleNotFoundError:                       # pragma: no cover
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from dataset_config import D as _DSCFG
TARGET = _DSCFG["target"]


class _NumpyStandardScaler:
    """ddof=0 standardizer, numerically identical to sklearn.StandardScaler.

    Fallback used only if scikit-learn is unavailable; ProtoLLM uses sklearn,
    and sklearn / numpy both use population std (ddof=0), so the target space
    matches either way.
    """

    def fit(self, x):
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, x):
        return (x - self.mean_) / self.scale_


def _make_scaler():
    return StandardScaler() if StandardScaler is not None else _NumpyStandardScaler()


class PRSA_TiDE_Dataset(Dataset):
    def __init__(self, csv_path, flag, lookback_len, pred_len,
                 target_scaler=None, cov_scaler=None):
        assert flag in ("train", "vali", "test"), f"bad flag {flag!r}"
        self.L = int(lookback_len)
        self.H = int(pred_len)
        self.length = self.L + self.H

        df = pd.read_csv(csv_path)
        if TARGET not in df.columns:
            raise KeyError(f"{TARGET!r} not found in {csv_path}; cols={list(df.columns)}")

        # Covariates = every column except the target, numeric, in CSV order.
        # For Aotizhongxin.csv this is exactly the 14 channels ProtoLLM feeds:
        # year, month, day, hour, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM.
        cov_df = df.drop(columns=[TARGET]).select_dtypes(include=[np.number])
        self.cov_cols = list(cov_df.columns)
        cov = cov_df.values.astype(np.float32)              # (N, C)
        tgt = df[[TARGET]].values.astype(np.float32)        # (N, 1)

        N = len(df)
        train_end = int(0.7 * N)
        val_end = int(0.8 * N)

        # Scalers fit on TRAIN slice only (no val/test leakage).
        if target_scaler is None:
            target_scaler = _make_scaler().fit(tgt[:train_end])
        if cov_scaler is None:
            cov_scaler = _make_scaler().fit(cov[:train_end])
        self.target_scaler = target_scaler
        self.cov_scaler = cov_scaler

        tgt_s = target_scaler.transform(tgt).astype(np.float32)
        cov_s = cov_scaler.transform(cov).astype(np.float32)

        if flag == "train":
            lo, hi = 0, train_end
        elif flag == "vali":
            lo, hi = train_end, val_end
        else:  # test
            lo, hi = val_end, N

        self.tgt = tgt_s[lo:hi]                              # (M, 1)
        self.cov = cov_s[lo:hi]                              # (M, C)
        self.n_cov = self.cov.shape[1]

    def __len__(self):
        return max(0, len(self.tgt) - self.length + 1)

    def __getitem__(self, idx):
        s = idx
        y_past = self.tgt[s : s + self.L]                       # (L, 1)
        x_cov = self.cov[s : s + self.L + self.H]              # (L+H, C)  KNOWN future covariates
        y_true = self.tgt[s + self.L : s + self.L + self.H]    # (H, 1)  held out (loss/metrics only)
        return (torch.from_numpy(y_past.copy()),
                torch.from_numpy(x_cov.copy()),
                torch.from_numpy(y_true.copy()))

    # Convenience for raw-unit (µg/m³) reporting.
    def target_scale(self) -> float:
        return float(np.asarray(self.target_scaler.scale_).reshape(-1)[0])

    def target_mean(self) -> float:
        return float(np.asarray(self.target_scaler.mean_).reshape(-1)[0])

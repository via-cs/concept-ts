"""
PRSA Aotizhongxin dataset for STRICT LOOKBACK-ONLY baselines.

Emits ONLY the lookback window as input; no future features of any kind
(no future calendar, no future covariates, no future target). This matches
the input contract of ProtoLLM's CBM in ``experiment_mode='lookback_only'``,
so reported MAE numbers can be compared directly against that run.

Comparability contract (matches PRSA_TiDE_Dataset / PRSA_EncDec_Dataset)
-----------------------------------------------------------------------
  * Split: chronological 70 / 10 / 20 on the raw CSV row order, with
    train_end = int(0.7*N), val_end = int(0.8*N). Each split emits only
    windows that fit entirely inside its slice (no cross-split windows).
  * TARGET (PM2.5) normalization: a single sklearn StandardScaler fit on the
    TRAIN target only. Same z-space so raw µg/m³ = z * scaler.scale_[0].
  * Covariates z-scored with a train-only StandardScaler (internal to the
    baseline models; does not affect the target metric space).
  * Calendar features (year, month, day, hour) are also z-scored -- these
    baselines don't need the [-0.5, 0.5] normalization the TFT / Informer
    ports use for their ``embed='timeF'`` layer, since DLinear/TimeXer
    don't consume calendar features separately.
  * Target is placed at the LAST column of x_lookback so TimeXer's
    "endogenous variable = last channel" convention works out of the box.

Per-window contract
-------------------
    __getitem__(i) -> (x_lookback, y_forecast)
        x_lookback : (L, C)  where C = 14 covariates + 1 target = 15
        y_forecast : (H, 1)  z-scored PM2.5

    NO x_dec, NO x_mark_enc, NO x_mark_dec. Baselines that expect the
    full TSL 4-tensor contract just pass ``None`` for the missing pieces
    (or the trainer wraps this up in a batch of two).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from sklearn.preprocessing import StandardScaler
except Exception:                                       # pragma: no cover
    StandardScaler = None


TARGET = "PM2.5"


class _NumpyStandardScaler:
    """ddof=0 standardizer, numerically identical to sklearn.StandardScaler."""

    def fit(self, x):
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, x):
        return (x - self.mean_) / self.scale_


def _make_scaler():
    return StandardScaler() if StandardScaler is not None else _NumpyStandardScaler()


class PRSA_LookbackOnly_Dataset(Dataset):
    """Strict lookback-only PRSA dataset.

    Emits ``(x_lookback, y_forecast)`` per sample. Nothing from the forecast
    window is exposed to the model. Same 70/10/20 split + train-only target
    StandardScaler as the other PRSA_* datasets in this repo so reported
    z-scored MAE / MSE lives in a single comparable space across all
    baselines and the CBM.
    """

    def __init__(self, csv_path, flag, lookback_len, pred_len,
                 target_scaler=None, feature_scaler=None):
        assert flag in ("train", "vali", "test"), f"bad flag {flag!r}"
        self.L = int(lookback_len)
        self.H = int(pred_len)
        self.length = self.L + self.H

        df = pd.read_csv(csv_path)
        if TARGET not in df.columns:
            raise KeyError(f"{TARGET!r} not found in {csv_path}; cols={list(df.columns)}")

        # Feature columns = everything numeric except target, in CSV order.
        # For Aotizhongxin.csv this is the 14 channels:
        # year, month, day, hour, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM.
        feat_df = df.drop(columns=[TARGET]).select_dtypes(include=[np.number])
        self.feature_cols = list(feat_df.columns)
        features = feat_df.values.astype(np.float32)             # (N, C-1)
        target   = df[[TARGET]].values.astype(np.float32)        # (N, 1)

        N = len(df)
        train_end = int(0.7 * N)
        val_end   = int(0.8 * N)

        # Fit scalers on TRAIN slice only.
        if target_scaler is None:
            target_scaler = _make_scaler().fit(target[:train_end])
        if feature_scaler is None:
            feature_scaler = _make_scaler().fit(features[:train_end])
        self.target_scaler  = target_scaler
        self.feature_scaler = feature_scaler

        features_s = feature_scaler.transform(features).astype(np.float32)
        target_s   = target_scaler.transform(target).astype(np.float32)

        if flag == "train":
            lo, hi = 0, train_end
        elif flag == "vali":
            lo, hi = train_end, val_end
        else:
            lo, hi = val_end, N

        # Concatenate features + target with TARGET AT THE LAST COLUMN.
        # This matches TimeXer's expectation of endogenous=last channel and
        # is a common TSL convention (`features='MS'`).
        self.x = np.concatenate([features_s[lo:hi], target_s[lo:hi]], axis=1)  # (M, C)
        self.y = target_s[lo:hi]                                                # (M, 1)
        self.C = self.x.shape[1]                                                # 15 for PRSA

    def __len__(self):
        return max(0, len(self.x) - self.length + 1)

    def __getitem__(self, idx):
        s = idx
        L, H = self.L, self.H
        x_lookback = self.x[s : s + L]                                          # (L, C)
        y_forecast = self.y[s + L : s + L + H]                                  # (H, 1)
        return (torch.from_numpy(x_lookback.copy()),
                torch.from_numpy(y_forecast.copy()))

    # ----- convenience helpers for raw-unit reporting -----
    def target_scale(self) -> float:
        return float(np.asarray(self.target_scaler.scale_).reshape(-1)[0])

    def target_mean(self) -> float:
        return float(np.asarray(self.target_scaler.mean_).reshape(-1)[0])

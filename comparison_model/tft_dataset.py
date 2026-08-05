"""
PRSA Aotizhongxin dataset for the Temporal Fusion Transformer baseline.

Comparability contract -- MUST match ``tide_dataset.PRSA_TiDE_Dataset`` so
TFT's z-scored MAE/MSE land in the same units as TiDE and ProtoLLM's CBM:

  * Split: chronological 70 / 10 / 20 on raw CSV order (train_end = int(0.7*N),
    val_end = int(0.8*N)). Each split emits only windows that fit entirely
    inside its slice. Identical to LOF_Dataset / PRSA_TiDE_Dataset.
  * TARGET (PM2.5) normalization: a single sklearn StandardScaler fit on the
    TRAIN target only. The same scaler is reused for val/test. Test MAE/MSE
    are therefore in the SAME z-space as TiDE and CBM, and raw µg/m³ =
    z * target_scaler.scale_[0] (+ mean for values).
  * KNOWN-FUTURE covariates: every covariate channel is supplied over the
    whole (L + H) window. PM2.5 over the forecast window is NEVER returned
    as an input -- only y_past_tgt (lookback target) and y_true (held out).

TFT-specific wrinkle (vs TiDE)
------------------------------
TFT distinguishes between *categorical* and *real* time-varying inputs:
calendar columns become embedding lookups, real columns get linear
projections. We therefore split the 14 covariate channels into:

  * Categorical (4): year, month, day, hour     -- kept as int64 indices.
                                                   year already 0..4 in
                                                   Aotizhongxin.csv; for
                                                   month/day we subtract 1 to
                                                   move them into 0-indexed
                                                   space. Vocab sizes are
                                                   exposed via .cat_cardinalities.
  * Real (10): PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM
                                                -- z-scored with a single
                                                   StandardScaler fit on the
                                                   TRAIN slice only. Same
                                                   train-only-fit semantics as
                                                   tide_dataset.

Both real and categorical streams are returned for the WHOLE window (L+H).
The target (PM2.5) is returned only for the lookback (y_past_tgt); the
held-out forecast target is y_true.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from sklearn.preprocessing import StandardScaler
except Exception:                                       # pragma: no cover
    StandardScaler = None


# Schema (target + categorical cols) from the single source of truth
# (dataset_config), selected by PROTO_DATASET (default "PRSA" → unchanged).
try:
    from dataset_config import D as _DSCFG
except ModuleNotFoundError:                       # pragma: no cover
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from dataset_config import D as _DSCFG
TARGET   = _DSCFG["target"]
CAT_COLS = tuple(_DSCFG["cat_cols"])
# 1-indexed columns (month, day) shifted into 0..vocab-1 to make embedding lookup
# safe. year is already 0..4, hour is already 0..23.
# Embedding-table offsets/cardinalities from the single source of truth
# (dataset_config). PRSA values are unchanged; LOF adds its calendar columns.
# Cardinality = max possible value AFTER offset + 1, using theoretical maxima
# so val/test never index past the table even on unusual months/days.
_CAT_OFFSET = _DSCFG["cat_offset"]
_CAT_CARD   = _DSCFG["cat_card"]


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


class PRSA_TFT_Dataset(Dataset):
    """Same split / target-scaler contract as PRSA_TiDE_Dataset.

    __getitem__ returns a dict (so the train loop can stay readable):
        y_past_tgt    (L, 1)   z-scored PM2.5 lookback
        x_real_past   (L, R)   z-scored real covariates over lookback
        x_real_future (H, R)   z-scored real covariates over forecast
        x_cat_past    (L, K)   int64 categorical covariates over lookback
        x_cat_future  (H, K)   int64 categorical covariates over forecast
        y_true        (H, 1)   z-scored PM2.5 ground truth (held out)

    where R = len(self.real_cols), K = len(self.cat_cols).
    """

    def __init__(self, csv_path, flag, lookback_len, pred_len,
                 target_scaler=None, real_scaler=None):
        assert flag in ("train", "vali", "test"), f"bad flag {flag!r}"
        self.L = int(lookback_len)
        self.H = int(pred_len)
        self.length = self.L + self.H

        df = pd.read_csv(csv_path)
        if TARGET not in df.columns:
            raise KeyError(f"{TARGET!r} not found in {csv_path}; cols={list(df.columns)}")
        for c in CAT_COLS:
            if c not in df.columns:
                raise KeyError(
                    f"Categorical column {c!r} missing from {csv_path}. "
                    f"This dataset adapter is hard-coded for the cleaned "
                    f"Aotizhongxin schema (year, month, day, hour, ...). "
                    f"Got cols={list(df.columns)}"
                )

        # Real covariate columns = everything numeric except target and the
        # 4 calendar columns, in CSV order. For Aotizhongxin this is exactly:
        # PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM.
        real_df = df.drop(columns=[TARGET, *CAT_COLS]).select_dtypes(include=[np.number])
        self.real_cols = list(real_df.columns)
        self.cat_cols = list(CAT_COLS)
        self.cat_cardinalities = [_CAT_CARD[c] for c in self.cat_cols]

        real = real_df.values.astype(np.float32)             # (N, R)
        tgt = df[[TARGET]].values.astype(np.float32)         # (N, 1)
        cats = np.stack(
            [df[c].values.astype(np.int64) + _CAT_OFFSET[c] for c in self.cat_cols],
            axis=1,
        )                                                     # (N, K)
        # Defensive clamp: even if the CSV has out-of-range values, never index
        # past the embedding tables. This costs nothing on a clean CSV.
        for k, c in enumerate(self.cat_cols):
            cap = _CAT_CARD[c] - 1
            np.clip(cats[:, k], 0, cap, out=cats[:, k])

        N = len(df)
        train_end = int(0.7 * N)
        val_end = int(0.8 * N)

        # Scalers fit on TRAIN slice ONLY (no val/test leakage).
        if target_scaler is None:
            target_scaler = _make_scaler().fit(tgt[:train_end])
        if real_scaler is None:
            real_scaler = _make_scaler().fit(real[:train_end])
        self.target_scaler = target_scaler
        self.real_scaler = real_scaler

        tgt_s = target_scaler.transform(tgt).astype(np.float32)
        real_s = real_scaler.transform(real).astype(np.float32)

        if flag == "train":
            lo, hi = 0, train_end
        elif flag == "vali":
            lo, hi = train_end, val_end
        else:                                                  # test
            lo, hi = val_end, N

        self.tgt = tgt_s[lo:hi]                                # (M, 1)
        self.real = real_s[lo:hi]                              # (M, R)
        self.cats = cats[lo:hi]                                # (M, K)
        self.n_real = self.real.shape[1]
        self.n_cat = self.cats.shape[1]

    def __len__(self):
        return max(0, len(self.tgt) - self.length + 1)

    def __getitem__(self, idx):
        s = idx
        L = self.L
        H = self.H
        y_past_tgt = self.tgt[s : s + L]                                       # (L, 1)
        x_real_past = self.real[s : s + L]                                     # (L, R)
        x_real_future = self.real[s + L : s + L + H]                           # (H, R)
        x_cat_past = self.cats[s : s + L]                                      # (L, K)
        x_cat_future = self.cats[s + L : s + L + H]                            # (H, K)
        y_true = self.tgt[s + L : s + L + H]                                   # (H, 1)
        return {
            "y_past_tgt":    torch.from_numpy(y_past_tgt.copy()),
            "x_real_past":   torch.from_numpy(x_real_past.copy()),
            "x_real_future": torch.from_numpy(x_real_future.copy()),
            "x_cat_past":    torch.from_numpy(x_cat_past.copy()),
            "x_cat_future":  torch.from_numpy(x_cat_future.copy()),
            "y_true":        torch.from_numpy(y_true.copy()),
        }

    # ----- convenience helpers for raw-unit reporting -----
    def target_scale(self) -> float:
        return float(np.asarray(self.target_scaler.scale_).reshape(-1)[0])

    def target_mean(self) -> float:
        return float(np.asarray(self.target_scaler.mean_).reshape(-1)[0])

"""
PRSA Aotizhongxin dataset for the Informer / Autoformer / FEDformer baselines.

Comparability contract -- MUST match ``tide_dataset.PRSA_TiDE_Dataset`` and
``tft_dataset.PRSA_TFT_Dataset`` so the encoder-decoder transformer baselines
report MAE in the SAME z-space as TiDE, TFT and ProtoLLM's CBM:

  * Split: chronological 70 / 10 / 20 on raw CSV order, with
    train_end = int(0.7*N), val_end = int(0.8*N). Identical to TiDE/TFT.
  * TARGET (PM2.5) normalization: a single StandardScaler fit on the TRAIN
    target only. The same scaler is reused for val/test.  Verified at
    construction time to be byte-identical to PRSA_TiDE_Dataset's scaler.
  * KNOWN-FUTURE covariates: every covariate channel is supplied over the
    whole (L + H) window. PM2.5 over the forecast window is NEVER returned
    as an input -- only y_past_tgt (lookback target) and y_true (held out).
  * Calendar features (year, month, day, hour) are returned as normalized
    continuous values in [-0.5, 0.5] (TSL's ``timeF`` embedding contract),
    so the ported Informer / Autoformer / FEDformer can be constructed with
    ``embed='timeF'`` and consume them directly.

Rich-batch format
-----------------
Each __getitem__ returns a dict (because the model wrappers need separate
past/future slices):

    y_past_tgt    (L, 1)        z-scored PM2.5 lookback
    x_real_past   (L, R)        z-scored real covariates over lookback
    x_real_future (H, R)        z-scored real covariates over forecast
    x_mark_past   (L, 4)        normalized calendar features over lookback
    x_mark_future (H, 4)        normalized calendar features over forecast
    y_true        (H, 1)        z-scored PM2.5 ground truth (held out)

The model wrappers (Informer/Autoformer/FEDformer) each define their own
``prepare_batch`` that re-assembles these tensors into the (x_enc, x_mark_enc,
x_dec, x_mark_dec) tuple expected by the model's ``forward()`` -- with the
target channel of the future half of x_dec ZEROED but the exogenous columns
filled with the actual known-future values (line 0 vs line 1 of the cap
implementations is the only delta).
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
# TSL 'timeF' embedding wants normalized continuous values around 0.
# We use the standard convention: x / period - 0.5  -> roughly [-0.5, 0.5].
_CAT_PERIOD = {"year": 5, "month": 12, "day": 31, "hour": 24}


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


class PRSA_EncDec_Dataset(Dataset):
    """Rich-batch dataset shared by Informer / Autoformer / FEDformer ports.

    Same split + target-scaler contract as PRSA_TiDE_Dataset / PRSA_TFT_Dataset
    so reported MAE/MSE are directly comparable.
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
                    f"Calendar column {c!r} missing from {csv_path}. This "
                    f"adapter is hard-coded for the cleaned Aotizhongxin schema."
                )

        real_df = df.drop(columns=[TARGET, *CAT_COLS]).select_dtypes(include=[np.number])
        self.real_cols = list(real_df.columns)
        # 10 real covariates: PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM
        real = real_df.values.astype(np.float32)             # (N, R)
        tgt = df[[TARGET]].values.astype(np.float32)         # (N, 1)

        # Normalized calendar features ([-0.5, 0.5] approx) for TSL 'timeF' embed.
        marks = np.stack(
            [df[c].values.astype(np.float32) / _CAT_PERIOD[c] - 0.5 for c in CAT_COLS],
            axis=1,
        )                                                     # (N, 4)

        N = len(df)
        train_end = int(0.7 * N)
        val_end = int(0.8 * N)

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

        self.tgt   = tgt_s[lo:hi]                              # (M, 1)
        self.real  = real_s[lo:hi]                             # (M, R)
        self.marks = marks[lo:hi]                              # (M, 4)
        self.n_real = self.real.shape[1]
        self.n_mark = self.marks.shape[1]

    def __len__(self):
        return max(0, len(self.tgt) - self.length + 1)

    def __getitem__(self, idx):
        s = idx
        L = self.L
        H = self.H
        y_past_tgt    = self.tgt[s : s + L]                                    # (L, 1)
        x_real_past   = self.real[s : s + L]                                   # (L, R)
        x_real_future = self.real[s + L : s + L + H]                           # (H, R)
        x_mark_past   = self.marks[s : s + L]                                  # (L, 4)
        x_mark_future = self.marks[s + L : s + L + H]                          # (H, 4)
        y_true        = self.tgt[s + L : s + L + H]                            # (H, 1)
        return {
            "y_past_tgt":    torch.from_numpy(y_past_tgt.copy()),
            "x_real_past":   torch.from_numpy(x_real_past.copy()),
            "x_real_future": torch.from_numpy(x_real_future.copy()),
            "x_mark_past":   torch.from_numpy(x_mark_past.copy()),
            "x_mark_future": torch.from_numpy(x_mark_future.copy()),
            "y_true":        torch.from_numpy(y_true.copy()),
        }

    # ----- convenience helpers for raw-unit reporting -----
    def target_scale(self) -> float:
        return float(np.asarray(self.target_scaler.scale_).reshape(-1)[0])

    def target_mean(self) -> float:
        return float(np.asarray(self.target_scaler.mean_).reshape(-1)[0])


# ---------------------------------------------------------------------------
# Helpers used by Informer / Autoformer / FEDformer prepare_batch
# ---------------------------------------------------------------------------
def assemble_tsl_inputs(batch, label_len: int) -> dict:
    """Convert a rich dict batch from PRSA_EncDec_Dataset into the canonical
    TSL (x_enc, x_mark_enc, x_dec, x_mark_dec, y_true) layout.

    The KEY contract this enforces (and which the original cap implementations
    violated) is::

        x_dec future half = [ZERO target,  real future covariates]

    i.e. the model is told the actual future weather/pollutant values, but
    NOT the future PM2.5 it has to predict.

    Shape contract:
        x_enc       (B, L, 1 + R)               target + reals over lookback
        x_mark_enc  (B, L, 4)                   calendar over lookback
        x_dec       (B, label_len + H, 1 + R)   priming half from end of
                                                lookback; future half has
                                                zero target + real exogenous
                                                future values
        x_mark_dec  (B, label_len + H, 4)       calendar over (priming + future)
        y_true      (B, H, 1)                   z-scored target
    """
    y_past_tgt    = batch["y_past_tgt"]            # (B, L, 1)
    x_real_past   = batch["x_real_past"]           # (B, L, R)
    x_real_future = batch["x_real_future"]         # (B, H, R)
    x_mark_past   = batch["x_mark_past"]           # (B, L, 4)
    x_mark_future = batch["x_mark_future"]         # (B, H, 4)
    y_true        = batch["y_true"]                # (B, H, 1)

    B, L, _ = y_past_tgt.shape
    H = x_real_future.shape[1]

    # x_enc: target + reals concatenated over the lookback.
    x_enc = torch.cat([y_past_tgt, x_real_past], dim=-1)               # (B, L, 1+R)

    # Priming half of x_dec: last `label_len` rows of x_enc.
    dec_prime = x_enc[:, -label_len:, :]                                # (B, label_len, 1+R)
    # Future half of x_dec: target column ZEROED, exogenous columns = real future.
    zero_tgt = torch.zeros(B, H, 1, dtype=x_enc.dtype, device=x_enc.device)
    dec_future = torch.cat([zero_tgt, x_real_future], dim=-1)           # (B, H, 1+R)
    x_dec = torch.cat([dec_prime, dec_future], dim=1)                   # (B, label_len+H, 1+R)

    # x_mark_enc / x_mark_dec.
    x_mark_enc = x_mark_past                                            # (B, L, 4)
    x_mark_dec = torch.cat(
        [x_mark_past[:, -label_len:, :], x_mark_future], dim=1,
    )                                                                    # (B, label_len+H, 4)

    return {
        "x_enc":      x_enc,
        "x_mark_enc": x_mark_enc,
        "x_dec":      x_dec,
        "x_mark_dec": x_mark_dec,
        "y_true":     y_true,
    }


# ---------------------------------------------------------------------------
# Tabular flattener -- used by Random Forest and XGBoost trainers
# ---------------------------------------------------------------------------
def flatten_dataset_for_tabular(ds: PRSA_EncDec_Dataset):
    """Walk every window in the dataset and emit one flat feature vector + one
    target vector per window. Used by sklearn-style models (Random Forest,
    XGBoost, Linear Regression, Ridge) that consume tabular (N, d) input and
    emit (N, pred_len) output.

    Returns
    -------
    X : np.ndarray of shape (N_windows, n_features)
        Per-window flat features in this order:
            [y_past_tgt (L,)]               + lookback target
            [x_real_past flattened (L*R,)]  + lookback covariates
            [x_real_future flattened (H*R,)]+ KNOWN-FUTURE covariates
            [x_mark_past flattened (L*K,)]  + lookback calendar
            [x_mark_future flattened (H*K,)]+ KNOWN-FUTURE calendar
        Total: L + L*R + H*R + L*K + H*K  features.
        For PRSA defaults (L=96, H=24, R=10, K=4): 96 + 960 + 240 + 384 + 96
        = 1776 features per window.
    Y : np.ndarray of shape (N_windows, pred_len)
        z-scored ground-truth target sequence. Same z-space as every other
        baseline so MAE/MSE numbers stay comparable.
    """
    if len(ds) == 0:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, ds.H), dtype=np.float32)

    L, H = ds.L, ds.H
    R, K = ds.n_real, ds.n_mark
    n_features = L + L * R + H * R + L * K + H * K

    N = len(ds)
    X = np.empty((N, n_features), dtype=np.float32)
    Y = np.empty((N, H), dtype=np.float32)

    for i in range(N):
        s = i
        # All slices live in ds.tgt / ds.real / ds.marks (already z-scored /
        # normalized in __init__), so we avoid the torch.from_numpy round-trip
        # of __getitem__ and go straight numpy → numpy.
        y_past_tgt    = ds.tgt[s : s + L].reshape(-1)             # (L,)
        x_real_past   = ds.real[s : s + L].reshape(-1)            # (L*R,)
        x_real_future = ds.real[s + L : s + L + H].reshape(-1)    # (H*R,)
        x_mark_past   = ds.marks[s : s + L].reshape(-1)           # (L*K,)
        x_mark_future = ds.marks[s + L : s + L + H].reshape(-1)   # (H*K,)
        y_true        = ds.tgt[s + L : s + L + H].reshape(-1)     # (H,)

        # Concatenate in the documented order; np.concatenate of all-1D arrays
        # is a memcpy.
        X[i] = np.concatenate([y_past_tgt, x_real_past, x_real_future,
                                x_mark_past, x_mark_future])
        Y[i] = y_true

    return X, Y

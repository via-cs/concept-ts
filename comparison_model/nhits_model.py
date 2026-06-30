"""
NHITS with exogenous future-known regressors (NHITSx).

NHITS (Challu et al. 2023, https://arxiv.org/abs/2201.12886) is a multi-rate
hierarchical MLP forecaster -- pure feed-forward, no recurrence, no attention.
The "x" variant feeds future-known exogenous values into each block so the
model can condition its forecast on known-future weather / calendar features.

Architecture (matches the paper)
--------------------------------
A stack of ``n_blocks`` NHITS blocks. Each block:

  1. Downsamples the (univariate) target lookback by an integer ``pool_kernel``
     using avg-pool (the paper uses MaxPool; we use AvgPool because it
     reconstructs the same low-frequency content but is less brittle to
     spike noise typical of PM2.5).
  2. Concatenates the downsampled signal with the flattened future-known
     exogenous values (the NHITSx contribution).
  3. Runs the concatenation through a small MLP (depth `mlp_depth`, width
     `mlp_hidden`).
  4. Projects to two coefficient vectors -- one for the backcast (length
     L / pool_kernel) and one for the forecast (length H / pool_kernel).
  5. Linearly INTERPOLATES the coefficient vectors back to full length L and H
     respectively.

Residual stacking (paper, Section 3.2): the running lookback is the original
target minus the sum of all previous blocks' backcasts; the running forecast
is the sum of all blocks' forecasts. Different blocks use different
pool_kernels so they specialize on different frequencies -- block 1 with a
large kernel sees only the multi-day trend, block N with kernel=1 sees the
hourly detail.

forward signature
-----------------
    forward(batch) -> y_hat
        batch is the rich-dict produced by PRSA_EncDec_Dataset, identical to
        TFT and DeepAR. We use y_past_tgt, x_real_future, x_mark_future. The
        lookback covariates (x_real_past, x_mark_past) are NOT consumed by
        this NHITSx implementation (the paper's "x" variant only uses
        future-known exogenous; lookback covariates would be a TINY-x or
        NHITS-Hist variant that we don't implement here).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------
class NHITSBlock(nn.Module):
    """One NHITS block: pool → mlp → (backcast_theta, forecast_theta) → upsample."""

    def __init__(self, lookback_len: int, pred_len: int, exog_size: int,
                 pool_kernel: int, mlp_hidden: int, mlp_depth: int,
                 dropout: float):
        super().__init__()
        self.lookback_len = int(lookback_len)
        self.pred_len = int(pred_len)
        self.pool_kernel = int(pool_kernel)

        # Pool sizes (ceil-mode so we never lose the tail).
        self.lookback_down = (lookback_len + pool_kernel - 1) // pool_kernel
        self.forecast_down = (pred_len + pool_kernel - 1) // pool_kernel

        self.pool = nn.AvgPool1d(
            kernel_size=pool_kernel, stride=pool_kernel, ceil_mode=True,
        )

        # MLP
        in_dim = self.lookback_down + exog_size
        layers: List[nn.Module] = []
        for i in range(mlp_depth):
            layers.append(nn.Linear(in_dim, mlp_hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = mlp_hidden
        self.mlp = nn.Sequential(*layers)

        # Backcast + forecast coefficient projection.
        self.theta_proj = nn.Linear(
            mlp_hidden, self.lookback_down + self.forecast_down,
        )

    def forward(self, lookback: torch.Tensor, exog_flat: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # lookback : (B, L)        running target residual
        # exog_flat: (B, exog_size) flat future-known exogenous values
        B = lookback.size(0)

        # Pool over time.
        x_down = self.pool(lookback.unsqueeze(1)).squeeze(1)         # (B, L_down)

        # MLP on [downsampled lookback | flat exog].
        inp = torch.cat([x_down, exog_flat], dim=-1)
        h = self.mlp(inp)
        theta = self.theta_proj(h)                                    # (B, L_down + F_down)
        bc_theta = theta[:, : self.lookback_down]                     # (B, L_down)
        fc_theta = theta[:, self.lookback_down:]                      # (B, F_down)

        # Linearly interpolate the coefficient vectors back to full resolution.
        backcast = F.interpolate(
            bc_theta.unsqueeze(1), size=self.lookback_len,
            mode="linear", align_corners=False,
        ).squeeze(1)                                                  # (B, L)
        forecast = F.interpolate(
            fc_theta.unsqueeze(1), size=self.pred_len,
            mode="linear", align_corners=False,
        ).squeeze(1)                                                  # (B, H)
        return backcast, forecast


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------
class NHITSx(nn.Module):
    """NHITSx forecaster with residual-stacking blocks at multiple rates.

    Args
    ----
    n_real, n_mark   : R, K dimensions of the real / calendar exogenous streams.
    lookback_len     : L.
    pred_len         : H.
    pool_kernels     : sequence of integers -- one per block. Larger kernel =
                        coarser downsample = lower-frequency block. The paper
                        recommends a multi-rate decreasing sequence such as
                        ``(8, 4, 1)`` which is also the default here.
    mlp_hidden       : width of each block's MLP.
    mlp_depth        : depth of each block's MLP.
    dropout          : applied between MLP layers.
    """

    def __init__(self, n_real: int, n_mark: int,
                 lookback_len: int, pred_len: int,
                 pool_kernels: Tuple[int, ...] = (8, 4, 1),
                 mlp_hidden: int = 512, mlp_depth: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.L = int(lookback_len)
        self.H = int(pred_len)
        self.R = int(n_real)
        self.K = int(n_mark)
        # Future-known exogenous flat size.
        self.exog_size = self.H * (self.R + self.K)

        self.blocks = nn.ModuleList([
            NHITSBlock(
                lookback_len=lookback_len, pred_len=pred_len,
                exog_size=self.exog_size, pool_kernel=k,
                mlp_hidden=mlp_hidden, mlp_depth=mlp_depth, dropout=dropout,
            )
            for k in pool_kernels
        ])

    def forward(self, batch):
        y_past_tgt    = batch["y_past_tgt"]            # (B, L, 1)
        x_real_future = batch["x_real_future"]         # (B, H, R)
        x_mark_future = batch["x_mark_future"]         # (B, H, K)

        B = y_past_tgt.size(0)

        # Build the flat future exog tensor (concatenate reals and calendar,
        # then flatten the (H, R+K) shape into a single vector).
        future_exog = torch.cat([x_real_future, x_mark_future], dim=-1)         # (B, H, R+K)
        exog_flat = future_exog.reshape(B, -1)                                   # (B, H*(R+K))

        # Residual stacking.
        residual = y_past_tgt.squeeze(-1)                                        # (B, L)
        forecast = torch.zeros(
            B, self.H, device=y_past_tgt.device, dtype=y_past_tgt.dtype,
        )
        for block in self.blocks:
            backcast, fc = block(residual, exog_flat)
            residual = residual - backcast
            forecast = forecast + fc

        return forecast.unsqueeze(-1)                                            # (B, H, 1)

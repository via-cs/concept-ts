"""
DLinear (Zeng et al. 2022, "Are Transformers Effective for Time Series
Forecasting?") ported into ProtoLLM/comparison_model_lookback as a strict
LOOKBACK-ONLY baseline for direct comparison with ProtoLLM's CBM run in
``experiment_mode='lookback_only'``.

Model class is byte-identical to ``cap/cap/models/DLinear.py`` (a series-
decomposition-based linear model). Two adaptations:

  1. Import path -- ``series_decomp`` is pulled from our sibling
     ``comparison_model.tsl_layers.Autoformer_EncDec`` package (same file as
     cap's, verbatim). Trainer sets PYTHONPATH so this import resolves.
  2. ``prepare_batch`` is rewritten to unpack the 2-tuple
     ``(x_lookback, y_forecast)`` emitted by
     ``PRSA_LookbackOnly_Dataset`` -- no x_dec / x_mark_* fed in.

The architecture itself has ZERO capacity to consume future features; DLinear
is naturally lookback-only. We forecast all channels in parallel (paper
convention) but only report metrics on the PM2.5 (last) channel.
"""

import torch
import torch.nn as nn

from comparison_model.tsl_layers.Autoformer_EncDec import series_decomp


class DLinear(nn.Module):
    """DLinear: series-decomposition + two linear heads, no future features."""

    def __init__(self, seq_len, pred_len, moving_avg, enc_in, individual=False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decompsition = series_decomp(moving_avg)
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal[-1].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[-1].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend    = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    # ----------------------------------------------------------------------
    # I/O glue: consumes the 2-tuple from PRSA_LookbackOnly_Dataset.
    # NO future features, NO calendar marks.
    # ----------------------------------------------------------------------
    def prepare_batch(self, batch):
        x_enc, y = batch
        return (x_enc,), y

    # ----------------------------------------------------------------------
    # Forward -- byte-identical to cap.
    # ----------------------------------------------------------------------
    def _encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init    = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype, device=seasonal_init.device,
            )
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype, device=trend_init.device,
            )
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :]    = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output    = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forward(self, x_enc):
        """Multi-channel forecast; downstream code slices channel -1 (PM2.5)."""
        dec_out = self._encoder(x_enc)
        return dec_out[:, -self.pred_len:, :]                                   # (B, H, C)

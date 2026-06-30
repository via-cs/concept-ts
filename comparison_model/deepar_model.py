"""
DeepAR (Salinas et al. 2020, https://www.sciencedirect.com/science/article/abs/pii/S0169207019301888)
implementation for PRSA PM2.5 forecasting with known-future covariates.

This is the classic autoregressive RNN forecaster from AWS Labs. The model
was DESIGNED from day one to consume future-known dynamic covariates as
inputs to the decoder at each step -- exactly the regime our CBM operates in,
so DeepAR is a strong apples-to-apples baseline.

Architecture
------------
Stacked LSTM, shared between encoder and decoder. At each timestep the LSTM
consumes a feature vector consisting of:

    [previous_target_value,  real_covariates_t,  calendar_features_t]

  * Encoder (lookback): the LSTM is fed the actual lookback target at each
    step (offset by 1 so the target at step t is fed at step t+1).
  * Decoder (forecast horizon):
      - Training -- teacher-forcing -- the LSTM is fed the GROUND TRUTH
        previous target at each step, so the model only learns one-step-ahead
        prediction conditional on the future-known covariates.
      - Inference -- autoregressive -- the LSTM is fed its OWN PREDICTED
        previous target, so prediction errors accumulate. This is the
        canonical DeepAR rollout.

Output
------
The original DeepAR predicts a Gaussian distribution (mu, sigma) at each
step. For direct MAE comparability with the other comparison_model/*
baselines (which all train with MSE on a point estimate) we predict the
mean only and train with MSE. The output shape is (B, pred_len, 1) -- z-
scored, in the same space as every other baseline.

Caveats
-------
  * Teacher forcing during training and autoregressive decoding at inference
    is the standard DeepAR setup; the train/test discrepancy can cause
    "exposure bias" but it's how the paper is written and is faithful.
  * For real-world deployment DeepAR's strength is its probabilistic output;
    we've sacrificed that for MAE-comparable point estimates here. If you
    want quantile predictions, use the --use_gaussian_nll flag added below
    and report from the median / mean.
"""

import torch
import torch.nn as nn


class DeepAR(nn.Module):
    """DeepAR with shared encoder/decoder LSTM.

    Args
    ----
    n_real         : R, number of real covariates over (lookback + forecast).
    n_mark         : K, number of calendar features over (lookback + forecast).
    lookback_len   : L.
    pred_len       : H.
    hidden_size    : LSTM hidden dim.
    num_layers     : LSTM depth.
    dropout        : applied between LSTM layers (no-op for num_layers=1).
    use_gaussian_nll : if True, predict (mu, sigma) and train with NLL; sigma
                       is parameterised via softplus. Default False = MSE +
                       point estimate (mu) for MAE comparability.
    """

    def __init__(self, n_real: int, n_mark: int,
                 lookback_len: int, pred_len: int,
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1,
                 use_gaussian_nll: bool = False):
        super().__init__()
        self.L = int(lookback_len)
        self.H = int(pred_len)
        self.R = int(n_real)
        self.K = int(n_mark)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.use_gaussian_nll = bool(use_gaussian_nll)

        # Per-timestep input: [target, R reals, K calendar]
        self.input_size = 1 + self.R + self.K
        self.input_proj = nn.Linear(self.input_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output heads
        self.mu_head = nn.Linear(hidden_size, 1)
        if use_gaussian_nll:
            self.sigma_head = nn.Linear(hidden_size, 1)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _step(self, x_t: torch.Tensor, hc):
        """One LSTM step. x_t: (B, 1, input_size). hc: (h, c) or None."""
        h = self.input_proj(x_t)
        out, hc = self.lstm(h, hc)
        mu = self.mu_head(out)
        sigma = None
        if self.use_gaussian_nll:
            sigma = torch.nn.functional.softplus(self.sigma_head(out)) + 1e-3
        return mu, sigma, hc

    # -----------------------------------------------------------------
    # Train forward (teacher-forced)
    # -----------------------------------------------------------------
    def forward(self, batch):
        """Teacher-forced training pass.

        Returns:
            y_mu     : (B, H, 1) predicted means over the forecast horizon
            y_sigma  : (B, H, 1) predicted sigmas if use_gaussian_nll else None
        """
        y_past_tgt    = batch["y_past_tgt"]               # (B, L, 1)
        x_real_past   = batch["x_real_past"]              # (B, L, R)
        x_real_future = batch["x_real_future"]            # (B, H, R)
        x_mark_past   = batch["x_mark_past"]              # (B, L, K)
        x_mark_future = batch["x_mark_future"]            # (B, H, K)
        y_true        = batch["y_true"]                   # (B, H, 1)

        B = y_past_tgt.size(0)

        # ENCODER -- run the LSTM over the lookback.
        # At each lookback step t the LSTM sees the target value at t (the
        # "previous" for the next step). For step 0 we'd need target_{-1},
        # which we don't have -- so we shift by one and use target_t at step t
        # as the "previous" value used to update the state. (This is the
        # standard DeepAR formulation: the LSTM consumes the target sequence
        # itself as input.)
        enc_input = torch.cat([y_past_tgt, x_real_past, x_mark_past], dim=-1)  # (B, L, input_size)
        enc_h = self.input_proj(enc_input)
        _, hc = self.lstm(enc_h)                                                # advances the state through the lookback

        # DECODER -- teacher-forced. The "previous target" at each forecast
        # step is the GROUND TRUTH from the previous step. For step 0 we use
        # the last lookback target.
        prev_target = torch.cat([y_past_tgt[:, -1:, :], y_true[:, :-1, :]],
                                dim=1)                                          # (B, H, 1)
        dec_input = torch.cat([prev_target, x_real_future, x_mark_future],
                              dim=-1)                                            # (B, H, input_size)
        dec_h = self.input_proj(dec_input)
        out, _ = self.lstm(dec_h, hc)                                            # (B, H, hidden_size)

        y_mu = self.mu_head(out)                                                 # (B, H, 1)
        y_sigma = None
        if self.use_gaussian_nll:
            y_sigma = torch.nn.functional.softplus(self.sigma_head(out)) + 1e-3  # (B, H, 1)
        return y_mu, y_sigma

    # -----------------------------------------------------------------
    # Inference (autoregressive)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def predict(self, batch):
        """Canonical DeepAR rollout: predicted target at step t-1 becomes the
        input at step t. Errors accumulate.

        Returns:
            y_mu : (B, H, 1)
        """
        y_past_tgt    = batch["y_past_tgt"]
        x_real_past   = batch["x_real_past"]
        x_real_future = batch["x_real_future"]
        x_mark_past   = batch["x_mark_past"]
        x_mark_future = batch["x_mark_future"]

        B = y_past_tgt.size(0)
        H = self.H

        # Encoder
        enc_input = torch.cat([y_past_tgt, x_real_past, x_mark_past], dim=-1)
        enc_h = self.input_proj(enc_input)
        _, hc = self.lstm(enc_h)

        # Autoregressive decoder
        prev = y_past_tgt[:, -1:, :]                                              # (B, 1, 1)
        outputs = []
        for t in range(H):
            step_in = torch.cat([
                prev,
                x_real_future[:, t:t + 1, :],
                x_mark_future[:, t:t + 1, :],
            ], dim=-1)                                                            # (B, 1, input_size)
            mu_t, _, hc = self._step(step_in, hc)                                 # (B, 1, 1)
            outputs.append(mu_t)
            prev = mu_t                                                           # autoregressive

        return torch.cat(outputs, dim=1)                                          # (B, H, 1)

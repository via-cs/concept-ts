"""
LSTM baseline (Hochreiter & Schmidhuber 1997, applied to multivariate PM2.5
forecasting) for ProtoLLM/comparison_model_lookback.

Why not just copy cap/cap/models/lstm.py verbatim
-------------------------------------------------
Cap's ``TimeSeriesLSTM`` is a per-timestep seq2seq wrapper: it runs the
LSTM over the input window and projects every hidden state via
``nn.Linear(d_model, output_dim)`` -- returning ``(B, seq_len, output_dim)``.
That contract works for auto-regressive next-step training on a fixed-
length input, but it does NOT emit a ``(B, pred_len, output_dim)`` forecast
for a ``lookback_len -> pred_len`` setup like this one (where
lookback_len can be equal to, or different from, pred_len).

This port keeps the LSTM encoder byte-identical to cap (same
``nn.LSTM(input_dim, d_model, num_layers, batch_first=True)`` core) and
adds a **direct multi-step decoder** on top:

  1. LSTM(x_enc) -> hidden states h[0..L-1] of shape (B, L, d_model)
  2. Take the final hidden state h[L-1] of shape (B, d_model)
     -- the summary of the entire lookback.
  3. Linear(d_model, pred_len) projects it to a pred_len-vector
     of z-scored PM2.5 values.
  4. Reshape to (B, pred_len, 1).

This is the standard "encoder-LSTM + linear head" forecaster used by every
Beijing PM2.5 paper that cites LSTM (LSTME, CNN-LSTM, AC-LSTM, VCBA,
etc.) and by the Time-Series-Library's LSTM benchmark. It fits the STRICT
lookback-only contract cleanly: only x_enc is consumed; no future features
of any kind ever enter the model.

Input contract
--------------
    prepare_batch((x_lookback, y_forecast))
        x_lookback : (B, L, C)   -- full lookback, target at column -1
        y_forecast : (B, H, 1)   -- z-scored PM2.5 forecast target
    forward(x_lookback) -> (B, H, 1)

    NO x_dec, NO x_mark_enc, NO x_mark_dec. Directly comparable to
    the CBM in ``experiment_mode='lookback_only'``.
"""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """Encoder-LSTM + linear multi-step decoder for univariate PM2.5 forecasting.

    Consumes the full multivariate lookback ``(B, L, C)`` and emits a
    univariate forecast ``(B, H, 1)``. Target (PM2.5) is assumed to be
    the LAST channel of the input; this matters only downstream when we
    slice the input for other baselines -- for the LSTM the specific
    channel ordering is irrelevant because every channel is fed to the
    recurrent cell equally.
    """

    def __init__(self, input_dim, d_model, pred_len,
                 num_layers=2, dropout=0.1, output_dim=1):
        super().__init__()
        self.d_model    = int(d_model)
        self.num_layers = int(num_layers)
        self.pred_len   = int(pred_len)
        self.output_dim = int(output_dim)

        # Torch's LSTM dropout only applies between stacked layers when
        # num_layers > 1; this matches cap and TSL conventions.
        _lstm_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=self.d_model,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=_lstm_dropout,
        )
        # A small MLP head reads the encoder's final hidden state and emits
        # the whole forecast horizon in one shot (direct multi-step).
        self.dropout_head = nn.Dropout(float(dropout))
        self.head = nn.Linear(self.d_model, self.pred_len * self.output_dim)

    # ----------------------------------------------------------------------
    # I/O glue -- strict lookback-only, matches DLinear / TimeXer.
    # ----------------------------------------------------------------------
    def prepare_batch(self, batch):
        x_enc, y = batch
        return (x_enc,), y

    def forward(self, x_enc):
        """
        Args
        ----
        x_enc : (B, L, C) full lookback (target at column -1, but any
                channel order works for the LSTM).

        Returns
        -------
        y_hat : (B, pred_len, output_dim)
        """
        B = x_enc.size(0)
        h0 = x_enc.new_zeros((self.num_layers, B, self.d_model))
        c0 = x_enc.new_zeros((self.num_layers, B, self.d_model))
        lstm_out, (h_n, _) = self.lstm(x_enc, (h0, c0))                         # (B, L, d_model)

        # Use the LAST hidden state of the top LSTM layer as the lookback
        # summary. Equivalent to lstm_out[:, -1, :] but more explicit.
        summary = h_n[-1]                                                       # (B, d_model)
        summary = self.dropout_head(summary)

        y_hat = self.head(summary)                                              # (B, pred_len * output_dim)
        y_hat = y_hat.view(B, self.pred_len, self.output_dim)                   # (B, H, output_dim)
        return y_hat

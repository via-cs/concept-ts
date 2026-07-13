"""
iTransformer (Liu et al. 2024, "iTransformer: Inverted Transformers Are
Effective for Time Series Forecasting", https://arxiv.org/abs/2310.06625)
ported into ProtoLLM/comparison_model_lookback as a strict LOOKBACK-ONLY
baseline. Directly comparable to ProtoLLM's CBM run in
``experiment_mode='lookback_only'``.

iTransformer is natively encoder-only and consumes NO future features: it
"inverts" the series so each VARIABLE becomes a token, embeds the whole
lookback of each variable, runs self-attention across variables, then
projects each variable token to ``pred_len`` future steps. So it's a clean
fit for lookback-only forecasting.

Two small adaptations from cap:
  1. Imports point at our sibling ``comparison_model.tsl_layers`` package.
  2. ``forward`` returns ONLY the target channel (last column) -> shape
     (B, pred_len, 1), matching our univariate PM2.5 setup.
     PRSA_LookbackOnly_Dataset places PM2.5 at column -1.
"""

import torch
import torch.nn as nn

from comparison_model.tsl_layers.Transformer_EncDec import Encoder, EncoderLayer
from comparison_model.tsl_layers.SelfAttention_Family import (
    FullAttention, AttentionLayer,
)
from comparison_model.tsl_layers.Embed import DataEmbedding_inverted


class iTransformer(nn.Module):
    """Lookback-only iTransformer for PRSA PM2.5 forecasting.

    ``x_enc`` has shape ``(B, seq_len, C)`` with PM2.5 (target) at column -1.
    Emits univariate ``(B, pred_len, 1)`` forecast (the target channel).
    """

    def __init__(self, enc_in, seq_len, pred_len, d_model=512, n_heads=8,
                 d_ff=2048, num_layers=2, dropout=0.05, embed="fixed",
                 freq="h", factor=5, activation="gelu"):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        # Inverted embedding: (B, L, N) -> (B, N, d_model) (one token per variable).
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads,
                    ),
                    d_model, d_ff, dropout=dropout, activation=activation,
                )
                for _ in range(num_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # Per-variable projection: d_model -> pred_len future steps.
        self.projection = nn.Linear(d_model, pred_len, bias=True)

    # ----------------------------------------------------------------------
    # I/O glue: takes ONLY the 2-tuple lookback batch. No calendar marks.
    # ----------------------------------------------------------------------
    def prepare_batch(self, batch):
        x_enc, y = batch
        return (x_enc,), y

    def forward(self, x_enc):
        # Non-stationary Transformer per-series normalization (RevIN-style).
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        _, _, N = x_enc.shape                                     # N = enc_in

        # No future/calendar marks in lookback-only mode -> pass None.
        enc_out = self.enc_embedding(x_enc, None)                 # (B, N, d_model)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)        # (B, N, d_model)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]   # (B, pred_len, N)

        # De-normalize each channel with its own lookback stats.
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # Univariate output: target is the LAST channel.
        return dec_out[:, -self.pred_len:, -1:]                   # (B, pred_len, 1)

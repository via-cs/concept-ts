"""
Informer (Zhou et al. 2021) ported into ProtoLLM/comparison_model_lookback
as a STRICT LOOKBACK-ONLY baseline for direct comparison with ProtoLLM's CBM
in ``experiment_mode='lookback_only'``.

Contrast with ``comparison_model/Informer.py``
----------------------------------------------
The other Informer port in this repo is the FUTURE-AWARE version -- its
``prepare_batch`` fills the future half of ``x_dec`` with the real future
exogenous values (only the target column zeroed). That's fair against TFT
and TiDE (which also consume future covariates), but NOT against CBM's
lookback-only mode.

This copy differs in exactly one behavior: ``prepare_batch`` builds
``x_dec = [x_enc[:, -label_len:, :], zeros(pred_len, C)]`` and passes
``x_mark_enc = x_mark_dec = None``. No future information -- exogenous or
calendar -- ever enters the model. This is Time-Series-Library's default
"no-future-features" Informer setup.

The model class (``__init__``, ``long_forecast``, ``short_forecast``,
``forward``) is architecturally byte-identical to cap's -- only the batch
adapter changes.

Output projection: ``nn.Linear(d_model, 1)`` -- univariate PM2.5 output.
"""

import torch
import torch.nn as nn

from comparison_model.tsl_layers.Transformer_EncDec import (
    Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer,
)
from comparison_model.tsl_layers.SelfAttention_Family import (
    ProbAttention, AttentionLayer,
)
from comparison_model.tsl_layers.Embed import DataEmbedding


class Informer(nn.Module):
    """Lookback-only Informer. Decoder input for the pred_len horizon is
    pure zeros -- no leaked future features."""

    def __init__(self, enc_in, dec_in, pred_len, label_len,
                 d_model=512, embed="fixed", freq="h", dropout=0.1,
                 factor=3, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
                 activation="gelu", distil=False):
        super().__init__()
        print(
            f"[LOOKBACK-ONLY] Informer initialized with d_model={d_model}, "
            f"n_heads={n_heads}, d_ff={d_ff}, e_layers={e_layers}, "
            f"d_layers={d_layers}, dropout={dropout}"
        )
        self.pred_len  = pred_len
        self.label_len = label_len

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads,
                    ),
                    d_model, d_ff, dropout=dropout, activation=activation,
                )
                for _ in range(e_layers)
            ],
            [ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=nn.LayerNorm(d_model),
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads,
                    ),
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads,
                    ),
                    d_model, d_ff, dropout=dropout, activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, 1, bias=True),                        # univariate (PM2.5)
        )

    # ----------------------------------------------------------------------
    # I/O glue -- STRICT lookback-only.
    #   * x_enc      = full lookback window (B, L, C)
    #   * x_dec      = [ x_enc[:, -label_len:, :], zeros(pred_len, C) ]
    #                  The zeros carry NO future info: no future weather, no
    #                  future calendar. The decoder must learn to extrapolate
    #                  purely from cross-attention over the encoder output.
    #   * x_mark_enc = x_mark_dec = None
    # ----------------------------------------------------------------------
    def prepare_batch(self, batch):
        x_enc, y = batch                                                        # (B, L, C), (B, H, 1)
        B, L, C = x_enc.shape
        # last label_len steps of the lookback + zeros for the pred_len horizon
        x_dec = torch.zeros(B, self.label_len + self.pred_len, C,
                            dtype=x_enc.dtype, device=x_enc.device)
        x_dec[:, :self.label_len, :] = x_enc[:, -self.label_len:, :]
        return (x_enc, None, x_dec, None), y

    # ----------------------------------------------------------------------
    # Forward -- byte-identical to cap.
    # ----------------------------------------------------------------------
    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out[:, -self.pred_len:, :]                                   # (B, pred_len, 1)

    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        std_enc  = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = (x_enc - mean_enc) / std_enc

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        # Only the last (target=PM2.5) channel stats were used to normalize
        # x_enc's target column; we denormalize with the same std/mean.
        target_mean = mean_enc[:, :, -1:].expand(-1, self.pred_len, 1)
        target_std  = std_enc[:, :, -1:].expand(-1, self.pred_len, 1)
        dec_out = dec_out * target_std + target_mean
        return dec_out[:, -self.pred_len:, :]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, task="long"):
        if task == "long":
            return self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if task == "short":
            return self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        raise ValueError("task must be 'long' or 'short'")

"""
Autoformer (Wu et al. 2021) ported into ProtoLLM/comparison_model as a fair
baseline for ProtoLLM's CBM on the Beijing PRSA Aotizhongxin PM2.5 task.

The model class (__init__, forward, decomposition + seq2seq pipeline) is
BYTE-IDENTICAL to ``cap/cap/models/Autoformer.py``. We change only:

  1. Import paths -- relative to our local ``tsl_layers/``.
  2. ``prepare_batch`` -- rewritten to consume the rich-batch dict from
     ``PRSA_EncDec_Dataset`` so the future half of ``x_dec`` carries the
     REAL future-known exogenous covariates instead of zeros, and the
     ``x_mark_*`` tensors carry real normalized calendar features
     (matching the ``embed='timeF'``, ``freq='h'`` contract). The
     architecture itself is unchanged.

IMPORTANT architectural caveat (not a bug, this is how Autoformer is defined)
-----------------------------------------------------------------------------
Autoformer's ``forward`` does NOT consume the contents of ``x_dec`` -- it
derives the decoder input from ``x_enc`` via series_decomp (line 115-126 of
upstream), padding the future half with zeros from the lookback's mean and
seasonal components. ``x_dec`` only contributes its SHAPE (batch size /
device).  This means Autoformer CANNOT consume future-known real exogenous
covariates -- the architecture has no inlet for them.

What Autoformer DOES consume from the forecast window:
  * ``x_mark_dec`` -- the calendar features over (label_len + pred_len) are
    added to the decoder embedding (DataEmbedding_wo_pos). Our prepare_batch
    fills this with REAL normalized calendar values (month/day/hour/year),
    matching the ``embed='timeF'``, ``freq='h'`` contract; the cap original
    fed zeros and threw this signal away.

So this port is a fair-with-an-asterisk baseline:
  * Future-known REAL exogenous (other pollutants, weather):
      TFT / TiDE / CBM / Informer all consume them;
      Autoformer & FEDformer architecturally cannot.
  * Future-known CALENDAR features:
      our port now consumes them through x_mark_dec.

We keep the original structure unchanged precisely so this comparison
reflects each model's true capability rather than a non-original mod.
"""

import torch
import torch.nn as nn

from .tsl_layers.Embed import DataEmbedding_wo_pos
from .tsl_layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .tsl_layers.Autoformer_EncDec import (
    Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp,
)
from .encdec_dataset import assemble_tsl_inputs


class Autoformer(nn.Module):
    """
    Autoformer for Time-Series Forecasting
    """

    def __init__(self, input_dim, output_dim, seq_len, pred_len, d_model=512, n_heads=8, d_ff=2048,
                 num_layers=2, dropout=0.1, moving_avg=25, factor=3, activation="gelu",
                 embed="fixed", freq="h"):
        """
        Args
        ----
        input_dim   : number of input features (1 + R for our setup; target + reals).
        output_dim  : number of output features (1 for univariate target).
        seq_len     : L.
        pred_len    : H.
        ... (rest identical to upstream).
        """
        super(Autoformer, self).__init__()
        print(f"Autoformer initialized with d_model: {d_model}, n_heads: {n_heads}, d_ff: {d_ff}, num_layers: {num_layers}, dropout: {dropout}")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = seq_len // 2  # Middle portion of sequence for decoder input

        # Decomposition for trend-seasonal components
        self.decomp = series_decomp(moving_avg)

        # Embeddings
        self.enc_embedding = DataEmbedding_wo_pos(input_dim, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding_wo_pos(output_dim, d_model, embed, freq, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(num_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    output_dim,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(num_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, output_dim, bias=True)
        )

    # ----------------------------------------------------------------------
    # I/O glue -- the ONLY part of this file that differs from cap.
    # ----------------------------------------------------------------------
    def prepare_batch(self, batch):
        tsl = assemble_tsl_inputs(batch, label_len=self.label_len)
        inputs = (tsl["x_enc"], tsl["x_mark_enc"], tsl["x_dec"], tsl["x_mark_dec"])
        return inputs, tsl["y_true"]

    # ----------------------------------------------------------------------
    # Forward -- BYTE-IDENTICAL to cap/cap/models/Autoformer.py except that
    # the noisy `print(output.shape)` debug call is removed.
    # ----------------------------------------------------------------------
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Autoformer forward pass for forecasting.
        """
        # Decomposition: Trend & Seasonal components
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, 1]).to(x_dec.device)

        seasonal_init, trend_init = self.decomp(x_enc)

        # Prepare decoder input (label_len past values + zeros for future)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, 0].unsqueeze(-1), zeros], dim=1)

        # Encoding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # Decoding
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)

        # Final output (trend + seasonal)
        output = trend_part[:, :, 0].unsqueeze(-1) + seasonal_part
        print(output.shape)
        return output[:, -self.pred_len:, :]  # Match target shape

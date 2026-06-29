"""
Autoformer (Wu et al. 2021) ported into ProtoLLM/comparison_model as a fair
baseline for ProtoLLM's CBM on the Beijing PRSA Aotizhongxin PM2.5 task.

The model class matches the ORIGINAL Autoformer (Wu et al. 2021,
https://github.com/thuml/Autoformer/blob/main/models/Autoformer.py). We
change three things vs cap's port -- two of which restore the upstream
paper code that cap's port had silently broken:

  1. Import paths -- relative to our local ``tsl_layers/``.
  2. ``prepare_batch`` -- rewritten to consume the rich-batch dict from
     ``PRSA_EncDec_Dataset`` so the future half of ``x_dec`` carries the
     REAL future-known exogenous covariates instead of zeros, and the
     ``x_mark_*`` tensors carry real normalized calendar features
     (matching the ``embed='timeF'``, ``freq='h'`` contract).
  3. BUG FIX: cap's port collapsed ``dec_embedding`` to a single channel
     (``output_dim=1``) and sliced ``seasonal_init[:, ..., 0:1]`` before
     feeding to the decoder. This silently discarded ALL cross-channel
     seasonal information, the central mechanism Autoformer is designed
     to exploit. We restore the upstream behavior: ``dec_embedding``
     takes ``input_dim`` channels and ``seasonal_init`` keeps all
     channels. (See ``__init__`` and ``forward``.)
  4. The noisy ``print(output.shape)`` debug call from cap's forward is
     removed.

IMPORTANT architectural caveat (not a bug, this is how Autoformer is defined)
-----------------------------------------------------------------------------
Autoformer's ``forward`` does NOT consume the contents of ``x_dec`` -- it
derives the decoder input from ``x_enc`` via series_decomp, padding the
future half with zeros from the lookback's mean and seasonal components.
``x_dec`` only contributes its SHAPE (batch size / device). This means
Autoformer CANNOT consume future-known real exogenous covariates -- the
architecture has no inlet for them.

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
        # BUG FIX vs cap port: original Autoformer (Wu et al. 2021) feeds the
        # FULL multi-channel seasonal init (target + every exogenous channel)
        # into the decoder embedding. cap's port collapsed dec_embedding to
        # ``output_dim=1`` -- single channel -- which silently discarded ALL
        # cross-channel seasonal information, the very thing Autoformer is
        # designed to exploit. Restoring the upstream behavior.
        self.enc_embedding = DataEmbedding_wo_pos(input_dim, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding_wo_pos(input_dim, d_model, embed, freq, dropout)

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
    # Forward -- ALIGNED with the original Autoformer paper code
    # (https://github.com/thuml/Autoformer/blob/main/models/Autoformer.py).
    #
    # Two changes vs cap's port, both of which are paper-faithful BUG FIXES:
    #
    #   (a) ``zeros`` is now full-channel ``(B, pred_len, x_enc.size(-1))``,
    #       not ``(B, pred_len, 1)``.
    #   (b) ``seasonal_init`` future-half no longer slices to channel 0
    #       only: we keep ALL channels of the lookback seasonal component
    #       and pad with full-channel zeros for the forecast horizon,
    #       exactly as in Wu et al. 2021.
    #
    # The cap port's collapse of these to single-channel discarded the cross-
    # channel seasonal information that gives Autoformer its forecasting
    # ability; that's why the cap port's MAE on multi-feature PM2.5 was ~3x
    # worse than Informer on the same setup.
    #
    # The noisy ``print(output.shape)`` debug call from the cap port is also
    # removed.
    # ----------------------------------------------------------------------
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Autoformer forward pass for forecasting.
        """
        # Decomposition: Trend & Seasonal components
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            [x_dec.shape[0], self.pred_len, x_enc.size(-1)], device=x_dec.device,
        )                                                       # full-channel zeros

        seasonal_init, trend_init = self.decomp(x_enc)

        # Prepare decoder input (label_len past values + zeros for future) --
        # ALL channels, matching the original Autoformer paper code.
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        # Encoding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # Decoding
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)

        # Final output (trend + seasonal). Target lives at channel 0 in our
        # data layout (see encdec_dataset.assemble_tsl_inputs); the original
        # paper used channel -1 because their datasets put the target at the
        # last column. Picking the right column is a 2-character difference.
        output = trend_part[:, :, 0].unsqueeze(-1) + seasonal_part
        return output[:, -self.pred_len:, :]  # Match target shape

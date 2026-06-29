"""
FEDformer (Zhou et al. 2022) ported into ProtoLLM/comparison_model as a fair
baseline for ProtoLLM's CBM on the Beijing PRSA Aotizhongxin PM2.5 task.

The model class is BYTE-IDENTICAL to ``cap/cap/models/FEDFormer.py`` EXCEPT
for two paper-faithful BUG FIXES that the cap port introduced relative to
the original FEDformer (Zhou et al. 2022). We change:

  1. Import paths -- relative to our local ``tsl_layers/``.
  2. ``prepare_batch`` -- rewritten to consume the rich-batch dict from
     ``PRSA_EncDec_Dataset`` so the future half of ``x_dec`` carries the
     REAL future-known exogenous covariates (target column zeroed) and the
     ``x_mark_*`` tensors carry real normalized calendar features
     (``embed='timeF'``, ``freq='h'``). Architecture is unchanged.
  3. BUG FIX: cap's port used ``DataEmbedding`` (with sinusoidal positional
     encoding) for both encoder and decoder embeddings. The original
     FEDformer paper specifies ``DataEmbedding_wo_pos`` -- WITHOUT
     positional encoding -- because the Fourier blocks inherently encode
     position via spectral coefficients. Stacking sinusoidal positional
     embedding on top of spectral analysis adds noise and degrades
     performance. We restore the upstream choice. (See ``__init__``.)

IMPORTANT architectural caveat (not a bug, this is how FEDformer is defined)
---------------------------------------------------------------------------
FEDformer's ``forecast`` does NOT consume the contents of ``x_dec`` -- it
derives the decoder input from ``x_enc`` via ``series_decomp`` and
``F.pad(seasonal_init[..., -label_len:, :], (0,0,0,pred_len))``. ``x_dec``
contributes nothing beyond batch shape, identical to Autoformer.  This means
FEDformer CANNOT consume future-known REAL exogenous covariates -- the
architecture has no inlet for them.

What FEDformer DOES consume from the forecast window:
  * ``x_mark_dec`` -- calendar features over (label_len + pred_len) are
    added to the decoder embedding (DataEmbedding). Our prepare_batch fills
    this with REAL normalized calendar values; the cap original fed zeros.

So FEDformer (like Autoformer) is a fair-with-an-asterisk baseline:
  * Future-known REAL exogenous: TFT / TiDE / CBM / Informer consume them;
    FEDformer architecturally cannot.
  * Future-known CALENDAR features: now consumed via x_mark_dec.

We keep the original structure unchanged precisely so this comparison
reflects each model's true capability rather than a non-original mod.

Note about the Fourier vs Wavelets variants
-------------------------------------------
Default ``version='fourier'`` works out of the box (only needs torch+numpy).
``version='Wavelets'`` requires sympy + scipy; tsl_layers/MultiWaveletCorrelation.py
wraps the optional imports so this module still loads cleanly without them,
but constructing the wavelet attention blocks will fail at that point if
sympy is not installed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tsl_layers.Embed import DataEmbedding_wo_pos                 # BUG FIX -- see __init__
from .tsl_layers.AutoCorrelation import AutoCorrelationLayer
from .tsl_layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from .tsl_layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from .tsl_layers.Autoformer_EncDec import (
    Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp,
)
from .encdec_dataset import assemble_tsl_inputs


class FEDformer(nn.Module):
    """
    FEDformer: Frequency-domain Transformer for Long Sequence Time-Series Forecasting
    Paper: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(self, enc_in, dec_in, pred_len, c_out, seq_len, label_len,
                 d_model=512, embed="fixed", freq="h", dropout=0.05, factor=3,
                 n_heads=8, e_layers=2, d_layers=1, d_ff=2048, activation="gelu",
                 moving_avg=25, distil=False, version='fourier', mode_select='random', modes=32):
        """
        Initializes the FEDformer model. See upstream docstring for arg meanings.
        """
        super(FEDformer, self).__init__()
        print(f"FEDformer initialized with d_model: {d_model}, n_heads: {n_heads}, d_ff: {d_ff}, e_layers: {e_layers}, d_layers: {d_layers}, dropout: {dropout}")
        self.pred_len = pred_len
        self.label_len = label_len
        self.seq_len = seq_len

        # Decomposition for time series
        self.decomp = series_decomp(moving_avg)

        # Embeddings
        # BUG FIX vs cap port: the original FEDformer (Zhou et al. 2022,
        # https://github.com/MAZiqing/FEDformer/blob/master/models/FEDformer.py)
        # uses ``DataEmbedding_wo_pos`` (no positional encoding) for BOTH the
        # encoder and decoder, because the Fourier blocks inherently encode
        # position via spectral coefficients. cap's port substituted
        # ``DataEmbedding`` (with sinusoidal positional encoding), adding
        # noise on top of the spectral signal and degrading the Fourier
        # block's representational quality. Restoring the upstream class.
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq, dropout)

        # Choose frequency-based attention mechanism
        if version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(
                in_channels=d_model, out_channels=d_model, seq_len_q=seq_len // 2 + pred_len,
                seq_len_kv=seq_len, modes=modes, ich=d_model, base='legendre', activation='tanh'
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=d_model, out_channels=d_model, n_heads=n_heads, seq_len=seq_len,
                modes=modes, mode_select_method=mode_select
            )
            decoder_self_att = FourierBlock(
                in_channels=d_model, out_channels=d_model, n_heads=n_heads,
                seq_len=seq_len // 2 + pred_len, modes=modes, mode_select_method=mode_select
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=d_model, out_channels=d_model, seq_len_q=seq_len // 2 + pred_len,
                seq_len_kv=seq_len, modes=modes, mode_select_method=mode_select, num_heads=n_heads
            )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(encoder_self_att, d_model, n_heads),
                    d_model, d_ff, moving_avg=moving_avg, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(decoder_self_att, d_model, n_heads),
                    AutoCorrelationLayer(decoder_cross_att, d_model, n_heads),
                    d_model, c_out, d_ff, moving_avg=moving_avg, dropout=dropout, activation=activation
                )
                for _ in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    # ----------------------------------------------------------------------
    # I/O glue -- the ONLY part of this file that differs from cap.
    # ----------------------------------------------------------------------
    def prepare_batch(self, batch):
        tsl = assemble_tsl_inputs(batch, label_len=self.label_len)
        inputs = (tsl["x_enc"], tsl["x_mark_enc"], tsl["x_dec"], tsl["x_mark_dec"])
        return inputs, tsl["y_true"]

    # ----------------------------------------------------------------------
    # forecast + forward -- BYTE-IDENTICAL to cap/cap/models/FEDFormer.py
    # except the commented-out debug `print` is removed.
    # ----------------------------------------------------------------------
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasts the next `pred_len` values given input sequences.
        """
        # Decomposition
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)

        # Prepare decoder inputs
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))

        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Decoder
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)

        # Final output
        dec_out = trend_part + seasonal_part
        # For univariate forecasting, return only the target feature (first feature)
        return dec_out[:, -self.pred_len:, :1]  # Shape: (batch, pred_len, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forward pass for forecasting.
        """
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

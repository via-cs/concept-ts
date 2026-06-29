"""
Informer (Zhou et al. 2021) ported into ProtoLLM/comparison_model as a fair
baseline for ProtoLLM's CBM on the Beijing PRSA Aotizhongxin PM2.5 task.

Why this port (vs simply re-running cap's Informer)
---------------------------------------------------
The model class itself (__init__, forward, long_forecast, short_forecast)
is BYTE-IDENTICAL to ``cap/cap/models/Informer.py``. We change only TWO things,
neither of which touches the original architecture:

  1. Import paths -- relative imports point at our local ``tsl_layers/``
     sibling package rather than ``cap.cap.layers``. The layer .py files are
     copied unmodified from cap (see tsl_layers/ for the exact files).
  2. ``prepare_batch`` is rewritten to actually consume the future-window
     KNOWN-FUTURE exogenous covariates. The cap implementation built the
     decoder input as ``[X[:, -label_len:, :], zeros(pred_len, C)]`` -- the
     exogenous columns of the future half were always zero. Our adapter
     emits a rich-batch dict (see encdec_dataset.PRSA_EncDec_Dataset) and
     the new ``prepare_batch`` fills the future half of ``x_dec`` with the
     ACTUAL future weather + other-pollutant values; only the target column
     (PM2.5) is zeroed.

Everything ProtoLLM compares against (TFT, TiDE, CBM) feeds the same future-
known covariates, so this change just brings Informer onto equal information
terms. The architecture itself is untouched.

Architectural capability note
-----------------------------
Unlike Autoformer and FEDformer, Informer's decoder embedding takes
``x_dec`` directly (via DataEmbedding's value_embedding), so the future-half
contents are genuinely consumed: the projected future exogenous values flow
through the masked self-attention + cross-attention stack alongside the
target placeholder. So Informer is a TRUE future-known-exogenous baseline,
on par with TFT / TiDE / CBM. (Autoformer & FEDformer architecturally
ignore x_dec content; see those files' docstrings.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tsl_layers.Transformer_EncDec import (
    Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer,
)
from .tsl_layers.SelfAttention_Family import ProbAttention, AttentionLayer
from .tsl_layers.Embed import DataEmbedding
from .encdec_dataset import assemble_tsl_inputs


class Informer(nn.Module):
    """
    Informer model optimized for forecasting (long & short-term).
    """

    def __init__(self, enc_in, dec_in, pred_len, label_len, d_model=512, embed="fixed", freq="h", dropout=0.1,
                 factor=3, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, activation="gelu", distil=False):
        super(Informer, self).__init__()
        print(f"Informer initialized with d_model: {d_model}, n_heads: {n_heads}, d_ff: {d_ff}, e_layers: {e_layers}, d_layers: {d_layers}, dropout: {dropout}")

        self.pred_len = pred_len
        self.label_len = label_len

        # Embedding layers
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [
                ConvLayer(d_model) for _ in range(e_layers - 1)
            ] if distil else None,
            norm_layer=nn.LayerNorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, 1, bias=True)  # Output shape: (B, seq, 1)
        )

    # ----------------------------------------------------------------------
    # I/O glue -- the ONLY part of this file that differs from cap.
    # ----------------------------------------------------------------------
    def prepare_batch(self, batch):
        """Convert a rich-batch dict from PRSA_EncDec_Dataset into the
        Informer ``forward`` input tuple, with REAL future-known exogenous
        covariates plumbed into ``x_dec``'s future half (target column kept
        zero). This is the contract TFT, TiDE and CBM use as well.
        """
        tsl = assemble_tsl_inputs(batch, label_len=self.label_len)
        inputs = (tsl["x_enc"], tsl["x_mark_enc"], tsl["x_dec"], tsl["x_mark_dec"])
        return inputs, tsl["y_true"]

    # ----------------------------------------------------------------------
    # Forward + forecast helpers -- BYTE-IDENTICAL to cap/cap/models/Informer.py.
    # ----------------------------------------------------------------------
    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out[:, -self.pred_len:, :]  # Shape: (B, pred_len, 1)

    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # Normalization
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = (x_enc - mean_enc) / std_enc

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        dec_out = dec_out * std_enc + mean_enc  # De-normalization
        return dec_out[:, -self.pred_len:, :]  # Shape: (B, pred_len, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, task="long"):
        if task == "long":
            return self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif task == "short":
            return self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            raise ValueError("Task must be either 'long' or 'short'.")

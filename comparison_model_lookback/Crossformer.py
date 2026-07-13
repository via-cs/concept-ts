"""
Crossformer (Zhang & Yan 2023, "Crossformer: Transformer Utilizing Cross-
Dimension Dependency for Multivariate Time Series Forecasting",
https://openreview.net/pdf?id=vSVLM2j9eie) ported into
ProtoLLM/comparison_model_lookback as a strict LOOKBACK-ONLY baseline.
Directly comparable to ProtoLLM's CBM run in
``experiment_mode='lookback_only'``.

Crossformer is encoder-decoder but the decoder queries are learned position
embeddings (``dec_pos_embedding``) -- it needs NO future inputs, so it is a
valid lookback-only model. It patch-embeds each variable's lookback into
segments and applies Two-Stage Attention (across time AND across variables).

Adaptations from cap:
  1. Imports point at our sibling ``comparison_model.tsl_layers`` package
     (Crossformer_EncDec was ported there too).
  2. Base class is plain ``nn.Module`` (dropped cap's BaseTimeSeriesModel)
     and the unused ``FlattenHead`` import was removed.
  3. ``l is 0`` -> ``l == 0`` (identity-compare-with-literal is a SyntaxWarning
     on modern Python).
  4. ``forward`` returns ONLY the target channel (last column) -> (B, pred_len, 1),
     matching our univariate PM2.5 setup. PRSA_LookbackOnly_Dataset places
     PM2.5 at column -1.

Requires ``einops`` (same as upstream Crossformer).
"""

import torch
import torch.nn as nn
from einops import repeat
from types import SimpleNamespace
from math import ceil

from comparison_model.tsl_layers.Crossformer_EncDec import (
    scale_block, Encoder, Decoder, DecoderLayer,
)
from comparison_model.tsl_layers.Embed import PatchEmbedding
from comparison_model.tsl_layers.SelfAttention_Family import (
    AttentionLayer, FullAttention, TwoStageAttentionLayer,
)


class Crossformer(nn.Module):
    """Lookback-only Crossformer for PRSA PM2.5 forecasting.

    ``x_enc`` has shape ``(B, seq_len, C)`` with PM2.5 (target) at column -1.
    Emits univariate ``(B, pred_len, 1)`` forecast (the target channel).
    """

    def __init__(self, enc_in, seq_len, pred_len, num_layers, d_model, n_heads,
                 d_ff, dropout, factor, seg_len=12, win_size=2):
        super(Crossformer, self).__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.seg_len = seg_len
        self.win_size = win_size

        # Padding so seq_len / pred_len are multiples of seg_len.
        self.pad_in_len = ceil(1.0 * seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (num_layers - 1)))
        self.head_nf = d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(
            d_model, self.seg_len, self.seg_len, self.pad_in_len - seq_len, 0,
        )
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, enc_in, self.in_seg_num, d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        configs = SimpleNamespace(factor=factor, dropout=dropout)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(
                    configs, 1 if l == 0 else self.win_size, d_model, n_heads, d_ff,
                    1, dropout,
                    self.in_seg_num if l == 0 else ceil(self.in_seg_num / self.win_size ** l),
                    factor,
                )
                for l in range(num_layers)
            ]
        )

        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, enc_in, (self.pad_out_len // self.seg_len), d_model))

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(
                        configs, (self.pad_out_len // self.seg_len), factor, d_model,
                        n_heads, d_ff, dropout,
                    ),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads,
                    ),
                    self.seg_len,
                    d_model,
                    d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers + 1)
            ],
        )

    # ----------------------------------------------------------------------
    # I/O glue: takes ONLY the 2-tuple lookback batch. No future inputs.
    # ----------------------------------------------------------------------
    def prepare_batch(self, batch):
        x_enc, y = batch
        return (x_enc, None, None, None), y

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: (B, L, C). PatchEmbedding wants (B, C, L).
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = x_enc.reshape(-1, n_vars, x_enc.shape[-2], x_enc.shape[-1])   # (b, d, seg_num, d_model)
        x_enc = x_enc + self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_in = repeat(self.dec_pos_embedding,
                        'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        dec_out = self.decoder(dec_in, enc_out)                               # (B, pad_out_len, C)
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # Univariate output: last pred_len steps, target = last channel.
        return dec_out[:, -self.pred_len:, -1:]                               # (B, pred_len, 1)

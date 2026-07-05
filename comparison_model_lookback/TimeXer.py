"""
TimeXer (Wang et al. 2024, "TimeXer: Empowering Transformers for Time Series
Forecasting with Exogenous Variables") ported into
ProtoLLM/comparison_model_lookback as a strict LOOKBACK-ONLY baseline.

TimeXer is natively encoder-only and consumes NO future features -- it
patch-embeds the endogenous target in the lookback window and cross-attends
to an inverted embedding of the lookback-window exogenous features. So it's
a natural fit for our lookback-only comparison; nothing about the
architecture had to be relaxed.

Two small adaptations from cap:
  1. Imports point at our sibling ``comparison_model.tsl_layers`` package.
  2. ``forward`` routes through ``forecast(...)`` (univariate: last-channel
     endogenous target) rather than cap's default ``forecast_multi(...)``,
     because we do univariate PM2.5 forecasting. Output shape is
     (B, pred_len, 1). PRSA_LookbackOnly_Dataset places PM2.5 at column -1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from comparison_model.tsl_layers.SelfAttention_Family import (
    FullAttention, AttentionLayer,
)
from comparison_model.tsl_layers.Embed import (
    DataEmbedding_inverted, PositionalEmbedding,
)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear  = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):                                                       # (B, nvars, d_model, patch_num+1)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    """Patch-based embedding of the endogenous (target) variable."""

    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding    = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token          = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout            = nn.Dropout(dropout)

    def forward(self, x):                                                       # (B, n_vars, L)
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))                      # (B, n_vars, 1, d_model)

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)    # (B, n_vars, P, patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) # (B*n_vars, P, patch_len)
        x = self.value_embedding(x) + self.position_embedding(x)                # (B*n_vars, P, d_model)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)                                          # (B, n_vars, P+1, d_model)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class _Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm   = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        return x


class _EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention  = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff,   kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff,   d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, _, D = cross.shape
        # self-attention over patches of the endogenous target
        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        )
        x = self.norm1(x)

        # global-token cross-attention over the inverted exogenous embedding
        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(
            self.cross_attention(x_glb, cross, cross,
                                 attn_mask=cross_mask, tau=tau, delta=delta)[0]
        )
        x_glb_attn = torch.reshape(
            x_glb_attn,
            (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2]),
        ).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class TimeXer(nn.Module):
    """Lookback-only TimeXer for PRSA PM2.5 forecasting.

    Assumes ``x_enc`` has shape ``(B, seq_len, C)`` with PM2.5 (target) at
    column -1 and other covariates at columns 0..C-2. Emits univariate
    (B, pred_len, 1) forecast.
    """

    def __init__(self, enc_in, seq_len, pred_len, use_norm=True, patch_len=16,
                 d_ff=2048, activation='gelu', num_layers=1, n_heads=8,
                 d_model=512, dropout=0.1, factor=3):
        super().__init__()
        assert seq_len % patch_len == 0, (
            f"seq_len ({seq_len}) must be divisible by patch_len ({patch_len})."
        )
        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.use_norm  = use_norm
        self.patch_len = patch_len
        self.patch_num = int(seq_len // patch_len)
        # ``n_vars`` for the endogenous embedding is 1 (only the target).
        self.n_vars    = 1

        self.en_embedding = EnEmbedding(1, d_model, self.patch_len, dropout)
        # DataEmbedding_inverted: converts (B, L, n_ex) -> (B, n_ex, d_model)
        # so cross-attention keys/values are one token per exogenous channel.
        self.ex_embedding = DataEmbedding_inverted(seq_len, d_model, 'fixed', 'h', dropout)

        self.encoder = _Encoder(
            [
                _EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads,
                    ),
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
        self.head_nf = d_model * (self.patch_num + 1)
        # ``head`` handles n_vars=1 -> pred_len values.
        self.head = FlattenHead(1, self.head_nf, pred_len, head_dropout=dropout)

    # ----------------------------------------------------------------------
    # I/O glue: takes ONLY the 2-tuple lookback batch. No calendar marks.
    # ----------------------------------------------------------------------
    def prepare_batch(self, batch):
        x_enc, y = batch
        return (x_enc,), y

    def forward(self, x_enc):
        if self.use_norm:
            # RevIN-style per-series normalization from the Non-stationary
            # Transformer paper; applied per channel of the lookback.
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # Endogenous = last channel (PM2.5).  Shape -> (B, 1, L).
        en_input = x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1)
        en_embed, n_vars = self.en_embedding(en_input)

        # Exogenous = all-but-last channels. ex_embedding handles x_mark_enc=None.
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], None)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)                                   # (B, n_vars, d_model, patch_num+1)

        dec_out = self.head(enc_out)                                            # (B, n_vars, pred_len)
        dec_out = dec_out.permute(0, 2, 1)                                      # (B, pred_len, n_vars=1)

        if self.use_norm:
            # De-normalize using only the endogenous (target = last) channel stats.
            dec_out = dec_out * stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out

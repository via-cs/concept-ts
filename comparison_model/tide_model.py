"""
TiDE (Das et al., 2023, https://arxiv.org/abs/2304.08424) adapted as a FAIR
baseline for ProtoLLM's Concept Bottleneck Model on the Beijing PRSA
Aotizhongxin PM2.5 task.

Why this adaptation (vs the upstream TiDE-main implementation)
--------------------------------------------------------------
The upstream TiDE-main only fed *calendar* time-features as the dynamic
covariates and flattened the lookback covariates as a "static attribute".
That is NOT the regime ProtoLLM uses. ProtoLLM's CBM consumes the real
exogenous drivers (meteorology + other pollutants + calendar) as KNOWN-FUTURE
covariates over the whole forecast horizon. To compare fairly, this TiDE:

  * Consumes the 14 PRSA exogenous channels (year, month, day, hour, PM10,
    SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM) as dynamic covariates over
    the WHOLE window (lookback + forecast) -- i.e. the future covariates are
    known, exactly as in ProtoLLM.
  * NEVER sees PM2.5 (the target) over the forecast window. Only the lookback
    target is an input. No target leakage.
  * Reads the batch size from the tensor (not a fixed arg), so the final
    partial batch does not crash.

Architecture (standard TiDE)
----------------------------
    x_cov (B, L+H, C) --featproj--> proj (B, L+H, p)
    enc_in = concat( flatten(proj), flatten(y_past(B,L,1)) )
    encoded   = ResidualBlock-stack encoder  (-> hidden)
    dense     = ResidualBlock-stack decoder  (-> H * decoder_output_dim)
    per step t in horizon:
        temporal_decoder( [ dense[t] ,  proj[L+t] ] ) -> y_hat[t]   (1 dim)
    y_hat += Linear(L -> H)(y_past)                    # lookback residual
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """TiDE's basic building block: 2-layer MLP + dropout + skip + LayerNorm."""

    def __init__(self, in_features, hid_features, out_features, dropout_prob=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hid_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hid_features, out_features)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.norm = nn.LayerNorm(out_features)
        self.shortcut = (nn.Identity() if in_features == out_features
                         else nn.Linear(in_features, out_features))

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.shortcut(residual)
        out = self.norm(out)
        return out


class _ResidualStack(nn.Module):
    """A stack of ``num_blocks`` ResidualBlocks (encoder / dense decoder)."""

    def __init__(self, in_features, hid_features, out_features,
                 drop_prob, num_blocks=1):
        super().__init__()
        blocks = []
        f = in_features
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(f, hid_features, out_features, drop_prob))
            f = out_features
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class TiDEModel(nn.Module):
    """TiDE for univariate-target forecasting with known-future covariates.

    Args
    ----
    n_covariates : C, number of dynamic covariate channels (14 for PRSA).
    lookback_len : L, lookback length.
    pred_len     : H, forecast horizon.
    The remaining args are standard TiDE hyperparameters.

    forward(y_past, x_cov) -> y_hat
        y_past : (B, L, 1)    scaled lookback target.
        x_cov  : (B, L+H, C)  scaled known covariates over the whole window.
        y_hat  : (B, H, 1)    scaled forecast (same z-space as y_past).
    """

    def __init__(self, n_covariates, lookback_len, pred_len,
                 feat_size=4, hidden_size=256, num_encoder_layers=2,
                 num_decoder_layers=2, decoder_output_dim=8,
                 temporal_decoder_hidden=128, drop_prob=0.3):
        super().__init__()
        self.L = int(lookback_len)
        self.H = int(pred_len)
        self.C = int(n_covariates)
        self.feat_size = int(feat_size)
        self.decoder_output_dim = int(decoder_output_dim)

        # Per-timestep covariate feature projection (shared across all steps).
        self.featproj = ResidualBlock(self.C, hidden_size, self.feat_size, drop_prob)

        # Encoder input = flattened projected covariates over (L+H) + past target (L).
        concat_dim = (self.L + self.H) * self.feat_size + self.L * 1
        self.encoder = _ResidualStack(
            concat_dim, hidden_size, hidden_size, drop_prob, num_encoder_layers,
        )
        self.denseDecoder = _ResidualStack(
            hidden_size, hidden_size, self.decoder_output_dim * self.H,
            drop_prob, num_decoder_layers,
        )
        # Temporal decoder: per-step [decoded_vec | projected_cov] -> 1.
        self.temporalDecoder = ResidualBlock(
            self.feat_size + self.decoder_output_dim, temporal_decoder_hidden,
            1, drop_prob,
        )
        # Linear lookback residual (global skip from past target to horizon).
        self.lookback_linear = nn.Linear(self.L, self.H)

    def forward(self, y_past, x_cov):
        B = y_past.size(0)
        proj = self.featproj(x_cov)                              # (B, L+H, p)

        enc_in = torch.cat(
            [proj.reshape(B, -1), y_past.reshape(B, -1)], dim=1,
        )                                                        # (B, (L+H)*p + L)
        encoded = self.encoder(enc_in)                           # (B, hidden)

        dense = self.denseDecoder(encoded)                       # (B, H*dod)
        dense = dense.reshape(B, self.H, self.decoder_output_dim)  # (B, H, dod)

        pred_feat = proj[:, -self.H:, :]                         # (B, H, p) future covariates
        temporal_in = torch.cat([dense, pred_feat], dim=2)       # (B, H, dod+p)
        temporal = self.temporalDecoder(temporal_in)             # (B, H, 1)

        res = self.lookback_linear(y_past.squeeze(-1)).unsqueeze(-1)  # (B, H, 1)
        return temporal + res

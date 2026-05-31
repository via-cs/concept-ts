"""
Per-variable channel embedding (Stage 1 of the ModernTCN encoder).

Lifts each (variable, timestep) scalar / integer to a D-dim vector while
preserving the variable axis. Output shape: (B, T, M, D).

Continuous channels go through ``Linear(1, D)`` — a single layer; the conv
stack downstream adds nonlinearity. Categorical channels go through
``nn.Embedding(num_classes, D)`` with the same hard-coded class counts as
ProtoTS (the dataset's calendar / holiday / weekday columns).

Rev_in normalization is applied to the target column's lookback portion,
and the forecast portion of the target is overwritten with a -5 sentinel
(the standard ProtoTS data contract).
"""

import os

import torch
import torch.nn as nn

from layer.rev_norm import Rev_in


# ---------------------------------------------------------------------------
# Dataset switch — flip to "AirQuality" to point the per-channel embedder at
# the Riverside PM2.5 schema. Keep this in sync with ``DATASET`` in
# ``dataloader/dataloader.py`` and ``prototype_proposing_loop/prompts.py``.
#
# Each entry of ``_CLASS_FEATURE_NUM_CLASSES`` is the number of categories
# used by ``nn.Embedding(num_classes, embed_dim)`` for that column index.
# Continuous columns get 0 (the Linear(1, embed_dim) path runs instead).
# ---------------------------------------------------------------------------
DATASET = "AirQuality"

if DATASET == "LOF":
    # 23-column LOF / ProtoTS schema (target = "load", last column).
    #   0:lunar_year   1:lunar_month  2:lunar_day  3:solar_year  4:solar_month
    #   5:solar_day    6:month(=hour) 7:minute     8-15:continuous weather
    #  16:is_special_workday  17:is_special_holiday  18:is_holiday
    #  19:is_workday  20:day_of_week  21:pricing  22:load
    _CLASS_FEATURE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20]
    _CLASS_FEATURE_NUM_CLASSES = [
        6, 13, 32, 6, 13,
        32, 25, 61, 0, 0,
        0, 0, 0, 0, 0,
        0, 2, 2, 2, 2,
        7, 0, 0,
    ]
elif DATASET == "AirQuality":
    # 17-column Riverside schema (target = "pm25_poc_3", last column).
    #   0:wind_spd_res     1:temp_outdoor   2:rel_humidity  3:solar_radiation
    #   4:uv_radiation     5:baro_pressure  6:sp            7:skt
    #   8:blh              9:tp            (all continuous so far)
    #  10:month (1-12)    11:day (1-31)   12:hour (0-23)   13:day_of_week (0-6)
    #  14:is_weekend      15:is_weekday   16:pm25_poc_3 (target, continuous)
    _CLASS_FEATURE_INDICES = [10, 11, 12, 13, 14, 15]
    _CLASS_FEATURE_NUM_CLASSES = [
        0, 0, 0, 0, 0,     # 0-4   wind_spd_res, temp_outdoor, rel_humidity, solar_radiation, uv_radiation
        0, 0, 0, 0, 0,     # 5-9   baro_pressure, sp, skt, blh, tp
        13, 32, 24, 7, 2,  # 10-14 month, day, hour, day_of_week, is_weekend
        2, 0,              # 15-16 is_weekday, pm25_poc_3 (target)
    ]
else:
    raise ValueError(
        f"Unknown DATASET={DATASET!r} in convolution_encoder.channel_embedding. "
        f"Expected 'LOF' or 'AirQuality'."
    )


class PerChannelEmbedding(nn.Module):
    """Lift each channel scalar to a D-dim vector, preserve channel axis.

    Args
    ----
    input_dim     : total channels (target + exogenous)
    ext_dim       : count of exogenous channels (target indices = [ext_dim, input_dim))
    embed_dim     : per-channel embedding dimension D
    input_length  : lookback length, used for Rev_in slicing
    output_length : forecast length, used for sentinel mask

    Forward
    -------
    x : (B, T, input_dim) → returns (B, T, input_dim, embed_dim)
    """

    def __init__(self, input_dim, ext_dim, embed_dim,
                 input_length, output_length):
        super().__init__()
        self.input_dim = input_dim
        self.ext_dim = ext_dim
        self.embed_dim = embed_dim
        self.input_length = input_length
        self.output_length = output_length

        self.input_time_index = list(range(input_length))
        self.target_feature_index = list(range(ext_dim, input_dim))
        self.class_feature = _CLASS_FEATURE_INDICES
        self.class_feature_map = _CLASS_FEATURE_NUM_CLASSES

        # Per-channel embedding modules
        self.embeding = nn.ModuleList()
        for i in range(input_dim):
            if i in self.class_feature:
                self.embeding.append(
                    nn.Embedding(self.class_feature_map[i], embed_dim)
                )
            else:
                self.embeding.append(nn.Linear(1, embed_dim))

        self.embeding.apply(self._init_weight)

        # Rev_in over the target slice (typically 1 channel)
        self.rev_in = Rev_in(input_dim - ext_dim)

    @staticmethod
    def _init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0,
                             std=1.0 / (m.weight.shape[1] ** 0.5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, use_rev_in: int = 1):
        device = os.environ.get("device", x.device)
        N, T = x.size(0), x.size(1)

        # Rev_in + sentinel on the target column
        input_feature = torch.index_select(
            x, 1, torch.tensor(self.input_time_index, device=device)
        )
        target_feature = torch.index_select(
            input_feature, 2, torch.tensor(self.target_feature_index, device=device)
        )
        target_feature = self.rev_in(target_feature)

        x[:, : self.input_length, -1] = target_feature.reshape(N, self.input_length)
        x[:, self.input_length :, -1] = -5

        # Per-channel embedding — channel axis preserved
        embedding_outputs = []
        for i in range(self.input_dim):
            now_feature = torch.index_select(
                x, 2, torch.tensor([i], device=device)
            )                                              # (N, T, 1)
            if i in self.class_feature:
                now_feature = now_feature.long()
            emb_i = self.embeding[i](now_feature)          # (N, T, 1, embed_dim)
            embedding_outputs.append(emb_i.reshape(N, T, 1, -1))

        embedding = torch.cat(embedding_outputs, dim=-2)   # (N, T, input_dim, embed_dim)
        return embedding

    def denorm(self, x):
        """Undo the Rev_in normalization cached during forward."""
        return self.rev_in.de_rev_in(x)

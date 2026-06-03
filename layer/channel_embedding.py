import os
import torch
import torch.nn as nn

from layer.rev_norm import Rev_in


# ---------------------------------------------------------------------------
# Dataset switch — flip to "AirQuality" to point the Mixer-encoder's
# per-channel embedder at the Riverside PM2.5 schema. Keep in sync with the
# ``DATASET`` constant in ``dataloader/dataloader.py``.
# ---------------------------------------------------------------------------
DATASET = "LOF"

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
        f"Unknown DATASET={DATASET!r} in layer.channel_embedding. "
        f"Expected 'LOF' or 'AirQuality'."
    )


class channel_embedding(nn.ModuleList):
    def __init__(
        self,
        input_dim,
        ext_dim,
        embeding_dim,
        input_length,
        output_length,
        dropout,
        norm_type='B'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.ext_dim = ext_dim
        self.embeding_dim = embeding_dim
        self.input_length = input_length
        self.output_length = output_length
        self.dropout = dropout
        self.norm_type = norm_type

        self.input_time_index = list(range(self.input_length))
        self.output_time_index = list(range(self.input_length, self.input_length + self.output_length))

        self.ext_feature_index = list(range(self.ext_dim))
        self.target_feature_index = list(range(self.ext_dim, self.input_dim))

        # Derived from the DATASET switch at the top of this module.
        self.class_feature = _CLASS_FEATURE_INDICES
        self.class_feature_map = _CLASS_FEATURE_NUM_CLASSES

        self.embeding = nn.ModuleList()
        for i in range(self.input_dim):
            if i in self.class_feature:
                self.embeding.append(nn.Embedding(self.class_feature_map[i], self.embeding_dim))
            else:
                self.embeding.append(
                    nn.Sequential(
                        nn.Linear(1, self.embeding_dim),
                        nn.GELU(),
                        nn.Linear(self.embeding_dim, self.embeding_dim),
                    )
                )

        self.embeding.apply(self.init_weight)

        self.rev_in = Rev_in(input_dim - ext_dim)

    def forward(self, x, use_rev_in=1):
        device = os.environ.get('device')
        batch_size, length = x.size(0), x.size(1)

        input_feature = torch.index_select(
            x, 1, torch.tensor(self.input_time_index).to(device)
        )
        target_feature = torch.index_select(
            input_feature, 2, torch.tensor(self.target_feature_index).to(device)
        )
        target_feature = self.rev_in(target_feature)

        x[:, :self.input_length, -1] = target_feature.reshape(batch_size, self.input_length)
        x[:, self.input_length:, -1] = -5

        embedding_outputs = []
        for i in range(self.input_dim):
            now_feature = torch.index_select(
                x, 2, torch.tensor([i]).to(device)
            )
            if i in self.class_feature:
                now_feature = now_feature.long()
            embedding_outputs.append(
                self.embeding[i](now_feature).reshape(batch_size, length, 1, -1)
            )

        embeding = torch.concat(embedding_outputs, dim=-2)
        embeding = embeding.mean(dim=-2)

        return embeding

    def denorm(self, x):
        return self.rev_in.de_rev_in(x)

    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            input_dim = m.weight.shape[1]
            nn.init.normal_(m.weight, mean=0, std=1 / (input_dim ** 0.5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

import torch
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(
        self,
        input_length,
        input_dim,
        bottle_dim_p,
        output_dim,
        norm_type='L',
        dropout=0.1
    ):
        super().__init__()

        self.input_length = input_length
        self.input_dim = input_dim
        self.bottle_dim_p = bottle_dim_p
        self.output_dim = output_dim
        self.dropout = dropout

        if norm_type == 'L':
            self.norm = nn.LayerNorm(self.input_dim)
        else:
            self.norm = nn.BatchNorm2d(1)

        self.time_mixer = nn.Sequential(
            nn.Linear(self.input_length, int(self.input_length * self.bottle_dim_p)),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(self.input_length * self.bottle_dim_p), self.input_length), 
            nn.Dropout(self.dropout)
        )

        self.feature_mixer = nn.Sequential(
            nn.Linear(self.input_dim, int(self.input_dim * self.bottle_dim_p)),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(self.input_dim * self.bottle_dim_p), self.output_dim),
            nn.Dropout(self.dropout)
        )

        self.time_mixer.apply(self.init_weight)
        self.feature_mixer.apply(self.init_weight)

    def forward(self, inputs):
        x = self.norm(inputs.reshape(inputs.size(0), 1, inputs.size(1), inputs.size(2)))
        x = x.reshape(inputs.size(0), inputs.size(1), inputs.size(2))
        x = self.time_mixer(x.transpose(1, 2)).transpose(1, 2)

        mid_ans = inputs + x  

        x = self.norm(mid_ans.reshape(mid_ans.size(0), 1, mid_ans.size(1), mid_ans.size(2)))
        x = x.reshape(mid_ans.size(0), mid_ans.size(1), mid_ans.size(2))
        x = self.feature_mixer(x)

        return mid_ans + x 

    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            input_dim = m.weight.shape[1]
            nn.init.normal_(m.weight, mean=0, std=1 / (input_dim ** 0.5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

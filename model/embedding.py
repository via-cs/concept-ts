import torch
import torch.nn as nn

from layer.rev_norm import Rev_in
from layer.backbone import Backbone
from layer.channel_embedding import channel_embedding


class Embedding_Learning(nn.Module):
    def __init__(
        self,
        input_dim,
        mid_dim,
        ext_dim,
        output_dim,
        bottle_dim_p,
        input_length,
        output_length,
        num_blocks,
        dropout,
        use_rev_in=1,
        norm_type='L'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.ext_dim = ext_dim
        self.mid_dim = mid_dim
        self.output_dim = output_dim
        self.bottle_dim_p = bottle_dim_p
        self.input_length = input_length
        self.output_length = output_length
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.norm_type = norm_type
        self.use_rev_in = use_rev_in

        self.embeding = channel_embedding(
            input_dim=self.input_dim,
            ext_dim=self.ext_dim,
            embeding_dim=self.mid_dim,
            input_length=self.input_length,
            output_length=self.output_length,
            dropout=self.dropout,
            norm_type=self.norm_type
        )

        self.res_blocks = nn.Sequential(
            *[
                Backbone(
                    input_length=self.input_length + self.output_length,
                    input_dim=self.mid_dim,
                    bottle_dim_p=self.bottle_dim_p,
                    output_dim=self.mid_dim,
                    dropout=self.dropout,
                    norm_type=self.norm_type
                ) for _ in range(self.num_blocks)
            ]
        )

        self.output_time_linear = nn.Linear(self.input_length + self.output_length, self.output_length)
        self.output_time_linear.apply(self.init_weight)

        self.output_feature_linear = nn.Linear(self.mid_dim, self.output_dim)

    def forward(self, x):
        embeding = self.embeding(x, self.use_rev_in)

        encoder = self.res_blocks(embeding)

        encoder = self.output_time_linear(encoder.transpose(1, 2)).transpose(1, 2)
        output = self.output_feature_linear(encoder)

        if self.use_rev_in:
            output = self.embeding.denorm(output)

        return output, encoder

    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            input_dim = m.weight.shape[1]
            nn.init.normal_(m.weight, mean=0, std=1 / (input_dim ** 0.5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

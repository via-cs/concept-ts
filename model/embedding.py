"""
Encoder export point + adapter-mode switch.

Three encoder variants are available; flip ONE line at the bottom of this
file to switch between them. ``ADAPTER_MODE`` is derived automatically from
whichever class is bound to ``Embedding_Learning``.

    Mode A — "none"             (current default)
        Embedding_Learning = ConvEmbedding_Learning
        Conv encoder with its own learnable 96→1 ``time_pool``.
        Returns (B, mid_dim). Prototype_Layer uses it directly as the
        match key — no adapter. align_layer is skipped (the encoder has
        already collapsed the time axis).

    Mode B — "matching_head"    (identity-init adapter, "ProtoLLM copy"-style)
# Embedding_Learning = ConvEmbedding_WithAdapter
        Conv encoder WITHOUT time_pool.
        Returns (B, output_length, mid_dim). Prototype_Layer applies
        ``matching_head`` — a Linear(mid_dim, mid_dim, bias=False) with
        ``nn.init.eye_`` init — over ``x.mean(dim=1)`` to produce the
        match key. At step 0 the adapter is a no-op, so the LLM-init
        coordinate system is preserved. align_layer IS applied (3D path).

    Mode C — "patch_embedding"  (original ProtoTS)
# Embedding_Learning = MixerEmbedding_Learning
        Mixer encoder (preserved verbatim below).
        Returns (B, output_length, mid_dim). Prototype_Layer applies the
        original ProtoTS ``patch_embedding`` — mean over the feature axis
        then a 96→192→128 MLP — to produce the match key. align_layer
        IS applied (3D path).

All three encoders share an identical constructor signature, so downstream
code (embedding_learning.py, prototype_learning.py, build_llm_init.py,
extract_top_k.py, etc.) needs zero changes.
"""

import torch
import torch.nn as nn

from layer.rev_norm import Rev_in
from layer.backbone import Backbone
from layer.channel_embedding import channel_embedding

from convolution_encoder.encoder import (
    ConvEmbedding_Learning,
    ConvEmbedding_WithAdapter,
)


# --------------------------------------------------------------------------
# Original ProtoTS encoder (TSMixer-style). Preserved verbatim so we can fall
# back / compare. To revert the global default, change the alias at the
# bottom of this file from ConvEmbedding_Learning to MixerEmbedding_Learning.
# --------------------------------------------------------------------------

class MixerEmbedding_Learning(nn.Module):
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


# --------------------------------------------------------------------------
# Mode switch — flip ONE of these three lines.
# ADAPTER_MODE below is derived automatically from your choice.
# --------------------------------------------------------------------------
# Embedding_Learning = ConvEmbedding_Learning        # Mode A: no adapter (default)
# Embedding_Learning = ConvEmbedding_WithAdapter   # Mode B: matching_head adapter
Embedding_Learning = MixerEmbedding_Learning     # Mode C: original ProtoTS

# Derived adapter mode — do not edit; flip the alias above instead.
_ADAPTER_BY_ENCODER = {
    ConvEmbedding_Learning:    "none",
    ConvEmbedding_WithAdapter: "matching_head",
    MixerEmbedding_Learning:   "patch_embedding",
}
ADAPTER_MODE = _ADAPTER_BY_ENCODER[Embedding_Learning]

import torch
import torch.nn as nn
from layer.prototype import Prototype_Layer
from layer.align_block import align

from model.embedding import ADAPTER_MODE


class Prototype(nn.Module):
    def __init__(
        self, prototype_num, prototype_dim, input_length,
        output_dim, prototype_split_num, top_k_activate, split_ratio
    ):
        super(Prototype, self).__init__()

        self.prototype_dim = prototype_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.adapter_mode = ADAPTER_MODE

        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.prototype = Prototype_Layer(
            prototype_dim=self.prototype_dim,
            init_prototype_num=prototype_num,
            input_length=self.input_length,
            output_dim=self.output_dim,
            prototype_split_num=prototype_split_num,
            top_k_activate=top_k_activate,
            split_ratio=split_ratio,
            adapter_mode=self.adapter_mode,
        )

        self.align_layer = align()

    def forward(self, embd, x):
        # ``embd`` shape depends on ADAPTER_MODE (set in model/embedding.py):
        #   * "patch_embedding" (Mode C, Mixer encoder): (B, T, D).
        #     Original ProtoTS path — roll the encoder output by
        #     −time_of_day before matching (canonicalise to a midnight-
        #     aligned reference), then roll the per-layer predictions by
        #     +time_of_day after the prototype layer. ``prototype_value``
        #     is learned in the canonical frame; align_layer.backward
        #     un-canonicalises it.
        #   * "matching_head"   (Mode B, ConvEmbedding without time_pool):
        #     (B, T, D). Same align dance — the encoder still preserves
        #     the time axis, and prototype_value is learned in the
        #     canonical frame so the match key needs to be in that frame
        #     too.
        #   * "none"            (Mode A, ConvEmbedding with time_pool):
        #     (B, D). The encoder has already aggregated over time, the
        #     match key is time-invariant by construction, and
        #     prototype_value is learned in the actual time frame. Skip
        #     align_layer.
        use_align = (self.adapter_mode != "none")
        if use_align:
            embd = self.align_layer(x, embd, self.input_length)
        output, score = self.prototype(embd)
        if use_align:
            output = self.align_layer.backward(x, output, self.input_length)
        return output, score

    # Thin pass-throughs so the training loop can call
    # ``prototype_model.set_initial_prototypes(...)`` /
    # ``prototype_model.anchor_loss(...)`` without reaching into the
    # nested ``self.prototype`` (Prototype_Layer).
    def set_initial_prototypes(self, embedding, value, anchor_mask):
        self.prototype.set_initial_prototypes(embedding, value, anchor_mask)

    def anchor_loss(self, lambda_emb=1.0, lambda_val=1.0):
        return self.prototype.anchor_loss(lambda_emb, lambda_val)

import torch
import torch.nn as nn
from layer.prototype import Prototype_Layer
from layer.align_block import align

class Prototype(nn.Module):
    def __init__(
        self, prototype_num, prototype_dim, input_length, 
        output_dim, prototype_split_num, top_k_activate, split_ratio
    ):
        super(Prototype, self).__init__()

        self.prototype_dim = prototype_dim
        self.output_dim = output_dim
        self.input_length = input_length

        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.prototype = Prototype_Layer(
            prototype_dim=self.prototype_dim,
            init_prototype_num=prototype_num,                                                           
            input_length=self.input_length,
            output_dim=self.output_dim,
            prototype_split_num=prototype_split_num,
            top_k_activate=top_k_activate,
            split_ratio=split_ratio,
        )

        self.align_layer = align()

    def forward(self, embd, x):
        # ``embd`` may be (B, T, D) — Mixer-encoder path, original ProtoTS —
        # or (B, D) — ConvEmbedding-encoder path.
        #
        # For the Mixer path we reproduce original ProtoTS exactly: roll the
        # encoder output by −time_of_day before matching (canonicalise to
        # midnight-aligned reference), then roll the per-layer predictions
        # by +time_of_day after the prototype layer to recover the input's
        # actual time-of-day frame. ``prototype_value`` is learned in the
        # canonical frame; ``align_layer.backward`` un-canonicalises it.
        #
        # For the ConvEmbedding path neither direction is meaningful — the
        # encoder has already aggregated over the time axis, so the match
        # key is time-invariant by construction and ``prototype_value`` is
        # learned in the actual time frame. We skip align_layer in that case.
        use_align = (embd.dim() == 3)
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

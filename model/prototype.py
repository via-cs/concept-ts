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
        # ``embd`` may be (B, T, D) — Mixer-encoder path — or (B, D) —
        # ConvEmbedding-encoder path. Both shapes are passed through to
        # the prototype layer, which dispatches internally: 3D inputs go
        # through patch_embedding; 2D inputs are used as the match key
        # directly. The original align_layer (cyclic-shift canonicalisation)
        # is bypassed regardless of encoder type.
        output, score = self.prototype(embd)
        return output, score

    # Thin pass-throughs so the training loop can call
    # ``prototype_model.set_initial_prototypes(...)`` /
    # ``prototype_model.anchor_loss(...)`` without reaching into the
    # nested ``self.prototype`` (Prototype_Layer).
    def set_initial_prototypes(self, embedding, value, anchor_mask):
        self.prototype.set_initial_prototypes(embedding, value, anchor_mask)

    def anchor_loss(self, lambda_emb=1.0, lambda_val=1.0):
        return self.prototype.anchor_loss(lambda_emb, lambda_val)

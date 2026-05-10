"""
Convolution-based replacement for ProtoTS's MLP-Mixer encoder.

Public class: ``ConvEmbedding_Learning`` — a drop-in replacement for
``model.embedding.Embedding_Learning`` (same constructor signature, same
forward contract: returns ``(prediction, encoder)`` of shapes
``(N, output_length, output_dim)`` and ``(N, output_length, mid_dim)``,
with ``.embeding.denorm(x)`` exposed for the prototype-side denorm path).

Internals are split across three small files:
    channel_embedding.py   per-channel embedding, NO channel-axis mean
    conv_block.py          one ModernTCN-style temporal block
    encoder.py             assembles everything into ConvEmbedding_Learning
"""

from convolution_encoder.encoder import ConvEmbedding_Learning

__all__ = ["ConvEmbedding_Learning"]

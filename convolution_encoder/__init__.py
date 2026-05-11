"""
ModernTCN-style convolution encoder for ProtoTS.

Architecture (following Luo & Wang, ICLR 2024, "ModernTCN: A Modern Pure
Convolution Structure for General Time Series Analysis"):

  Stage 0  : Rev_in target normalization + sentinel masking
  Stage 1  : per-variable embedding (categorical → nn.Embedding, continuous →
             Linear). Output shape (B, T, M, D); channel axis preserved.
  Stage 2  : depthwise patchification — Conv1d(M*D, M*D, kernel=P, stride=S,
             groups=M*D). Each (variable, feature) pair gets its own
             temporal aggregation filter. Output (B, M, D, N) with N = T/S.
  Stage 3  : K ModernTCN blocks, each containing:
             * DWConv (groups=M*D): temporal mixing per (variable, feature)
             * ConvFFN1 (groups=M, inverted bottleneck × expand): feature
               mixing per variable
             * ConvFFN2 (groups=D, inverted bottleneck × expand): variable
               mixing per feature
             with reshape/permute between FFN1 and FFN2 to switch group axes.
  Stage 4  : decode (B, M, D, N) → (B, output_length, mid_dim)
  Stage 5  : prediction head Linear(mid_dim, output_dim)
  Stage 6  : Rev_in.de_rev_in to un-normalize the prediction

Public API: ConvEmbedding_Learning — drop-in replacement for
``model.embedding.Embedding_Learning``. Same constructor signature, same
forward contract.
"""

from convolution_encoder.encoder import ConvEmbedding_Learning

__all__ = ["ConvEmbedding_Learning"]

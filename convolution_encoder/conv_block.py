"""
ModernTCN-style temporal block.

One block, given input ``h`` of shape ``(B, D, T)`` (PyTorch's Conv1d
convention: batch × channels × length):

    residual = h
    h = LayerNorm(h)                  # along the D axis
    h = DepthwiseConv1d(D → D, kernel=K, padding=K//2, groups=D)(h)
    h = GELU(h)
    h = PointwiseConv1d(D → D*expand)(h)
    h = GELU(h)
    h = PointwiseConv1d(D*expand → D)(h)
    h = Dropout(h)
    return residual + h

The depthwise-then-pointwise factorisation (MobileNet / ConvNeXt) keeps
parameter count low while letting the temporal kernel grow large. The
shared-weight per-channel use happens at the outer level — the encoder
flattens the channel axis into the batch dim before calling this block.
"""

import torch
import torch.nn as nn


class ModernTCNBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 51,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd for symmetric 'same' padding, got {kernel_size}"
            )

        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,        # 'same' length output
            groups=dim,                       # depthwise: one filter per dim
        )
        self.pointwise_expand = nn.Conv1d(dim, dim * expand, kernel_size=1)
        self.pointwise_project = nn.Conv1d(dim * expand, dim, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, D, T)
        residual = h

        # LayerNorm wants D on the last axis
        h = h.transpose(1, 2)               # (B, T, D)
        h = self.norm(h)
        h = h.transpose(1, 2)               # (B, D, T)

        h = self.depthwise(h)               # (B, D, T)
        h = self.act(h)
        h = self.pointwise_expand(h)        # (B, D*expand, T)
        h = self.act(h)
        h = self.pointwise_project(h)       # (B, D, T)
        h = self.dropout(h)

        return residual + h

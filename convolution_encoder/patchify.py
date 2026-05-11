"""
Depthwise patchification stem (Stage 2 of the ModernTCN encoder).

Takes the per-channel embedded tensor (B, T, M, D) and produces a patched
representation (B, M, D, N) where N = ceil(T / stride) — each patch
summarising ``patch_size`` consecutive timesteps via a learned, channel-
specific filter.

Key design choice: the patchify conv is **depthwise** (groups = M*D), so
each (variable, feature) channel gets its own length-``patch_size`` filter
for aggregating consecutive timesteps. This preserves both the variable
axis and the feature axis through the patchification step — neither
channel mixing nor variable mixing happens here. Those happen later
inside the ModernTCN blocks (via ConvFFN1 and ConvFFN2 respectively).

Equivalent role in the original ModernTCN paper: their stem is
``Conv1d(1, D, kernel=P, stride=S)`` per variable, applied to a scalar
time series. Here, each variable has already been embedded to ``D`` dims
by the upstream ``PerChannelEmbedding`` (which is necessary because we
have categorical channels that can't be patchified as raw scalars), so
the patchify step compresses the already-embedded representation.
"""

import torch
import torch.nn as nn


class DepthwisePatchify(nn.Module):
    def __init__(self, M: int, D: int, patch_size: int, stride: int):
        super().__init__()
        self.M = M
        self.D = D
        self.patch_size = patch_size
        self.stride = stride

        # Pad so the conv produces ceil(T / stride) patches.
        # For our default (P=4, S=4, T=480) this is a no-op (N = 120 exactly).
        # For mismatched (P, S, T) it ensures we don't silently drop tail timesteps.
        self.padding_left = (patch_size - stride) // 2 if patch_size > stride else 0
        self.padding_right = self.padding_left

        self.patch_conv = nn.Conv1d(
            in_channels=M * D,
            out_channels=M * D,
            kernel_size=patch_size,
            stride=stride,
            groups=M * D,        # depthwise: one filter per (variable, feature)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, M, D) — output of PerChannelEmbedding
        returns : (B, M, D, N) where N = floor((T - patch_size) / stride) + 1
        """
        B, T, M, D = x.shape

        # Bring time to the last axis: (B, M, D, T) → (B, M*D, T)
        x = x.permute(0, 2, 3, 1).contiguous()           # (B, M, D, T)
        x = x.reshape(B, M * D, T)

        # Optional symmetric padding on the time axis
        if self.padding_left > 0 or self.padding_right > 0:
            x = nn.functional.pad(x, (self.padding_left, self.padding_right))

        x = self.patch_conv(x)                            # (B, M*D, N)
        N = x.shape[-1]

        x = x.reshape(B, M, D, N)
        return x

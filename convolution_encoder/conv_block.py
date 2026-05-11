"""
ModernTCN block — Stage 3 of the encoder.

Structure (Figure 2(d) of the paper):

    Input  Z ∈ R^{B × M × D × N}
    │
    ├── Reshape → (B, M*D, N)         # M groups of D channels, time axis = N
    │
    ├── DWConv (kernel=K, groups=M*D)  # temporal mixing per (variable, feature)
    ├── BatchNorm1d(M*D)
    │
    ├── ConvFFN1 (groups=M, expand=r)  # feature mixing per variable
    │     Conv1d(M*D → M*r*D, k=1, groups=M)
    │     GELU
    │     Conv1d(M*r*D → M*D, k=1, groups=M)
    │
    ├── Reshape & Permute → (B, D*M, N)  # D groups of M channels (variables in group)
    │
    ├── ConvFFN2 (groups=D, expand=r)  # variable mixing per feature
    │     Conv1d(D*M → D*r*M, k=1, groups=D)
    │     GELU
    │     Conv1d(D*r*M → D*M, k=1, groups=D)
    │
    ├── Reshape & Permute → (B, M, D, N)
    ├── Dropout
    └── + Z (residual)

Each of DWConv / ConvFFN1 / ConvFFN2 mixes information along exactly ONE
of {temporal, feature, variable} dimensions — the "decoupling design"
that gives ModernTCN its property of being deep, parameter-efficient, and
preserving all three structural axes.
"""

import torch
import torch.nn as nn


class ModernTCNBlock(nn.Module):
    def __init__(self, M: int, D: int, kernel_size: int = 51,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd for symmetric padding, got {kernel_size}"
            )

        self.M = M
        self.D = D

        # ---- DWConv: temporal mixing per (variable, feature) ----
        self.dwconv = nn.Conv1d(
            in_channels=M * D,
            out_channels=M * D,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=M * D,
        )
        self.bn = nn.BatchNorm1d(M * D)

        # ---- ConvFFN1: feature mixing per variable, with M groups ----
        # In each of the M groups, the conv sees D input channels and r*D output channels
        self.convffn1_expand = nn.Conv1d(
            in_channels=M * D,
            out_channels=M * D * expand,
            kernel_size=1,
            groups=M,
        )
        self.convffn1_project = nn.Conv1d(
            in_channels=M * D * expand,
            out_channels=M * D,
            kernel_size=1,
            groups=M,
        )

        # ---- ConvFFN2: variable mixing per feature, with D groups ----
        # After reshape&permute, each of the D groups sees M input channels and r*M output channels
        self.convffn2_expand = nn.Conv1d(
            in_channels=D * M,
            out_channels=D * M * expand,
            kernel_size=1,
            groups=D,
        )
        self.convffn2_project = nn.Conv1d(
            in_channels=D * M * expand,
            out_channels=D * M,
            kernel_size=1,
            groups=D,
        )

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (B, M, D, N) — patched embedded tensor
        returns : (B, M, D, N) — same shape, with all three mixings applied
        """
        B, M, D, N = z.shape
        assert M == self.M and D == self.D, (
            f"shape mismatch: got M={M}, D={D}; block was built with M={self.M}, D={self.D}"
        )
        residual = z

        # ---- Flatten channel axis: (B, M, D, N) → (B, M*D, N) ----
        # In this arrangement, channels are M groups of D consecutive channels.
        h = z.reshape(B, M * D, N)

        # ---- DWConv + BN ----
        h = self.dwconv(h)
        h = self.bn(h)

        # ---- ConvFFN1: per-variable feature mixing (groups=M) ----
        h = self.convffn1_expand(h)
        h = self.act(h)
        h = self.convffn1_project(h)

        # ---- Reshape & Permute: (B, M*D, N) → (B, D*M, N) ----
        # Switch grouping so we have D groups of M consecutive channels.
        h = h.reshape(B, M, D, N)
        h = h.permute(0, 2, 1, 3).contiguous()           # (B, D, M, N)
        h = h.reshape(B, D * M, N)

        # ---- ConvFFN2: per-feature variable mixing (groups=D) ----
        h = self.convffn2_expand(h)
        h = self.act(h)
        h = self.convffn2_project(h)

        # ---- Reshape & Permute back: (B, D*M, N) → (B, M, D, N) ----
        h = h.reshape(B, D, M, N)
        h = h.permute(0, 2, 1, 3).contiguous()           # (B, M, D, N)

        # ---- Dropout + residual ----
        h = self.dropout(h)
        return residual + h

"""
TSEncoder — the per-timestep TS2Vec encoder.

Pipeline:

    x ∈ (B, T, input_dims)
    │
    ├── Linear(input_dims, hidden_dims)        # input_fc — channel mixing
    │
    ├── random masking                         # zero-out some timesteps (training-time SSL augmentation)
    │
    ├── DilatedConvEncoder                     # depth × ConvBlocks, dilation grows 1, 2, 4, ...
    │
    └── repr_dropout                           # final dropout on the representation
        output ∈ (B, T, output_dims)

This is the unsupervised representation. To use it as a forecasting backbone
inside ProtoLLM, wrap it with a forecast head (see ``ts2vec_embedding.py``).
"""

import numpy as np
import torch
from torch import nn

from .dilated_conv import DilatedConvEncoder


def generate_continuous_mask(B, T, n=5, l=0.1):
    """Zero-out ``n`` random contiguous chunks of length ``l*T`` per sample."""
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t:t + l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    """Independently keep each timestep with probability ``p``."""
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class TSEncoder(nn.Module):
    """The TS2Vec backbone. Returns ``(B, T, output_dims)`` representations."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: int = 64,
        depth: int = 10,
        mask_mode: str = "binomial",
        repr_dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode

        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3,
        )
        self.repr_dropout = nn.Dropout(p=repr_dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """x : (B, T, input_dims) → (B, T, output_dims)."""
        # Standard TS2Vec NaN handling (we don't have NaNs in LOF, but keep
        # the path for shape-compatibility with the original).
        nan_mask = ~x.isnan().any(axis=-1)
        x = x.clone()
        x[~nan_mask] = 0
        x = self.input_fc(x)                              # (B, T, hidden_dims)

        # Decide which mask scheme to apply.
        if mask is None:
            mask = self.mask_mode if self.training else "all_true"

        if mask == "binomial":
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "continuous":
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "all_true":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == "all_false":
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == "mask_last":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask = mask & nan_mask
        x[~mask] = 0

        # Conv stack: (B, T, hidden_dims) → (B, hidden_dims, T) → (B, output_dims, T) → (B, T, output_dims)
        x = x.transpose(1, 2)
        x = self.repr_dropout(self.feature_extractor(x))
        x = x.transpose(1, 2)
        return x

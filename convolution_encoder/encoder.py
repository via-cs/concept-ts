"""
Drop-in replacement for ``model.embedding.Embedding_Learning``.

Architecture (matches the sketch from the brainstorm):

    Stage 0 + 1 : per-channel embedding (Rev_in target, sentinel forecast,
                   per-channel lookups / Linear)              → (N, T, in_dim, D_pc)
    Stage 2     : channel-independent ModernTCN temporal stack
                   (K blocks, shared weights across channels) → (N, T, in_dim, D_pc)
    Stage 3     : late cross-channel mixing (Option 3A:
                   flatten + 2-layer MLP)                      → (N, T, mid_dim)
    Stage 4     : time-axis projection (T → output_length)    → (N, output_length, mid_dim)
    Stage 5     : prediction head (Linear D → output_dim)     → (N, output_length, output_dim)
    Stage 6     : Rev_in.de_rev_in                             → (N, output_length, output_dim)

The constructor signature exactly matches ``Embedding_Learning`` so the rest
of the pipeline (embedding_learning.py, prototype_learning.py, etc.) needs
no changes. Two of the parameters — ``bottle_dim_p`` and ``norm_type`` —
were Mixer-specific knobs and are accepted for compatibility but ignored
by this conv-based encoder.
"""

import torch
import torch.nn as nn

from convolution_encoder.channel_embedding import PerChannelEmbedding
from convolution_encoder.conv_block import ModernTCNBlock


class ConvEmbedding_Learning(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mid_dim: int,
        ext_dim: int,
        output_dim: int,
        bottle_dim_p,           # ignored — kept for signature parity
        input_length: int,
        output_length: int,
        num_blocks: int,
        dropout: float,
        use_rev_in: int = 1,
        norm_type: str = "B",   # ignored — kept for signature parity
        # Conv-encoder specific knobs (defaults sized for the RE dataset)
        d_pc: int = 64,
        kernel_size: int = 51,
        expand: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.ext_dim = ext_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.output_length = output_length
        self.use_rev_in = use_rev_in
        self.d_pc = d_pc
        self.total_length = input_length + output_length

        # Stage 0+1: per-channel embedding (Rev_in inside)
        # Named ``embeding`` (typo preserved from ProtoTS) so downstream code
        # that calls ``model.embeding.denorm(...)`` keeps working unchanged.
        self.embeding = PerChannelEmbedding(
            input_dim=input_dim,
            ext_dim=ext_dim,
            embed_dim=d_pc,
            input_length=input_length,
            output_length=output_length,
        )

        # Stage 2: K ModernTCN blocks, shared weights across channels
        self.temporal_blocks = nn.ModuleList([
            ModernTCNBlock(
                dim=d_pc,
                kernel_size=kernel_size,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        # Stage 3: late cross-channel mixing (Option 3A)
        self.cross_channel_mix = nn.Sequential(
            nn.Linear(input_dim * d_pc, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, mid_dim),
        )

        # Stage 4: time-axis projection
        self.output_time_linear = nn.Linear(self.total_length, output_length)

        # Stage 5: prediction head
        self.output_feature_linear = nn.Linear(mid_dim, output_dim)

        self._init_linear_weights()

    def _init_linear_weights(self):
        """Match ProtoTS's per-Linear init: N(0, 1/sqrt(fan_in)), zero biases.
        Conv1d weights keep PyTorch's default Kaiming-uniform init."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / (fan_in ** 0.5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (N, total_length, input_dim) — the dataloader's combined window.

        Returns
        -------
        prediction : (N, output_length, output_dim) — denormed forecast
                     (in z-scored target space, same as ProtoTS).
        encoder    : (N, output_length, mid_dim) — features consumed by
                     the prototype layer.
        """
        N = x.size(0)

        # Stage 0+1: per-channel embedding, channel axis preserved
        h = self.embeding(x, self.use_rev_in)          # (N, T, in_dim, d_pc)
        T = h.size(1)
        in_dim = h.size(2)
        d_pc = h.size(3)

        # Stage 2: channel-independent temporal stack
        # (N, T, in_dim, d_pc) → (N*in_dim, d_pc, T)
        h = h.permute(0, 2, 3, 1).contiguous()         # (N, in_dim, d_pc, T)
        h = h.reshape(N * in_dim, d_pc, T)
        for block in self.temporal_blocks:
            h = block(h)                                # (N*in_dim, d_pc, T)
        # Reshape back, restore (N, T, in_dim, d_pc)
        h = h.reshape(N, in_dim, d_pc, T)
        h = h.permute(0, 3, 1, 2).contiguous()         # (N, T, in_dim, d_pc)

        # Stage 3: late cross-channel mixing (flatten → MLP)
        h = h.reshape(N, T, in_dim * d_pc)             # (N, T, in_dim*d_pc)
        h = self.cross_channel_mix(h)                   # (N, T, mid_dim)

        # Stage 4: time-axis projection (T → output_length)
        h = h.transpose(1, 2)                           # (N, mid_dim, T)
        h = self.output_time_linear(h)                  # (N, mid_dim, output_length)
        encoder = h.transpose(1, 2)                     # (N, output_length, mid_dim)

        # Stage 5: prediction head
        output = self.output_feature_linear(encoder)    # (N, output_length, output_dim)

        # Stage 6: undo Rev_in
        if self.use_rev_in:
            output = self.embeding.denorm(output)

        return output, encoder

"""
ConvEmbedding_Learning — drop-in replacement for the Mixer-based encoder,
following the ModernTCN design (Luo & Wang, ICLR 2024).

Pipeline:

    x ∈ (B, T, M)                              # raw input from LOF_Dataset
    │
    ├── Stage 0+1 : PerChannelEmbedding        # Rev_in + sentinel + per-variable embed
    │   output      (B, T, M, D)                # channel axis preserved
    │
    ├── Stage 2  : DepthwisePatchify           # length-P kernel, stride S, groups=M*D
    │   output     (B, M, D, N)                # N = T // S patches
    │
    ├── Stage 3  : K × ModernTCNBlock           # DWConv → ConvFFN1 → ConvFFN2 → +residual
    │   output     (B, M, D, N)
    │
    ├── Stage 4  : decode                       # (B, M, D, N) → (B, output_length, mid_dim)
    │
    ├── Stage 5  : Linear(mid_dim, output_dim)  # prediction head
    │
    └── Stage 6  : Rev_in.de_rev_in             # un-normalize → (B, output_length, output_dim)

Constructor signature matches ``Embedding_Learning`` exactly, so downstream
code (embedding_learning.py, prototype_learning.py, build_llm_init.py,
extract_top_k.py, etc.) needs zero changes. The Mixer-only constructor
args ``bottle_dim_p`` and ``norm_type`` are accepted and ignored.

ModernTCN-specific knobs (``d_pc``, ``patch_size``, ``patch_stride``,
``kernel_size``, ``expand``) have defaults sized for the RE dataset
(T=480, M=23). To tune, edit the defaults below or pass via kwargs.
"""

import torch
import torch.nn as nn

from convolution_encoder.channel_embedding import PerChannelEmbedding
from convolution_encoder.patchify import DepthwisePatchify
from convolution_encoder.conv_block import ModernTCNBlock


class ConvEmbedding_Learning(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mid_dim: int,
        ext_dim: int,
        output_dim: int,
        bottle_dim_p,                # ignored — Mixer-only knob
        input_length: int,
        output_length: int,
        num_blocks: int,
        dropout: float,
        use_rev_in: int = 1,
        norm_type: str = "B",        # ignored — Mixer-only knob
        # --- ModernTCN-specific knobs ---
        d_pc: int = 32,              # per-variable embedding dim D
        patch_size: int = 4,         # patch length P
        patch_stride: int = 4,       # patch stride S; T must be divisible by S
        kernel_size: int = 51,       # DWConv kernel size (odd)
        expand: int = 2,             # ConvFFN expand factor r
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
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.total_length = input_length + output_length

        # Number of patches after stem (assuming aligned T and S; pad in
        # DepthwisePatchify handles minor mismatch).
        self.n_patches = self.total_length // patch_stride
        if self.total_length % patch_stride != 0:
            # Padded version will be slightly larger; recompute conservatively.
            pad = (patch_size - patch_stride) if patch_size > patch_stride else 0
            self.n_patches = ((self.total_length + pad) - patch_size) // patch_stride + 1

        # ---- Stage 0+1: per-channel embedding ----
        # Named ``embeding`` (typo preserved from ProtoTS) so downstream code
        # that calls ``model.embeding.denorm(...)`` keeps working unchanged.
        self.embeding = PerChannelEmbedding(
            input_dim=input_dim,
            ext_dim=ext_dim,
            embed_dim=d_pc,
            input_length=input_length,
            output_length=output_length,
        )

        # ---- Stage 2: depthwise patchification ----
        self.patchify = DepthwisePatchify(
            M=input_dim,
            D=d_pc,
            patch_size=patch_size,
            stride=patch_stride,
        )

        # ---- Stage 3: K ModernTCN blocks ----
        self.blocks = nn.ModuleList([
            ModernTCNBlock(
                M=input_dim,
                D=d_pc,
                kernel_size=kernel_size,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        # ---- Stage 4: decode to (B, output_length, mid_dim) ----
        # First project N patches → output_length along the time axis,
        # then mix M*D channels → mid_dim per output timestep.
        self.decode_time = nn.Linear(self.n_patches, output_length)
        self.decode_feature = nn.Linear(input_dim * d_pc, mid_dim)

        # ---- Stage 5: prediction head ----
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
        x : (B, total_length, input_dim) — combined window from LOF_Dataset.

        Returns
        -------
        prediction : (B, output_length, output_dim) — denormed forecast
                     (z-scored target space, matching ProtoTS).
        encoder    : (B, output_length, mid_dim) — features consumed by the
                     prototype layer.
        """
        B = x.size(0)

        # Stage 0+1: per-channel embedding (channel axis preserved)
        z = self.embeding(x, self.use_rev_in)              # (B, T, M, D)

        # Stage 2: depthwise patchification
        z = self.patchify(z)                                # (B, M, D, N)

        # Stage 3: K ModernTCN blocks
        for block in self.blocks:
            z = block(z)                                    # (B, M, D, N)

        # Stage 4: decode
        _, M, D, N = z.shape
        z = z.reshape(B, M * D, N)                          # (B, M*D, N)
        z = self.decode_time(z)                             # (B, M*D, output_length)
        z = z.transpose(1, 2)                                # (B, output_length, M*D)
        encoder = self.decode_feature(z)                     # (B, output_length, mid_dim)

        # Stage 5: prediction head
        output = self.output_feature_linear(encoder)         # (B, output_length, output_dim)

        # Stage 6: undo Rev_in
        if self.use_rev_in:
            output = self.embeding.denorm(output)            # back to z-scored target space

        return output, encoder

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
    """
    Conv encoder shared by Mode A ("none") and Mode B ("matching_head").

    The only structural difference between the two modes is whether the
    encoder's own ``time_pool`` is attached at the end:

    - ``use_time_pool=True``  (Mode A, default) — the encoder applies a
      learnable 96→1 mean-equivalent Linear and returns ``(B, mid_dim)``.
      The Prototype_Layer takes this 2D match-key directly (no adapter).

    - ``use_time_pool=False`` (Mode B) — the encoder skips ``time_pool`` and
      returns the per-timestep tensor ``(B, output_length, mid_dim)``. The
      Prototype_Layer applies an identity-initialized ``matching_head``
      adapter (``Linear(mid_dim, mid_dim)`` with ``eye`` init) over the
      time-mean of this tensor to produce the match key. At step 0 the
      adapter is a no-op, so the LLM-init coordinate system is preserved;
      gradient descent is free to drift it.

    See ``ConvEmbedding_WithAdapter`` below for the Mode-B alias.
    """
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
        # --- adapter-mode knob ---
        use_time_pool: bool = True,  # False → Mode B, encoder returns (B, T, D)
        # --- target-handling knob (see PerChannelEmbedding.target_mode) ---
        # "split" (default) keeps the legacy lookback+forecast contract.
        # "revin_all" / "mask_all" let this class be used as a single-window
        # encoder over a pure lookback or pure forecast window (the two-encoder
        # CBM design).
        target_mode: str = "split",
    ):
        super().__init__()
        self.target_mode = target_mode
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.ext_dim = ext_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.output_length = output_length
        self.use_rev_in = use_rev_in
        self.use_time_pool = use_time_pool
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
            target_mode=target_mode,
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

        # ---- Stage 6 (Mode A only): learnable shared time pooling ----
        # Replaces the fixed ``mean(dim=1)`` aggregation with a learnable
        # Linear(output_length, 1) that gives the time axis a learned single
        # weighting vector, shared across the mid_dim channels. Following the
        # same simple-linear-projection pattern as ``decode_time`` (480 → 96)
        # and ``decode_feature`` (M·D → mid_dim).
        # Initialized to a constant 1/output_length so this layer is exactly
        # equivalent to the mean at training start — pretraining therefore
        # starts from the same coordinate system the previous mean-based
        # encoder produced, and gradient descent is free to drift away
        # from uniform weights wherever a non-uniform temporal weighting
        # helps the L1 prediction loss.
        #
        # In Mode B (use_time_pool=False) the encoder returns the per-
        # timestep ``encoder_full`` instead, and the downstream adapter
        # (``matching_head`` in Prototype_Layer) does the time aggregation.
        if self.use_time_pool:
            self.time_pool = nn.Linear(output_length, 1, bias=False)

        self._init_linear_weights()

        # Mean-equivalent init — applied AFTER _init_linear_weights so the
        # generic Linear-init pass over all Linear modules doesn't clobber it.
        if self.use_time_pool:
            nn.init.constant_(self.time_pool.weight, 1.0 / output_length)

        # ---- Locality prior for decode_time (Gaussian banded init) ----
        # decode_time projects N patches (covering the full input+output window)
        # to output_length timesteps. With random init, each output timestep
        # is a uniform mix of all N patches and has no preferred temporal
        # locality. That makes per-sub-window slicing downstream (in the
        # CBM's per-sub-window concept predictors) less effective because
        # encoder_full[:, k*T_sw:(k+1)*T_sw, :] then contains essentially
        # the same global signal at every position.
        #
        # We initialize decode_time with a row-stochastic Gaussian band:
        # row t (output timestep t) is centered on patch index t * (N/T_out)
        # with a small sigma, so at init the encoder is already temporally
        # localized. Gradient descent is free to drift this anywhere it wants;
        # the goal is just to provide a useful prior so locality doesn't have
        # to be discovered from scratch.
        self._banded_init_decode_time()

    def _banded_init_decode_time(self) -> None:
        T_out = self.decode_time.out_features      # output_length
        N     = self.decode_time.in_features        # n_patches
        ratio = N / T_out
        # Sigma is set so adjacent output timesteps' bands overlap by ~half
        # their width. A smaller sigma localizes more aggressively; a larger
        # one is closer to uniform mixing.
        sigma = max(2.0, ratio * 1.5)
        positions = torch.arange(N, dtype=torch.float).unsqueeze(0)         # (1, N)
        peaks     = torch.arange(T_out, dtype=torch.float).unsqueeze(1) * ratio  # (T, 1)
        W = torch.exp(-0.5 * ((positions - peaks) / sigma) ** 2)            # (T, N)
        W = W / W.sum(dim=1, keepdim=True)                                  # row-stochastic
        with torch.no_grad():
            self.decode_time.weight.copy_(W)
            self.decode_time.bias.zero_()

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
        encoder    : in Mode A (use_time_pool=True), (B, mid_dim) — the
                     time-pooled match key consumed directly by the
                     prototype layer.
                     In Mode B (use_time_pool=False), (B, output_length,
                     mid_dim) — the per-timestep tensor handed to the
                     prototype layer's matching_head adapter.
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
        encoder_full = self.decode_feature(z)                # (B, output_length, mid_dim)

        # Stage 5: prediction head — operates on the per-timestep features,
        # so the L1-loss pretraining objective and the (B, output_length, 1)
        # prediction contract are unchanged regardless of adapter mode.
        output = self.output_feature_linear(encoder_full)    # (B, output_length, output_dim)

        # Stage 6: undo Rev_in
        if self.use_rev_in:
            output = self.embeding.denorm(output)            # back to z-scored target space

        # Stage 7: match-key extraction.
        #   Mode A — apply the learnable 96→1 time_pool. At init the weights
        #            are constant 1/output_length, so this is equivalent to
        #            encoder_full.mean(dim=1). Returned shape: (B, mid_dim).
        #   Mode B — return the per-timestep encoder_full unchanged and let
        #            the Prototype_Layer's matching_head adapter handle
        #            time aggregation + coordinate alignment.
        if self.use_time_pool:
            encoder_t = encoder_full.transpose(1, 2)         # (B, mid_dim, output_length)
            encoder = self.time_pool(encoder_t).squeeze(-1)   # (B, mid_dim)
        else:
            encoder = encoder_full                            # (B, output_length, mid_dim)

        return output, encoder


# --------------------------------------------------------------------------
# Mode-B alias: conv encoder *without* the internal time_pool. Pairs with
# the matching_head adapter in Prototype_Layer (see ADAPTER_MODE resolution
# in model/embedding.py).
# --------------------------------------------------------------------------
class ConvEmbedding_WithAdapter(ConvEmbedding_Learning):
    """Conv encoder variant that omits ``time_pool`` and returns the
    per-timestep ``(B, output_length, mid_dim)`` tensor.

    Intended to be paired with ADAPTER_MODE='matching_head'. Constructor
    signature is identical to ConvEmbedding_Learning, so downstream
    instantiation in exp/prototype_learning.py does not change.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_time_pool", False)
        super().__init__(*args, **kwargs)

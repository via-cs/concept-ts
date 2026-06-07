"""
TS2VecEmbedding_Learning — drop-in replacement for the Mixer / Conv encoders.

Internally uses a TS2Vec dilated-conv backbone (``TSEncoder``) instead of
TSMixer / ModernTCN. Constructor signature and forward contract are
identical to ``MixerEmbedding_Learning``:

    forward(x) → (prediction, encoder)
        prediction : (B, output_length, output_dim) — denormalized forecast
        encoder    : (B, output_length, mid_dim)    — per-timestep features
                                                       consumed by the prototype
                                                       layer

So downstream code (embedding_learning.py / prototype_learning.py /
build_llm_init.py / experiments/*) needs zero changes to use this encoder.
Flip the alias in ``model/embedding.py`` to point at this class.

Three TS2Vec-specific knobs (``hidden_dims``, ``mask_mode``,
``contrastive_repr``) have defaults; they're constructor-level only and not
threaded through the existing CLI flags.

When ``contrastive_repr=True``, the forward path *also* returns the per-
timestep representation BEFORE the time-projection head, which is what
the hierarchical contrastive loss is computed on during self-supervised
pretraining. See ``train_ts2vec.py``.
"""

import torch
import torch.nn as nn

from layer.rev_norm import Rev_in
from .tsencoder import TSEncoder


class _RevInWrapper(nn.Module):
    """Exposes ``.denorm(x)`` so downstream code that calls
    ``model.embeding.denorm(...)`` (a pattern used by both Mixer and Conv
    encoders) keeps working unchanged."""

    def __init__(self, target_dim: int):
        super().__init__()
        self.rev_in = Rev_in(target_dim)

    def forward(self, x):
        return self.rev_in(x)

    def denorm(self, x):
        return self.rev_in.de_rev_in(x)


class TS2VecEmbedding_Learning(nn.Module):
    """TS2Vec-style encoder + standard forecast head, packaged behind the
    existing ProtoLLM Embedding_Learning interface."""

    def __init__(
        self,
        input_dim: int,
        mid_dim: int,
        ext_dim: int,
        output_dim: int,
        bottle_dim_p,               # ignored — Mixer-only knob
        input_length: int,
        output_length: int,
        num_blocks: int,            # → TSEncoder depth (= number of dilated ConvBlocks)
        dropout: float,             # → repr_dropout
        use_rev_in: int = 1,
        norm_type: str = "B",       # ignored
        # ---- TS2Vec-specific knobs ----
        hidden_dims: int = 64,      # internal channel width inside TSEncoder
        mask_mode: str = "binomial",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.ext_dim = ext_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.output_length = output_length
        self.use_rev_in = use_rev_in
        self.total_length = input_length + output_length

        # ---- Rev_in adapter on the target column (last channel) ----
        # Wrapped so that ``self.embeding.denorm(...)`` still works
        # downstream (matches the contract of Mixer / Conv encoders).
        self.embeding = _RevInWrapper(target_dim=input_dim - ext_dim)

        # ---- Core TS2Vec backbone ----
        # Takes (B, T_total, input_dim) → (B, T_total, mid_dim).
        self.tsencoder = TSEncoder(
            input_dims=input_dim,
            output_dims=mid_dim,
            hidden_dims=hidden_dims,
            depth=num_blocks,
            mask_mode=mask_mode,
            repr_dropout=dropout,
        )

        # ---- Forecast head ----
        # Same shape contract as MixerEmbedding_Learning:
        #   - output_time_linear : project T_input+T_output → T_output along time axis
        #   - output_feature_linear : project mid_dim → output_dim
        self.output_time_linear = nn.Linear(self.total_length, output_length)
        self.output_feature_linear = nn.Linear(mid_dim, output_dim)

    # ------------------------------------------------------------------
    # Forward — task-mode (forecasting) path used by embedding_learning,
    # prototype_learning, build_llm_init, and the experiments scripts.
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, mask=None):
        """x : (B, T_input+T_output, input_dim)

        Returns
        -------
        prediction : (B, output_length, output_dim)  — denormed if use_rev_in
        encoder    : (B, output_length, mid_dim)     — per-output-timestep features
        """
        # ---- Rev_in on the lookback portion of the target column ----
        # The forecast portion of the target column is overwritten with a
        # sentinel value (-5), matching ProtoTS's data contract so the
        # encoder doesn't "see" the answer at inference time.
        x = x.clone()
        if self.use_rev_in:
            target_lookback = x[:, : self.input_length, -1:].clone()  # (B, L_in, 1)
            target_normed = self.embeding(target_lookback)             # caches stats
            x[:, : self.input_length, -1:] = target_normed
            x[:, self.input_length :, -1:] = -5

        # ---- TS2Vec dilated-conv encoder ----
        z = self.tsencoder(x, mask=mask)            # (B, T_total, mid_dim)

        # ---- Forecast head: time projection then channel projection ----
        encoder = self.output_time_linear(z.transpose(1, 2)).transpose(1, 2)
        # encoder : (B, output_length, mid_dim)
        output = self.output_feature_linear(encoder)
        # output  : (B, output_length, output_dim)

        if self.use_rev_in:
            output = self.embeding.denorm(output)

        return output, encoder

    # ------------------------------------------------------------------
    # Pretraining-only path — returns the raw per-timestep representation
    # BEFORE the forecast head, which is what the hierarchical contrastive
    # loss operates on. Called by train_ts2vec.py.
    # ------------------------------------------------------------------

    def encode_repr(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """x : (B, T, input_dim) → (B, T, mid_dim).

        No Rev_in, no forecast head — pure representation. Used during
        self-supervised pretraining where the two augmented views of the
        same sample are compared via contrastive loss.
        """
        return self.tsencoder(x, mask=mask)

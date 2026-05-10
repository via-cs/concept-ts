"""
Per-channel embedding with channel axis preserved.

Differences from ``layer/channel_embedding.py``:

  - Output is (N, T, in_dim, embed_dim) — we do NOT collapse the channel axis
    via ``.mean(dim=-2)``. Channel identity is preserved into the temporal
    stack, so each channel can carry its own learned representation through
    the rest of the encoder.

  - Continuous channels use a single ``Linear(1, embed_dim)`` instead of a
    2-layer MLP. The temporal convolution stack downstream adds nonlinearity,
    so a richer per-value embedding is redundant.

  - Categorical channels keep the same ``nn.Embedding`` lookups as ProtoTS
    (cyclic / discrete features need them).

  - The Rev_in normalisation on the lookback target and the −5 sentinel
    masking on the forecast target are kept verbatim — those are part of
    the data contract that downstream code (Rev_in.denorm, prototype layer)
    expects.
"""

import os

import torch
import torch.nn as nn

from layer.rev_norm import Rev_in


# Same hard-coded categorical setup as ProtoTS (kept for compatibility with
# the existing dataset's column ordering).
_CLASS_FEATURE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20]
_CLASS_FEATURE_NUM_CLASSES = [
    6, 13, 32, 6, 13,
    32, 25, 61, 0, 0,
    0, 0, 0, 0, 0,
    0, 2, 2, 2, 2,
    7, 0, 0,
]


class PerChannelEmbedding(nn.Module):
    """Embed each of the ``input_dim`` channels into ``embed_dim`` features.

    Output preserves the channel axis: (N, T, input_dim, embed_dim).

    Args
    ----
    input_dim       : total channels in the raw input (target + exogenous)
    ext_dim         : count of exogenous channels — target channels are
                      assumed to occupy indices [ext_dim, input_dim).
    embed_dim       : per-channel embedding dimension (D_pc in the design doc)
    input_length    : lookback window length (used for Rev_in slicing)
    output_length   : forecast window length (used for sentinel masking)
    """

    def __init__(self, input_dim, ext_dim, embed_dim,
                 input_length, output_length):
        super().__init__()
        self.input_dim = input_dim
        self.ext_dim = ext_dim
        self.embed_dim = embed_dim
        self.input_length = input_length
        self.output_length = output_length

        self.input_time_index = list(range(input_length))
        self.target_feature_index = list(range(ext_dim, input_dim))
        self.class_feature = _CLASS_FEATURE_INDICES
        self.class_feature_map = _CLASS_FEATURE_NUM_CLASSES

        # Per-channel embedding modules
        self.embeding = nn.ModuleList()
        for i in range(input_dim):
            if i in self.class_feature:
                self.embeding.append(
                    nn.Embedding(self.class_feature_map[i], embed_dim)
                )
            else:
                # Single Linear instead of 2-layer MLP — the conv stack
                # downstream adds the nonlinearity.
                self.embeding.append(nn.Linear(1, embed_dim))

        self.embeding.apply(self._init_weight)

        # Rev_in over the target-channel slice (typically input_dim - ext_dim = 1)
        self.rev_in = Rev_in(input_dim - ext_dim)

    @staticmethod
    def _init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0,
                             std=1.0 / (m.weight.shape[1] ** 0.5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, use_rev_in: int = 1):
        """
        Parameters
        ----------
        x : (N, T, input_dim) float tensor — the dataloader's combined window.
        use_rev_in : kept for signature parity; Rev_in is always applied so
                     the cached (x_mean, x_std) is available for ``denorm``
                     downstream. Pass anything; the value is unused.

        Returns
        -------
        embedding : (N, T, input_dim, embed_dim) — channel axis preserved.
        """
        device = os.environ.get("device", x.device)
        N, T = x.size(0), x.size(1)

        # ---- Stage 0: Rev_in on lookback target + sentinel on forecast target ----
        input_feature = torch.index_select(
            x, 1, torch.tensor(self.input_time_index, device=device)
        )
        target_feature = torch.index_select(
            input_feature, 2, torch.tensor(self.target_feature_index, device=device)
        )
        target_feature = self.rev_in(target_feature)

        x[:, : self.input_length, -1] = target_feature.reshape(N, self.input_length)
        x[:, self.input_length :, -1] = -5

        # ---- Stage 1: per-channel embedding (NO mean reduction) ----
        embedding_outputs = []
        for i in range(self.input_dim):
            now_feature = torch.index_select(
                x, 2, torch.tensor([i], device=device)
            )                                              # (N, T, 1)
            if i in self.class_feature:
                now_feature = now_feature.long()
            emb_i = self.embeding[i](now_feature)          # (N, T, 1, embed_dim)
            embedding_outputs.append(emb_i.reshape(N, T, 1, -1))

        embedding = torch.cat(embedding_outputs, dim=-2)   # (N, T, input_dim, embed_dim)
        return embedding

    def denorm(self, x):
        """Undo the per-instance Rev_in normalisation cached during forward.
        Used by prototype_learning's eval / training loop to bring prototype-
        mixture predictions back to z-scored target space."""
        return self.rev_in.de_rev_in(x)

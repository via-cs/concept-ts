"""
Per-variable channel embedding (Stage 1 of the ModernTCN encoder).

Lifts each (variable, timestep) scalar / integer to a D-dim vector while
preserving the variable axis. Output shape: (B, T, M, D).

Continuous channels go through ``Linear(1, D)`` — a single layer; the conv
stack downstream adds nonlinearity. Categorical channels go through
``nn.Embedding(num_classes, D)`` with the same hard-coded class counts as
ProtoTS (the dataset's calendar / holiday / weekday columns).

Rev_in normalization is applied to the target column's lookback portion,
and the forecast portion of the target is overwritten with a -5 sentinel
(the standard ProtoTS data contract).
"""

import os

import torch
import torch.nn as nn

from layer.rev_norm import Rev_in


# ---------------------------------------------------------------------------
# Dataset switch — flip to "AirQuality" to point the per-channel embedder at
# the Riverside PM2.5 schema. Keep this in sync with ``DATASET`` in
# ``dataloader/dataloader.py`` and ``prototype_proposing_loop/prompts.py``.
#
# Each entry of ``_CLASS_FEATURE_NUM_CLASSES`` is the number of categories
# used by ``nn.Embedding(num_classes, embed_dim)`` for that column index.
# Continuous columns get 0 (the Linear(1, embed_dim) path runs instead).
# ---------------------------------------------------------------------------
# Categorical-column layout comes from the single source of truth
# (dataset_config), selected by PROTO_DATASET (default "PRSA" → unchanged).
from dataset_config import D as _DSCFG

DATASET                    = _DSCFG["name"]
_CLASS_FEATURE_INDICES     = _DSCFG["class_feature_indices"]
_CLASS_FEATURE_NUM_CLASSES = _DSCFG["class_feature_num_classes"]


class PerChannelEmbedding(nn.Module):
    """Lift each channel scalar to a D-dim vector, preserve channel axis.

    Args
    ----
    input_dim     : total channels (target + exogenous)
    ext_dim       : count of exogenous channels (target indices = [ext_dim, input_dim))
    embed_dim     : per-channel embedding dimension D
    input_length  : lookback length, used for Rev_in slicing
    output_length : forecast length, used for sentinel mask

    Forward
    -------
    x : (B, T, input_dim) → returns (B, T, input_dim, embed_dim)
    """

    def __init__(self, input_dim, ext_dim, embed_dim,
                 input_length, output_length, target_mode: str = "split"):
        super().__init__()
        # target_mode controls how the TARGET column is normalized / masked:
        #   "split"     (default, legacy): RevIN the lookback slice
        #               x[:, :input_length, target] and overwrite the forecast
        #               slice x[:, input_length:, target] with the -5 sentinel.
        #               Used by the single full-window encoder.
        #   "revin_all" : the WHOLE window is lookback -> RevIN the entire
        #               target over all T steps, no sentinel. Used by the
        #               two-encoder design's LOOKBACK encoder.
        #   "mask_all"  : the WHOLE window is forecast -> overwrite the entire
        #               target with the -5 sentinel, no RevIN. Used by the
        #               two-encoder design's FORECAST encoder (the target is
        #               unknown at forecast time). Pair with use_rev_in=0 so
        #               the encoder's denorm path (which needs RevIN stats) is
        #               skipped.
        if target_mode not in ("split", "revin_all", "mask_all"):
            raise ValueError(
                f"Unknown target_mode={target_mode!r}. Expected "
                f"'split' | 'revin_all' | 'mask_all'."
            )
        self.target_mode = target_mode
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
                self.embeding.append(nn.Linear(1, embed_dim))

        self.embeding.apply(self._init_weight)

        # Rev_in over the target slice (typically 1 channel)
        self.rev_in = Rev_in(input_dim - ext_dim)

    @staticmethod
    def _init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0,
                             std=1.0 / (m.weight.shape[1] ** 0.5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, use_rev_in: int = 1):
        device = os.environ.get("device", x.device)
        N, T = x.size(0), x.size(1)

        # Rev_in + sentinel on the target column (behavior depends on target_mode)
        tgt_idx = torch.tensor(self.target_feature_index, device=device)
        if self.target_mode == "split":
            input_feature = torch.index_select(
                x, 1, torch.tensor(self.input_time_index, device=device)
            )
            target_feature = torch.index_select(input_feature, 2, tgt_idx)
            target_feature = self.rev_in(target_feature)
            x[:, : self.input_length, -1] = target_feature.reshape(N, self.input_length)
            x[:, self.input_length :, -1] = -5
        elif self.target_mode == "revin_all":
            # Whole window is lookback: RevIN the entire target, no sentinel.
            target_feature = torch.index_select(x, 2, tgt_idx)      # (N, T, 1)
            target_feature = self.rev_in(target_feature)
            x[:, :, -1] = target_feature.reshape(N, T)
        else:  # "mask_all"
            # Whole window is forecast: mask the entire target, no RevIN.
            x[:, :, -1] = -5

        # Per-channel embedding — channel axis preserved
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
        """Undo the Rev_in normalization cached during forward."""
        return self.rev_in.de_rev_in(x)

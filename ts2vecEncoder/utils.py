"""
Helper utilities for TS2Vec-style training.

Only ``take_per_row`` is used by the training script — it implements the
"sample two overlapping crops per sequence" augmentation from the original
TS2Vec paper.
"""

import numpy as np
import torch


def take_per_row(A: torch.Tensor, indx: np.ndarray, num_elem: int) -> torch.Tensor:
    """Gather ``num_elem`` consecutive timesteps starting at ``indx[i]`` for
    each row ``i`` of ``A`` (shape ``(B, T, C)``).

    Used during training to produce two overlapping crops of each sequence
    in the batch. Each row gets its own starting index, so the crops are
    sample-specific. Returns shape ``(B, num_elem, C)``.
    """
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]

"""
TS2Vec hierarchical contrastive loss — verbatim from the TS2Vec repo
(models/losses.py).

Two complementary contrastive objectives:

  * instance_contrastive_loss
      For each timestep t, the two views z1[:, t] and z2[:, t] should be
      identifiable as the SAME sample against (2B-1) other samples in the
      batch. Pushes embeddings of different samples apart.

  * temporal_contrastive_loss
      For each sample b, the two views z1[b] and z2[b] are concatenated
      along the time axis and the (t, t)-th pair must be identifiable as
      the same timestep against (2T-1) other timesteps. Pushes embeddings
      of different timesteps apart within the same sample.

  * hierarchical_contrastive_loss
      Applies both losses, then max-pools the representation along the
      time axis (kernel=2) and applies them again. Repeats until T=1. This
      captures the multi-scale structure of time series.

Identical to the TS2Vec implementation; tested in the original paper across
universal representation tasks.
"""

import torch
from torch import nn
import torch.nn.functional as F


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0.0, device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss = loss + alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss = loss + (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss = loss + alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=0)        # (2B, T, C)
    z = z.transpose(0, 1)                  # (T, 2B, C)
    sim = torch.matmul(z, z.transpose(1, 2))   # (T, 2B, 2B)
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=1)        # (B, 2T, C)
    sim = torch.matmul(z, z.transpose(1, 2))   # (B, 2T, 2T)
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

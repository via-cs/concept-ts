"""
Self-supervised TS2Vec pretraining for ProtoLLM.

Trains the TS2Vec backbone inside ``TS2VecEmbedding_Learning`` using the
hierarchical contrastive loss only — no forecasting label is consulted.
After this runs, the encoder produces representations where semantically
similar segments cluster together by construction, without needing LLM-
derived cluster targets.

Workflow:
  1. (this script)    train_ts2vec.py   → produces a contrastive-pretrained encoder
  2. (existing code)  embedding_learning.py
                      Load the contrastive checkpoint, fine-tune the
                      forecast head with L1 loss. (Optional — can skip if
                      you want to use the encoder as-is.)
  3. (existing code)  build_llm_init.py → compute LLM-init μ from the new encoder
  4. (existing code)  prototype_learning.py → train prototype layer

The two-crop augmentation follows the original TS2Vec paper exactly:
  - Pick two overlapping sub-windows from each batch sample.
  - Encode each independently with binomial random masking.
  - Compare them on their overlap region via hierarchical contrastive loss.

Command (LOF / RE example):

    PYTHONPATH=. python -u ts2vecEncoder/train_ts2vec.py \
        --dataset ./dataset/LOF/RE.csv --epoch 30 \
        --batch_size 32 --input_length 384 --output_length 96 \
        --input_dim 23 --output_dim 1 \
        --num_layer 10 --dropout 0.1 --embed_dim 128 \
        --hidden_dims 64

(num_layer here drives the depth of the TS2Vec dilated-conv stack — the
original paper uses depth=10. dropout maps to repr_dropout, default 0.1.)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make the project root importable when run as `python ts2vecEncoder/train_ts2vec.py`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataloader.dataloader import LOF_Dataset                  # noqa: E402
from ts2vecEncoder.ts2vec_embedding import TS2VecEmbedding_Learning  # noqa: E402
from ts2vecEncoder.losses import hierarchical_contrastive_loss      # noqa: E402
from ts2vecEncoder.utils import take_per_row                        # noqa: E402


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",        type=str, required=True)
    p.add_argument("--epoch",          type=int, required=True)
    p.add_argument("--batch_size",     type=int, default=32)
    p.add_argument("--input_length",   type=int, default=384)
    p.add_argument("--output_length",  type=int, default=96)
    p.add_argument("--input_dim",      type=int, default=23)
    p.add_argument("--output_dim",     type=int, default=1)

    # TS2Vec backbone hyperparameters
    p.add_argument("--num_layer",     type=int,   default=10,
                   help="Depth of the dilated-conv stack (TS2Vec paper default = 10).")
    p.add_argument("--dropout",       type=float, default=0.1,
                   help="repr_dropout on the final representation (TS2Vec default = 0.1).")
    p.add_argument("--embed_dim",     type=int,   default=128,
                   help="Output representation dimension (mid_dim).")
    p.add_argument("--hidden_dims",   type=int,   default=64,
                   help="Internal channel width inside TSEncoder.")
    p.add_argument("--bottle_dim_p",  type=float, default=0.1,
                   help="Ignored — only present so the constructor signature matches "
                        "the other ProtoLLM encoders.")

    # Contrastive loss hyperparameters
    p.add_argument("--alpha",         type=float, default=0.5,
                   help="Weight between instance and temporal contrastive loss.")
    p.add_argument("--temporal_unit", type=int,   default=0,
                   help="Minimum hierarchy level to apply temporal contrast.")

    # Training hyperparameters
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--checkpoint_dir", type=str,  default="./checkpoint/ts2vec")
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def make_two_crops(x: torch.Tensor, temporal_unit: int = 0):
    """Pick two overlapping sub-windows from each row of ``x``.

    Reproduces the exact crop-overlap scheme used in the original TS2Vec
    train.py. Both crops share a sub-region of length ``crop_l`` and the
    contrastive loss is computed only on that overlap.

    Returns (left_crop, right_crop, crop_l) where the two crops are
    sliced to (B, crop_l, C) on the overlap region after encoding.

    Specifically:
        out1 = encoder(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
        out1 = out1[:, -crop_l:]                 # right edge of left crop
        out2 = encoder(take_per_row(x, crop_offset + crop_left,  crop_eright - crop_left))
        out2 = out2[:, :crop_l]                  # left edge of right crop
    """
    ts_l = x.size(1)
    crop_l    = np.random.randint(low=2 ** (temporal_unit + 1), high=ts_l + 1)
    crop_left  = np.random.randint(ts_l - crop_l + 1)
    crop_right = crop_left + crop_l
    crop_eleft  = np.random.randint(crop_left + 1)
    crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
    crop_offset = np.random.randint(low=-crop_eleft,
                                    high=ts_l - crop_eright + 1, size=x.size(0))

    left_crop_input  = take_per_row(x, crop_offset + crop_eleft,
                                       crop_right - crop_eleft)
    right_crop_input = take_per_row(x, crop_offset + crop_left,
                                        crop_eright - crop_left)
    return left_crop_input, right_crop_input, crop_l


def train_loop(model, dataset, batch_size, n_epochs, lr,
               temporal_unit, alpha, device, checkpoint_path):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loss_log = []
    model.train()
    for epoch in range(n_epochs):
        cum_loss = 0.0
        n_iters = 0
        for x_full, _ in loader:
            # x_full : (B, input_length + output_length, input_dim)
            x_full = x_full.to(device)

            left_in, right_in, crop_l = make_two_crops(x_full,
                                                        temporal_unit=temporal_unit)

            # Encode each crop independently with binomial masking
            # (the encoder is in train() mode, mask_mode='binomial' by default).
            z1 = model.encode_repr(left_in)         # (B, len1, mid_dim)
            z2 = model.encode_repr(right_in)        # (B, len2, mid_dim)

            # Trim to the overlap region of length crop_l.
            z1 = z1[:, -crop_l:]
            z2 = z2[:, :crop_l]

            loss = hierarchical_contrastive_loss(
                z1, z2,
                alpha=alpha,
                temporal_unit=temporal_unit,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()
            n_iters += 1

        epoch_loss = cum_loss / max(1, n_iters)
        loss_log.append(epoch_loss)
        print(f"[Epoch {epoch + 1}/{n_epochs}]  contrastive loss = {epoch_loss:.4f}")

        # Always save the latest — TS2Vec doesn't have a val signal during
        # pretraining (it's pure SSL), so we keep the most recent weights.
        torch.save(model.state_dict(), checkpoint_path)

    print(f"Final TS2Vec checkpoint saved to {checkpoint_path}")
    return loss_log


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["device"] = device
    print(f"Using device: {device}")

    set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    fname = (
        f"{os.path.splitext(os.path.basename(args.dataset))[0]}_"
        f"OL{args.output_length}_"
        f"IL{args.input_length}_"
        f"L{args.num_layer}_"
        f"D{args.dropout}_"
        f"EMB{args.embed_dim}_"
        f"H{args.hidden_dims}_"
        f"ts2vec.pth"
    )
    checkpoint_path = os.path.join(args.checkpoint_dir, fname)
    print(f"Will save checkpoint to {checkpoint_path}")

    # Build the same wrapper class that's used downstream by the existing
    # ProtoLLM scripts. Pretraining uses encode_repr; downstream uses
    # forward(x). The state_dict is identical, so the checkpoint is
    # consumable by embedding_learning.py and beyond.
    model = TS2VecEmbedding_Learning(
        input_dim=args.input_dim,
        mid_dim=args.embed_dim,
        ext_dim=args.input_dim - args.output_dim,
        output_dim=args.output_dim,
        bottle_dim_p=args.bottle_dim_p,
        input_length=args.input_length,
        output_length=args.output_length,
        num_blocks=args.num_layer,
        dropout=args.dropout,
        norm_type="B",
        use_rev_in=1,
        hidden_dims=args.hidden_dims,
        mask_mode="binomial",
    ).to(device)

    # SSL pretraining uses train data only — no val / test signal needed.
    train_dataset = LOF_Dataset(
        args.input_length, args.output_length,
        "train", args.input_dim, args.dataset,
    )
    print(f"Loaded {len(train_dataset)} training windows from {args.dataset}")

    train_loop(
        model=model,
        dataset=train_dataset,
        batch_size=args.batch_size,
        n_epochs=args.epoch,
        lr=args.lr,
        temporal_unit=args.temporal_unit,
        alpha=args.alpha,
        device=device,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()

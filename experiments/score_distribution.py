"""
score_distribution.py — layer-0 (parent) prototype activation scores over the
full test set.

Diagnostic for prototype collapse: if every test segment routes to the same
small subset of prototypes with near-uniform softmax weight, the model is not
doing discriminative routing — regardless of how good the MAE looks.

Compatible with whatever encoder is currently active in ``model/embedding.py``:

  * Mode A — ConvEmbedding with internal time_pool   (returns (B, D))
  * Mode B — ConvEmbedding without time_pool         (returns (B, T, D))
  * Mode C — MixerEmbedding (original ProtoTS)       (returns (B, T, D))

The forward path in ``_common.forward_capture`` dispatches on ``adapter_mode``
so no encoder-specific code lives here.

Outputs
-------
  * Console: aggregate stats — mean top-k score, mean entropy, prototype
    usage histogram, # unique top-1 prototypes.
  * CSV ``score_distribution.csv``: one row per test segment
    (top-1/2/3 score, top-1 prototype index, entropy, normalised entropy).
  * PNG ``score_distribution.png``: two panels —
       (a) per-prototype top-1 usage count (how often each is the winner)
       (b) per-prototype mean activation score across all segments
    Anchored (LLM-named) prototypes are coloured separately from blanks
    when an ``--anchor_mask`` is supplied.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataloader.dataloader import LOF_Dataset            # noqa: E402
from experiments._common import (                         # noqa: E402
    forward_capture,
    load_encoder_and_prototype,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, required=True,
                   help="Path to the CSV the model was trained on.")
    p.add_argument("--pretrain_checkpoint",  type=str, required=True,
                   help="Trained encoder .pth (saved by exp/prototype_learning.py).")
    p.add_argument("--prototype_checkpoint", type=str, required=True,
                   help="Trained prototype .pth (saved by exp/prototype_learning.py).")
    p.add_argument("--anchor_mask", type=str, default=None,
                   help="Optional anchor_mask.pt — used only to colour blank "
                        "vs anchored prototypes in the PNG. Skip for original "
                        "ProtoTS (no LLM init).")
    p.add_argument("--final_prototypes", type=str, default=None,
                   help="Optional final_prototypes JSON for prototype names "
                        "on x-axis ticks. Skip for original ProtoTS.")
    p.add_argument("--output_dir", type=str, default="./experiments/output",
                   help="Where to write the CSV + PNG.")
    p.add_argument("--tag", type=str, default="",
                   help="Optional suffix on output filenames so multiple "
                        "encoder/checkpoint comparisons don't overwrite each "
                        "other (e.g. --tag protoTS / --tag protoLLM).")

    # Shape knobs — must match training.
    p.add_argument("--input_length",  type=int, default=384)
    p.add_argument("--output_length", type=int, default=96)
    p.add_argument("--input_dim",     type=int, default=23)
    p.add_argument("--output_dim",    type=int, default=1)
    p.add_argument("--embed_dim",     type=int, default=128)
    p.add_argument("--num_layer",     type=int, default=4)
    p.add_argument("--dropout",       type=float, default=0.5)
    p.add_argument("--bottle_dim_p",  type=float, default=0.1)
    p.add_argument("--prototype_num",       type=int, default=8)
    p.add_argument("--prototype_split_num", type=int, default=2)
    p.add_argument("--top_k_activate",      type=int, default=2)
    p.add_argument("--split_ratio",         type=float, default=0.4)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--top_k_report", type=int, default=3,
                   help="How many top scores to record per segment in the CSV.")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    # ------------------------------------------------------------------
    # Load model (encoder dispatch is handled inside _common via the
    # ADAPTER_MODE derived from the alias in model/embedding.py).
    # ------------------------------------------------------------------
    encoder, prototype_model = load_encoder_and_prototype(
        pretrain_checkpoint=args.pretrain_checkpoint,
        prototype_checkpoint=args.prototype_checkpoint,
        prototype_num=args.prototype_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        embed_dim=args.embed_dim,
        input_length=args.input_length,
        output_length=args.output_length,
        num_layer=args.num_layer,
        dropout=args.dropout,
        bottle_dim_p=args.bottle_dim_p,
        prototype_split_num=args.prototype_split_num,
        top_k_activate=args.top_k_activate,
        split_ratio=args.split_ratio,
        device=device,
    )
    P = prototype_model.prototype.prototype_embedding[0].shape[0]
    print(f"adapter_mode = {prototype_model.adapter_mode!r},  layer-0 prototypes = {P}")

    # ------------------------------------------------------------------
    # Optional anchor mask + names (cosmetic — for the PNG only).
    # ------------------------------------------------------------------
    mask = None
    if args.anchor_mask and os.path.exists(args.anchor_mask):
        mask = torch.load(args.anchor_mask, map_location="cpu").float()
        if mask.shape[0] != P:
            print(f"  [warn] anchor_mask has {mask.shape[0]} entries but "
                  f"layer 0 has {P} prototypes — ignoring mask.")
            mask = None

    names = [f"proto_{i}" for i in range(P)]
    if args.final_prototypes and os.path.exists(args.final_prototypes):
        try:
            from experiments._common import load_llm_prototypes
            llm = load_llm_prototypes(args.final_prototypes)
            for spec in llm[:P]:
                names[spec["anchor_idx"]] = spec["name"]
        except Exception as e:
            print(f"  [warn] could not parse final_prototypes for names: {e}")

    # ------------------------------------------------------------------
    # Test dataset — feed every window through the model
    # ------------------------------------------------------------------
    test_dataset = LOF_Dataset(
        args.input_length, args.output_length,
        "test", args.input_dim, args.dataset,
    )
    N = len(test_dataset)
    print(f"test dataset windows = {N}")

    window_indices = list(range(N))
    cap = forward_capture(
        encoder, prototype_model, test_dataset,
        window_indices, device, batch_size=args.batch_size,
    )
    scores = cap["layer0_score"]   # (N, P), softmax-normalised
    assert scores.shape == (N, P), f"unexpected scores shape {tuple(scores.shape)}"

    # Row-sum sanity check — should be ~1.0 per segment
    row_sums = scores.sum(dim=1)
    print(f"sanity: row sums  min={row_sums.min():.4f}  max={row_sums.max():.4f}  "
          f"(should be ≈ 1.0)")

    # ------------------------------------------------------------------
    # Per-segment stats — top-k, entropy
    # ------------------------------------------------------------------
    K = max(1, min(args.top_k_report, P))
    top_vals, top_idx = torch.topk(scores, K, dim=1)        # (N, K)

    # Shannon entropy in nats
    eps = 1e-8
    entropy = -(scores * (scores + eps).log()).sum(dim=1)  # (N,)
    max_entropy = float(torch.log(torch.tensor(float(P))))  # log(P)
    norm_entropy = entropy / max_entropy                    # 1.0 = uniform, 0 = one-hot

    # ------------------------------------------------------------------
    # Aggregate stats — printed and saved to console
    # ------------------------------------------------------------------
    uniform_baseline = 1.0 / P
    print()
    print("──────────  AGGREGATE SCORE DISTRIBUTION  ──────────")
    print(f"  uniform baseline (1/P)           = {uniform_baseline:.4f}")
    print(f"  mean top-1 score                 = {top_vals[:, 0].mean():.4f}    "
          f"(uniform = {uniform_baseline:.4f}, one-hot = 1.0)")
    if K >= 2:
        print(f"  mean top-2 score                 = {top_vals[:, 1].mean():.4f}")
    if K >= 3:
        print(f"  mean top-3 score                 = {top_vals[:, 2].mean():.4f}")
    print(f"  mean Shannon entropy             = {entropy.mean():.4f}   "
          f"(uniform = log P = {max_entropy:.4f})")
    print(f"  mean normalised entropy          = {norm_entropy.mean():.4f}   "
          f"(1.0 = uniform, 0.0 = one-hot)")
    print(f"  median top-1 score               = {top_vals[:, 0].median():.4f}")
    print(f"  min   top-1 score                = {top_vals[:, 0].min():.4f}")
    print(f"  max   top-1 score                = {top_vals[:, 0].max():.4f}")

    # ------------------------------------------------------------------
    # Top-1 prototype usage histogram
    # ------------------------------------------------------------------
    top1 = top_idx[:, 0]
    usage = torch.bincount(top1, minlength=P)        # (P,)
    unique_top1 = int((usage > 0).sum().item())
    print(f"\n  # unique prototypes used as top-1 = {unique_top1} / {P}")
    print("  Per-prototype top-1 win count:")
    for i in range(P):
        bar = "█" * int(40 * float(usage[i]) / max(1, int(usage.max())))
        tag = " (blank)" if mask is not None and mask[i].item() < 0.5 else ""
        print(f"    [{i:>2}] {names[i]:30s}{tag}  {int(usage[i]):>6d}  {bar}")

    # ------------------------------------------------------------------
    # Per-prototype mean score across all segments
    # ------------------------------------------------------------------
    mean_per_proto = scores.mean(dim=0)               # (P,)

    # ------------------------------------------------------------------
    # CSV — one row per segment
    # ------------------------------------------------------------------
    cols = {
        "segment_idx":   list(range(N)),
        "top1_proto":    top_idx[:, 0].tolist(),
        "top1_name":     [names[int(i)] for i in top_idx[:, 0].tolist()],
        "top1_score":    [round(v, 4) for v in top_vals[:, 0].tolist()],
    }
    if K >= 2:
        cols["top2_proto"] = top_idx[:, 1].tolist()
        cols["top2_score"] = [round(v, 4) for v in top_vals[:, 1].tolist()]
    if K >= 3:
        cols["top3_proto"] = top_idx[:, 2].tolist()
        cols["top3_score"] = [round(v, 4) for v in top_vals[:, 2].tolist()]
    cols["entropy"]      = [round(v, 4) for v in entropy.tolist()]
    cols["norm_entropy"] = [round(v, 4) for v in norm_entropy.tolist()]
    csv_path = out_dir / f"score_distribution{suffix}.csv"
    pd.DataFrame(cols).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}  ({N} rows)")

    # ------------------------------------------------------------------
    # PNG — two-panel diagnostic
    # ------------------------------------------------------------------
    xs = list(range(P))
    if mask is not None:
        colors = ["tab:blue" if mask[i].item() > 0.5 else "0.6" for i in range(P)]
    else:
        colors = ["tab:blue"] * P

    fig, axes = plt.subplots(2, 1, figsize=(max(7, P * 0.55), 6.6), sharex=True)
    axes[0].bar(xs, usage.tolist(), color=colors)
    axes[0].set_ylabel("top-1 win count")
    title = "Layer-0 prototype activation distribution"
    if args.tag:
        title += f" — {args.tag}"
    axes[0].set_title(title)
    axes[0].axhline(N / P, color="black", linestyle="--", linewidth=0.8,
                    label=f"uniform baseline (N/P = {N/P:.0f})")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].bar(xs, mean_per_proto.tolist(), color=colors)
    axes[1].axhline(uniform_baseline, color="black", linestyle="--", linewidth=0.8,
                    label=f"uniform baseline (1/P = {uniform_baseline:.4f})")
    axes[1].set_ylabel("mean activation score")
    axes[1].set_xlabel("prototype")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([n if len(n) <= 18 else n[:15] + "…" for n in names],
                            rotation=45, ha="right", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    png_path = out_dir / f"score_distribution{suffix}.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()

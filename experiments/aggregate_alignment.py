"""
aggregate_alignment.py — quantitative semantic-alignment metric for the
LLM-named layer-0 prototypes in a trained ProtoLLM.

For each LLM-named prototype P:

  1.  Build the test-set segments (same schema the LLM saw at proposal time).
  2.  Partition them into "matching" (P's filter function returns True) and
      "non-matching".
  3.  Run a forward pass over the matched dataset windows and compute the
      mean layer-0 score that P receives, separately on the matching and the
      non-matching partitions.
  4.  Report the ratio
          alignment_ratio(P) = mean_score_match / mean_score_other.
      Random behaviour ≈ 1.0. Strong semantic alignment ≫ 1.0.

Outputs
-------
  * Console: a per-prototype table + an overall summary line.
  * CSV    : ``<output_dir>/aggregate_alignment.csv``.

The script never trains anything — it just loads the trained checkpoints
and runs a single forward pass over the test set.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch

from experiments._common import (
    build_test_segments,
    find_matching_test_segments,
    forward_capture,
    load_encoder_and_prototype,
    load_llm_prototypes,
)
from dataloader.dataloader import LOF_Dataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, required=True,
                   help="Path to the CSV that was used for training.")
    p.add_argument("--pretrain_checkpoint", type=str, required=True,
                   help="Trained encoder .pth (model_path saved by "
                        "exp/prototype_learning.py).")
    p.add_argument("--prototype_checkpoint", type=str, required=True,
                   help="Trained prototype .pth (prototype_path saved by "
                        "exp/prototype_learning.py).")
    p.add_argument("--final_prototypes", type=str, required=True,
                   help="final_prototypes_<timestamp>.json from the LLM "
                        "proposing loop.")
    p.add_argument("--output_dir", type=str, default="./experiments/output")

    # Shape knobs (must match training).
    p.add_argument("--input_length",  type=int, default=384)
    p.add_argument("--output_length", type=int, default=96)
    p.add_argument("--input_dim",     type=int, default=23)
    p.add_argument("--output_dim",    type=int, default=1)
    p.add_argument("--embed_dim",     type=int, default=128)
    p.add_argument("--num_layer",     type=int, default=4)
    p.add_argument("--dropout",       type=float, default=0.5)
    p.add_argument("--bottle_dim_p",  type=float, default=0.1)

    # Prototype-layer knobs (must match training).
    p.add_argument("--prototype_num",       type=int, default=8)
    p.add_argument("--prototype_split_num", type=int, default=2)
    p.add_argument("--top_k_activate",      type=int, default=2)
    p.add_argument("--split_ratio",         type=float, default=0.4)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--segment_stride", type=int, default=96,
                   help="Stride used when building test segments (default 96 "
                        "= non-overlapping daily windows, matching Part-A).")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load LLM prototype specs + trained model
    # ------------------------------------------------------------------
    llm_protos = load_llm_prototypes(args.final_prototypes)
    print(f"loaded {len(llm_protos)} LLM prototypes from {args.final_prototypes}")

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
    P0 = prototype_model.prototype.prototype_embedding[0].shape[0]
    print(f"layer-0 prototype count = {P0}")

    # ------------------------------------------------------------------
    # Build test segments + test dataset
    # ------------------------------------------------------------------
    test_segments = build_test_segments(
        csv_path=args.dataset,
        input_length=args.input_length,
        output_length=args.output_length,
        stride=args.segment_stride,
        split="test",
    )
    print(f"built {len(test_segments)} test segments "
          f"(stride={args.segment_stride})")

    test_dataset = LOF_Dataset(
        args.input_length, args.output_length,
        "test", args.input_dim, args.dataset,
    )
    print(f"test dataset has {len(test_dataset)} sliding windows")

    # Map each test segment to its dataset window index. The split slice in
    # LOF_Dataset starts at row 0 of the test sub-CSV, and build_test_segments
    # uses the same row indexing, so seg.start_idx is the dataset index.
    window_indices = [int(s["start_idx"]) for s in test_segments]
    # Drop any segments whose window would fall outside the dataset (can
    # happen at the tail when stride and length don't align cleanly).
    valid_pairs = [(i, w) for i, w in enumerate(window_indices) if w < len(test_dataset)]
    seg_indices_used = [p[0] for p in valid_pairs]
    window_indices    = [p[1] for p in valid_pairs]
    if len(window_indices) < len(test_segments):
        print(f"  (dropped {len(test_segments) - len(window_indices)} segments "
              f"whose window exceeds the dataset's last valid start)")

    # ------------------------------------------------------------------
    # Single forward pass over the full test segment set
    # ------------------------------------------------------------------
    print("running forward pass …")
    cap = forward_capture(encoder, prototype_model, test_dataset,
                          window_indices, device, batch_size=args.batch_size)
    layer0 = cap["layer0_score"]            # (K, P0), softmax-normalized
    K = layer0.shape[0]
    print(f"captured layer0 scores: {layer0.shape}")
    print(f"sanity: row sums (should be ≈1): "
          f"min={layer0.sum(dim=1).min():.4f} max={layer0.sum(dim=1).max():.4f}")

    # ------------------------------------------------------------------
    # Per-prototype alignment metric
    # ------------------------------------------------------------------
    used_segments = [test_segments[i] for i in seg_indices_used]
    rows = []
    for proto in llm_protos:
        idx_p = proto["anchor_idx"]
        if idx_p >= P0:
            print(f"  [skip] {proto['name']}: anchor idx {idx_p} >= P0 ({P0})")
            continue

        matched_seg_idx, err = find_matching_test_segments(proto, used_segments)
        if err:
            print(f"  [skip] {proto['name']}: function failed → {err.splitlines()[0]}")
            continue

        matched_mask = torch.zeros(K, dtype=torch.bool)
        if matched_seg_idx:
            matched_mask[torch.tensor(matched_seg_idx)] = True
        n_match = int(matched_mask.sum().item())
        n_other = int((~matched_mask).sum().item())

        scores_p = layer0[:, idx_p]
        mean_match = (scores_p[matched_mask].mean().item()
                      if n_match > 0 else float("nan"))
        mean_other = (scores_p[~matched_mask].mean().item()
                      if n_other > 0 else float("nan"))
        ratio = mean_match / mean_other if (mean_other and mean_other > 0) else float("nan")
        baseline = 1.0 / P0   # uniform-attention baseline

        rows.append({
            "anchor_idx":              idx_p,
            "name":                    proto["name"],
            "n_match":                 n_match,
            "n_other":                 n_other,
            "mean_score_match":        round(mean_match, 6) if mean_match == mean_match else None,
            "mean_score_other":        round(mean_other, 6) if mean_other == mean_other else None,
            "ratio_match_over_other":  round(ratio, 4) if ratio == ratio else None,
            "uniform_baseline":        round(baseline, 6),
        })

        print(f"  {proto['name']:35s}  "
              f"matched={n_match:4d}  "
              f"mean_score(match)={mean_match:.4f}  "
              f"mean_score(other)={mean_other:.4f}  "
              f"ratio={ratio:.2f}×")

    # ------------------------------------------------------------------
    # Aggregate + save
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)
    out_csv = Path(args.output_dir) / "aggregate_alignment.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    finite_ratios = [r["ratio_match_over_other"] for r in rows
                     if r["ratio_match_over_other"] is not None
                     and r["ratio_match_over_other"] == r["ratio_match_over_other"]]
    if finite_ratios:
        print(f"\noverall (across {len(finite_ratios)} prototypes): "
              f"mean ratio = {sum(finite_ratios)/len(finite_ratios):.2f}×, "
              f"min = {min(finite_ratios):.2f}×, "
              f"max = {max(finite_ratios):.2f}×")
        print("interpretation: ratio = 1.0 means no semantic alignment "
              "(prototype activates equally on matching & non-matching "
              "test segments). Higher = stronger alignment with the LLM "
              "filter that defined the prototype.")


if __name__ == "__main__":
    main()

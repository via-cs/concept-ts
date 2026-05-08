"""
One-shot orchestrator: LLM prototype proposal loop  →  init tensors.

Wraps two stages into a single command:

    1. prototype_proposing_loop/run_loop.run_loop()
       — builds summary vectors, clusters, asks Claude to propose Python
       select_segments() functions, validates them, revises if needed,
       writes final_prototypes_<timestamp>.json

    2. LLM_prototype/prototype_initializer.py main flow
       — runs each prototype's function, picks the typical_num most typical
       segments via Mahalanobis-to-centroid, encodes them, averages →
       (init_embedding.pt, init_value.pt, anchor_mask.pt)

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python exp/build_llm_init.py \
        --dataset             dataset/LOF/RE.csv \
        --segments_path       preprocessing/output/RE_full_segments.json \
        --pretrain_checkpoint checkpoint/embedding_learning/RE_..._BDP0.1.pth \
        --output_dir          LLM_prototype/output/init \
        --clustering_num 20  --n_prototypes 8  --typical_num 20 \
        --num_blank 2  --max_iterations 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataloader.dataloader import LOF_Dataset
from model.embedding import Embedding_Learning

# Local helpers
sys.path.insert(0, os.path.join(_REPO_ROOT, "prototype_proposing_loop"))
from prototype_proposing_loop.run_loop import run_loop
from LLM_prototype.prototype_initializer import build_one_prototype
from prototype_proposing_loop.summary_vector import load_artifacts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="Run the LLM proposing loop and produce the init tensors "
                    "expected by exp/prototype_learning.py."
    )

    # Data + encoder
    p.add_argument("--dataset",             type=str, required=True)
    p.add_argument("--segments_path",       type=str, required=True,
                   help="preprocessing/output/<DATASET>_full_segments.json")
    p.add_argument("--pretrain_checkpoint", type=str, required=True)
    p.add_argument("--output_dir",          type=str, required=True)

    # Clustering / proposal-loop
    p.add_argument("--clustering_num", type=int, default=20)
    p.add_argument("--n_prototypes",   type=int, default=8)
    p.add_argument("--top_k",          type=int, default=10,
                   help="K used inside the rubric (forecast-coherence cohort)")
    p.add_argument("--max_iterations", type=int, default=3)
    p.add_argument("--min_match_ratio", type=float, default=0.5)
    p.add_argument("--generality_cap",    type=float, default=0.4)
    p.add_argument("--max_jaccard",       type=float, default=0.5)
    p.add_argument("--max_forecast_cv",   type=float, default=0.3)
    p.add_argument("--min_constrained_dims", type=int, default=2)
    p.add_argument("--random_state",      type=int, default=0)

    # Typicality + init
    p.add_argument("--typical_num", type=int, default=20,
                   help="Number of typical segments per prototype to encode + average")
    p.add_argument("--num_blank",   type=int, default=2)

    # Encoder shape (must match embedding_learning.py)
    p.add_argument("--input_length",  type=int, default=96 * 4)
    p.add_argument("--output_length", type=int, default=96)
    p.add_argument("--input_dim",     type=int, default=23)
    p.add_argument("--output_dim",    type=int, default=1)
    p.add_argument("--embed_dim",     type=int, default=128)
    p.add_argument("--num_layer",     type=int, default=4)
    p.add_argument("--dropout",       type=float, default=0.5)
    p.add_argument("--bottle_dim_p",  type=float, default=0.1)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["device"] = device

    os.makedirs(args.output_dir, exist_ok=True)
    loop_dir = os.path.join(args.output_dir, "loop_artifacts")

    # ==================================================================
    # Phase 1 — propose-validate-revise loop
    # ==================================================================
    print("\n" + "#" * 64)
    print("# Phase 1 — propose-validate-revise loop")
    print("#" * 64)
    loop_result = run_loop(
        segments_path=args.segments_path,
        output_dir=loop_dir,
        clustering_num=args.clustering_num,
        n_prototypes=args.n_prototypes,
        top_k=args.top_k,
        min_match_ratio=args.min_match_ratio,
        max_iterations=args.max_iterations,
        generality_cap=args.generality_cap,
        max_jaccard=args.max_jaccard,
        max_forecast_cv=args.max_forecast_cv,
        min_constrained_dims=args.min_constrained_dims,
        random_state=args.random_state,
    )
    prototypes = loop_result["prototypes"]
    print(f"\nLoop produced {len(prototypes)} prototype specs.")

    # ==================================================================
    # Phase 2 — typicality + encoder + init tensors
    # ==================================================================
    print("\n" + "#" * 64)
    print("# Phase 2 — typicality (Mahalanobis) + encoder + init tensors")
    print("#" * 64)

    # Load encoder
    model = Embedding_Learning(
        input_dim=args.input_dim, mid_dim=args.embed_dim,
        ext_dim=args.input_dim - args.output_dim, output_dim=args.output_dim,
        bottle_dim_p=args.bottle_dim_p,
        input_length=args.input_length, output_length=args.output_length,
        num_blocks=args.num_layer, dropout=args.dropout,
        norm_type="B", use_rev_in=1,
    ).to(device)
    model.load_state_dict(torch.load(args.pretrain_checkpoint, map_location=device))
    model.eval()
    print(f"Loaded encoder      <- {args.pretrain_checkpoint}")

    # Training-window dataset (for the encoder pass)
    train_dataset = LOF_Dataset(args.input_length, args.output_length,
                                 "train", args.input_dim, args.dataset)
    print(f"Loaded {len(train_dataset)} training windows from {args.dataset}")

    # Part-A segments + summary-vector artifacts
    with open(args.segments_path) as f:
        segments = json.load(f)["segments"]
    summary = load_artifacts(loop_result["summary_artifacts_dir"])
    summary_vectors_z = summary["vectors_z"]
    print(f"Loaded {len(segments)} segments + {summary_vectors_z.shape[0]} summary vectors")

    # Per-prototype build
    mu_list, p_list, names = [], [], []
    for spec in prototypes:
        result = build_one_prototype(
            spec=spec, segments=segments,
            summary_vectors_z=summary_vectors_z,
            dataset=train_dataset, model=model, device=device,
            input_length=args.input_length,
            output_length=args.output_length,
            typical_num=args.typical_num,
        )
        if result is None:
            continue
        mu_list.append(result["mu"])
        p_list.append(result["p"])
        names.append(result["name"])
        print(f"  ✓ {result['name']:30s}  matched={result['n_matched']:4d}  "
              f"typical={result['n_typical']:3d}  ‖μ‖={result['mu'].norm():.3f}  "
              f"mean(p)={result['p'].mean():+.3f}")

    if not mu_list:
        raise RuntimeError("No prototypes initialised. Check function execution + matches.")

    # Stack + append blanks
    mu_init = torch.stack(mu_list, dim=0)
    p_init  = torch.stack(p_list, dim=0)
    N = mu_init.shape[0]
    M = args.num_blank
    if M > 0:
        mu_blank = torch.empty(M, args.embed_dim)
        nn.init.kaiming_uniform_(mu_blank, nonlinearity="relu")
        p_blank = torch.randn(M, args.output_length, args.output_dim)
        mu_init = torch.cat([mu_init, mu_blank], dim=0)
        p_init  = torch.cat([p_init,  p_blank],  dim=0)
    anchor_mask = torch.zeros(N + M)
    anchor_mask[:N] = 1.0

    torch.save(mu_init,     os.path.join(args.output_dir, "init_embedding.pt"))
    torch.save(p_init,      os.path.join(args.output_dir, "init_value.pt"))
    torch.save(anchor_mask, os.path.join(args.output_dir, "anchor_mask.pt"))
    with open(os.path.join(args.output_dir, "prototype_names.json"), "w") as f:
        json.dump({"names": names, "num_anchored": N, "num_blank": M,
                   "total": N + M}, f, indent=2)

    print("\n" + "=" * 64)
    print(f"Wrote init tensors for {N + M} prototypes ({N} LLM-anchored + {M} blank)")
    print(f"  {os.path.join(args.output_dir, 'init_embedding.pt'):60s} {tuple(mu_init.shape)}")
    print(f"  {os.path.join(args.output_dir, 'init_value.pt'):60s} {tuple(p_init.shape)}")
    print(f"  {os.path.join(args.output_dir, 'anchor_mask.pt'):60s} {tuple(anchor_mask.shape)}")
    print(f"\nNext: pass --prototype_num {N + M} to prototype_learning.py")
    print("=" * 64)


if __name__ == "__main__":
    main()

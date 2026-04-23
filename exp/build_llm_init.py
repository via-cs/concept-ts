"""
One-shot orchestrator: LLM prototype suggestion loop  →  init tensors.

This folds three previously separate steps into a single command:

    1. propose-validate-revise loop (prototype_proposing_loop/run_loop.py)
    2. per-spec top-K instance finding (LLM_prototype/instance_finder.find_instances)
    3. encoder-based prototype initialization + blank appending
       (LLM_prototype/prototype_initializer)

Output:
    <output_dir>/init_embedding.pt
    <output_dir>/init_value.pt
    <output_dir>/anchor_mask.pt
    <output_dir>/prototype_names.json
    <output_dir>/loop_artifacts/       # full LLM loop log (specs / feedback / evals)

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python exp/build_llm_init.py \
        --dataset              dataset/LOF/RE.csv \
        --segments_path        preprocessing/output/RE_full_segments.json \
        --pretrain_checkpoint  checkpoint/embedding_learning/RE_...BDP0.1.pth \
        --output_dir           LLM_prototype/output/init \
        --n_prototypes         10 \
        --top_k                20 \
        --max_iterations       3 \
        --num_blank            2
"""

import argparse
import json
import os
import sys

# Make sibling packages importable when this is run from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from dataloader.dataloader import LOF_Dataset
from model.embedding import Embedding_Learning
from prototype_proposing_loop.run_loop import run_loop
from LLM_prototype.instance_finder import find_instances
from LLM_prototype.prototype_initializer import (
    build_one_prototype_from_indices,
    save_init_tensors,
)


def get_args():
    p = argparse.ArgumentParser(
        description="Run the LLM prototype loop and produce the init tensors "
                    "needed by exp/prototype_learning.py"
    )

    # Data + encoder
    p.add_argument("--dataset",             type=str, required=True)
    p.add_argument("--segments_path",       type=str, required=True,
                   help="preprocessing/output/<DATASET>_full_segments.json")
    p.add_argument("--pretrain_checkpoint", type=str, required=True)
    p.add_argument("--output_dir",          type=str, required=True)

    # LLM loop
    p.add_argument("--n_prototypes",   type=int, default=10)
    p.add_argument("--top_k",          type=int, default=20)
    p.add_argument("--max_iterations", type=int, default=3)
    p.add_argument("--min_match_ratio", type=float, default=0.5)

    # Rubric knobs (applied inside run_loop → evaluate_all_prototypes)
    p.add_argument("--generality_cap",    type=float, default=0.4)
    p.add_argument("--max_jaccard",       type=float, default=0.5)
    p.add_argument("--max_forecast_cv",   type=float, default=0.3)
    p.add_argument("--min_active_reqs",   type=int,   default=2)
    p.add_argument("--min_req_rejection", type=float, default=0.15)

    # Init phase
    p.add_argument("--num_blank",      type=int,   default=2)
    p.add_argument("--relaxation_steps", type=int, default=3)

    # Model shape (must match embedding_learning.py)
    p.add_argument("--input_length",  type=int, default=96 * 4)
    p.add_argument("--output_length", type=int, default=96)
    p.add_argument("--input_dim",     type=int, default=23)
    p.add_argument("--output_dim",    type=int, default=1)
    p.add_argument("--embed_dim",     type=int, default=128)
    p.add_argument("--num_layer",     type=int, default=4)
    p.add_argument("--dropout",       type=float, default=0.5)
    p.add_argument("--bottle_dim_p",  type=float, default=0.1)

    return p.parse_args()


def load_encoder(args, device):
    model = Embedding_Learning(
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
    ).to(device)
    model.load_state_dict(torch.load(args.pretrain_checkpoint, map_location=device))
    model.eval()
    return model


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["device"] = device

    os.makedirs(args.output_dir, exist_ok=True)
    loop_out = os.path.join(args.output_dir, "loop_artifacts")

    # ==================================================================
    # Phase 1 — LLM propose-validate-revise loop.
    # Writes its iteration logs under <output_dir>/loop_artifacts and
    # returns the final accepted spec list in-memory.
    # ==================================================================
    print("\n" + "#" * 64)
    print("# Phase 1 — LLM propose-validate-revise loop")
    print("#" * 64)
    loop_result = run_loop(
        segments_path=args.segments_path,
        n_prototypes=args.n_prototypes,
        top_k=args.top_k,
        min_match_ratio=args.min_match_ratio,
        max_iterations=args.max_iterations,
        output_dir=loop_out,
        generality_cap=args.generality_cap,
        max_jaccard=args.max_jaccard,
        max_forecast_cv=args.max_forecast_cv,
        min_active_reqs=args.min_active_reqs,
        min_req_rejection=args.min_req_rejection,
    )
    specs = loop_result["prototypes"]
    print(f"\nLoop converged with {len(specs)} prototype specs.")

    # ==================================================================
    # Phase 2 — Per-spec top-K instance finding (in-memory).
    # ==================================================================
    print("\n" + "#" * 64)
    print("# Phase 2 — Instance finding")
    print("#" * 64)

    with open(args.segments_path) as f:
        segments = json.load(f)["segments"]

    spec_to_indices = []          # list of (name, [start_idx...])
    for spec in specs:
        name = spec["name"]
        print(f"\n-- {name} --")
        result = find_instances(
            segments,
            spec["channel_requirements"],
            top_k=args.top_k,
            relaxation_steps=args.relaxation_steps,
        )
        start_indices = [inst["start_idx"] for inst in result["selected_instances"]]
        spec_to_indices.append((name, start_indices))
        print(f"  selected {len(start_indices)} instances")

    # ==================================================================
    # Phase 3 — Encoder pass + init tensors.
    # ==================================================================
    print("\n" + "#" * 64)
    print("# Phase 3 — Encoder-based prototype initialization")
    print("#" * 64)

    model = load_encoder(args, device)
    print(f"Loaded encoder from {args.pretrain_checkpoint}")
    train_dataset = LOF_Dataset(
        args.input_length, args.output_length,
        "train", args.input_dim, args.dataset,
    )
    print(f"Loaded {len(train_dataset)} training windows from {args.dataset}")

    mu_list, p_list, names = [], [], []
    for name, start_indices in spec_to_indices:
        r = build_one_prototype_from_indices(
            name, start_indices, model, train_dataset, device,
            args.input_length, args.output_length, args.top_k,
        )
        if r is None:
            print(f"  [skip] {name} — no usable instances")
            continue
        mu_list.append(r["mu"])
        p_list.append(r["p"])
        names.append(r["name"])
        print(f"  ✓ {r['name']:30s}  n={r['n_instances']:3d}  "
              f"‖μ‖={r['mu'].norm():.3f}  mean(p)={r['p'].mean():+.3f}")

    mu_init, p_init, anchor_mask = save_init_tensors(
        mu_list, p_list, names,
        output_dir=args.output_dir,
        num_blank=args.num_blank,
        embed_dim=args.embed_dim,
        output_length=args.output_length,
        output_dim=args.output_dim,
    )

    total = mu_init.shape[0]
    print("\n" + "=" * 64)
    print(f"Wrote init tensors for {total} prototypes "
          f"({len(mu_list)} LLM-anchored + {args.num_blank} blank)")
    print(f"  {os.path.join(args.output_dir, 'init_embedding.pt')}  "
          f"shape {tuple(mu_init.shape)}")
    print(f"  {os.path.join(args.output_dir, 'init_value.pt')}      "
          f"shape {tuple(p_init.shape)}")
    print(f"  {os.path.join(args.output_dir, 'anchor_mask.pt')}     "
          f"shape {tuple(anchor_mask.shape)}")
    print(f"\nNext: pass --prototype_num {total} to prototype_learning.py")
    print("=" * 64)


if __name__ == "__main__":
    main()

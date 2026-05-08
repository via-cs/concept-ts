"""
Main loop controller (function-based pipeline).

Pipeline:
    1. Load Part-A segments + global stats.
    2. Build per-segment summary vectors, z-score-standardise, K-means
       cluster into `clustering_num` regimes.
    3. Ask Claude to propose `n_prototypes` prototypes, each as a Python
       `select_segments` function. The prompt sees the cluster summaries
       (not just global stats) so it can write realistic predicates.
    4. Validate each prototype via the rubric in segment_matcher.py.
    5. If any prototype fails, send formatted feedback to Claude and ask
       for revisions. Repeat up to `max_iterations`.
    6. Save the final prototype list (with their functions, matched
       indices, and evaluation) for the initializer to consume.

Usage:
    python prototype_proposing_loop/run_loop.py \
        --segments_path preprocessing/output/RE_full_segments.json \
        --output_dir prototype_proposing_loop/output/RE \
        --clustering_num 20 \
        --n_prototypes 8 \
        --top_k 10 \
        --max_iterations 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

# Make imports work when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.data_preprocessing import CHANNEL_META
from summary_vector import build_and_cluster
from segment_matcher import evaluate_all_prototypes
from feedback_generator import generate_feedback, generate_compact_feedback
from llm_proposer import propose_initial_prototypes, propose_revisions
from prompts import INITIAL_PROMPT, REVISION_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_iteration(output_dir: str, timestamp: str, iteration: int,
                     prototypes: list, evaluation: dict | None,
                     feedback_text: str | None = None) -> None:
    iter_dir = os.path.join(output_dir, f"run_{timestamp}", f"iter_{iteration}")
    os.makedirs(iter_dir, exist_ok=True)
    with open(os.path.join(iter_dir, "prototypes.json"), "w") as f:
        json.dump(prototypes, f, indent=2)
    if evaluation is not None:
        with open(os.path.join(iter_dir, "evaluation.json"), "w") as f:
            json.dump(evaluation, f, indent=2)
    if feedback_text is not None:
        with open(os.path.join(iter_dir, "feedback.txt"), "w") as f:
            f.write(feedback_text)


def _print_eval_summary(evaluation: dict) -> None:
    for r in evaluation["results"]:
        icon = "✓" if r["status"] == "good" else "✗"
        modes = r.get("failure_modes", [])
        tag = "[good]" if not modes else f"[low_quality: {','.join(modes)}]"
        print(f"  {icon} P{r['prototype_id']} \"{r['name']}\": "
              f"n={r['n_matches']}, gen={r['generality']}, "
              f"cv={r.get('forecast_coherence_cv')}, "
              f"sharp={r.get('spec_sharpness', {}).get('n_constrained', 0)}, "
              f"J={r.get('max_jaccard_overlap', 0)} {tag}")
    print(f"\n  Summary: {evaluation['n_good']} good, "
          f"{evaluation['n_low_quality']} need revision")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(segments_path: str, output_dir: str, *,
             clustering_num: int = 20,
             n_prototypes: int = 8,
             top_k: int = 10,
             min_match_ratio: float = 0.5,
             max_iterations: int = 3,
             generality_cap: float = 0.4,
             max_jaccard: float = 0.5,
             max_forecast_cv: float = 0.3,
             min_constrained_dims: int = 2,
             random_state: int = 0) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- 1-2. Build summary vectors + cluster ----
    print("=" * 60)
    print("STAGE 1+2 — SUMMARY VECTORS & CLUSTERING")
    print("=" * 60)
    artifacts_dir = os.path.join(output_dir, "summary")
    bundle = build_and_cluster(
        segments_path=segments_path,
        output_dir=artifacts_dir,
        n_clusters=clustering_num,
        channel_meta=CHANNEL_META,
        random_state=random_state,
    )
    segments = bundle["segments"]
    global_stats = bundle["global_stats"]
    summary_vectors_z = bundle["vectors_z"]
    cluster_summaries = bundle["cluster_summaries"]

    # Pull metadata back from the segments JSON (we drop it during build_and_cluster)
    with open(segments_path) as f:
        metadata = json.load(f)["metadata"]

    # ---- 3. Initial proposal ----
    print("\n" + "=" * 60)
    print("STAGE 3 — INITIAL PROPOSAL")
    print("=" * 60)
    prototypes = propose_initial_prototypes(
        n_prototypes=n_prototypes,
        global_statistics=global_stats,
        cluster_summaries=cluster_summaries,
        metadata=metadata,
        initial_prompt_template=INITIAL_PROMPT,
    )
    if len(prototypes) != n_prototypes:
        print(f"  WARNING: expected {n_prototypes} prototypes, got {len(prototypes)}")
    _save_iteration(output_dir, timestamp, 0, prototypes, None)

    # ---- 4-5. Validate-revise loop ----
    iteration = 0
    final_eval = None
    for iteration in range(1, max_iterations + 1):
        print("\n" + "=" * 60)
        print(f"STAGE 4 — VALIDATION (iteration {iteration})")
        print("=" * 60)
        evaluation = evaluate_all_prototypes(
            prototypes, segments, summary_vectors_z,
            top_k=top_k, min_match_ratio=min_match_ratio,
            generality_cap=generality_cap, max_jaccard=max_jaccard,
            max_forecast_cv=max_forecast_cv,
            min_constrained_dims=min_constrained_dims,
            target_variable=metadata.get("target_variable", "load"),
        )
        _print_eval_summary(evaluation)
        final_eval = evaluation

        if evaluation["all_good"]:
            print("\n  ALL PROTOTYPES PASSED.")
            _save_iteration(output_dir, timestamp, iteration, prototypes, evaluation)
            break

        feedback_text = (generate_feedback(evaluation) if iteration == 1
                         else generate_compact_feedback(evaluation))
        print(f"\n  Feedback generated ({len(feedback_text)} chars).")
        _save_iteration(output_dir, timestamp, iteration, prototypes,
                        evaluation, feedback_text)

        if iteration == max_iterations:
            print(f"\n  MAX ITERATIONS reached. "
                  f"{evaluation['n_low_quality']} prototypes still flagged.")
            break

        print("\n  Requesting revision from Claude...")
        prototypes = propose_revisions(
            current_prototypes=prototypes,
            feedback_text=feedback_text,
            revision_prompt_template=REVISION_PROMPT,
        )

    # ---- 6. Save final ----
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    _print_eval_summary(final_eval)

    final_output = {
        "metadata": {
            "timestamp": timestamp,
            "dataset": metadata.get("dataset_name"),
            "clustering_num": clustering_num,
            "n_prototypes": n_prototypes,
            "top_k": top_k,
            "min_match_ratio": min_match_ratio,
            "iterations_used": iteration,
            "all_passed": final_eval["all_good"],
        },
        "prototypes": prototypes,
        "evaluation": final_eval,
        "summary_artifacts_dir": artifacts_dir,
        "segments_path": segments_path,
    }
    final_path = os.path.join(output_dir, f"final_prototypes_{timestamp}.json")
    with open(final_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n  Saved final prototypes -> {final_path}")
    return final_output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the function-based prototype proposing loop."
    )
    parser.add_argument("--segments_path", type=str, required=True,
                        help="Part-A full-segments JSON")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to write summary artifacts + prototype JSONs")
    parser.add_argument("--clustering_num", type=int, default=20)
    parser.add_argument("--n_prototypes", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--min_match_ratio", type=float, default=0.5)
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--generality_cap", type=float, default=0.4)
    parser.add_argument("--max_jaccard", type=float, default=0.5)
    parser.add_argument("--max_forecast_cv", type=float, default=0.3)
    parser.add_argument("--min_constrained_dims", type=int, default=2)
    parser.add_argument("--random_state", type=int, default=0)
    args = parser.parse_args()

    run_loop(
        segments_path=args.segments_path,
        output_dir=args.output_dir,
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


if __name__ == "__main__":
    main()

"""
Main Loop Controller: The propose-validate-revise loop.

Flow:
  1. Load preprocessing outputs (global stats + full segment bank)
  2. Call LLM to propose N initial prototypes
  3. Validate all prototypes via segment matcher
  4. If all good → done
  5. If some low-quality → generate feedback, ask LLM to revise
  6. Repeat 3-5 until all good or max iterations reached
  7. Save final prototypes to disk

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python prototype_proposing_loop/run_loop.py \
        --segments_path preprocessing/output/RE_full_segments.json \
        --n_prototypes 10 \
        --top_k 10 \
        --max_iterations 3
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add parent dir to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segment_matcher import evaluate_all_prototypes
from feedback_generator import generate_feedback, generate_compact_feedback
from llm_proposer import propose_initial_prototypes, propose_revisions
from prompts import INITIAL_PROMPT, REVISION_PROMPT


def run_loop(
    segments_path: str,
    n_prototypes: int = 10,
    top_k: int = 10,
    min_match_ratio: float = 0.5,
    max_iterations: int = 3,
    output_dir: str = None,
    *,
    generality_cap: float = 0.4,
    max_jaccard: float = 0.5,
    max_forecast_cv: float = 0.3,
    min_active_reqs: int = 2,
    min_req_rejection: float = 0.15,
):
    """Run the full propose-validate-revise loop.

    Args:
        segments_path: Path to RE_full_segments.json from Part A
        n_prototypes: Number of prototypes to generate
        top_k: Desired instances per prototype
        min_match_ratio: Below this ratio of top_k = criterion-1 fail
        max_iterations: Maximum revision rounds
        output_dir: Where to save outputs (default: prototype_proposing_loop/output/)
        generality_cap: Criterion 2 — max fraction of segments a spec can match
        max_jaccard: Criterion 3 — max Jaccard overlap vs any earlier accepted
        max_forecast_cv: Criterion 4 — max coefficient-of-variation of
                         forecast-mean across the top-K matched segments
        min_active_reqs: Criterion 6 — min count of discriminative requirements
        min_req_rejection: Criterion 6 — a requirement is "discriminative"
                           if it individually rejects this fraction of segments
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ============================================================
    # Step 1: Load preprocessing data
    # ============================================================
    print("=" * 60)
    print("PROTOTYPE PROPOSING LOOP")
    print("=" * 60)
    print(f"\nLoading segments from {segments_path}...")
    with open(segments_path) as f:
        seg_data = json.load(f)

    segments = seg_data["segments"]
    global_stats = seg_data["global_statistics"]
    metadata = seg_data["metadata"]
    print(f"  Loaded {len(segments)} segments")
    print(f"  Dataset: {metadata.get('dataset_name', 'unknown')}")

    # ============================================================
    # Step 2: Initial proposal
    # ============================================================
    print("\n" + "=" * 60)
    print("ITERATION 0: Initial Proposal")
    print("=" * 60)

    prototypes = propose_initial_prototypes(
        n_prototypes=n_prototypes,
        global_statistics=global_stats,
        metadata=metadata,
        initial_prompt_template=INITIAL_PROMPT,
    )

    # Validate prototype count
    if len(prototypes) != n_prototypes:
        print(f"  WARNING: Expected {n_prototypes} prototypes, got {len(prototypes)}")

    # Save initial proposal
    _save_iteration(output_dir, timestamp, 0, prototypes, None)

    # ============================================================
    # Step 3-6: Validate-Revise Loop
    # ============================================================
    for iteration in range(1, max_iterations + 1):
        print("\n" + "=" * 60)
        print(f"ITERATION {iteration}: Validation")
        print("=" * 60)

        # Validate
        evaluation = evaluate_all_prototypes(
            prototypes, segments,
            top_k=top_k, min_match_ratio=min_match_ratio,
            generality_cap=generality_cap,
            max_jaccard=max_jaccard,
            max_forecast_cv=max_forecast_cv,
            min_active_reqs=min_active_reqs,
            min_req_rejection=min_req_rejection,
            target_variable=metadata.get("target_variable", "load"),
        )

        # Print summary
        for r in evaluation["results"]:
            status_icon = "✓" if r["status"] == "good" else "✗"
            modes = r.get("failure_modes", [])
            tag = "[good]" if not modes else f"[low_quality: {','.join(modes)}]"
            print(
                f"  {status_icon} Prototype {r['prototype_id']} \"{r['name']}\": "
                f"n={r['n_matches']}, gen={r['generality']}, "
                f"cv={r.get('forecast_coherence_cv')}, "
                f"J={r.get('max_jaccard_overlap', 0)} {tag}"
            )

        print(f"\n  Summary: {evaluation['n_good']} good, "
              f"{evaluation['n_low_quality']} need revision")

        # Check if done
        if evaluation["all_good"]:
            print("\n  ALL PROTOTYPES PASSED!")
            _save_iteration(output_dir, timestamp, iteration, prototypes, evaluation)
            break

        # Generate feedback
        if iteration == 1:
            feedback_text = generate_feedback(evaluation)
        else:
            # Use compact feedback after first iteration to save tokens
            feedback_text = generate_compact_feedback(evaluation)

        print(f"\n  Feedback generated ({len(feedback_text)} chars)")

        # Save this iteration's state
        _save_iteration(output_dir, timestamp, iteration, prototypes, evaluation,
                        feedback_text)

        # Check if this is the last iteration
        if iteration == max_iterations:
            print(f"\n  MAX ITERATIONS ({max_iterations}) reached.")
            print(f"  {evaluation['n_low_quality']} prototypes still need revision.")
            print("  Proceeding with current prototypes (some may be low-quality).")
            break

        # Ask LLM to revise
        print(f"\n  Requesting revision from Claude...")
        prototypes = propose_revisions(
            current_prototypes=prototypes,
            feedback_text=feedback_text,
            revision_prompt_template=REVISION_PROMPT,
        )

    # ============================================================
    # Step 7: Save final results
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    # Final validation
    final_eval = evaluate_all_prototypes(
        prototypes, segments,
        top_k=top_k, min_match_ratio=min_match_ratio,
        generality_cap=generality_cap,
        max_jaccard=max_jaccard,
        max_forecast_cv=max_forecast_cv,
        min_active_reqs=min_active_reqs,
        min_req_rejection=min_req_rejection,
        target_variable=metadata.get("target_variable", "load"),
    )

    for r in final_eval["results"]:
        status_icon = "✓" if r["status"] == "good" else "✗"
        modes = r.get("failure_modes", [])
        tag = "good" if not modes else f"low_quality: {','.join(modes)}"
        print(
            f"  {status_icon} Prototype {r['prototype_id']} \"{r['name']}\": "
            f"n={r['n_matches']}, gen={r['generality']}, "
            f"cv={r.get('forecast_coherence_cv')}, "
            f"J={r.get('max_jaccard_overlap', 0)} [{tag}] | "
            f"selected {len(r['selected_indices'])} instances"
        )

    # Save final outputs
    final_output = {
        "metadata": {
            "timestamp": timestamp,
            "dataset": metadata.get("dataset_name"),
            "n_prototypes": n_prototypes,
            "top_k": top_k,
            "min_match_ratio": min_match_ratio,
            "iterations_used": iteration if not final_eval["all_good"] else iteration,
            "all_passed": final_eval["all_good"],
        },
        "prototypes": prototypes,
        "evaluation": final_eval,
    }

    final_path = os.path.join(output_dir, f"final_prototypes_{timestamp}.json")
    with open(final_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n  Saved final prototypes to {final_path}")

    # Also save a clean prototypes-only file for downstream use
    specs_path = os.path.join(output_dir, f"prototype_specs_{timestamp}.json")
    with open(specs_path, "w") as f:
        json.dump(prototypes, f, indent=2)
    print(f"  Saved clean specs to {specs_path}")

    return final_output


def _save_iteration(output_dir, timestamp, iteration, prototypes, evaluation,
                    feedback_text=None):
    """Save the state of one iteration for debugging/inspection."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Run the prototype propose-validate-revise loop"
    )
    parser.add_argument("--segments_path", type=str, required=True,
                        help="Path to full segments JSON from Part A")
    parser.add_argument("--n_prototypes", type=int, default=10,
                        help="Number of prototypes to generate (default: 10)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Desired instances per prototype (default: 10)")
    parser.add_argument("--min_match_ratio", type=float, default=0.5,
                        help="Min matches / top_k ratio for 'good' (default: 0.5)")
    parser.add_argument("--max_iterations", type=int, default=3,
                        help="Max revision rounds (default: 3)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: prototype_proposing_loop/output/)")

    # Rubric knobs
    parser.add_argument("--generality_cap", type=float, default=0.4,
                        help="Criterion 2: max n_matches/total (default: 0.4)")
    parser.add_argument("--max_jaccard", type=float, default=0.5,
                        help="Criterion 3: max Jaccard vs earlier accepted "
                             "(default: 0.5)")
    parser.add_argument("--max_forecast_cv", type=float, default=0.3,
                        help="Criterion 4: max CV of forecast-mean over top-K "
                             "(default: 0.3)")
    parser.add_argument("--min_active_reqs", type=int, default=2,
                        help="Criterion 6: min discriminative requirements "
                             "(default: 2)")
    parser.add_argument("--min_req_rejection", type=float, default=0.15,
                        help="Criterion 6: fraction of segments a requirement "
                             "must individually reject to count as discriminative "
                             "(default: 0.15)")
    args = parser.parse_args()

    run_loop(
        segments_path=args.segments_path,
        n_prototypes=args.n_prototypes,
        top_k=args.top_k,
        min_match_ratio=args.min_match_ratio,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
        generality_cap=args.generality_cap,
        max_jaccard=args.max_jaccard,
        max_forecast_cv=args.max_forecast_cv,
        min_active_reqs=args.min_active_reqs,
        min_req_rejection=args.min_req_rejection,
    )


if __name__ == "__main__":
    main()

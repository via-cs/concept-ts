"""
Part B/C: Instance Finder for LLM-Guided Prototype Initialization

Given a prototype spec (JSON with channel_requirements), this script:
1. Loads the full segment bank from Part A preprocessing
2. Filters segments by the prototype's channel requirements
3. Ranks matching segments by "margin" (how strongly they satisfy each threshold)
4. Selects top-K instances
5. Averages their forecast windows → prototype_value (p) initialization
6. Stores segment indices for later encoder-based prototype_embedding (μ) initialization

Usage:
    python instance_finder.py \
        --spec_path LLM_prototype/prototype_specs/RE_prototype_001.json \
        --segments_path preprocessing/output/RE_full_segments.json \
        --data_path dataset/LOF/RE.csv \
        --output_dir LLM_prototype/output \
        --top_k 20
"""

import os
import json
import argparse
import numpy as np
import pandas as pd


def evaluate_requirement(value, requirement: dict) -> tuple:
    """Check if a value satisfies a channel requirement.

    Returns:
        (matches: bool, margin: float)
        margin > 0 means the value exceeds the threshold by that amount.
        Higher margin = stronger match.
    """
    op = requirement.get("operator")

    if op is None:
        # No requirement on this channel
        return True, 0.0

    threshold = requirement.get("threshold")
    values_list = requirement.get("values")

    if op == "<":
        return value < threshold, max(0, threshold - value)
    elif op == ">":
        return value > threshold, max(0, value - threshold)
    elif op == "<=":
        return value <= threshold, max(0, threshold - value)
    elif op == ">=":
        return value >= threshold, max(0, value - threshold)
    elif op == "==":
        return value == threshold, (1.0 if value == threshold else 0.0)
    elif op == "!=":
        return value != threshold, (1.0 if value != threshold else 0.0)
    elif op == "in":
        return value in values_list, (1.0 if value in values_list else 0.0)
    elif op == "between":
        low = requirement["low"]
        high = requirement["high"]
        in_range = low <= value <= high
        if in_range:
            margin = min(value - low, high - value)
        else:
            margin = 0.0
        return in_range, margin
    else:
        raise ValueError(f"Unknown operator: {op}")


def match_segment(segment: dict, requirements: dict) -> tuple:
    """Check if a segment matches all channel requirements.

    Evaluates requirements against the lookback window statistics.
    For continuous channels, matches against the mean.
    For discrete channels, matches against the mode.

    Returns:
        (matches: bool, total_margin: float, per_channel_results: dict)
    """
    lb_stats = segment["lookback_stats"]
    per_channel = {}
    total_margin = 0.0
    all_match = True

    for channel, req in requirements.items():
        if req.get("operator") is None:
            # No requirement on this channel, skip
            continue

        if channel not in lb_stats:
            # Channel not in segment data — can't evaluate, skip
            per_channel[channel] = {"match": False, "reason": "channel_missing"}
            all_match = False
            continue

        ch_stats = lb_stats[channel]

        # Pick the right summary statistic to compare
        if "mean" in ch_stats:
            # Continuous channel — compare against mean
            value = ch_stats["mean"]
        elif "mode" in ch_stats:
            # Discrete channel — compare against mode
            value = ch_stats["mode"]
        else:
            per_channel[channel] = {"match": False, "reason": "no_stat"}
            all_match = False
            continue

        matches, margin = evaluate_requirement(value, req)
        per_channel[channel] = {
            "match": matches,
            "value": value,
            "margin": round(margin, 4),
        }

        if not matches:
            all_match = False
        total_margin += margin

    return all_match, total_margin, per_channel


def find_instances(
    segments: list,
    requirements: dict,
    top_k: int = 20,
    relaxation_steps: int = 3,
) -> dict:
    """Find the top-K segments matching a prototype's requirements.

    If fewer than top_k segments match, progressively relaxes the least
    important requirements (those with operator=null are already excluded).

    Args:
        segments: Full list of segment dicts from Part A
        requirements: channel_requirements dict from prototype spec
        top_k: Number of instances to select
        relaxation_steps: How many times to try relaxing thresholds

    Returns:
        Dict with matching results and selected indices
    """
    # Phase 1: Strict matching
    active_reqs = {ch: req for ch, req in requirements.items()
                   if req.get("operator") is not None}

    matches = []
    for seg in segments:
        all_match, margin, details = match_segment(seg, active_reqs)
        if all_match:
            matches.append({
                "segment_id": seg["segment_id"],
                "start_idx": seg["start_idx"],
                "end_idx": seg["end_idx"],
                "margin": margin,
                "details": details,
            })

    print(f"  Strict matching: {len(matches)} segments match all {len(active_reqs)} requirements")

    # Phase 2: Relaxation if needed
    relaxation_log = []
    if len(matches) < top_k and relaxation_steps > 0:
        # Rank requirements by how many segments they individually reject
        rejection_counts = {}
        for ch, req in active_reqs.items():
            n_reject = 0
            for seg in segments:
                single_match, _, _ = match_segment(seg, {ch: req})
                if not single_match:
                    n_reject += 1
            rejection_counts[ch] = n_reject

        # Drop the most restrictive requirement first
        sorted_channels = sorted(rejection_counts.keys(),
                                 key=lambda c: rejection_counts[c], reverse=True)

        relaxed_reqs = dict(active_reqs)
        for step in range(min(relaxation_steps, len(sorted_channels))):
            if len(matches) >= top_k:
                break

            drop_ch = sorted_channels[step]
            relaxation_log.append({
                "step": step + 1,
                "dropped_channel": drop_ch,
                "rejection_count": rejection_counts[drop_ch],
            })
            print(f"  Relaxation step {step+1}: dropping '{drop_ch}' "
                  f"(rejected {rejection_counts[drop_ch]} segments)")
            del relaxed_reqs[drop_ch]

            matches = []
            for seg in segments:
                all_match, margin, details = match_segment(seg, relaxed_reqs)
                if all_match:
                    matches.append({
                        "segment_id": seg["segment_id"],
                        "start_idx": seg["start_idx"],
                        "end_idx": seg["end_idx"],
                        "margin": margin,
                        "details": details,
                    })
            print(f"    Now: {len(matches)} matches")

    # Phase 3: Rank by margin, select top-K
    matches.sort(key=lambda m: m["margin"], reverse=True)
    selected = matches[:top_k]

    return {
        "n_total_segments": len(segments),
        "n_strict_matches": len([m for m in matches]),
        "n_selected": len(selected),
        "top_k": top_k,
        "relaxation_log": relaxation_log,
        "selected_instances": selected,
    }


def compute_prototype_init(
    df: pd.DataFrame,
    selected_instances: list,
    input_length: int = 384,
    output_length: int = 96,
) -> dict:
    """Compute prototype_value (p) initialization from selected instances.

    Averages the forecast window (last 96 steps) of the target variable
    across all selected instances.

    Also extracts the raw data windows for later encoder-based μ initialization.

    Args:
        df: Full training dataframe
        selected_instances: List of dicts with start_idx and end_idx
        input_length: Lookback window length
        output_length: Forecast window length

    Returns:
        Dict with:
          - prototype_value: averaged forecast curve [output_length]
          - prototype_value_std: std across instances [output_length]
          - raw_windows: list of raw data arrays for encoder pass
    """
    target_col = "load"
    forecast_curves = []
    raw_lookback_windows = []

    for inst in selected_instances:
        start = inst["start_idx"]
        end = inst["end_idx"]
        forecast_start = start + input_length

        # Forecast curve (target only)
        forecast = df[target_col].iloc[forecast_start:end].values
        if len(forecast) == output_length:
            forecast_curves.append(forecast)

        # Full lookback window (all channels, for encoder pass later)
        lookback = df.iloc[start:forecast_start].values
        raw_lookback_windows.append(lookback)

    if not forecast_curves:
        return {"error": "No valid forecast windows found"}

    forecast_curves = np.array(forecast_curves)  # [K, 96]

    # Trimmed mean: drop top/bottom 10% per timestep for robustness
    if len(forecast_curves) >= 5:
        trim_n = max(1, len(forecast_curves) // 10)
        sorted_curves = np.sort(forecast_curves, axis=0)
        trimmed = sorted_curves[trim_n:-trim_n]
        proto_value = trimmed.mean(axis=0)
    else:
        proto_value = forecast_curves.mean(axis=0)

    proto_std = forecast_curves.std(axis=0)

    return {
        "prototype_value": proto_value.tolist(),
        "prototype_value_std": proto_std.tolist(),
        "n_instances": len(forecast_curves),
        "individual_means": forecast_curves.mean(axis=1).tolist(),
    }


def run_instance_finding(
    spec_path: str,
    segments_path: str,
    data_path: str,
    output_dir: str,
    top_k: int = 20,
    input_length: int = 384,
    output_length: int = 96,
):
    """Run the full instance finding pipeline for one prototype spec."""
    os.makedirs(output_dir, exist_ok=True)

    # Load prototype spec
    print(f"Loading prototype spec from {spec_path}...")
    with open(spec_path) as f:
        spec = json.load(f)
    print(f"  Prototype: {spec['name']}")
    print(f"  Description: {spec['description'][:100]}...")

    # Load full segment bank
    print(f"\nLoading segments from {segments_path}...")
    with open(segments_path) as f:
        seg_data = json.load(f)
    segments = seg_data["segments"]
    print(f"  Total segments: {len(segments)}")

    # Find matching instances
    print(f"\nFinding instances (top_k={top_k})...")
    results = find_instances(
        segments,
        spec["channel_requirements"],
        top_k=top_k,
    )
    print(f"  Selected: {results['n_selected']} instances")

    if results["n_selected"] == 0:
        print("ERROR: No matching instances found. Check requirements.")
        return

    # Load raw data and compute prototype_value
    print(f"\nComputing prototype_value from raw data...")
    df = pd.read_csv(data_path)
    train_end = int(0.7 * len(df))
    train_df = df.iloc[:train_end]

    proto_init = compute_prototype_init(
        train_df,
        results["selected_instances"],
        input_length=input_length,
        output_length=output_length,
    )
    print(f"  Averaged {proto_init['n_instances']} forecast curves")
    proto_value = np.array(proto_init["prototype_value"])
    print(f"  Prototype value: mean={proto_value.mean():.0f}, "
          f"min={proto_value.min():.0f}, max={proto_value.max():.0f}")

    # Save results
    output = {
        "prototype_spec": spec,
        "instance_finding": results,
        "prototype_init": {
            "prototype_value": proto_init["prototype_value"],
            "prototype_value_std": proto_init["prototype_value_std"],
            "n_instances": proto_init["n_instances"],
            "individual_means": proto_init["individual_means"],
        },
        "selected_segment_indices": [
            inst["start_idx"] for inst in results["selected_instances"]
        ],
    }

    output_filename = f"{spec['name']}_instances.json"
    output_path = os.path.join(output_dir, output_filename)
    print(f"\nSaving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Also save the prototype_value as a standalone .npy for easy loading
    npy_path = os.path.join(output_dir, f"{spec['name']}_proto_value.npy")
    np.save(npy_path, proto_value)
    print(f"Saved prototype_value to {npy_path}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Instance Finding Complete: {spec['name']}")
    print("=" * 60)
    print(f"  Matched: {results['n_selected']}/{len(segments)} segments")
    print(f"  Forecast curve mean: {proto_value.mean():.0f} MW")
    print(f"  Forecast curve range: {proto_value.min():.0f} - {proto_value.max():.0f} MW")
    if results["relaxation_log"]:
        print(f"  Relaxation steps: {len(results['relaxation_log'])}")
        for r in results["relaxation_log"]:
            print(f"    - Dropped '{r['dropped_channel']}'")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Part B/C: Find instances matching a prototype spec"
    )
    parser.add_argument("--spec_path", type=str, required=True,
                        help="Path to prototype spec JSON")
    parser.add_argument("--segments_path", type=str, required=True,
                        help="Path to full segments JSON from Part A")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to raw dataset CSV")
    parser.add_argument("--output_dir", type=str, default="./LLM_prototype/output",
                        help="Directory to save output")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of instances to select")
    parser.add_argument("--input_length", type=int, default=384)
    parser.add_argument("--output_length", type=int, default=96)
    args = parser.parse_args()

    run_instance_finding(
        spec_path=args.spec_path,
        segments_path=args.segments_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        input_length=args.input_length,
        output_length=args.output_length,
    )


if __name__ == "__main__":
    main()

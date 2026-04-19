"""
Segment Matcher: Evaluates prototype specs against the segment bank.

For each prototype, this module:
1. Checks every segment against the prototype's channel_requirements
2. Counts how many segments match
3. Ranks matches by margin (how strongly they satisfy thresholds)
4. Tags the prototype as "good" or "low_quality" based on match count
5. Identifies the bottleneck requirement (the one rejecting the most segments)

Used by the main loop to validate LLM-proposed prototypes.
"""

import numpy as np


def evaluate_requirement(value, requirement: dict) -> tuple:
    """Check if a value satisfies a channel requirement.

    Returns:
        (matches: bool, margin: float)
    """
    op = requirement.get("operator")
    if op is None:
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
        low, high = requirement["low"], requirement["high"]
        in_range = low <= value <= high
        margin = min(value - low, high - value) if in_range else 0.0
        return in_range, margin
    else:
        raise ValueError(f"Unknown operator: {op}")


def _get_stat_value(ch_stats: dict, requirement: dict) -> float:
    """Extract the right statistic from channel stats based on requirement's 'stat' field.

    Defaults to 'mean' for continuous channels, 'mode' for discrete channels.
    Supports: mean, median, std, min, max, range, slope, mode.
    """
    stat_key = requirement.get("stat", None)

    if stat_key is not None:
        if stat_key in ch_stats:
            return ch_stats[stat_key]
        else:
            return None

    # Default: mean for continuous, mode for discrete
    if "mean" in ch_stats:
        return ch_stats["mean"]
    elif "mode" in ch_stats:
        return ch_stats["mode"]
    return None


def match_segment(segment: dict, requirements: dict) -> tuple:
    """Check if a segment matches all active channel requirements.

    Returns:
        (all_match: bool, total_margin: float, per_channel: dict)
    """
    lb_stats = segment["lookback_stats"]
    per_channel = {}
    total_margin = 0.0
    all_match = True

    for channel, req in requirements.items():
        if req.get("operator") is None:
            continue

        if channel not in lb_stats:
            per_channel[channel] = {"match": False, "reason": "channel_missing"}
            all_match = False
            continue

        value = _get_stat_value(lb_stats[channel], req)
        if value is None:
            per_channel[channel] = {"match": False, "reason": "stat_missing"}
            all_match = False
            continue

        matches, margin = evaluate_requirement(value, req)
        per_channel[channel] = {
            "match": matches,
            "value": round(value, 4),
            "margin": round(margin, 4),
        }
        if not matches:
            all_match = False
        total_margin += margin

    return all_match, total_margin, per_channel


def analyze_bottleneck(segments: list, requirements: dict) -> dict:
    """For each active requirement, count how many segments it individually rejects.

    Returns a dict mapping channel_name -> {
        "rejection_count": int,
        "rejection_pct": float,
        "pass_count": int,
        "sample_values": list of actual values from segments (for LLM context)
    }
    """
    active_reqs = {ch: req for ch, req in requirements.items()
                   if req.get("operator") is not None}

    analysis = {}
    for ch, req in active_reqs.items():
        reject_count = 0
        pass_count = 0
        sample_values = []

        for seg in segments:
            lb_stats = seg["lookback_stats"]
            if ch not in lb_stats:
                reject_count += 1
                continue

            value = _get_stat_value(lb_stats[ch], req)
            if value is None:
                reject_count += 1
                continue

            matches, _ = evaluate_requirement(value, req)
            if matches:
                pass_count += 1
            else:
                reject_count += 1

            # Collect sample values for the LLM's reference
            if len(sample_values) < 20:
                sample_values.append(round(value, 2))

        analysis[ch] = {
            "rejection_count": reject_count,
            "rejection_pct": round(reject_count / len(segments) * 100, 1),
            "pass_count": pass_count,
            "threshold": req.get("threshold", req.get("values")),
            "operator": req.get("operator"),
            "percentiles": _compute_percentiles(segments, ch, req),
        }

    return analysis


def _compute_percentiles(segments: list, channel: str, req: dict) -> dict:
    """Compute percentiles of the channel's values across all segments.

    Helps the LLM understand the data distribution to set better thresholds.
    """
    stat_key = req.get("stat", None)
    values = []
    for seg in segments:
        lb_stats = seg["lookback_stats"]
        if channel not in lb_stats:
            continue
        val = _get_stat_value(lb_stats[channel], req)
        if val is not None:
            values.append(val)

    if not values:
        return {}

    values = np.array(values)
    return {
        "p10": round(float(np.percentile(values, 10)), 2),
        "p25": round(float(np.percentile(values, 25)), 2),
        "p50": round(float(np.percentile(values, 50)), 2),
        "p75": round(float(np.percentile(values, 75)), 2),
        "p90": round(float(np.percentile(values, 90)), 2),
        "min": round(float(values.min()), 2),
        "max": round(float(values.max()), 2),
    }


def evaluate_prototype(
    prototype_spec: dict,
    segments: list,
    top_k: int = 10,
    min_match_ratio: float = 0.5,
) -> dict:
    """Full evaluation of a single prototype spec against the segment bank.

    Args:
        prototype_spec: A prototype dict with "channel_requirements"
        segments: Full list of segments from Part A
        top_k: Desired number of instances per prototype
        min_match_ratio: Minimum ratio of matches to top_k (below this = "low_quality")

    Returns:
        Dict with:
          - status: "good" or "low_quality"
          - n_matches: total segments matching all requirements
          - min_required: top_k * min_match_ratio
          - selected_indices: start_idx of top-K matched segments
          - bottleneck: analysis of which requirement is most restrictive
          - matched_load_stats: mean/std of matched segments' load (for quality check)
    """
    requirements = prototype_spec.get("channel_requirements", {})
    min_required = int(top_k * min_match_ratio)

    # Match all segments
    matches = []
    for seg in segments:
        all_match, margin, details = match_segment(seg, requirements)
        if all_match:
            matches.append({
                "segment_id": seg["segment_id"],
                "start_idx": seg["start_idx"],
                "end_idx": seg["end_idx"],
                "margin": margin,
            })

    # Sort by margin, select top-K
    matches.sort(key=lambda m: m["margin"], reverse=True)
    selected = matches[:top_k]

    # Determine status
    n_matches = len(matches)
    status = "good" if n_matches >= min_required else "low_quality"

    # Bottleneck analysis (always compute — useful even for "good" prototypes)
    bottleneck = analyze_bottleneck(segments, requirements)

    # Find the single most restrictive requirement
    worst_channel = None
    worst_rejection = 0
    for ch, analysis in bottleneck.items():
        if analysis["rejection_count"] > worst_rejection:
            worst_rejection = analysis["rejection_count"]
            worst_channel = ch

    result = {
        "prototype_id": prototype_spec.get("prototype_id"),
        "name": prototype_spec.get("name", "unnamed"),
        "status": status,
        "n_matches": n_matches,
        "n_total_segments": len(segments),
        "top_k": top_k,
        "min_required": min_required,
        "selected_indices": [m["start_idx"] for m in selected],
        "bottleneck_analysis": bottleneck,
        "worst_bottleneck": {
            "channel": worst_channel,
            "rejection_pct": bottleneck[worst_channel]["rejection_pct"] if worst_channel else 0,
            "threshold": bottleneck[worst_channel]["threshold"] if worst_channel else None,
            "percentiles": bottleneck[worst_channel]["percentiles"] if worst_channel else {},
        } if worst_channel else None,
    }

    return result


def evaluate_all_prototypes(
    prototype_specs: list,
    segments: list,
    top_k: int = 10,
    min_match_ratio: float = 0.5,
) -> dict:
    """Evaluate all N prototypes and return a summary.

    Returns:
        Dict with:
          - results: list of per-prototype evaluation results
          - n_good: count of "good" prototypes
          - n_low_quality: count of "low_quality" prototypes
          - all_good: bool, True if all prototypes pass
          - low_quality_ids: list of prototype_ids that need revision
    """
    results = []
    for spec in prototype_specs:
        result = evaluate_prototype(spec, segments, top_k, min_match_ratio)
        results.append(result)

    good = [r for r in results if r["status"] == "good"]
    low_quality = [r for r in results if r["status"] == "low_quality"]

    return {
        "results": results,
        "n_good": len(good),
        "n_low_quality": len(low_quality),
        "all_good": len(low_quality) == 0,
        "low_quality_ids": [r["prototype_id"] for r in low_quality],
    }

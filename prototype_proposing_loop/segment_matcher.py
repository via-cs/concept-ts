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


def _forecast_coherence_cv(matches_top_k: list, segments_by_id: dict,
                           target_variable: str) -> float:
    """Coefficient of variation of the forecast-window target mean
    across the top-K selected segments. Low CV = coherent forecasts.

    CV = std(forecast_mean_i) / |mean(forecast_mean_i)|  (over i = 1..K)

    Returns None if there aren't enough segments, or the target channel
    isn't continuous in forecast_stats.
    """
    if not matches_top_k:
        return None

    means = []
    for m in matches_top_k:
        seg = segments_by_id.get(m["segment_id"])
        if seg is None:
            continue
        fc = seg.get("forecast_stats", {}).get(target_variable, {})
        if "mean" in fc:
            means.append(fc["mean"])

    if len(means) < 2:
        return None

    arr = np.asarray(means, dtype=float)
    mu = arr.mean()
    sd = arr.std(ddof=0)
    denom = max(abs(mu), 1e-9)
    return float(sd / denom)


def _discriminative_requirements(bottleneck: dict, min_rejection: float) -> dict:
    """From bottleneck analysis, list requirements whose individual
    rejection rate is at least `min_rejection` (as a fraction, not pct).

    Returns dict:
      - n_discriminative: int
      - discriminative: list of (channel, rejection_pct)
      - weak: list of (channel, rejection_pct) with rejection_pct < min_rejection*100
    """
    cutoff_pct = min_rejection * 100.0
    discriminative, weak = [], []
    for ch, an in bottleneck.items():
        entry = (ch, an["rejection_pct"])
        if an["rejection_pct"] >= cutoff_pct:
            discriminative.append(entry)
        else:
            weak.append(entry)
    return {
        "n_discriminative": len(discriminative),
        "discriminative": discriminative,
        "weak": weak,
    }


def evaluate_prototype(
    prototype_spec: dict,
    segments: list,
    top_k: int = 10,
    min_match_ratio: float = 0.5,
    target_variable: str = "load",
    segments_by_id: dict = None,
) -> dict:
    """Evaluate a single prototype spec. Returns per-prototype metrics
    needed by the quality rubric. Inter-prototype checks (Jaccard) are
    deferred to evaluate_all_prototypes.

    Args:
        prototype_spec: A prototype dict with "channel_requirements"
        segments: Full list of segments from Part A
        top_k: Desired number of instances per prototype
        min_match_ratio: K_min / top_k — below this a prototype fails criterion 1
        target_variable: Name of the forecast target channel (for coherence CV)
        segments_by_id: Optional precomputed {segment_id: segment} dict
    """
    requirements = prototype_spec.get("channel_requirements", {})
    min_required = int(top_k * min_match_ratio)

    if segments_by_id is None:
        segments_by_id = {s["segment_id"]: s for s in segments}

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

    n_matches = len(matches)
    n_total = len(segments)
    generality = n_matches / n_total if n_total > 0 else 0.0

    # Bottleneck analysis (always compute — useful even for "good" prototypes)
    bottleneck = analyze_bottleneck(segments, requirements)

    # Find the single most restrictive requirement
    worst_channel = None
    worst_rejection = 0
    for ch, analysis in bottleneck.items():
        if analysis["rejection_count"] > worst_rejection:
            worst_rejection = analysis["rejection_count"]
            worst_channel = ch

    # Forecast coherence CV (criterion 4)
    forecast_cv = _forecast_coherence_cv(selected, segments_by_id, target_variable)

    result = {
        "prototype_id": prototype_spec.get("prototype_id"),
        "name": prototype_spec.get("name", "unnamed"),
        "n_matches": n_matches,
        "n_total_segments": n_total,
        "top_k": top_k,
        "min_required": min_required,
        "generality": round(generality, 4),
        "forecast_coherence_cv": None if forecast_cv is None else round(forecast_cv, 4),
        "selected_indices": [m["start_idx"] for m in selected],
        "matched_segment_ids": [m["segment_id"] for m in matches],
        "bottleneck_analysis": bottleneck,
        "worst_bottleneck": {
            "channel": worst_channel,
            "rejection_pct": bottleneck[worst_channel]["rejection_pct"] if worst_channel else 0,
            "threshold": bottleneck[worst_channel]["threshold"] if worst_channel else None,
            "percentiles": bottleneck[worst_channel]["percentiles"] if worst_channel else {},
        } if worst_channel else None,
    }

    return result


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u > 0 else 0.0


def _apply_rubric(
    per_proto: list,
    min_required: int,
    generality_cap: float,
    max_jaccard: float,
    max_forecast_cv: float,
    min_active_reqs: int,
    min_req_rejection: float,
) -> list:
    """Apply the full quality rubric. Mutates each result in `per_proto`
    with:
        - failure_modes: list[str] drawn from
            {"too_few", "too_general", "overlap", "incoherent", "weak_requirements"}
        - status: "good" | "low_quality"
        - max_jaccard_overlap: float
        - max_jaccard_with: prototype_id of the overlap culprit (or None)
        - discriminative_requirements: dict with n_discriminative / weak lists

    Jaccard is computed against *earlier-accepted* prototypes only, in
    input order. An accepted prototype is one whose final status is
    "good" (i.e. all criteria passed).
    """
    accepted_sets = []    # list of (pid, set_of_matched_ids) for good prototypes

    for r in per_proto:
        fails = []

        # Criterion 1: K_min ≤ n_matches
        if r["n_matches"] < min_required:
            fails.append("too_few")

        # Criterion 2: generality cap
        if r["generality"] > generality_cap:
            fails.append("too_general")

        # Criterion 4: forecast coherence CV
        cv = r.get("forecast_coherence_cv")
        if cv is not None and cv > max_forecast_cv:
            fails.append("incoherent")

        # Criterion 6: ≥ min_active_reqs requirements with ≥ min_req_rejection each
        disc = _discriminative_requirements(
            r["bottleneck_analysis"], min_req_rejection
        )
        r["discriminative_requirements"] = disc
        if disc["n_discriminative"] < min_active_reqs:
            fails.append("weak_requirements")

        # Criterion 3: Jaccard overlap with any earlier-accepted prototype
        my_ids = set(r["matched_segment_ids"])
        best_jacc, best_pid = 0.0, None
        for (pid, prev_ids) in accepted_sets:
            j = _jaccard(my_ids, prev_ids)
            if j > best_jacc:
                best_jacc, best_pid = j, pid
        r["max_jaccard_overlap"] = round(best_jacc, 4)
        r["max_jaccard_with"] = best_pid
        if best_jacc > max_jaccard:
            fails.append("overlap")

        r["failure_modes"] = fails
        r["status"] = "good" if not fails else "low_quality"
        r["rubric_thresholds"] = {
            "min_required": min_required,
            "generality_cap": generality_cap,
            "max_jaccard": max_jaccard,
            "max_forecast_cv": max_forecast_cv,
            "min_active_reqs": min_active_reqs,
            "min_req_rejection": min_req_rejection,
        }

        if r["status"] == "good":
            accepted_sets.append((r["prototype_id"], my_ids))

    return per_proto


def evaluate_all_prototypes(
    prototype_specs: list,
    segments: list,
    top_k: int = 10,
    min_match_ratio: float = 0.5,
    *,
    generality_cap: float = 0.4,
    max_jaccard: float = 0.5,
    max_forecast_cv: float = 0.3,
    min_active_reqs: int = 2,
    min_req_rejection: float = 0.15,
    target_variable: str = "load",
) -> dict:
    """Evaluate all N prototypes with the full quality rubric.

    A prototype is "good" only if ALL of:
        (1) n_matches ≥ top_k * min_match_ratio
        (2) n_matches / total ≤ generality_cap
        (3) max Jaccard overlap with any earlier-accepted prototype ≤ max_jaccard
        (4) forecast coherence CV ≤ max_forecast_cv
        (6) ≥ min_active_reqs requirements with per-requirement
            rejection rate ≥ min_req_rejection

    Returns:
        Dict with:
          - results: list of per-prototype evaluation results (with
            `failure_modes` and new rubric fields)
          - n_good: count of "good" prototypes
          - n_low_quality: count of "low_quality" prototypes
          - all_good: bool
          - low_quality_ids: list of prototype_ids that need revision
          - rubric: the threshold config used (for downstream logging)
    """
    segments_by_id = {s["segment_id"]: s for s in segments}
    min_required = int(top_k * min_match_ratio)

    # Phase 1: per-prototype metrics (no cross-prototype dependencies yet)
    per_proto = []
    for spec in prototype_specs:
        r = evaluate_prototype(
            spec, segments,
            top_k=top_k,
            min_match_ratio=min_match_ratio,
            target_variable=target_variable,
            segments_by_id=segments_by_id,
        )
        per_proto.append(r)

    # Phase 2: apply the full rubric (including Jaccard against earlier accepted)
    results = _apply_rubric(
        per_proto,
        min_required=min_required,
        generality_cap=generality_cap,
        max_jaccard=max_jaccard,
        max_forecast_cv=max_forecast_cv,
        min_active_reqs=min_active_reqs,
        min_req_rejection=min_req_rejection,
    )

    good = [r for r in results if r["status"] == "good"]
    low_quality = [r for r in results if r["status"] == "low_quality"]

    return {
        "results": results,
        "n_good": len(good),
        "n_low_quality": len(low_quality),
        "all_good": len(low_quality) == 0,
        "low_quality_ids": [r["prototype_id"] for r in low_quality],
        "rubric": {
            "min_match_ratio": min_match_ratio,
            "min_required": min_required,
            "top_k": top_k,
            "generality_cap": generality_cap,
            "max_jaccard": max_jaccard,
            "max_forecast_cv": max_forecast_cv,
            "min_active_reqs": min_active_reqs,
            "min_req_rejection": min_req_rejection,
            "target_variable": target_variable,
        },
    }

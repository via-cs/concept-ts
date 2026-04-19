"""
Feedback Generator: Converts segment matcher results into structured
feedback text that the LLM can use to revise its prototype specs.

The feedback is designed to be:
1. Specific — tells the LLM exactly which prototype and which requirement is the problem
2. Data-grounded — includes actual percentiles so the LLM can set realistic thresholds
3. Actionable — suggests concrete threshold relaxations
"""


def generate_feedback(evaluation_summary: dict) -> str:
    """Generate a complete feedback report for the LLM.

    Args:
        evaluation_summary: Output from segment_matcher.evaluate_all_prototypes()

    Returns:
        A formatted string to include in the LLM revision prompt
    """
    results = evaluation_summary["results"]
    n_good = evaluation_summary["n_good"]
    n_low = evaluation_summary["n_low_quality"]
    total = len(results)

    lines = []
    lines.append("=" * 60)
    lines.append("PROTOTYPE EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Total prototypes: {total}")
    lines.append(f"Good: {n_good} | Need revision: {n_low}")
    lines.append("")

    for r in results:
        lines.append("-" * 60)
        lines.append(f"Prototype {r['prototype_id']}: \"{r['name']}\"")
        lines.append(f"  Status: {r['status'].upper()}")
        lines.append(f"  Matched segments: {r['n_matches']} / {r['n_total_segments']}")
        lines.append(f"  Required minimum: {r['min_required']} (top_k={r['top_k']})")

        if r["status"] == "low_quality":
            lines.append("")
            lines.append("  >>> REVISION NEEDED <<<")
            lines.append(_format_bottleneck_feedback(r))
        else:
            lines.append("  No revision needed.")

        lines.append("")

    # Overall summary for the LLM
    if n_low > 0:
        lines.append("=" * 60)
        lines.append("ACTION REQUIRED")
        lines.append("=" * 60)
        low_ids = evaluation_summary["low_quality_ids"]
        lines.append(f"Please revise the following prototypes: {low_ids}")
        lines.append("For each, relax the bottleneck requirement using the percentile")
        lines.append("data provided above. Keep the prototype's conceptual identity,")
        lines.append("but adjust thresholds so it matches at least "
                     f"{results[0]['min_required']} segments.")
        lines.append("")
        lines.append("Return the COMPLETE list of all prototypes (both revised and")
        lines.append("unchanged) in the same JSON format as before.")
    else:
        lines.append("=" * 60)
        lines.append("ALL PROTOTYPES PASSED — no revision needed.")
        lines.append("=" * 60)

    return "\n".join(lines)


def _format_bottleneck_feedback(result: dict) -> str:
    """Format detailed bottleneck feedback for a single low-quality prototype."""
    lines = []

    bottleneck = result.get("bottleneck_analysis", {})
    worst = result.get("worst_bottleneck")

    # Show per-requirement breakdown
    lines.append("  Requirement-by-requirement breakdown:")
    for ch, analysis in bottleneck.items():
        status_icon = "✗" if analysis["rejection_pct"] > 50 else "✓"
        lines.append(
            f"    {status_icon} {ch}: "
            f"operator={analysis['operator']}, threshold={analysis['threshold']} "
            f"→ rejects {analysis['rejection_pct']}% of segments "
            f"(passes {analysis['pass_count']})"
        )
        # Show percentiles for context
        pcts = analysis.get("percentiles", {})
        if pcts:
            lines.append(
                f"      Data distribution: "
                f"min={pcts.get('min')}, p10={pcts.get('p10')}, "
                f"p25={pcts.get('p25')}, p50={pcts.get('p50')}, "
                f"p75={pcts.get('p75')}, p90={pcts.get('p90')}, "
                f"max={pcts.get('max')}"
            )

    # Highlight the worst bottleneck with a suggestion
    if worst and worst["channel"]:
        lines.append("")
        lines.append(f"  PRIMARY BOTTLENECK: {worst['channel']}")
        lines.append(f"    This requirement alone rejects {worst['rejection_pct']}% of segments.")

        pcts = worst.get("percentiles", {})
        if pcts:
            suggestion = _suggest_relaxation(worst, pcts)
            if suggestion:
                lines.append(f"    SUGGESTION: {suggestion}")

    return "\n".join(lines)


def _suggest_relaxation(worst: dict, percentiles: dict) -> str:
    """Generate a concrete threshold suggestion based on percentile data."""
    op = worst.get("channel")
    threshold = worst.get("threshold")

    if threshold is None or not percentiles:
        return None

    # For requirements that are too strict, suggest moving toward the median
    if isinstance(threshold, (int, float)):
        p25 = percentiles.get("p25")
        p75 = percentiles.get("p75")
        p50 = percentiles.get("p50")

        if p25 is not None and p75 is not None:
            return (
                f"The current threshold {threshold} is very restrictive. "
                f"Consider using a value between p25={p25} and p75={p75} "
                f"(median={p50}) to capture a more representative set."
            )

    return None


def generate_compact_feedback(evaluation_summary: dict) -> str:
    """Generate a shorter feedback string focused only on problems.

    Use this for later loop iterations to save tokens.
    """
    results = evaluation_summary["results"]
    low_quality = [r for r in results if r["status"] == "low_quality"]

    if not low_quality:
        return "ALL PROTOTYPES PASSED. No revision needed."

    lines = []
    lines.append(f"{len(low_quality)} prototypes need revision:\n")

    for r in low_quality:
        worst = r.get("worst_bottleneck", {})
        ch = worst.get("channel", "unknown")
        pcts = worst.get("percentiles", {})

        lines.append(
            f"- Prototype {r['prototype_id']} \"{r['name']}\": "
            f"only {r['n_matches']} matches (need {r['min_required']}). "
            f"Bottleneck: {ch} rejects {worst.get('rejection_pct', '?')}%. "
            f"Data p25={pcts.get('p25')}, p50={pcts.get('p50')}, p75={pcts.get('p75')}."
        )

    lines.append("\nRevise these prototypes and return the full list.")
    return "\n".join(lines)

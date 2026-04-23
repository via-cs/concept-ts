"""
Feedback Generator: Converts segment matcher results into structured
feedback text that the LLM can use to revise its prototype specs.

The feedback is designed to be:
1. Specific — tells the LLM exactly which prototype and which criterion failed
2. Data-grounded — includes actual percentiles / CV / overlap numbers so the
   LLM can set realistic thresholds
3. Actionable — every failure mode is paired with a concrete fix direction

Failure modes understood here:
    too_few              — criterion 1: n_matches < min_required
    too_general          — criterion 2: n_matches / total > generality_cap
    overlap              — criterion 3: Jaccard vs an earlier-accepted
                           prototype > max_jaccard
    incoherent           — criterion 4: forecast CV over top-K > max_forecast_cv
    weak_requirements    — criterion 6: < min_active_reqs requirements with
                           per-requirement rejection rate ≥ min_req_rejection
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
    rubric = evaluation_summary.get("rubric", {})

    lines = []
    lines.append("=" * 60)
    lines.append("PROTOTYPE EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Total prototypes: {total}")
    lines.append(f"Good: {n_good} | Need revision: {n_low}")
    if rubric:
        lines.append("")
        lines.append("Rubric thresholds:")
        lines.append(
            f"  (1) n_matches >= {rubric.get('min_required')} "
            f"(top_k={rubric.get('top_k')}, min_match_ratio={rubric.get('min_match_ratio')})"
        )
        lines.append(
            f"  (2) n_matches / total_segments <= {rubric.get('generality_cap')}"
        )
        lines.append(
            f"  (3) max Jaccard vs any earlier-accepted prototype <= "
            f"{rubric.get('max_jaccard')}"
        )
        lines.append(
            f"  (4) forecast-mean CV over top-K <= {rubric.get('max_forecast_cv')} "
            f"(target = {rubric.get('target_variable')})"
        )
        lines.append(
            f"  (6) >= {rubric.get('min_active_reqs')} requirements, each "
            f"rejecting >= {int(round(rubric.get('min_req_rejection', 0) * 100))}% "
            f"of segments"
        )
    lines.append("")

    for r in results:
        lines.append("-" * 60)
        lines.append(f"Prototype {r['prototype_id']}: \"{r['name']}\"")
        lines.append(f"  Status: {r['status'].upper()}")
        lines.append(
            f"  Matched segments: {r['n_matches']} / {r['n_total_segments']} "
            f"(generality = {r['generality']})"
        )
        lines.append(f"  Required minimum: {r['min_required']} (top_k={r['top_k']})")
        cv = r.get("forecast_coherence_cv")
        lines.append(
            f"  Forecast-mean CV over top-K: "
            f"{'n/a' if cv is None else cv}"
        )
        disc = r.get("discriminative_requirements", {})
        if disc:
            lines.append(
                f"  Discriminative requirements (rejection >= cutoff): "
                f"{disc.get('n_discriminative', 0)}"
            )
        lines.append(
            f"  Max Jaccard w/ earlier accepted: "
            f"{r.get('max_jaccard_overlap', 0.0)} "
            f"(vs prototype {r.get('max_jaccard_with')})"
        )

        if r["status"] == "low_quality":
            lines.append("")
            lines.append(
                f"  >>> REVISION NEEDED — failed: "
                f"{', '.join(r['failure_modes'])} <<<"
            )
            lines.append(_format_failure_feedback(r))
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
        lines.append(
            "For each, address EVERY listed failure_mode. Keep the prototype's "
            "conceptual identity but adjust thresholds and channel requirements "
            "so it passes all of criteria (1), (2), (3), (4), (6) above."
        )
        lines.append("")
        lines.append("Return the COMPLETE list of all prototypes (both revised and")
        lines.append("unchanged) in the same JSON format as before.")
    else:
        lines.append("=" * 60)
        lines.append("ALL PROTOTYPES PASSED — no revision needed.")
        lines.append("=" * 60)

    return "\n".join(lines)


# ------------------------------------------------------------------
# Per-failure-mode formatters
# ------------------------------------------------------------------


def _format_failure_feedback(result: dict) -> str:
    """Dispatch to the right per-failure-mode formatter and concatenate."""
    blocks = []
    for mode in result["failure_modes"]:
        fn = _FAILURE_FORMATTERS.get(mode)
        if fn is None:
            blocks.append(f"    ! unknown failure_mode: {mode}")
            continue
        blocks.append(fn(result))
    return "\n".join(blocks)


def _format_too_few(r: dict) -> str:
    lines = ["  [FAIL 1 — too_few]"]
    lines.append(
        f"    Only {r['n_matches']} segments match (need >= {r['min_required']})."
    )
    bottleneck = r.get("bottleneck_analysis", {})
    worst = r.get("worst_bottleneck")
    if bottleneck:
        lines.append("    Requirement-by-requirement rejection:")
        for ch, an in bottleneck.items():
            icon = "X" if an["rejection_pct"] > 50 else "."
            lines.append(
                f"      {icon} {ch}: op={an['operator']}, thr={an['threshold']} "
                f"→ rejects {an['rejection_pct']}% (passes {an['pass_count']})"
            )
            pcts = an.get("percentiles", {})
            if pcts:
                lines.append(
                    f"        data: min={pcts.get('min')}, p10={pcts.get('p10')}, "
                    f"p25={pcts.get('p25')}, p50={pcts.get('p50')}, "
                    f"p75={pcts.get('p75')}, p90={pcts.get('p90')}, "
                    f"max={pcts.get('max')}"
                )
    if worst and worst["channel"]:
        lines.append(f"    PRIMARY BOTTLENECK: {worst['channel']} "
                     f"(alone rejects {worst['rejection_pct']}%).")
        suggestion = _suggest_relaxation(worst)
        if suggestion:
            lines.append(f"    SUGGESTION: {suggestion}")
    return "\n".join(lines)


def _format_too_general(r: dict) -> str:
    cap = r["rubric_thresholds"]["generality_cap"]
    lines = ["  [FAIL 2 — too_general]"]
    lines.append(
        f"    This spec matches {r['n_matches']} / {r['n_total_segments']} "
        f"segments ({r['generality'] * 100:.1f}%), exceeding the "
        f"{cap * 100:.0f}% generality cap."
    )
    # Identify which requirements are "free" (low rejection) — those are the
    # ones the LLM should tighten.
    soft = []
    for ch, an in r.get("bottleneck_analysis", {}).items():
        if an["rejection_pct"] < 30:
            soft.append((ch, an))
    if soft:
        lines.append("    Loosely-restricted requirements (rejection < 30%):")
        for ch, an in soft:
            pcts = an.get("percentiles", {})
            lines.append(
                f"      - {ch}: op={an['operator']}, thr={an['threshold']} "
                f"→ only rejects {an['rejection_pct']}%"
            )
            if pcts:
                lines.append(
                    f"        data: p25={pcts.get('p25')}, p50={pcts.get('p50')}, "
                    f"p75={pcts.get('p75')}"
                )
    lines.append(
        "    SUGGESTION: tighten one or more of the soft requirements "
        "(e.g., move thresholds from p25/p75 toward p10/p90), or add a "
        "new discriminative requirement, so n_matches lands between "
        f"{r['min_required']} and {int(r['n_total_segments'] * cap)}."
    )
    return "\n".join(lines)


def _format_overlap(r: dict) -> str:
    cap = r["rubric_thresholds"]["max_jaccard"]
    lines = ["  [FAIL 3 — overlap]"]
    lines.append(
        f"    Jaccard overlap with prototype {r.get('max_jaccard_with')} is "
        f"{r.get('max_jaccard_overlap', 0.0):.3f}, exceeding cap {cap}."
    )
    lines.append(
        "    This prototype is mostly redundant with an earlier-accepted one "
        "(it selects largely the same segments)."
    )
    lines.append(
        "    SUGGESTION: differentiate this prototype conceptually — pick a "
        "regime the overlapping prototype does NOT cover. Concretely: flip the "
        "sign/direction of one key requirement, or replace a requirement with "
        "one operating on a different channel (e.g. temperature → humidity, "
        "or weekday → weekend)."
    )
    return "\n".join(lines)


def _format_incoherent(r: dict) -> str:
    cap = r["rubric_thresholds"]["max_forecast_cv"]
    cv = r.get("forecast_coherence_cv")
    lines = ["  [FAIL 4 — incoherent]"]
    lines.append(
        f"    Forecast-mean CV across the top-{r['top_k']} matched segments is "
        f"{cv}, exceeding cap {cap}."
    )
    lines.append(
        "    The matched segments produce very different future curves — i.e. "
        "the spec is not selecting a coherent behavioral regime, so averaging "
        "their forecasts into p_i will wash out."
    )
    lines.append(
        "    SUGGESTION: add or tighten a requirement that captures the "
        "operating-regime driver (time-of-day / season / weekday-weekend / "
        "pricing tier). If the spec already constrains these, the residual "
        "variance likely comes from a continuous covariate — tighten its range."
    )
    return "\n".join(lines)


def _format_weak_requirements(r: dict) -> str:
    thr = r["rubric_thresholds"]
    cutoff_pct = int(round(thr["min_req_rejection"] * 100))
    need = thr["min_active_reqs"]
    lines = ["  [FAIL 6 — weak_requirements]"]
    disc = r.get("discriminative_requirements", {})
    n_disc = disc.get("n_discriminative", 0)
    lines.append(
        f"    Only {n_disc} requirement(s) individually reject "
        f">= {cutoff_pct}% of segments; need >= {need}."
    )
    strong = disc.get("discriminative", [])
    weak = disc.get("weak", [])
    if strong:
        lines.append("    Discriminative requirements already present:")
        for ch, pct in strong:
            lines.append(f"      + {ch}: rejects {pct}%")
    if weak:
        lines.append("    Non-discriminative requirements "
                     f"(< {cutoff_pct}% rejection):")
        for ch, pct in weak:
            lines.append(f"      - {ch}: rejects {pct}%")
    lines.append(
        "    SUGGESTION: either tighten one of the weak requirements until it "
        "individually rejects >= {cutoff}% of segments, or add a new "
        "requirement on a different channel that does so. The prototype must "
        "be genuinely selective on at least {need} axes.".format(
            cutoff=cutoff_pct, need=need
        )
    )
    return "\n".join(lines)


_FAILURE_FORMATTERS = {
    "too_few": _format_too_few,
    "too_general": _format_too_general,
    "overlap": _format_overlap,
    "incoherent": _format_incoherent,
    "weak_requirements": _format_weak_requirements,
}


def _suggest_relaxation(worst: dict) -> str:
    """Generate a concrete threshold suggestion based on percentile data."""
    threshold = worst.get("threshold")
    percentiles = worst.get("percentiles", {})

    if threshold is None or not percentiles:
        return None

    if isinstance(threshold, (int, float)):
        p25 = percentiles.get("p25")
        p75 = percentiles.get("p75")
        p50 = percentiles.get("p50")

        if p25 is not None and p75 is not None:
            return (
                f"The current threshold {threshold} is very restrictive. "
                f"Consider a value between p25={p25} and p75={p75} "
                f"(median={p50}) to capture a more representative set."
            )
    return None


# ------------------------------------------------------------------
# Compact feedback
# ------------------------------------------------------------------


def generate_compact_feedback(evaluation_summary: dict) -> str:
    """Shorter feedback for later iterations — still surfaces every failure
    mode with concrete numbers, but drops the full requirement breakdown."""
    results = evaluation_summary["results"]
    low_quality = [r for r in results if r["status"] == "low_quality"]

    if not low_quality:
        return "ALL PROTOTYPES PASSED. No revision needed."

    lines = []
    lines.append(f"{len(low_quality)} prototypes need revision:\n")

    for r in low_quality:
        modes = r["failure_modes"]
        bits = []
        thr = r["rubric_thresholds"]
        if "too_few" in modes:
            bits.append(
                f"too_few: {r['n_matches']}<{thr['min_required']}"
            )
        if "too_general" in modes:
            bits.append(
                f"too_general: {r['generality'] * 100:.1f}%>"
                f"{thr['generality_cap'] * 100:.0f}%"
            )
        if "overlap" in modes:
            bits.append(
                f"overlap: J={r.get('max_jaccard_overlap', 0):.2f}>{thr['max_jaccard']} "
                f"(vs #{r.get('max_jaccard_with')})"
            )
        if "incoherent" in modes:
            bits.append(
                f"incoherent: CV={r.get('forecast_coherence_cv')}>"
                f"{thr['max_forecast_cv']}"
            )
        if "weak_requirements" in modes:
            disc = r.get("discriminative_requirements", {})
            bits.append(
                f"weak_requirements: only "
                f"{disc.get('n_discriminative', 0)} discriminative "
                f"req(s) (need {thr['min_active_reqs']})"
            )

        worst = r.get("worst_bottleneck") or {}
        worst_hint = ""
        if "too_few" in modes and worst.get("channel"):
            pcts = worst.get("percentiles", {})
            worst_hint = (
                f"  primary bottleneck: {worst['channel']} "
                f"(rejects {worst['rejection_pct']}%, p25={pcts.get('p25')}, "
                f"p50={pcts.get('p50')}, p75={pcts.get('p75')})"
            )

        lines.append(
            f"- P{r['prototype_id']} \"{r['name']}\" [{' | '.join(bits)}]"
        )
        if worst_hint:
            lines.append(worst_hint)

    lines.append("\nRevise every listed prototype so ALL failure modes are fixed. "
                 "Return the full prototype list.")
    return "\n".join(lines)

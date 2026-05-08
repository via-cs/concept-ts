"""
Feedback generator for the function-based propose-validate-revise loop.

For each low-quality prototype, emit concrete numbers + a one-line suggestion
per failure mode. Failure modes (from segment_matcher.py):

    execution_error     prototype's function crashed or returned junk
    too_few             n_matches < min_required
    too_general         generality > generality_cap
    overlap             Jaccard with an earlier-accepted prototype > max_jaccard
    incoherent          forecast-mean CV across top-K > max_forecast_cv
    weak_spec           too few summary-vector dims showed std-shrinkage
                        (the spec doesn't constrain enough features)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Per-failure-mode formatters
# ---------------------------------------------------------------------------

def _fmt_execution_error(r: dict) -> str:
    err = r.get("execution_error", "?")
    return (
        "  [FAIL — execution_error]\n"
        f"    The function failed to run. Trace:\n      {err}\n"
        "    SUGGESTION: rewrite the function. Make sure it is named "
        "`select_segments`, takes `segments: list[dict]`, returns `list[int]`, "
        "and uses `.get(...)` for every nested field access."
    )


def _fmt_too_few(r: dict) -> str:
    thr = r["rubric_thresholds"]
    return (
        "  [FAIL — too_few]\n"
        f"    Only {r['n_matches']} segments match (need >= {thr['min_required']} "
        f"= top_k * min_match_ratio = {thr['top_k']} * {thr['min_match_ratio']}).\n"
        "    SUGGESTION: relax one or more thresholds in the function body. "
        "Use the cluster summaries / global percentiles in the original prompt "
        "to pick more realistic cutoffs."
    )


def _fmt_too_general(r: dict) -> str:
    thr = r["rubric_thresholds"]
    return (
        "  [FAIL — too_general]\n"
        f"    Spec matches {r['n_matches']} / {r['n_total_segments']} = "
        f"{r['generality'] * 100:.1f}% of segments, exceeding the "
        f"{thr['generality_cap'] * 100:.0f}% cap.\n"
        "    SUGGESTION: tighten the function — add a missing feature "
        "constraint (e.g. season, day-of-week, holiday flag) or narrow an "
        "existing range. Aim for ~3-15% generality."
    )


def _fmt_overlap(r: dict) -> str:
    thr = r["rubric_thresholds"]
    return (
        "  [FAIL — overlap]\n"
        f"    Jaccard overlap with prototype {r.get('max_jaccard_with')} is "
        f"{r.get('max_jaccard_overlap', 0):.3f}, exceeding cap {thr['max_jaccard']}.\n"
        "    SUGGESTION: this prototype is selecting largely the same segments "
        "as an earlier-accepted one. Differentiate it conceptually — flip the "
        "direction of one key predicate (cold→hot, weekday→weekend) or replace "
        "it with one operating on a different channel."
    )


def _fmt_incoherent(r: dict) -> str:
    thr = r["rubric_thresholds"]
    return (
        "  [FAIL — incoherent]\n"
        f"    Forecast-mean CV across the top-{thr['top_k']} matched segments is "
        f"{r.get('forecast_coherence_cv')}, exceeding cap {thr['max_forecast_cv']}.\n"
        "    SUGGESTION: the matched segments produce inconsistent forecasts, "
        "so averaging them into the prototype curve will smear. Add or tighten "
        "a regime-defining predicate (time-of-day, season, weekday/weekend) so "
        "all matches truly share a forecast shape."
    )


def _fmt_weak_spec(r: dict) -> str:
    thr = r["rubric_thresholds"]
    sharp = r.get("spec_sharpness", {})
    return (
        "  [FAIL — weak_spec]\n"
        f"    Only {sharp.get('n_constrained', 0)} summary-vector dimension(s) "
        f"shrink to <= 50% of their global std under this spec; need at least "
        f"{thr['min_constrained_dims']}.\n"
        "    Translation: the spec barely constrains the matched cohort — the "
        "K segments still vary a lot across most features. The prototype is "
        "not selective enough.\n"
        "    SUGGESTION: add at least one more discriminative predicate "
        "(constrain a new channel that wasn't bound), or tighten an existing "
        "loose range, until the matched cohort is genuinely concentrated."
    )


_FORMATTERS = {
    "execution_error": _fmt_execution_error,
    "too_few":         _fmt_too_few,
    "too_general":     _fmt_too_general,
    "overlap":         _fmt_overlap,
    "incoherent":      _fmt_incoherent,
    "weak_spec":       _fmt_weak_spec,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_feedback(evaluation: dict) -> str:
    """Full per-prototype feedback report for the LLM revision step."""
    results = evaluation["results"]
    rubric = evaluation.get("rubric", {})

    lines = []
    lines.append("=" * 60)
    lines.append("PROTOTYPE EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Total: {len(results)}    "
                 f"good: {evaluation['n_good']}    "
                 f"need revision: {evaluation['n_low_quality']}")
    if rubric:
        lines.append("")
        lines.append("Rubric thresholds:")
        lines.append(f"  n_matches >= {rubric.get('min_required')}  "
                     f"(top_k={rubric.get('top_k')}, "
                     f"min_match_ratio={rubric.get('min_match_ratio')})")
        lines.append(f"  generality <= {rubric.get('generality_cap')}")
        lines.append(f"  max Jaccard vs earlier accepted <= {rubric.get('max_jaccard')}")
        lines.append(f"  forecast_coherence_cv <= {rubric.get('max_forecast_cv')}")
        lines.append(f"  spec_sharpness.n_constrained >= {rubric.get('min_constrained_dims')}")
    lines.append("")

    for r in results:
        lines.append("-" * 60)
        lines.append(f"Prototype {r['prototype_id']}: \"{r['name']}\"")
        lines.append(f"  Status: {r['status'].upper()}")
        lines.append(f"  n_matches: {r['n_matches']} / {r['n_total_segments']}  "
                     f"(generality = {r['generality']})")
        lines.append(f"  forecast_coherence_cv: {r.get('forecast_coherence_cv')}")
        lines.append(f"  spec_sharpness.n_constrained: "
                     f"{r.get('spec_sharpness', {}).get('n_constrained', 0)}")
        lines.append(f"  max_jaccard_overlap: {r.get('max_jaccard_overlap', 0)} "
                     f"(vs prototype {r.get('max_jaccard_with')})")

        if r["status"] == "low_quality":
            lines.append("")
            lines.append(f"  >>> REVISION NEEDED — failed: "
                         f"{', '.join(r['failure_modes'])} <<<")
            for mode in r["failure_modes"]:
                fn = _FORMATTERS.get(mode)
                if fn is not None:
                    lines.append(fn(r))
        else:
            lines.append("  No revision needed.")
        lines.append("")

    if evaluation["n_low_quality"] > 0:
        lines.append("=" * 60)
        lines.append("ACTION REQUIRED")
        lines.append("=" * 60)
        lines.append(f"Revise prototypes: {evaluation['low_quality_ids']}")
        lines.append("Address EVERY listed failure mode for each. Return the "
                     "complete prototype list as JSON.")
    else:
        lines.append("=" * 60)
        lines.append("ALL PROTOTYPES PASSED")
        lines.append("=" * 60)

    return "\n".join(lines)


def generate_compact_feedback(evaluation: dict) -> str:
    """One-line-per-prototype version for later iterations."""
    low = [r for r in evaluation["results"] if r["status"] == "low_quality"]
    if not low:
        return "ALL PROTOTYPES PASSED. No revision needed."

    lines = [f"{len(low)} prototypes need revision:"]
    for r in low:
        thr = r["rubric_thresholds"]
        bits = []
        for mode in r["failure_modes"]:
            if mode == "too_few":
                bits.append(f"too_few: {r['n_matches']} < {thr['min_required']}")
            elif mode == "too_general":
                bits.append(f"too_general: {r['generality'] * 100:.1f}% > "
                             f"{thr['generality_cap'] * 100:.0f}%")
            elif mode == "overlap":
                bits.append(f"overlap: J={r.get('max_jaccard_overlap', 0):.2f} > "
                             f"{thr['max_jaccard']} (vs #{r.get('max_jaccard_with')})")
            elif mode == "incoherent":
                bits.append(f"incoherent: cv={r.get('forecast_coherence_cv')} > "
                             f"{thr['max_forecast_cv']}")
            elif mode == "weak_spec":
                n = r.get("spec_sharpness", {}).get("n_constrained", 0)
                bits.append(f"weak_spec: n_constrained={n} < "
                             f"{thr['min_constrained_dims']}")
            elif mode == "execution_error":
                bits.append("execution_error")
        lines.append(f"  - P{r['prototype_id']} \"{r['name']}\": "
                     f"[{' | '.join(bits)}]")
    lines.append("Revise so every listed failure is fixed. Return the full "
                 "prototype list as JSON.")
    return "\n".join(lines)

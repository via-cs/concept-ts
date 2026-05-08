"""
Segment Matcher (function-based version).

Given a list of prototype specs (each carrying a Python `select_segments`
function as a string) and the segment bank, this module:

  1. Runs each prototype's function to get its matching index set.
  2. Computes per-prototype rubric metrics (n_matches, generality,
     forecast coherence CV, spec sharpness).
  3. Computes inter-prototype Jaccard against earlier-accepted prototypes.
  4. Tags each prototype as "good" or "low_quality" with a list of failure
     modes drawn from:
         too_few              n_matches < min_required
         too_general          generality > generality_cap
         overlap              max Jaccard vs earlier accepted > max_jaccard
         incoherent           forecast-mean CV across top-K > max_forecast_cv
         weak_spec            fewer than min_constrained_dims summary-vector
                              dimensions show std-shrinkage of >= 0.5x global
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from function_executor import execute_select_function


# ---------------------------------------------------------------------------
# Per-prototype metrics
# ---------------------------------------------------------------------------

def _forecast_coherence_cv(matched_indices: list, segments: list,
                            target_variable: str, top_k: int) -> float | None:
    """CV of forecast_stats[target]['mean'] across the first top_k matches."""
    if not matched_indices:
        return None
    selection = matched_indices[:top_k]
    means = []
    for idx in selection:
        seg = segments[idx]
        m = seg.get("forecast_stats", {}).get(target_variable, {}).get("mean")
        if m is not None:
            means.append(float(m))
    if len(means) < 2:
        return None
    arr = np.asarray(means, dtype=np.float64)
    mu = arr.mean()
    sd = arr.std(ddof=0)
    return float(sd / max(abs(mu), 1e-9))


def _spec_sharpness(matched_indices: list, summary_vectors_z: np.ndarray,
                     shrink_threshold: float = 0.5) -> dict:
    """How many summary-vector dims show std-shrinkage in the matched cohort.

    Replaces the old "weak_requirements" criterion which inspected per-channel
    bottlenecks. A dim is "constrained" if std_K(dim) / std_N(dim) <= threshold.
    Since we standardise globally so std_N = 1 per dim, this simplifies to
    std_K(dim) <= threshold.
    """
    if not matched_indices:
        return {"n_constrained": 0, "constrained_dims": []}
    sub = summary_vectors_z[matched_indices]                 # (k, D)
    if sub.shape[0] < 2:
        return {"n_constrained": 0, "constrained_dims": []}
    std_K = sub.std(axis=0, ddof=0)                          # (D,)
    constrained = np.where(std_K <= shrink_threshold)[0].tolist()
    return {"n_constrained": int(len(constrained)),
            "constrained_dims": constrained}


def evaluate_prototype(spec: dict, segments: list,
                        summary_vectors_z: np.ndarray,
                        top_k: int, target_variable: str) -> dict:
    """Run the prototype's function + compute its per-prototype metrics."""
    function_code = spec.get("function", "")
    exec_result = execute_select_function(function_code, segments)
    matched = exec_result["matched_indices"]
    error = exec_result["error"]

    if matched is None:
        return {
            "prototype_id": spec.get("prototype_id"),
            "name": spec.get("name", "unnamed"),
            "execution_error": error,
            "matched_indices": [],
            "n_matches": 0,
            "n_total_segments": len(segments),
            "generality": 0.0,
            "forecast_coherence_cv": None,
            "spec_sharpness": {"n_constrained": 0, "constrained_dims": []},
        }

    n_matches = len(matched)
    n_total = len(segments)
    generality = n_matches / n_total if n_total else 0.0
    cv = _forecast_coherence_cv(matched, segments, target_variable, top_k)
    sharp = _spec_sharpness(matched, summary_vectors_z)

    return {
        "prototype_id": spec.get("prototype_id"),
        "name": spec.get("name", "unnamed"),
        "execution_error": None,
        "matched_indices": matched,
        "n_matches": n_matches,
        "n_total_segments": n_total,
        "generality": round(generality, 4),
        "forecast_coherence_cv": None if cv is None else round(cv, 4),
        "spec_sharpness": sharp,
    }


# ---------------------------------------------------------------------------
# Cross-prototype rubric
# ---------------------------------------------------------------------------

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _apply_rubric(per_proto: list, *, min_required: int, generality_cap: float,
                   max_jaccard: float, max_forecast_cv: float,
                   min_constrained_dims: int) -> list:
    accepted = []   # list of (pid, set_of_matched_ids) for prototypes already passed

    for r in per_proto:
        fails = []

        # Execution error → always low-quality (no recovery possible)
        if r["execution_error"] is not None:
            fails.append("execution_error")

        # 1) too_few
        if r["n_matches"] < min_required:
            fails.append("too_few")

        # 2) too_general
        if r["generality"] > generality_cap:
            fails.append("too_general")

        # 4) incoherent
        cv = r.get("forecast_coherence_cv")
        if cv is not None and cv > max_forecast_cv:
            fails.append("incoherent")

        # 5) weak_spec — too few constrained dims
        n_constr = r.get("spec_sharpness", {}).get("n_constrained", 0)
        if n_constr < min_constrained_dims:
            fails.append("weak_spec")

        # 3) overlap with earlier-accepted prototypes
        my_ids = set(r["matched_indices"])
        best_j, best_pid = 0.0, None
        for pid, prev in accepted:
            j = _jaccard(my_ids, prev)
            if j > best_j:
                best_j, best_pid = j, pid
        r["max_jaccard_overlap"] = round(best_j, 4)
        r["max_jaccard_with"] = best_pid
        if best_j > max_jaccard:
            fails.append("overlap")

        r["failure_modes"] = fails
        r["status"] = "good" if not fails else "low_quality"

        if r["status"] == "good":
            accepted.append((r["prototype_id"], my_ids))

    return per_proto


def evaluate_all_prototypes(prototypes: list, segments: list,
                             summary_vectors_z: np.ndarray,
                             *, top_k: int = 10, min_match_ratio: float = 0.5,
                             generality_cap: float = 0.4, max_jaccard: float = 0.5,
                             max_forecast_cv: float = 0.3,
                             min_constrained_dims: int = 2,
                             target_variable: str = "load") -> dict:
    """Evaluate every prototype + apply the cross-prototype rubric."""
    min_required = int(top_k * min_match_ratio)

    per_proto = [
        evaluate_prototype(spec, segments, summary_vectors_z,
                           top_k=top_k, target_variable=target_variable)
        for spec in prototypes
    ]

    results = _apply_rubric(
        per_proto,
        min_required=min_required,
        generality_cap=generality_cap,
        max_jaccard=max_jaccard,
        max_forecast_cv=max_forecast_cv,
        min_constrained_dims=min_constrained_dims,
    )

    rubric = {
        "top_k": top_k,
        "min_match_ratio": min_match_ratio,
        "min_required": min_required,
        "generality_cap": generality_cap,
        "max_jaccard": max_jaccard,
        "max_forecast_cv": max_forecast_cv,
        "min_constrained_dims": min_constrained_dims,
        "target_variable": target_variable,
    }
    for r in results:
        r["rubric_thresholds"] = rubric

    n_good = sum(r["status"] == "good" for r in results)
    n_low = len(results) - n_good
    return {
        "results": results,
        "n_good": n_good,
        "n_low_quality": n_low,
        "all_good": n_low == 0,
        "low_quality_ids": [r["prototype_id"] for r in results
                            if r["status"] == "low_quality"],
        "rubric": rubric,
    }

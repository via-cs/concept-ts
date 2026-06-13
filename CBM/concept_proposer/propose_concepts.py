"""
Single-command LLM-driven concept proposal for the 1 + N CBM.

End-to-end flow
---------------

  1. Load Part-A segments and global statistics.
  2. Build per-segment summary vectors + K-means cluster into ``--n_clusters``
     regimes (reuses the prototype_proposing_loop summary_vector module).
  3. Render the prompt in ``prompts.INITIAL_PROMPT`` with global statistics
     + cluster summaries baked in. Asks Claude for TWO concept sets:
     ``lookback_concepts`` (size --n_lookback_concepts) and
     ``forecast_concepts`` (size --n_forecast_concepts, applied per
     sub-window in the CBM).
  4. Call Claude (Anthropic API) and parse the JSON object.
  5. Validate BOTH sets:
       - lookback rules run against each full segment dict
       - forecast rules run against each sub_seg (one per sub-window of each segment)
     Tag each rule with (positive_rate, n_positive, status) and print a
     two-table report.

Usage
-----

    export ANTHROPIC_API_KEY=...
    PYTHONPATH=. python -u CBM/concept_proposer/propose_concepts.py \\
        --n_lookback_concepts 10 \\
        --n_forecast_concepts 25 \\
        --n_clusters 20 \\
        --segments_path preprocessing/output/RE/RE_full_segments.json \\
        --output_dir CBM/output

Outputs (written to --output_dir):
  - concepts_raw_<TS>.json        : as returned by Claude
  - concepts_validated_<TS>.json  : same content + per-rule status tags
  - _last_prompt.md               : the rendered prompt (for inspection)

Pass ``concepts_validated_<TS>.json`` to ``CBM/train_cbm.py --concepts_path``.

Validate-only mode
------------------

To skip clustering + LLM and just re-validate an existing two-set concepts JSON:

    PYTHONPATH=. python -u CBM/concept_proposer/propose_concepts.py \\
        --validate_only \\
        --concepts_path CBM/output/concepts_raw_<TS>.json \\
        --segments_path preprocessing/output/RE/RE_full_segments.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Sibling-style imports so we don't trigger CBM/__init__.py (which imports torch).
_ROOT = Path(__file__).resolve().parent.parent.parent
_PROPOSER_DIR = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_PROPOSER_DIR) not in sys.path:
    sys.path.insert(0, str(_PROPOSER_DIR))

from prototype_proposing_loop.function_executor import _SAFE_BUILTINS

from prompts import INITIAL_PROMPT
from llm_caller import call_claude, extract_json_object, validate_concept_schema


# ===========================================================================
# Step 1 — clustering
# ===========================================================================

def _cluster_segments(segments_path: str,
                      output_dir: str,
                      n_clusters: int,
                      random_state: int = 0) -> dict:
    from preprocessing.data_preprocessing import CHANNEL_META
    from prototype_proposing_loop.summary_vector import build_and_cluster

    print("=" * 72)
    print(f"STAGE 1+2 — SUMMARY VECTORS & K-MEANS ({n_clusters} clusters)")
    print("=" * 72)
    artifacts_dir = os.path.join(output_dir, "cluster_artifacts")
    bundle = build_and_cluster(
        segments_path=segments_path,
        output_dir=artifacts_dir,
        n_clusters=n_clusters,
        channel_meta=CHANNEL_META,
        random_state=random_state,
    )
    return bundle


# ===========================================================================
# Step 2 — render the prompt
# ===========================================================================

def _render_prompt(n_lookback_concepts: int,
                   n_forecast_concepts: int,
                   cluster_bundle: dict,
                   metadata: dict) -> str:
    n_sw = int(metadata.get("n_subwindows", 4))
    out_len = int(metadata.get("output_length", 96))
    sw_len = int(metadata.get("subwindow_length", out_len // n_sw))
    sw_hours = sw_len * 15 / 60  # 15-min resolution

    return INITIAL_PROMPT.format(
        n_lookback_concepts=n_lookback_concepts,
        n_forecast_concepts=n_forecast_concepts,
        n_subwindows=n_sw,
        subwindow_length=sw_len,
        subwindow_hours=f"{sw_hours:.1f}",
        n_clusters=len(cluster_bundle["cluster_summaries"]),
        global_statistics=json.dumps(cluster_bundle["global_stats"], indent=2),
        cluster_summaries=json.dumps(cluster_bundle["cluster_summaries"], indent=2),
        channel_descriptions=json.dumps(metadata.get("channels", {}), indent=2),
        dataset_name=metadata.get("dataset_name", "LOF/RE"),
        target_variable=metadata.get("target_variable", "load"),
        target_description=metadata.get("target_description",
            "electricity load (MW) for a province in northern China"),
        time_resolution=metadata.get("time_resolution", "15 minutes"),
        input_length=metadata.get("input_length", 384),
        output_length=out_len,
        total_segments=metadata.get("total_segments", len(cluster_bundle["segments"])),
    )


# ===========================================================================
# Step 3 — call Claude
# ===========================================================================

def _call_llm(prompt: str,
              n_lookback_concepts: int,
              n_forecast_concepts: int) -> tuple[list[dict], list[dict]]:
    """Returns (lookback_concepts, forecast_concepts)."""
    print("\n" + "=" * 72)
    print(f"STAGE 3 — CALLING CLAUDE  "
          f"(requesting {n_lookback_concepts} lookback + {n_forecast_concepts} forecast concepts)")
    print("=" * 72)
    system_prompt = (
        "You are a domain expert in time-series forecasting. You produce "
        "Python filter functions and structured JSON. Always return a valid "
        "JSON object as your entire response — no surrounding prose."
    )
    response_text = call_claude(prompt, system_prompt=system_prompt)
    obj = extract_json_object(response_text)

    lb_list = obj.get("lookback_concepts")
    fc_list = obj.get("forecast_concepts")
    if not isinstance(lb_list, list) or not isinstance(fc_list, list):
        raise ValueError(
            "Claude response missing required keys 'lookback_concepts' and "
            f"'forecast_concepts'. Got keys: {list(obj.keys())}"
        )
    print(f"  Received {len(lb_list)} lookback + {len(fc_list)} forecast concepts.")
    if len(lb_list) != n_lookback_concepts:
        print(f"  WARNING: requested {n_lookback_concepts} lookback, got {len(lb_list)}.")
    if len(fc_list) != n_forecast_concepts:
        print(f"  WARNING: requested {n_forecast_concepts} forecast, got {len(fc_list)}.")

    for c in lb_list: validate_concept_schema(c)
    for c in fc_list: validate_concept_schema(c)
    return lb_list, fc_list


# ===========================================================================
# Step 4 — validation
# ===========================================================================

def _compile_match_function(code: str):
    safe_globals = {"__builtins__": _SAFE_BUILTINS}
    safe_locals: dict = {}
    try:
        compiled = compile(code, filename="<concept>", mode="exec")
        exec(compiled, safe_globals, safe_locals)
    except Exception as e:
        return None, f"compile/exec error: {type(e).__name__}: {e}"
    fn = safe_locals.get("matches")
    if fn is None:
        callables = [v for v in safe_locals.values() if callable(v)]
        if not callables:
            return None, "no `matches` (or any callable) found in function string"
        fn = callables[0]
    return fn, None


def _classify_status(positive_rate: float,
                      n_pos: int,
                      n_total: int,
                      runtime_err: str | None) -> tuple[str, str | None]:
    if runtime_err:
        return "runtime_error", runtime_err
    if n_pos == 0:
        return "warning_empty", "filter matched 0 inputs"
    if n_pos == n_total:
        return "warning_universal", "filter matched ALL inputs"
    if positive_rate < 0.05 or positive_rate > 0.60:
        return "warning_balance", f"positive rate {positive_rate} outside 5-60% band"
    return "ok", None


def _validate_one_lookback(c: dict, segments: list) -> dict:
    rec = dict(c)
    fn, err = _compile_match_function(c.get("function", ""))
    if err:
        rec["n_positive"]    = 0
        rec["positive_rate"] = 0.0
        rec["validation"]    = {"status": "error", "error": err}
        return rec
    n_pos, runtime_err = 0, None
    for seg in segments:
        try:
            if bool(fn(seg)):
                n_pos += 1
        except Exception as e:
            runtime_err = f"{type(e).__name__}: {e}"
            break
    rec["n_positive"]    = n_pos
    rec["positive_rate"] = round(n_pos / max(1, len(segments)), 4)
    status, errmsg = _classify_status(rec["positive_rate"], n_pos, len(segments), runtime_err)
    rec["validation"]   = {"status": status, "error": errmsg} if errmsg else {"status": status}
    return rec


def _build_sub_segs(segments: list) -> list:
    """Flatten segments into a list of sub_seg dicts: one per (seg, sub-window)."""
    sub_segs = []
    for seg in segments:
        cross = seg.get("cross_channel_stats", {})
        for k, sw in enumerate(seg.get("subwindow_stats") or []):
            sub_segs.append({
                "subwindow_idx":       int(sw.get("subwindow_idx", k)),
                "stats":               sw.get("stats", {}),
                "time_context":        sw.get("time_context", {}),
                "cross_channel_stats": cross,
            })
    return sub_segs


def _validate_one_forecast(c: dict, sub_segs: list) -> dict:
    rec = dict(c)
    fn, err = _compile_match_function(c.get("function", ""))
    if err:
        rec["n_positive"]    = 0
        rec["positive_rate"] = 0.0
        rec["validation"]    = {"status": "error", "error": err}
        return rec
    n_pos, runtime_err = 0, None
    for ss in sub_segs:
        try:
            if bool(fn(ss)):
                n_pos += 1
        except Exception as e:
            runtime_err = f"{type(e).__name__}: {e}"
            break
    rec["n_positive"]    = n_pos
    rec["positive_rate"] = round(n_pos / max(1, len(sub_segs)), 4)
    status, errmsg = _classify_status(rec["positive_rate"], n_pos, len(sub_segs), runtime_err)
    rec["validation"]   = {"status": status, "error": errmsg} if errmsg else {"status": status}
    return rec


def _status_flag(status: str) -> str:
    if status == "ok":            return " + "
    if status.startswith("warning"): return " ~ "
    return " x "


def _print_table(title: str, n_inputs_label: str, validated: list[dict]) -> dict[str, int]:
    bar = "─" * 96
    print(f"\n{title}  (evaluated against {n_inputs_label})")
    print(bar)
    print(f"     {'id':>3}  {'name':32s}  {'pos%':>6}  {'count':>6}  status")
    print(bar)
    for c in validated:
        status = c["validation"]["status"]
        print(f"  {_status_flag(status)}[{c['id']:>2}] {c['name']:32s}  "
              f"{c['positive_rate']*100:>5.1f}%  "
              f"{c['n_positive']:>6d}  {status}")
    print(bar)
    by_status: dict[str, int] = {}
    for c in validated:
        s = c["validation"]["status"]
        by_status[s] = by_status.get(s, 0) + 1
    n_usable = by_status.get("ok", 0) + by_status.get("warning_balance", 0)
    print(f"Summary: {by_status}    Usable (ok + warning_balance): {n_usable}/{len(validated)}")
    return by_status


def _validate_and_print(lookback_concepts: list[dict],
                        forecast_concepts: list[dict],
                        segments: list[dict]
                        ) -> tuple[list[dict], list[dict], dict, dict]:
    print("\n" + "=" * 72)
    print(f"STAGE 4 — VALIDATING CONCEPTS")
    print("=" * 72)

    sub_segs = _build_sub_segs(segments)
    n_sub = sum(len(s.get("subwindow_stats") or []) for s in segments)
    assert n_sub == len(sub_segs)

    lb_validated = [_validate_one_lookback(c, segments) for c in lookback_concepts]
    fc_validated = [_validate_one_forecast(c, sub_segs) for c in forecast_concepts]

    lb_summary = _print_table(
        "LOOKBACK concepts",
        f"{len(segments)} segments",
        lb_validated,
    )
    fc_summary = _print_table(
        "FORECAST concepts (one rule applied to each sub-window)",
        f"{len(segments)} segments x {n_sub // max(1, len(segments))} sub-windows = {n_sub} sub_segs",
        fc_validated,
    )
    return lb_validated, fc_validated, lb_summary, fc_summary


# ===========================================================================
# Step 5 — save
# ===========================================================================

def _save_results(lookback_raw, forecast_raw,
                  lookback_validated, forecast_validated,
                  output_dir: str, metadata_extra: dict
                  ) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_path = os.path.join(output_dir, f"concepts_raw_{timestamp}.json")
    with open(raw_path, "w") as f:
        json.dump({
            "metadata": {"timestamp": timestamp, **metadata_extra},
            "lookback_concepts": lookback_raw,
            "forecast_concepts": forecast_raw,
        }, f, indent=2)

    def _summarise(validated):
        by_status = {}
        for c in validated:
            s = c["validation"]["status"]
            by_status[s] = by_status.get(s, 0) + 1
        n_usable = by_status.get("ok", 0) + by_status.get("warning_balance", 0)
        return by_status, n_usable

    lb_summary, lb_usable = _summarise(lookback_validated)
    fc_summary, fc_usable = _summarise(forecast_validated)

    val_path = os.path.join(output_dir, f"concepts_validated_{timestamp}.json")
    with open(val_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "lookback_validation_summary": lb_summary,
                "forecast_validation_summary": fc_summary,
                "n_lookback_usable": lb_usable,
                "n_forecast_usable": fc_usable,
                **metadata_extra,
            },
            "lookback_concepts": lookback_validated,
            "forecast_concepts": forecast_validated,
        }, f, indent=2)

    print(f"\nWrote raw concepts       -> {raw_path}")
    print(f"Wrote validated concepts -> {val_path}")
    print(f"Pass --concepts_path {val_path} to CBM/train_cbm.py.")
    return raw_path, val_path


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n_lookback_concepts", type=int, default=10,
                   help="K_lb -- concepts for the lookback bottleneck.")
    p.add_argument("--n_forecast_concepts", type=int, default=25,
                   help="K_fc -- concepts shared across the forecast sub-window bottlenecks.")
    p.add_argument("--n_clusters",          type=int, default=20,
                   help="K-means cluster count for the prompt's regime summaries.")
    p.add_argument("--segments_path",       type=str, required=True,
                   help="Part-A full segments JSON (must include subwindow_stats blocks).")
    p.add_argument("--output_dir",          type=str, default="CBM/output")
    p.add_argument("--random_state",        type=int, default=0)
    p.add_argument("--validate_only",       action="store_true",
                   help="Skip clustering + LLM. Re-validate an existing two-set "
                        "concepts JSON at --concepts_path.")
    p.add_argument("--concepts_path",       type=str, default=None,
                   help="Required with --validate_only.")
    args = p.parse_args()

    # ----- Validate-only branch -----
    if args.validate_only:
        if not args.concepts_path:
            p.error("--concepts_path is required when --validate_only is set.")
        with open(args.concepts_path) as f:
            blob = json.load(f)
        lb_raw = blob.get("lookback_concepts") or []
        fc_raw = blob.get("forecast_concepts") or []
        with open(args.segments_path) as f:
            segments = json.load(f)["segments"]
        print(f"Loaded {len(lb_raw)} lookback + {len(fc_raw)} forecast concepts.")
        print(f"Loaded {len(segments)} segments.")
        lb_val, fc_val, _, _ = _validate_and_print(lb_raw, fc_raw, segments)
        _save_results(
            lookback_raw=lb_raw, forecast_raw=fc_raw,
            lookback_validated=lb_val, forecast_validated=fc_val,
            output_dir=args.output_dir,
            metadata_extra={
                "mode": "validate_only",
                "source_concepts": args.concepts_path,
                "segments_path":   args.segments_path,
            },
        )
        return

    # ----- Full propose+validate pipeline -----
    bundle = _cluster_segments(
        segments_path=args.segments_path,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
    )

    with open(args.segments_path) as f:
        seg_blob = json.load(f)
    metadata = seg_blob.get("metadata", {})

    prompt = _render_prompt(
        n_lookback_concepts=args.n_lookback_concepts,
        n_forecast_concepts=args.n_forecast_concepts,
        cluster_bundle=bundle,
        metadata=metadata,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    prompt_path = os.path.join(args.output_dir, "_last_prompt.md")
    with open(prompt_path, "w") as f:
        f.write(prompt)
    print(f"  (Rendered prompt saved to {prompt_path})")

    lb_raw, fc_raw = _call_llm(
        prompt,
        n_lookback_concepts=args.n_lookback_concepts,
        n_forecast_concepts=args.n_forecast_concepts,
    )
    lb_val, fc_val, _, _ = _validate_and_print(lb_raw, fc_raw, bundle["segments"])

    _save_results(
        lookback_raw=lb_raw, forecast_raw=fc_raw,
        lookback_validated=lb_val, forecast_validated=fc_val,
        output_dir=args.output_dir,
        metadata_extra={
            "mode": "propose",
            "n_lookback_concepts_requested": args.n_lookback_concepts,
            "n_forecast_concepts_requested": args.n_forecast_concepts,
            "n_clusters":                    args.n_clusters,
            "n_subwindows":                  metadata.get("n_subwindows"),
            "segments_path":                 args.segments_path,
            "dataset":                       metadata.get("dataset_name", "LOF/RE"),
        },
    )


if __name__ == "__main__":
    main()

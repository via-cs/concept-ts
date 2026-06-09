"""
Inspect + validate the concepts JSON that Claude wrote into the project.

Workflow (two commands total)
-----------------------------

  Step 1 (this script) — view and validate concepts:

      PYTHONPATH=. python -u CBM/concept_proposer/propose_concepts.py \\
          --concepts_input CBM/output/concepts.json \\
          --segments_path  preprocessing/output/RE/RE_full_segments.json

      Compiles each concept's ``matches(seg)`` filter in a sandboxed
      namespace, runs it against your training segments, and prints a
      summary table of (id, name, positive_rate, count, validation_status)
      so you can eyeball the proposed concepts. A copy of the input with
      validation tags attached is saved to
      ``CBM/output/concepts_validated_<TS>.json``.

  Step 2 — train (after you're happy with the concepts):

      PYTHONPATH=. python -u CBM/train_cbm.py \\
          --concepts_path CBM/output/concepts_validated_<TS>.json \\
          ... (other flags) ...

Validation tags attached to each concept
----------------------------------------
    ok                  positive_rate in [0.05, 0.60], runs without error
    warning_empty       matched 0 segments
    warning_universal   matched all segments
    warning_balance     positive_rate outside [0.05, 0.60]
    error               compile/exec exception (filter source is broken)
    runtime_error       the filter raised on some segment

``train_cbm.py`` only loads concepts with status ``ok`` or
``warning_balance``; the rest are skipped automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prototype_proposing_loop.function_executor import _SAFE_BUILTINS


# ---------------------------------------------------------------------------
# Sandboxed compile + per-segment evaluation
# ---------------------------------------------------------------------------

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
            return None, "no `matches` (or any callable) found in the function string"
        fn = callables[0]
    return fn, None


def validate_concepts(concepts: list, segments: list) -> list:
    n_seg = len(segments)
    out = []
    for c in concepts:
        rec = dict(c)
        fn, err = _compile_match_function(c.get("function", ""))
        if err:
            rec["n_positive"]    = 0
            rec["positive_rate"] = 0.0
            rec["validation"]    = {"status": "error", "error": err}
            out.append(rec); continue

        n_pos = 0
        runtime_err = None
        for seg in segments:
            try:
                if bool(fn(seg)):
                    n_pos += 1
            except Exception as e:
                runtime_err = f"{type(e).__name__}: {e}"
                break

        rec["n_positive"]    = n_pos
        rec["positive_rate"] = round(n_pos / max(1, n_seg), 4)
        if runtime_err:
            rec["validation"] = {"status": "runtime_error", "error": runtime_err}
        elif n_pos == 0:
            rec["validation"] = {"status": "warning_empty",
                                  "error": "filter matched 0 segments"}
        elif n_pos == n_seg:
            rec["validation"] = {"status": "warning_universal",
                                  "error": "filter matched ALL segments"}
        elif rec["positive_rate"] < 0.05 or rec["positive_rate"] > 0.60:
            rec["validation"] = {"status": "warning_balance",
                                  "error": f"positive rate {rec['positive_rate']} outside 5-60% band"}
        else:
            rec["validation"] = {"status": "ok"}
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concepts_input", type=str, required=True,
                   help="Path to the raw Claude-generated concepts JSON.")
    p.add_argument("--segments_path", type=str, required=True,
                   help="Path to <dataset>_full_segments.json (Part-A output).")
    p.add_argument("--output_dir", type=str, default="./CBM/output",
                   help="Where to write the validated concepts JSON.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load inputs
    with open(args.concepts_input) as f:
        blob = json.load(f)
    concepts = blob.get("concepts") if isinstance(blob, dict) else blob
    print(f"Loaded {len(concepts)} concepts from {args.concepts_input}")

    with open(args.segments_path) as f:
        segments = json.load(f)["segments"]
    print(f"Loaded {len(segments)} segments from {args.segments_path}\n")

    print(f"Validating each concept's matches(seg) against the segment bank ...\n")
    validated = validate_concepts(concepts, segments)

    # Header
    print("─" * 92)
    print(f"  {'id':>4}  {'name':30s}  {'pos%':>6}  {'count':>6}  status")
    print("─" * 92)
    for c in validated:
        status = c["validation"]["status"]
        flag = " ✓ " if status == "ok" else (" △ " if status.startswith("warning") else " ✗ ")
        print(f"  {flag}[{c['id']:>2}] {c['name']:30s}  "
              f"{c['positive_rate']*100:>5.1f}%  "
              f"{c['n_positive']:>6d}  {status}")
    print("─" * 92)

    by_status = {}
    for c in validated:
        s = c["validation"]["status"]
        by_status[s] = by_status.get(s, 0) + 1
    print(f"\nSummary: {by_status}")

    # Save validated copy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"concepts_validated_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "source_file": args.concepts_input,
                "n_concepts_input": len(concepts),
                "n_concepts_ok":    by_status.get("ok", 0),
                "validation_summary": by_status,
            },
            "concepts": validated,
        }, f, indent=2)
    print(f"\nWrote validated concepts to {out_path}")
    print(f"Use that path as --concepts_path in CBM/train_cbm.py.")


if __name__ == "__main__":
    main()

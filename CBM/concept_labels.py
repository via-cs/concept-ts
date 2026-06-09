"""
Compute concept labels for training/val/test segments and map them to the
sliding-window indexing used by ``LOF_Dataset``.

Public surface
--------------
    load_concepts(concepts_path)
        Read concepts JSON and return the list of OK-validated concepts and
        their compiled match functions.

    compute_segment_labels(segments, match_fns)
        Apply each match function to every segment → (N_segments, K) binary
        numpy array.

    segments_to_window_labels(seg_labels, n_windows, segment_stride)
        Broadcast per-segment labels to per-window labels. Window w uses the
        label of segment ⌊w / segment_stride⌋ — same convention as
        ``exp.cluster_supervision.compute_cluster_assignments_from_llm``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prototype_proposing_loop.function_executor import _SAFE_BUILTINS


# ---------------------------------------------------------------------------
# Load + compile
# ---------------------------------------------------------------------------

def _compile_one(code: str):
    safe_globals = {"__builtins__": _SAFE_BUILTINS}
    safe_locals: dict = {}
    compiled = compile(code, filename="<concept>", mode="exec")
    exec(compiled, safe_globals, safe_locals)
    fn = safe_locals.get("matches")
    if fn is None:
        callables = [v for v in safe_locals.values() if callable(v)]
        if not callables:
            raise ValueError("Concept function has no `matches` and no callable.")
        fn = callables[0]
    return fn


def load_concepts(concepts_path: str, only_ok: bool = True) -> tuple[list[dict], list]:
    """Load the concepts JSON and return (concept_metadata_list, match_functions).

    only_ok : when True, drop concepts whose validation.status != 'ok'. This is
              the safe default — concepts with status 'error', 'warning_empty',
              or 'warning_universal' produce useless labels (all zeros or all
              ones), which hurt training. 'warning_balance' concepts are kept
              by default since they often turn out to be useful even when
              their positive_rate is slightly outside the 5-60% band.
    """
    with open(concepts_path) as f:
        blob = json.load(f)
    raw_concepts = blob["concepts"]

    accepted, match_fns = [], []
    for c in raw_concepts:
        status = c.get("validation", {}).get("status", "ok")
        if only_ok and status not in ("ok", "warning_balance"):
            continue
        try:
            fn = _compile_one(c["function"])
        except Exception as e:
            print(f"  [skip] concept {c.get('name', '?')}: compile error: {e}")
            continue
        accepted.append(c)
        match_fns.append(fn)
    return accepted, match_fns


# ---------------------------------------------------------------------------
# Label computation
# ---------------------------------------------------------------------------

def compute_segment_labels(segments: list[dict], match_fns: list) -> np.ndarray:
    """Apply each match function to every segment. Returns (N_segments, K) binary array."""
    n, k = len(segments), len(match_fns)
    labels = np.zeros((n, k), dtype=np.float32)
    for i, seg in enumerate(segments):
        for j, fn in enumerate(match_fns):
            try:
                labels[i, j] = 1.0 if bool(fn(seg)) else 0.0
            except Exception:
                labels[i, j] = 0.0
    return labels


def segments_to_window_labels(seg_labels: np.ndarray,
                              n_windows: int,
                              segment_stride: int) -> np.ndarray:
    """Broadcast per-segment labels to per-window labels via floor division.

    Each dataset window w starts at row w and spans (input_length+output_length)
    rows. Segment k spans rows [k*stride, k*stride + segment_length), so we
    map window w to segment w // stride. Windows whose segment index is out
    of bounds get an all-zero label (rare; only at the tail).
    """
    n_seg, k = seg_labels.shape
    out = np.zeros((n_windows, k), dtype=np.float32)
    for w in range(n_windows):
        seg_idx = w // segment_stride
        if seg_idx < n_seg:
            out[w] = seg_labels[seg_idx]
    return out


# ---------------------------------------------------------------------------
# Convenience: end-to-end label table
# ---------------------------------------------------------------------------

def build_window_label_table(concepts_path: str,
                             segments_path: str,
                             n_windows: int,
                             segment_stride: int | None = None,
                             only_ok: bool = True) -> tuple[np.ndarray, list[dict]]:
    """Top-level helper used by train_cbm.py:
        1. Load + compile concepts.
        2. Apply each match function to every segment in segments_path.
        3. Broadcast to dataset windows.

    Returns (window_labels (n_windows, K), accepted_concepts_metadata)."""
    accepted, match_fns = load_concepts(concepts_path, only_ok=only_ok)
    if not accepted:
        raise RuntimeError(f"No OK concepts loaded from {concepts_path}.")
    print(f"  loaded {len(accepted)} concepts from {concepts_path}")

    with open(segments_path) as f:
        segments = json.load(f)["segments"]
    print(f"  loaded {len(segments)} segments from {segments_path}")

    if segment_stride is None:
        if len(segments) < 2:
            raise ValueError("Cannot infer segment_stride with <2 segments.")
        segment_stride = int(segments[1]["start_idx"] - segments[0]["start_idx"])
    print(f"  segment stride = {segment_stride}")

    seg_labels = compute_segment_labels(segments, match_fns)
    win_labels = segments_to_window_labels(seg_labels, n_windows, segment_stride)
    print(f"  per-segment positive rates: "
          f"{[round(float(seg_labels[:, j].mean()), 3) for j in range(len(accepted))]}")
    return win_labels, accepted

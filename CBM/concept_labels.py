"""
Compute concept labels for the 1 + N concept bottleneck CBM.

Two label sets per segment
--------------------------

  Lookback labels   shape (N_seg, K_lb)
    Each rule is a ``def matches(seg) -> bool`` predicate that consumes
    a full segment dict (the standard ``seg`` from Part-A preprocessing).
    Use these rules to probe the HISTORY -- e.g. ``is_history_cold`` checks
    ``seg['lookback_stats']['Skin_temperature']['mean'] < 0``.

  Forecast labels   shape (N_seg, K_sw, K_fc)
    Each rule is a ``def matches(sub_seg) -> bool`` predicate that consumes
    a sub-window dict produced by ``preprocessing/data_preprocessing.py``.
    A ``sub_seg`` has this shape:

        {
          "subwindow_idx":      int,
          "stats":              {channel -> {mean, std, slope, ...}},
          "time_context":       {subwindow_idx, start{...}, end{...}, season},
          "cross_channel_stats": dict (inherited from parent segment)
        }

    Use these rules to probe each forecast sub-window -- e.g.
    ``is_temperature_rising`` checks
    ``sub_seg['stats']['Skin_temperature']['slope'] > 0.05``.

Windows vs segments
-------------------
The CBM trainer consumes WINDOWS from LOF_Dataset (overlapping slices of
the full series), not SEGMENTS. We broadcast per-segment labels to
per-window labels with window w -> segment w // segment_stride, matching
the convention in ``exp/cluster_supervision.py``.
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


# ===========================================================================
# Compile + load
# ===========================================================================

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


def _filter_and_compile(concepts: list[dict],
                        only_ok: bool) -> tuple[list[dict], list]:
    accepted, match_fns = [], []
    for c in concepts:
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


def load_concept_sets(concepts_path: str,
                      only_ok: bool = True
                      ) -> tuple[tuple[list, list], tuple[list, list]]:
    """Load the two-set concepts JSON written by ``concept_proposer``.

    Returns
    -------
    (lookback_accepted, lookback_match_fns), (forecast_accepted, forecast_match_fns)

    Validated concepts JSON layout:
        {
          "metadata": {...},
          "lookback_concepts":  [<concept>, ...],
          "forecast_concepts":  [<concept>, ...]
        }
    """
    with open(concepts_path) as f:
        blob = json.load(f)
    if "lookback_concepts" not in blob or "forecast_concepts" not in blob:
        raise ValueError(
            f"{concepts_path} is not a two-set concepts file. Expected keys "
            f"'lookback_concepts' and 'forecast_concepts', got {list(blob.keys())}."
        )
    lb_accepted, lb_fns = _filter_and_compile(blob["lookback_concepts"], only_ok)
    fc_accepted, fc_fns = _filter_and_compile(blob["forecast_concepts"], only_ok)
    return (lb_accepted, lb_fns), (fc_accepted, fc_fns)


# ===========================================================================
# Build the per-segment / per-window label tensors
# ===========================================================================

def _safe_call(fn, x) -> int:
    try:
        return 1 if bool(fn(x)) else 0
    except Exception:
        return 0


def compute_segment_label_tensors(
    segments: list[dict],
    lookback_match_fns: list,
    forecast_match_fns: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply both sets of rules to every segment.

    Returns
    -------
    lb_labels : (N_seg, K_lb)
    fc_labels : (N_seg, K_sw, K_fc)
    """
    n_seg = len(segments)
    K_lb  = len(lookback_match_fns)
    K_fc  = len(forecast_match_fns)
    if n_seg == 0:
        return np.zeros((0, K_lb), dtype=np.float32), \
               np.zeros((0, 0, K_fc), dtype=np.float32)
    K_sw = len(segments[0].get("subwindow_stats") or [])
    if K_sw == 0:
        raise ValueError(
            "segments[0] has no 'subwindow_stats' field. Re-run preprocessing "
            "with --n_subwindows so each segment carries per-sub-window stats."
        )

    lb_labels = np.zeros((n_seg, K_lb),       dtype=np.float32)
    fc_labels = np.zeros((n_seg, K_sw, K_fc), dtype=np.float32)

    for i, seg in enumerate(segments):
        # Lookback labels: rule sees the whole segment dict.
        for j, fn in enumerate(lookback_match_fns):
            lb_labels[i, j] = _safe_call(fn, seg)

        # Forecast labels: rule sees a sub_seg dict per sub-window.
        cross = seg.get("cross_channel_stats", {})
        for k, sw in enumerate(seg.get("subwindow_stats", [])):
            sub_seg = {
                "subwindow_idx":       int(sw.get("subwindow_idx", k)),
                "stats":               sw.get("stats", {}),
                "time_context":        sw.get("time_context", {}),
                "cross_channel_stats": cross,
            }
            for j, fn in enumerate(forecast_match_fns):
                fc_labels[i, k, j] = _safe_call(fn, sub_seg)

    return lb_labels, fc_labels


# ===========================================================================
# Per-segment -> per-window broadcast
# ===========================================================================

def _broadcast_seg_to_windows(seg_labels: np.ndarray,
                              n_windows: int,
                              segment_stride: int) -> np.ndarray:
    """Window w uses the label of segment w // segment_stride.
    Trailing windows whose segment index is out of bounds get all-zero labels.
    Works for any trailing label-axis shape."""
    n_seg = seg_labels.shape[0]
    out = np.zeros((n_windows,) + seg_labels.shape[1:], dtype=np.float32)
    for w in range(n_windows):
        si = w // segment_stride
        if si < n_seg:
            out[w] = seg_labels[si]
    return out


def build_window_label_tables(
    concepts_path: str,
    segments_path: str,
    n_windows: int,
    segment_stride: int | None = None,
    only_ok: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    """Top-level helper used by ``train_cbm.py``.

    Returns
    -------
    lb_window_labels : (n_windows, K_lb)
    fc_window_labels : (n_windows, K_sw, K_fc)
    lb_accepted      : list of accepted lookback concept metadata dicts
    fc_accepted      : list of accepted forecast concept metadata dicts
    """
    (lb_accepted, lb_fns), (fc_accepted, fc_fns) = load_concept_sets(
        concepts_path, only_ok=only_ok
    )
    if not lb_accepted and not fc_accepted:
        raise RuntimeError(f"No OK concepts loaded from {concepts_path}.")
    print(f"  loaded {len(lb_accepted)} lookback + {len(fc_accepted)} forecast concepts")

    with open(segments_path) as f:
        segments = json.load(f)["segments"]
    print(f"  loaded {len(segments)} segments from {segments_path}")

    if segment_stride is None:
        if len(segments) < 2:
            raise ValueError("Cannot infer segment_stride with <2 segments.")
        segment_stride = int(segments[1]["start_idx"] - segments[0]["start_idx"])
    print(f"  segment stride = {segment_stride}")

    lb_seg, fc_seg = compute_segment_label_tensors(segments, lb_fns, fc_fns)
    print(f"  lookback label positive rates: "
          f"{[round(float(lb_seg[:, j].mean()), 3) for j in range(lb_seg.shape[1])]}")
    if fc_seg.size:
        fc_flat = fc_seg.reshape(-1, fc_seg.shape[-1])
        print(f"  forecast label positive rates (avg across sub-windows): "
              f"{[round(float(fc_flat[:, j].mean()), 3) for j in range(fc_flat.shape[1])]}")

    lb_win = _broadcast_seg_to_windows(lb_seg, n_windows, segment_stride)
    fc_win = _broadcast_seg_to_windows(fc_seg, n_windows, segment_stride)
    return lb_win, fc_win, lb_accepted, fc_accepted

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
                        only_ok: bool,
                        min_pos_rate: float = 0.0,
                        max_pos_rate: float = 1.0) -> tuple[list[dict], list]:
    """Filter concepts by validation status AND by positive-rate band.

    A concept is kept iff:
      - its validation status is in {ok, warning_balance} (when only_ok=True)
      - its positive_rate is in [min_pos_rate, max_pos_rate]
      - its function code compiles

    The positive-rate band exists to drop two failure modes the trainer is
    sensitive to:
      - Very-rare concepts (pos_rate < min_pos_rate, e.g. 0.08): the BCE
        loss can be minimized by trivially predicting False, contributing
        almost no useful gradient.
      - Near-universal concepts (pos_rate > max_pos_rate, e.g. 0.75):
        same story in reverse; trivial-baseline AUROC ~ 0.5 and the encoder
        just learns 'always say yes'. These also tend to be very brittle
        to distribution shift between train/val/test.
    """
    accepted, match_fns = [], []
    for c in concepts:
        status = c.get("validation", {}).get("status", "ok")
        if only_ok and status not in ("ok", "warning_balance"):
            continue
        pr = c.get("positive_rate")
        if pr is not None and (pr < min_pos_rate or pr > max_pos_rate):
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
                      only_ok: bool = True,
                      min_pos_rate: float = 0.0,
                      max_pos_rate: float = 1.0,
                      ) -> tuple[tuple[list, list], tuple[list, list], tuple[list, list]]:
    """Load the three-set concepts JSON written by ``concept_proposer``.

    Returns
    -------
    (lookback_accepted, lookback_match_fns),
    (forecast_accepted, forecast_match_fns),
    (forecast_global_accepted, forecast_global_match_fns)

    The third (``forecast_global_concepts``) is OPTIONAL for backward
    compatibility with older concepts JSONs. If the key is missing it returns
    ``([], [])`` and downstream code treats K_fc_global = 0 as "no global
    forecast bottleneck".
    """
    with open(concepts_path) as f:
        blob = json.load(f)
    if "lookback_concepts" not in blob or "forecast_concepts" not in blob:
        raise ValueError(
            f"{concepts_path} is missing required keys. Expected "
            f"'lookback_concepts' and 'forecast_concepts'; got {list(blob.keys())}."
        )
    lb_accepted, lb_fns = _filter_and_compile(
        blob["lookback_concepts"], only_ok, min_pos_rate, max_pos_rate,
    )
    fc_accepted, fc_fns = _filter_and_compile(
        blob["forecast_concepts"], only_ok, min_pos_rate, max_pos_rate,
    )
    # New optional set -- absent for older concept files; treated as empty.
    raw_global = blob.get("forecast_global_concepts", [])
    gfc_accepted, gfc_fns = _filter_and_compile(
        raw_global, only_ok, min_pos_rate, max_pos_rate,
    )
    return (lb_accepted, lb_fns), (fc_accepted, fc_fns), (gfc_accepted, gfc_fns)


# ===========================================================================
# Build the per-segment / per-window label tensors
# ===========================================================================

def _safe_call(fn, x) -> int:
    try:
        return 1 if bool(fn(x)) else 0
    except Exception:
        return 0


def _whole_forecast_sub_seg(seg: dict) -> dict:
    """Build a sub_seg-shaped dict that represents the WHOLE forecast window.

    Used for the single-forecast-bottleneck ablation: forecast concept rules
    are applied to the whole-forecast stats and a single segment-level time
    context, producing one label per (segment, concept) instead of one per
    (segment, sub-window, concept).
    """
    tc = seg.get("time_context", {})
    return {
        "subwindow_idx":       0,            # not meaningful in this mode
        "stats":               seg.get("forecast_stats", {}),
        "time_context": {
            "subwindow_idx": 0,
            "start":  tc.get("forecast_start", {}),
            "end":    tc.get("forecast_end",   {}),
            "season": tc.get("season"),
        },
        "cross_channel_stats": seg.get("cross_channel_stats", {}),
    }


def compute_segment_label_tensors(
    segments: list[dict],
    lookback_match_fns: list,
    forecast_match_fns: list,
    forecast_global_match_fns: list | None = None,
    single_forecast_bottleneck: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (lb_labels, fc_labels, gfc_labels).

    ``gfc_labels`` has shape (N_seg, K_fc_global); rules are applied to a
    whole-forecast sub_seg synthesized from the segment's ``forecast_stats``.
    If ``forecast_global_match_fns`` is None / empty, gfc_labels has shape
    (N_seg, 0) -- safe to ignore downstream when K_fc_global = 0.
    """
    forecast_global_match_fns = forecast_global_match_fns or []
    n_seg = len(segments)
    K_lb  = len(lookback_match_fns)
    K_fc  = len(forecast_match_fns)
    K_gfc = len(forecast_global_match_fns)
    if n_seg == 0:
        lb_zero  = np.zeros((0, K_lb),  dtype=np.float32)
        gfc_zero = np.zeros((0, K_gfc), dtype=np.float32)
        if single_forecast_bottleneck:
            return lb_zero, np.zeros((0, K_fc), dtype=np.float32), gfc_zero
        return lb_zero, np.zeros((0, 0, K_fc), dtype=np.float32), gfc_zero

    lb_labels  = np.zeros((n_seg, K_lb),  dtype=np.float32)
    gfc_labels = np.zeros((n_seg, K_gfc), dtype=np.float32)
    if single_forecast_bottleneck:
        fc_labels = np.zeros((n_seg, K_fc), dtype=np.float32)
    else:
        K_sw = len(segments[0].get("subwindow_stats") or [])
        if K_sw == 0:
            raise ValueError(
                "segments[0] has no 'subwindow_stats' field. Re-run preprocessing "
                "with --n_subwindows so each segment carries per-sub-window stats."
            )
        fc_labels = np.zeros((n_seg, K_sw, K_fc), dtype=np.float32)

    for i, seg in enumerate(segments):
        # Lookback labels: rule sees the whole segment dict.
        for j, fn in enumerate(lookback_match_fns):
            lb_labels[i, j] = _safe_call(fn, seg)

        if single_forecast_bottleneck:
            # Forecast labels: rule sees ONE whole-window sub_seg per segment.
            sub_seg = _whole_forecast_sub_seg(seg)
            for j, fn in enumerate(forecast_match_fns):
                fc_labels[i, j] = _safe_call(fn, sub_seg)
        else:
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

        # Global-forecast labels: rule sees a sub_seg covering the WHOLE
        # forecast window (synthesized from seg.forecast_stats and the
        # segment-level forecast_start/end time context).
        if K_gfc > 0:
            whole = _whole_forecast_sub_seg(seg)
            for j, fn in enumerate(forecast_global_match_fns):
                gfc_labels[i, j] = _safe_call(fn, whole)

    return lb_labels, fc_labels, gfc_labels


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
    single_forecast_bottleneck: bool = False,
    min_pos_rate: float = 0.0,
    max_pos_rate: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           list[dict], list[dict], list[dict]]:
    """Top-level helper used by ``train_cbm.py``.

    Returns
    -------
    lb_window_labels   : (n_windows, K_lb)
    fc_window_labels   :
        - shape (n_windows, K_sw, K_fc) when single_forecast_bottleneck=False
        - shape (n_windows, K_fc)       when single_forecast_bottleneck=True
    gfc_window_labels  : (n_windows, K_fc_global)  -- K_fc_global = 0 when the
                          concepts JSON has no ``forecast_global_concepts`` key
                          (backward compat).
    lb_accepted        : list of accepted lookback concept metadata dicts
    fc_accepted        : list of accepted forecast (sub-window) concept dicts
    gfc_accepted       : list of accepted global-forecast concept dicts
    """
    (lb_accepted, lb_fns), (fc_accepted, fc_fns), (gfc_accepted, gfc_fns) = \
        load_concept_sets(
            concepts_path, only_ok=only_ok,
            min_pos_rate=min_pos_rate, max_pos_rate=max_pos_rate,
        )
    if not lb_accepted and not fc_accepted and not gfc_accepted:
        raise RuntimeError(f"No OK concepts loaded from {concepts_path}.")
    print(f"  loaded {len(lb_accepted)} lookback + {len(fc_accepted)} forecast "
          f"+ {len(gfc_accepted)} forecast-global concepts "
          f"(pos_rate band [{min_pos_rate:.2f}, {max_pos_rate:.2f}])")

    with open(segments_path) as f:
        segments = json.load(f)["segments"]
    print(f"  loaded {len(segments)} segments from {segments_path}")

    if segment_stride is None:
        if len(segments) < 2:
            raise ValueError("Cannot infer segment_stride with <2 segments.")
        segment_stride = int(segments[1]["start_idx"] - segments[0]["start_idx"])
    print(f"  segment stride = {segment_stride}")

    lb_seg, fc_seg, gfc_seg = compute_segment_label_tensors(
        segments, lb_fns, fc_fns, gfc_fns,
        single_forecast_bottleneck=single_forecast_bottleneck,
    )
    print(f"  lookback label positive rates: "
          f"{[round(float(lb_seg[:, j].mean()), 3) for j in range(lb_seg.shape[1])]}")
    if fc_seg.size:
        fc_flat = fc_seg.reshape(-1, fc_seg.shape[-1])
        scope_note = ("whole forecast window"
                       if single_forecast_bottleneck else
                       "avg across sub-windows")
        print(f"  forecast label positive rates ({scope_note}): "
              f"{[round(float(fc_flat[:, j].mean()), 3) for j in range(fc_flat.shape[1])]}")
    if gfc_seg.size:
        print(f"  forecast-global label positive rates: "
              f"{[round(float(gfc_seg[:, j].mean()), 3) for j in range(gfc_seg.shape[1])]}")

    lb_win  = _broadcast_seg_to_windows(lb_seg,  n_windows, segment_stride)
    fc_win  = _broadcast_seg_to_windows(fc_seg,  n_windows, segment_stride)
    gfc_win = _broadcast_seg_to_windows(gfc_seg, n_windows, segment_stride)
    return lb_win, fc_win, gfc_win, lb_accepted, fc_accepted, gfc_accepted

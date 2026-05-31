"""
Part-A preprocessing pipeline for the Riverside air-quality dataset.

Mirrors ``data_preprocessing.py`` but adapted for the new schema:

  * target column     : ``pm25_poc_3``   (PM2.5 concentration)
  * time resolution   : 1 hour            (vs LOF's 15 minutes)
  * calendar columns  : ``month``, ``day``, ``hour``, ``day_of_week``,
                        ``is_weekday`` / ``is_weekend``  (no lunar, no solar_*,
                        no minute, no is_holiday, no pricing)
  * default windows   : input_length = 96, output_length = 96 (4 days each
                        at hourly resolution; non-overlapping daily stride)
  * meteorological    : ``wind_spd_res``, ``temp_outdoor``, ``rel_humidity``,
                        ``solar_radiation``, ``uv_radiation``,
                        ``baro_pressure``, ``sp``, ``skt``, ``blh``, ``tp``

Outputs three JSON files into ``--output_dir`` (same naming scheme as
the LOF pipeline so downstream tools work unchanged):

  * ``<dataset>_full_segments.json`` — every segment with full stats.
  * ``<dataset>_llm_context.json``   — stratified sample for LLM prompt.
  * ``<dataset>_llm_compact.json``   — minimal-footprint version of above.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


# ============================================================
# Channel metadata — Riverside air-quality schema
# ============================================================

TARGET_COLUMN      = "pm25_poc_3"
TARGET_DESCRIPTION = "PM2.5 air-pollutant concentration (µg/m³)"
TIME_RESOLUTION    = "1 hour"

CHANNEL_META = {
    # ---- discrete (categorical) ----
    "month":        {"type": "discrete",  "description": "Month of year (1-12)"},
    "day":          {"type": "discrete",  "description": "Day of month (1-31)"},
    "hour":         {"type": "discrete",  "description": "Hour of day (0-23)"},
    "day_of_week":  {"type": "discrete",  "description": "Day of week (0=Monday … 6=Sunday)"},
    "is_weekend":   {"type": "discrete",  "description": "1 if Saturday/Sunday, else 0"},
    "is_weekday":   {"type": "discrete",  "description": "1 if Mon-Fri, else 0"},
    # ---- continuous ----
    "wind_spd_res":     {"type": "continuous", "description": "Wind speed (resultant, m/s)"},
    "temp_outdoor":     {"type": "continuous", "description": "Outdoor air temperature (°F)"},
    "rel_humidity":     {"type": "continuous", "description": "Relative humidity (%)"},
    "solar_radiation":  {"type": "continuous", "description": "Solar irradiance (kW/m²)"},
    "uv_radiation":     {"type": "continuous", "description": "UV irradiance (W/m²)"},
    "baro_pressure":    {"type": "continuous", "description": "Barometric pressure at station (hPa)"},
    "sp":               {"type": "continuous", "description": "ERA5 surface pressure (Pa)"},
    "skt":              {"type": "continuous", "description": "ERA5 skin temperature (K)"},
    "blh":              {"type": "continuous", "description": "ERA5 boundary layer height (m) — vertical mixing depth"},
    "tp":               {"type": "continuous", "description": "ERA5 total precipitation (m)"},
    # ---- target ----
    "pm25_poc_3":       {"type": "continuous", "description": TARGET_DESCRIPTION},
}

# Channels whose correlation with the target is interpretable for air quality.
TARGET_CORR_CHANNELS = [
    "temp_outdoor", "rel_humidity", "wind_spd_res", "baro_pressure", "blh",
]


# ============================================================
# Statistical helpers (identical to data_preprocessing.py)
# ============================================================

def _skewness(values: np.ndarray) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    m = np.mean(values)
    s = np.std(values, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((values - m) / s) ** 3))


def _kurtosis(values: np.ndarray) -> float:
    n = len(values)
    if n < 4:
        return 0.0
    m = np.mean(values)
    s = np.std(values, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((values - m) / s) ** 4) - 3.0)


def _linregress_slope(values: np.ndarray) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = values.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-12:
        return 0.0
    return float(np.sum((x - x_mean) * (values - y_mean)) / denom)


def extract_continuous_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean":     round(float(np.mean(values)),   2),
        "median":   round(float(np.median(values)), 2),
        "std":      round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 2),
        "min":      round(float(np.min(values)),    2),
        "max":      round(float(np.max(values)),    2),
        "range":    round(float(np.max(values) - np.min(values)), 2),
        "skewness": round(_skewness(values),        2),
        "kurtosis": round(_kurtosis(values),        2),
        "slope":    round(_linregress_slope(values), 4),
        "volume":   round(float(np.sum(np.abs(np.diff(values)))) if len(values) > 1 else 0.0, 2),
    }


def extract_discrete_stats(values: np.ndarray) -> dict:
    values = np.asarray(values)
    unique, counts = np.unique(values, return_counts=True)
    n = float(len(values))
    return {
        "mode":         int(unique[counts.argmax()]),
        "n_unique":     int(len(unique)),
        "distribution": {str(int(v)): round(float(c) / n, 2)
                         for v, c in zip(unique, counts)},
    }


# ============================================================
# Per-segment stats
# ============================================================

def extract_segment_stats(segment_df: pd.DataFrame, channel_meta: dict) -> dict:
    """Per-channel statistics for a single segment window."""
    stats = {}
    for col in segment_df.columns:
        if col not in channel_meta:
            continue
        values = segment_df[col].values
        if channel_meta[col]["type"] == "continuous":
            stats[col] = extract_continuous_stats(values)
        else:
            stats[col] = extract_discrete_stats(values)
    return stats


def extract_cross_channel_stats(segment_df: pd.DataFrame) -> dict:
    """Air-quality-specific cross-channel statistics.

    Differences from the LOF pipeline:
      * correlations are computed against ``pm25_poc_3`` (target), not
        ``load``;
      * day-type is derived from ``is_weekday`` (Riverside has no
        ``is_workday`` / holiday columns).
    """
    cross = {}

    # Target-vs-driver correlations.
    if TARGET_COLUMN in segment_df.columns:
        target = segment_df[TARGET_COLUMN].values
        for col in TARGET_CORR_CHANNELS:
            if col in segment_df.columns:
                other = segment_df[col].values
                if np.std(target) > 1e-8 and np.std(other) > 1e-8:
                    corr = float(np.corrcoef(target, other)[0, 1])
                    cross[f"target_corr_{col}"] = round(corr, 2)

    # Uniform day type within the segment.
    if "is_weekday" in segment_df.columns:
        wd_vals = segment_df["is_weekday"].unique()
        cross["is_uniform_day_type"] = bool(len(wd_vals) == 1)
        if len(wd_vals) == 1:
            cross["uniform_day_type"] = "weekday" if wd_vals[0] == 1 else "weekend"

    return cross


# ============================================================
# Global statistics (over the training slice)
# ============================================================

def compute_global_statistics(df: pd.DataFrame, channel_meta: dict) -> dict:
    """Global per-channel stats over the training portion. Continuous
    channels also get five percentiles for the LLM to use when proposing
    thresholds."""
    global_stats = {}
    for col in df.columns:
        if col not in channel_meta:
            continue
        values = df[col].values
        if channel_meta[col]["type"] == "continuous":
            stats = extract_continuous_stats(values)
            for p in [5, 10, 25, 75, 90, 95]:
                stats[f"p{p}"] = round(float(np.percentile(values, p)), 2)
            global_stats[col] = stats
        else:
            global_stats[col] = extract_discrete_stats(values)
    return global_stats


# ============================================================
# Segmentation
# ============================================================

def segment_time_series(
    df: pd.DataFrame,
    input_length: int = 96,
    output_length: int = 96,
    stride: int = 96,
) -> list:
    """Slide a (lookback + forecast) window over ``df`` with the given stride.

    Defaults are tuned for hourly Riverside data: 96-hour lookback + 96-hour
    forecast = 4 days each, with non-overlapping windows.
    """
    total_window = input_length + output_length
    n_rows = len(df)
    segments = []

    seg_id = 0
    for start in range(0, n_rows - total_window + 1, stride):
        end = start + total_window
        lookback_end = start + input_length

        window_df  = df.iloc[start:end]
        lookback_df = df.iloc[start:lookback_end]
        forecast_df = df.iloc[lookback_end:end]

        segments.append({
            "segment_id": seg_id,
            "start_idx":  int(start),
            "end_idx":    int(end),
            "lookback_stats":      extract_segment_stats(lookback_df, CHANNEL_META),
            "forecast_stats":      extract_segment_stats(forecast_df, CHANNEL_META),
            "cross_channel_stats": extract_cross_channel_stats(window_df),
            "time_context":        _extract_time_context(lookback_df, forecast_df),
        })
        seg_id += 1

    return segments


def _extract_time_context(lookback_df: pd.DataFrame, forecast_df: pd.DataFrame) -> dict:
    """Human-readable calendar context — uses the actual Riverside columns
    (``month``, ``day``, ``hour``) rather than the LOF lunar/solar trick."""
    lb_start = lookback_df.iloc[0]
    lb_end   = lookback_df.iloc[-1]
    fc_start = forecast_df.iloc[0]
    fc_end   = forecast_df.iloc[-1]

    def _stamp(row):
        return {
            "month": int(row.get("month", 0)),
            "day":   int(row.get("day",   0)),
            "hour":  int(row.get("hour",  0)),
        }

    context = {
        "lookback_start": _stamp(lb_start),
        "lookback_end":   _stamp(lb_end),
        "forecast_start": _stamp(fc_start),
        "forecast_end":   _stamp(fc_end),
    }

    # Season from forecast-start month (Northern Hemisphere).
    month = int(fc_start.get("month", 1))
    if month in (3, 4, 5):
        context["season"] = "spring"
    elif month in (6, 7, 8):
        context["season"] = "summer"
    elif month in (9, 10, 11):
        context["season"] = "autumn"
    else:
        context["season"] = "winter"
    return context


# ============================================================
# Sampling for LLM context budget
# ============================================================

def sample_segments_for_llm(
    segments: list,
    max_segments: int = 300,
    strategy: str = "stratified",
) -> list:
    """Optionally subsample segments to stay within an LLM prompt budget.

    Stratification key for Riverside: (season, weekday-vs-weekend).
    """
    if len(segments) <= max_segments:
        return segments

    if strategy == "uniform":
        idx = np.linspace(0, len(segments) - 1, max_segments, dtype=int)
        return [segments[i] for i in idx]

    if strategy == "stratified":
        buckets = {}
        for seg in segments:
            season   = seg["time_context"].get("season", "unknown")
            day_type = seg["cross_channel_stats"].get("uniform_day_type", "mixed")
            buckets.setdefault(f"{season}_{day_type}", []).append(seg)

        per_bucket = max(1, max_segments // max(1, len(buckets)))
        sampled = []
        for bucket_segs in buckets.values():
            n = min(per_bucket, len(bucket_segs))
            idx = np.linspace(0, len(bucket_segs) - 1, n, dtype=int)
            sampled.extend([bucket_segs[i] for i in idx])
        return sampled[:max_segments]

    return segments[:max_segments]


# ============================================================
# Compact representation (Riverside-specific channel list)
# ============================================================

COMPACT_CONTINUOUS_CHANNELS = [
    "temp_outdoor", "rel_humidity", "wind_spd_res",
    "baro_pressure", "blh", "tp", "pm25_poc_3",
]
COMPACT_DISCRETE_CHANNELS = [
    "month", "hour", "day_of_week", "is_weekday",
]


def compact_segment(seg: dict) -> dict:
    """Trim a segment summary down for LLM prompt budget."""
    compact = {
        "id":     seg["segment_id"],
        "ctx":    seg["time_context"]["season"],
        "fc_hr":  seg["time_context"]["forecast_start"]["hour"],
    }

    lb = {}
    for ch in COMPACT_CONTINUOUS_CHANNELS:
        if ch in seg["lookback_stats"]:
            s = seg["lookback_stats"][ch]
            lb[ch] = {"m": s["mean"], "s": s["std"], "sl": s["slope"]}
    for ch in COMPACT_DISCRETE_CHANNELS:
        if ch in seg["lookback_stats"]:
            s = seg["lookback_stats"][ch]
            lb[ch] = {"mode": s["mode"]}
    compact["lb"] = lb

    if TARGET_COLUMN in seg["forecast_stats"]:
        fs = seg["forecast_stats"][TARGET_COLUMN]
        compact["fc"] = {"m": fs["mean"], "s": fs["std"], "sl": fs["slope"]}

    compact["xc"] = seg["cross_channel_stats"]
    return compact


# ============================================================
# Main pipeline
# ============================================================

def run_preprocessing(
    data_path: str,
    output_dir: str,
    input_length: int = 96,
    output_length: int = 96,
    stride: int = 96,
    max_segments_for_llm: int = 300,
):
    """Run the full Part-A pipeline against Riverside-style hourly data."""
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]

    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Shape: {df.shape}")

    # Training slice: first 70 % of rows (same convention as the LOF pipeline).
    train_end = int(0.7 * len(df))
    train_df  = df.iloc[:train_end].copy()
    print(f"  Training set: {len(train_df)} rows ({len(train_df)/24:.0f} days at hourly resolution)")

    # 1. Global statistics over training slice.
    print("\nComputing global statistics...")
    global_stats = compute_global_statistics(train_df, CHANNEL_META)
    print(f"  Computed for {len(global_stats)} channels")

    # 2. Segmentation.
    print(f"\nSegmenting (window={input_length}+{output_length}, stride={stride})...")
    all_segments = segment_time_series(
        train_df,
        input_length=input_length,
        output_length=output_length,
        stride=stride,
    )
    print(f"  Total segments: {len(all_segments)}")

    # 3. Sampling for LLM budget.
    print(f"\nSampling up to {max_segments_for_llm} segments for LLM context...")
    sampled_segments = sample_segments_for_llm(
        all_segments, max_segments=max_segments_for_llm, strategy="stratified"
    )
    print(f"  Sampled: {len(sampled_segments)} segments")

    # 4. Metadata.
    metadata = {
        "dataset_name":        dataset_name,
        "data_path":           data_path,
        "total_rows":          len(df),
        "training_rows":       len(train_df),
        "time_resolution":     TIME_RESOLUTION,
        "input_length":        input_length,
        "output_length":       output_length,
        "stride":              stride,
        "total_segments":      len(all_segments),
        "sampled_segments":    len(sampled_segments),
        "target_variable":     TARGET_COLUMN,
        "target_description":  TARGET_DESCRIPTION,
        "channels": {
            col: CHANNEL_META[col] for col in df.columns if col in CHANNEL_META
        },
    }

    # 5. Save full segments JSON.
    full_output = {
        "metadata":          metadata,
        "global_statistics": global_stats,
        "segments":          all_segments,
    }
    full_path = os.path.join(output_dir, f"{dataset_name}_full_segments.json")
    print(f"\nSaving full segments to {full_path}...")
    with open(full_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"  Size: {os.path.getsize(full_path) / (1024 * 1024):.1f} MB")

    # 6. LLM-context JSON (sampled, full per-segment stats).
    llm_output = {
        "metadata":          metadata,
        "global_statistics": global_stats,
        "segment_summaries": sampled_segments,
    }
    llm_path = os.path.join(output_dir, f"{dataset_name}_llm_context.json")
    print(f"Saving LLM context to {llm_path}...")
    with open(llm_path, "w") as f:
        json.dump(llm_output, f, indent=2)
    print(f"  Size: {os.path.getsize(llm_path) / (1024 * 1024):.1f} MB")

    # 7. Compact LLM JSON (trimmed channels + stats).
    compact_segments = [compact_segment(s) for s in sampled_segments]
    compact_output = {
        "metadata": {
            "dataset_name":      dataset_name,
            "target_variable":   TARGET_COLUMN,
            "target_description": TARGET_DESCRIPTION,
            "time_resolution":   TIME_RESOLUTION,
            "input_length":      input_length,
            "output_length":     output_length,
            "total_segments":    len(all_segments),
            "sampled_segments":  len(compact_segments),
            "compact_key": {
                "lb": "lookback window stats",
                "fc": "forecast window stats (target only)",
                "xc": "cross-channel stats",
                "m":  "mean", "s": "std", "sl": "slope",
                "ctx":   "season",
                "fc_hr": "forecast start hour",
            },
            "channels": {
                col: CHANNEL_META[col]
                for col in COMPACT_CONTINUOUS_CHANNELS + COMPACT_DISCRETE_CHANNELS
                if col in CHANNEL_META
            },
        },
        "global_statistics": global_stats,
        "segments":          compact_segments,
    }
    compact_path = os.path.join(output_dir, f"{dataset_name}_llm_compact.json")
    print(f"Saving compact LLM context to {compact_path}...")
    with open(compact_path, "w") as f:
        json.dump(compact_output, f, indent=2)
    print(f"  Size: {os.path.getsize(compact_path) / 1024:.0f} KB "
          f"(~{os.path.getsize(compact_path) // 4} tokens)")

    # 8. Summary.
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"  Dataset:          {dataset_name}")
    print(f"  Training rows:    {len(train_df)}")
    print(f"  Total segments:   {len(all_segments)}")
    print(f"  Sampled for LLM:  {len(sampled_segments)}")
    print(f"  Full output:      {full_path}")
    print(f"  LLM context:      {llm_path}")
    print(f"  Compact LLM ctx:  {compact_path}")

    return full_output, llm_output


def main():
    parser = argparse.ArgumentParser(
        description="Part-A preprocessing for the Riverside air-quality dataset."
    )
    parser.add_argument("--data_path",     type=str, required=True,
                        help="Path to the CSV (e.g. dataset/AirQuality/Riverside.csv)")
    parser.add_argument("--output_dir",    type=str, default="./preprocessing/output_air",
                        help="Directory to save the three output JSONs")
    parser.add_argument("--input_length",  type=int, default=96,
                        help="Lookback window size (default 96 = 4 days at 1h resolution)")
    parser.add_argument("--output_length", type=int, default=96,
                        help="Forecast window size (default 96 = 4 days at 1h resolution)")
    parser.add_argument("--stride",        type=int, default=96,
                        help="Step between segments (default 96 = non-overlapping)")
    parser.add_argument("--max_segments",  type=int, default=300,
                        help="Max segments to include in the LLM-context JSON")
    args = parser.parse_args()

    run_preprocessing(
        data_path=args.data_path,
        output_dir=args.output_dir,
        input_length=args.input_length,
        output_length=args.output_length,
        stride=args.stride,
        max_segments_for_llm=args.max_segments,
    )


if __name__ == "__main__":
    main()

"""
Part A: Data Preprocessing for LLM-Guided Prototype Initialization

This script performs:
1. Segmentation of multivariate time series into windows
   (384 lookback + 96 forecast = 480 steps)
2. Per-segment statistical extraction for each channel
3. Global statistical extraction across the full training set
4. JSON serialization for LLM consumption

Output: A JSON file containing:
  - "metadata": dataset info, channel descriptions, feature types
  - "global_statistics": per-channel statistics across entire training set
  - "segment_summaries": list of per-segment statistical summaries (capped for LLM context)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd


# ============================================================
# Channel Metadata
# ============================================================

# ---------------------------------------------------------------------------
# Dataset switch — keep in sync with the DATASET constants in
# ``dataloader/dataloader.py`` and ``convolution_encoder/channel_embedding.py``.
# CHANNEL_META describes per-column type + a one-line semantic description
# used by the LLM concept proposer.
# ---------------------------------------------------------------------------
DATASET = "PRSA"

if DATASET == "LOF":
    CHANNEL_META = {
        # Discrete (categorical) channels
        "lunar_year":       {"type": "discrete", "description": "Year in Chinese lunar calendar (1-5, cyclical)"},
        "lunar_month":      {"type": "discrete", "description": "Month in Chinese lunar calendar (1-12)"},
        "lunar_day":        {"type": "discrete", "description": "Day in Chinese lunar calendar (1-30)"},
        "solar_year":       {"type": "discrete", "description": "Solar calendar year index (2-5)"},
        "solar_month":      {"type": "discrete", "description": "Month of year (1-12)"},
        "solar_day":        {"type": "discrete", "description": "Day of month (1-31)"},
        "month":            {"type": "discrete", "description": "Hour of day (0-23, despite column name 'month')"},
        "minute":           {"type": "discrete", "description": "Minute within hour (0, 15, 30, 45)"},
        "is_special_workday":  {"type": "discrete", "description": "Whether this is a special working day (0 or 1, e.g., weekend rescheduled as workday)"},
        "is_special_holiday":  {"type": "discrete", "description": "Whether this is a special holiday (0 or 1, e.g., weekday declared as holiday)"},
        "is_holiday":       {"type": "discrete", "description": "Whether this is a holiday (0 or 1)"},
        "is_workday":       {"type": "discrete", "description": "Whether this is a workday (0 or 1)"},
        "day_of_week":      {"type": "discrete", "description": "Day of week (0=Monday to 6=Sunday)"},
        "pricing":          {"type": "discrete", "description": "Electricity pricing tier (0.1, 0.3, 1.0, 1.7, 2.0)"},
        # Continuous channels
        "Skin_temperature":              {"type": "continuous", "description": "Surface skin temperature (°C)"},
        "Surface_pressure":              {"type": "continuous", "description": "Surface atmospheric pressure (kPa)"},
        "Surface_sensible_heat_flux":    {"type": "continuous", "description": "Surface sensible heat flux (W/m²)"},
        "Total_cloud_cover":             {"type": "continuous", "description": "Total cloud cover"},
        "Surface_net_solar_radiation":   {"type": "continuous", "description": "Net solar radiation at surface (W/m²)"},
        "Total_precipitation_4h_ahead":  {"type": "continuous", "description": "Forecasted total precipitation 4h ahead (mm)"},
        "10_metre_wind":                 {"type": "continuous", "description": "Wind speed at 10m height (m/s)"},
        "Relative_humidity":             {"type": "continuous", "description": "Relative humidity (%)"},
        # Target
        "load":             {"type": "continuous", "description": "Electricity load (target variable, MW)"},
    }
elif DATASET == "PRSA":
    CHANNEL_META = {
        # Discrete (categorical) channels
        "year":  {"type": "discrete", "description": "Year index 0..4 (mapped from raw 2013..2017 during cleaning)"},
        "month": {"type": "discrete", "description": "Calendar month (1-12)"},
        "day":   {"type": "discrete", "description": "Day of month (1-31)"},
        "hour":  {"type": "discrete", "description": "Hour of day (0-23)"},
        # Continuous channels — air-quality covariates
        "PM10":  {"type": "continuous", "description": "PM10 concentration (μg/m³)"},
        "SO2":   {"type": "continuous", "description": "SO2 concentration (μg/m³)"},
        "NO2":   {"type": "continuous", "description": "NO2 concentration (μg/m³)"},
        "CO":    {"type": "continuous", "description": "CO concentration (μg/m³)"},
        "O3":    {"type": "continuous", "description": "O3 concentration (μg/m³)"},
        # Continuous channels — meteorology
        "TEMP":  {"type": "continuous", "description": "Surface temperature (°C)"},
        "PRES":  {"type": "continuous", "description": "Surface atmospheric pressure (hPa)"},
        "DEWP":  {"type": "continuous", "description": "Dew-point temperature (°C); higher = more humid air"},
        "RAIN":  {"type": "continuous", "description": "Hourly precipitation (mm)"},
        "WSPM":  {"type": "continuous", "description": "Wind speed (m/s) — low wind suppresses pollutant dispersion"},
        # Target
        "PM2.5": {"type": "continuous", "description": "PM2.5 concentration (target, μg/m³)"},
    }
else:
    raise ValueError(
        f"Unknown DATASET={DATASET!r} in preprocessing.data_preprocessing. "
        f"Expected 'LOF' | 'PRSA'."
    )


# ============================================================
# Statistical Extraction Functions
# ============================================================

def _skewness(values: np.ndarray) -> float:
    """Compute skewness without scipy."""
    n = len(values)
    if n < 3:
        return 0.0
    m = np.mean(values)
    s = np.std(values, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((values - m) / s) ** 3))


def _kurtosis(values: np.ndarray) -> float:
    """Compute excess kurtosis without scipy."""
    n = len(values)
    if n < 4:
        return 0.0
    m = np.mean(values)
    s = np.std(values, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((values - m) / s) ** 4) - 3.0)


def _linregress_slope(values: np.ndarray) -> float:
    """Compute linear regression slope without scipy."""
    n = len(values)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    v_mean = values.mean()
    denom = np.sum((t - t_mean) ** 2)
    if denom < 1e-12:
        return 0.0
    return float(np.sum((t - t_mean) * (values - v_mean)) / denom)


def extract_continuous_stats(values: np.ndarray) -> dict:
    """
    mean, median, std, min, max, range,
    skewness, kurtosis, slope (linear trend), plus volume (sum of abs changes).
    Values are rounded to 2 decimal places for compact LLM representation.
    """
    n = len(values)
    if n == 0:
        return {}

    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if n > 1 else 0.0
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    slope = _linregress_slope(values)

    # Volume: sum of absolute differences (measures total variation)
    volume = float(np.sum(np.abs(np.diff(values)))) if n > 1 else 0.0

    return {
        "mean": round(mean_val, 2),
        "median": round(float(np.median(values)), 2),
        "std": round(std_val, 2),
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "range": round(max_val - min_val, 2),
        "skewness": round(_skewness(values), 2),
        "kurtosis": round(_kurtosis(values), 2),
        "slope": round(slope, 4),
        "volume": round(volume, 2),
    }


def extract_discrete_stats(values: np.ndarray) -> dict:
    """
    mode, unique count, and value distribution.
    """
    n = len(values)
    if n == 0:
        return {}

    unique, counts = np.unique(values, return_counts=True)
    mode_idx = np.argmax(counts)

    # Value distribution (top values by frequency)
    distribution = {}
    for val, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        distribution[str(int(val))] = round(float(cnt) / n, 2)

    return {
        "mode": int(unique[mode_idx]),
        "n_unique": int(len(unique)),
        "distribution": distribution,
    }


def extract_segment_stats(segment_df: pd.DataFrame, channel_meta: dict) -> dict:
    """Extract statistics for all channels in a single segment."""
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


if DATASET == "LOF":
    _TARGET_COL          = "load"
    _TARGET_CORR_COLS    = ["Skin_temperature", "Relative_humidity",
                             "Surface_net_solar_radiation"]
elif DATASET == "PRSA":
    _TARGET_COL          = "PM2.5"
    _TARGET_CORR_COLS    = ["PM10", "WSPM", "TEMP", "DEWP", "PRES", "RAIN"]
else:
    _TARGET_COL          = None
    _TARGET_CORR_COLS    = []


def extract_cross_channel_stats(segment_df: pd.DataFrame) -> dict:
    """Extract cross-channel statistics that capture variable interactions.

    The cross-channel block is the place where the LLM concept proposer
    learns about target-correlated covariates and (LOF-only) day-type uniformity.
    """
    cross = {}

    # Correlation between target and key continuous covariates.
    if _TARGET_COL and _TARGET_COL in segment_df.columns:
        tgt = segment_df[_TARGET_COL].values
        for col in _TARGET_CORR_COLS:
            if col in segment_df.columns:
                other = segment_df[col].values
                if np.std(tgt) > 1e-8 and np.std(other) > 1e-8:
                    corr = float(np.corrcoef(tgt, other)[0, 1])
                    cross[f"target_corr_{col}"] = round(corr, 2)

    # LOF-only: day-type / holiday flags. These columns don't exist in PRSA.
    if "is_workday" in segment_df.columns:
        workday_vals = segment_df["is_workday"].unique()
        cross["is_uniform_day_type"] = bool(len(workday_vals) == 1)
        if len(workday_vals) == 1:
            cross["uniform_day_type"] = "workday" if workday_vals[0] == 1.0 else "non_workday"
    if "is_holiday" in segment_df.columns:
        holiday_vals = segment_df["is_holiday"].unique()
        cross["contains_holiday"] = bool(1.0 in holiday_vals)
        cross["is_all_holiday"]   = bool(len(holiday_vals) == 1 and holiday_vals[0] == 1.0)

    return cross


# ============================================================
# Global Statistics
# ============================================================

def compute_global_statistics(df: pd.DataFrame, channel_meta: dict) -> dict:
    """
    Compute global statistics across the entire training set.
    Following ShapeTalk, we cap this at a manageable summary so the LLM
    can use it for Dynamic Feature Threshold Inference.
    """
    global_stats = {}
    for col in df.columns:
        if col not in channel_meta:
            continue
        values = df[col].values
        if channel_meta[col]["type"] == "continuous":
            stats = extract_continuous_stats(values)
            # Add percentiles for threshold inference
            for p in [5, 10, 25, 75, 90, 95]:
                stats[f"p{p}"] = round(float(np.percentile(values, p)), 2)
            global_stats[col] = stats
        else:
            global_stats[col] = extract_discrete_stats(values)

    return global_stats


# ============================================================
# Segmentation
# ============================================================

def _season_from_month(month: int) -> str:
    if month in (3, 4, 5):   return "spring"
    if month in (6, 7, 8):   return "summer"
    if month in (9, 10, 11): return "autumn"
    return "winter"


def _row_calendar(row) -> dict:
    """Pull a (calendar_month, day_of_month, hour) tuple out of one dataframe row,
    handling both schemas:
      LOF  uses solar_month / solar_day / 'month'(=hour) / minute.
      PRSA uses month        / day        / hour.
    """
    if DATASET == "LOF":
        return {
            "solar_month": int(row.get("solar_month", 0)),
            "solar_day":   int(row.get("solar_day", 0)),
            "hour":        int(row.get("month", 0)),   # 'month' column is hour-of-day
            "minute":      int(row.get("minute", 0)),
        }
    if DATASET == "PRSA":
        return {
            "month": int(row.get("month", 0)),
            "day":   int(row.get("day", 0)),
            "hour":  int(row.get("hour", 0)),
        }
    raise ValueError(f"Unknown DATASET={DATASET!r}")


def _calendar_month_of(row) -> int:
    """Return the CALENDAR month (1-12) of a dataframe row, regardless of schema."""
    if DATASET == "LOF":
        return int(row.get("solar_month", 1))
    if DATASET == "PRSA":
        return int(row.get("month", 1))
    raise ValueError(f"Unknown DATASET={DATASET!r}")


def _extract_subwindow_time_context(sw_df: pd.DataFrame, sw_idx: int) -> dict:
    """Time context for a forecast sub-window. Resolves the calendar/hour
    columns through the DATASET-aware ``_row_calendar``."""
    sw_start = sw_df.iloc[0]
    sw_end = sw_df.iloc[-1]
    return {
        "subwindow_idx": int(sw_idx),
        "start": _row_calendar(sw_start),
        "end":   _row_calendar(sw_end),
        "season": _season_from_month(_calendar_month_of(sw_start)),
    }


def build_subwindow_blocks(forecast_df: pd.DataFrame,
                           n_subwindows: int) -> list:
    """Slice the forecast window into ``n_subwindows`` equal chunks and
    compute per-chunk per-channel stats + a local time_context.

    Each returned block has shape:
        {
          "subwindow_idx":      int,
          "stats":              {channel -> {mean, std, slope, ...}},
          "time_context":       {subwindow_idx, start{...}, end{...}, season},
        }

    The forecast concept rules consume these blocks (one rule × four blocks
    = four labels per segment per concept). See CBM/concept_labels.py.
    """
    out_len = len(forecast_df)
    if out_len % n_subwindows != 0:
        raise ValueError(
            f"forecast length {out_len} not divisible by n_subwindows={n_subwindows}"
        )
    sw_len = out_len // n_subwindows
    blocks = []
    for k in range(n_subwindows):
        sw_df = forecast_df.iloc[k * sw_len : (k + 1) * sw_len]
        blocks.append({
            "subwindow_idx": int(k),
            "stats":         extract_segment_stats(sw_df, CHANNEL_META),
            "time_context":  _extract_subwindow_time_context(sw_df, k),
        })
    return blocks


def segment_time_series(
    df: pd.DataFrame,
    input_length: int = 384,
    output_length: int = 96,
    stride: int = 96,
    n_subwindows: int = 4,
) -> list:
    """Segment the time series into non-overlapping (or strided) windows.

    Each segment = 384 (lookback) + 96 (forecast) = 480 time steps.

    Args:
        df: Full dataframe with all channels + target
        input_length: Lookback window size (384 = 4 days at 15-min resolution)
        output_length: Forecast window size (96 = 1 day)
        stride: Step size between consecutive segments. Default 96 (1 day)
                to get non-overlapping forecast windows.

    Returns:
        List of dicts, each containing:
          - "segment_id": int
          - "start_idx": int (start row in original df)
          - "end_idx": int (end row, exclusive)
          - "lookback_stats": per-channel stats for the lookback window
          - "forecast_stats": per-channel stats for the forecast window (target only)
          - "cross_channel_stats": interaction statistics
          - "time_context": calendar context at segment start/end
    """
    total_window = input_length + output_length
    n_rows = len(df)
    segments = []

    seg_id = 0
    for start in range(0, n_rows - total_window + 1, stride):
        end = start + total_window
        lookback_end = start + input_length

        window_df = df.iloc[start:end]
        lookback_df = df.iloc[start:lookback_end]
        forecast_df = df.iloc[lookback_end:end]

        # Time context: what calendar period does this segment cover
        time_context = _extract_time_context(lookback_df, forecast_df)

        # Per-channel statistics for lookback window (all channels)
        lookback_stats = extract_segment_stats(lookback_df, CHANNEL_META)

        # Per-channel statistics for forecast window (target + key covariates)
        forecast_stats = extract_segment_stats(forecast_df, CHANNEL_META)

        # Cross-channel statistics on full window
        cross_stats = extract_cross_channel_stats(window_df)

        # Per-sub-window stats over the forecast window — consumed by the
        # forecast-side concept layers in the CBM (one rule per sub-window).
        subwindow_blocks = build_subwindow_blocks(forecast_df, n_subwindows=n_subwindows)

        segments.append({
            "segment_id": seg_id,
            "start_idx": int(start),
            "end_idx": int(end),
            "lookback_stats": lookback_stats,
            "forecast_stats": forecast_stats,
            "subwindow_stats": subwindow_blocks,
            "cross_channel_stats": cross_stats,
            "time_context": time_context,
        })

        seg_id += 1

    return segments


def _extract_time_context(lookback_df: pd.DataFrame, forecast_df: pd.DataFrame) -> dict:
    """Extract human-readable time context for a segment. Resolves the
    calendar columns through the DATASET-aware ``_row_calendar`` helper."""
    lb_start = lookback_df.iloc[0]
    lb_end   = lookback_df.iloc[-1]
    fc_start = forecast_df.iloc[0]
    fc_end   = forecast_df.iloc[-1]
    return {
        "lookback_start": _row_calendar(lb_start),
        "lookback_end":   _row_calendar(lb_end),
        "forecast_start": _row_calendar(fc_start),
        "forecast_end":   _row_calendar(fc_end),
        "season":         _season_from_month(_calendar_month_of(fc_start)),
    }


# ============================================================
# Sampling for LLM Context Budget
# ============================================================

def sample_segments_for_llm(
    segments: list,
    max_segments: int = 300,
    strategy: str = "stratified",
) -> list:
    """Sample a subset of segments to fit within LLM context budget.

    Following ShapeTalk's approach of capping patch counts (300 for local,
    100 for global) to keep the LLM prompt manageable.

    Args:
        segments: Full list of segment summaries
        max_segments: Maximum number to include
        strategy: "stratified" (sample evenly across seasons/day types) or "uniform"

    Returns:
        Sampled subset of segments
    """
    if len(segments) <= max_segments:
        return segments

    if strategy == "uniform":
        indices = np.linspace(0, len(segments) - 1, max_segments, dtype=int)
        return [segments[i] for i in indices]

    elif strategy == "stratified":
        # Group by season and day type for balanced representation
        buckets = {}
        for seg in segments:
            season = seg["time_context"].get("season", "unknown")
            day_type = seg["cross_channel_stats"].get("uniform_day_type", "mixed")
            key = f"{season}_{day_type}"
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(seg)

        # Sample proportionally from each bucket
        sampled = []
        per_bucket = max(1, max_segments // len(buckets))
        for key, bucket_segs in buckets.items():
            n_sample = min(per_bucket, len(bucket_segs))
            indices = np.linspace(0, len(bucket_segs) - 1, n_sample, dtype=int)
            sampled.extend([bucket_segs[i] for i in indices])

        # Trim to exact budget if oversampled
        return sampled[:max_segments]

    return segments[:max_segments]


# ============================================================
# Compact Representation for LLM Context Budget
# ============================================================

# Key channels to keep in the compact LLM representation.
# DATASET-aware: the compact view changes per schema.
if DATASET == "LOF":
    COMPACT_CONTINUOUS_CHANNELS = [
        "Skin_temperature", "Relative_humidity",
        "Surface_net_solar_radiation", "Total_precipitation_4h_ahead",
        "10_metre_wind", "load",
    ]
    COMPACT_DISCRETE_CHANNELS = [
        "solar_month", "month", "is_holiday", "is_workday", "day_of_week", "pricing",
    ]
elif DATASET == "PRSA":
    COMPACT_CONTINUOUS_CHANNELS = [
        "PM10", "TEMP", "PRES", "DEWP", "WSPM", "RAIN", "O3", "PM2.5",
    ]
    COMPACT_DISCRETE_CHANNELS = [
        "month", "day", "hour", "year",
    ]
else:
    COMPACT_CONTINUOUS_CHANNELS = []
    COMPACT_DISCRETE_CHANNELS   = []
# For continuous channels, keep only mean, std, slope (3 numbers instead of 10)
# For discrete channels, keep only mode


def compact_segment(seg: dict) -> dict:
    """Reduce a segment summary to minimal representation for LLM context.

    Shrinks each segment from ~2KB to ~300 bytes by keeping only the most
    informative statistics for the most important channels.
    """
    compact = {
        "id": seg["segment_id"],
        "ctx": seg["time_context"]["season"],
        "fc_hr": seg["time_context"]["forecast_start"]["hour"],
    }

    # Lookback: compact continuous stats
    lb = {}
    for ch in COMPACT_CONTINUOUS_CHANNELS:
        if ch in seg["lookback_stats"]:
            s = seg["lookback_stats"][ch]
            lb[ch] = {"m": s["mean"], "s": s["std"], "sl": s["slope"]}
    # Lookback: compact discrete stats
    for ch in COMPACT_DISCRETE_CHANNELS:
        if ch in seg["lookback_stats"]:
            s = seg["lookback_stats"][ch]
            lb[ch] = {"mode": s["mode"]}
    compact["lb"] = lb

    # Forecast: just the target variable.
    if _TARGET_COL and _TARGET_COL in seg["forecast_stats"]:
        fs = seg["forecast_stats"][_TARGET_COL]
        compact["fc"] = {"m": fs["mean"], "s": fs["std"], "sl": fs["slope"]}

    # Cross-channel: keep as-is (already small)
    compact["xc"] = seg["cross_channel_stats"]

    return compact


# ============================================================
# Main Pipeline
# ============================================================

def run_preprocessing(
    data_path: str,
    output_dir: str,
    input_length: int = 384,
    output_length: int = 96,
    stride: int = 96,
    max_segments_for_llm: int = 300,
    n_subwindows: int = 4,
    split: str = "train",
):
    """Run the full Part A preprocessing pipeline.

    Args:
        data_path: Path to the CSV file (e.g., dataset/LOF/RE.csv)
        output_dir: Directory to save output JSON files
        input_length: Lookback window size
        output_length: Forecast window size
        stride: Stride between segments
        max_segments_for_llm: Max segments to include in LLM-facing JSON
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]

    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Shape: {df.shape}")

    # Slice the dataframe according to --split. Matches the legacy 70/10/20
    # chronological split used by dataloader/dataloader.py when
    # USE_LAST_YEAR_SPLIT is False (the active default).
    N = len(df)
    train_end = int(0.7 * N)
    val_end   = int(0.8 * N)
    if split == "train":
        slice_df = df.iloc[:train_end].copy()
        split_desc = f"training set (first 70%): rows [0, {train_end})"
    elif split == "vali":
        slice_df = df.iloc[train_end:val_end].copy()
        split_desc = f"validation set (70%-80%): rows [{train_end}, {val_end})"
    elif split == "test":
        slice_df = df.iloc[val_end:].copy()
        split_desc = f"test set (last 20%): rows [{val_end}, {N})"
    else:
        raise ValueError(f"Unknown --split {split!r}. Expected train | vali | test.")
    print(f"  Split: {split_desc} -> {len(slice_df)} rows "
          f"({len(slice_df)/96:.1f} days)")
    suffix = "" if split == "train" else f"_{split}"
    # Always compute global statistics on TRAIN only -- val/test must not
    # leak into the reference distribution used by the LLM prompt.
    train_df = df.iloc[:train_end].copy()

    # Step 1: Global statistics
    print("\nComputing global statistics (always on train slice)...")
    global_stats = compute_global_statistics(train_df, CHANNEL_META)
    print(f"  Computed for {len(global_stats)} channels")

    # Step 2: Segmentation and per-segment extraction (on the requested split)
    print(f"\nSegmenting {split} slice (window={input_length}+{output_length}, stride={stride})...")
    all_segments = segment_time_series(
        slice_df,
        input_length=input_length,
        output_length=output_length,
        stride=stride,
        n_subwindows=n_subwindows,
    )
    print(f"  Total segments in {split}: {len(all_segments)}")

    # Step 3: Sample for LLM context budget
    print(f"\nSampling {max_segments_for_llm} segments for LLM context...")
    sampled_segments = sample_segments_for_llm(
        all_segments, max_segments=max_segments_for_llm, strategy="stratified"
    )
    print(f"  Sampled: {len(sampled_segments)} segments")

    # DATASET-specific metadata fragments shared between full and compact outputs.
    if DATASET == "LOF":
        _time_res, _tgt, _tgt_desc = "15 minutes", "load", "Electricity load in MW"
    elif DATASET == "PRSA":
        _time_res = "1 hour"
        _tgt     = "PM2.5"
        _tgt_desc = ("PM2.5 concentration in micrograms per cubic meter at a "
                     "Beijing air-quality monitoring station")
    else:
        _time_res, _tgt, _tgt_desc = "unknown", "unknown", "unknown"

    # Step 4: Build metadata
    metadata = {
        "dataset_name": dataset_name,
        "data_path": data_path,
        "total_rows": len(df),
        "training_rows": len(train_df),
        "time_resolution": _time_res,
        "input_length": input_length,
        "output_length": output_length,
        "stride": stride,
        "n_subwindows": n_subwindows,
        "subwindow_length": output_length // n_subwindows,
        "total_segments": len(all_segments),
        "sampled_segments": len(sampled_segments),
        "target_variable": _tgt,
        "target_description": _tgt_desc,
        "channels": {
            col: CHANNEL_META[col] for col in df.columns if col in CHANNEL_META
        },
    }

    # Step 5: Save outputs
    # Full output (all segments — for Instance Finding in Part C)
    full_output = {
        "metadata": metadata,
        "global_statistics": global_stats,
        "segments": all_segments,
    }
    full_path = os.path.join(output_dir, f"{dataset_name}_full_segments{suffix}.json")
    print(f"\nSaving full segments to {full_path}...")
    with open(full_path, "w") as f:
        json.dump(full_output, f, indent=2)
    file_size_mb = os.path.getsize(full_path) / (1024 * 1024)
    print(f"  Size: {file_size_mb:.1f} MB")

    # LLM-facing output (sampled segments + global stats — for Part B)
    llm_output = {
        "metadata": metadata,
        "global_statistics": global_stats,
        "segment_summaries": sampled_segments,
    }
    llm_path = os.path.join(output_dir, f"{dataset_name}_llm_context{suffix}.json")
    print(f"Saving LLM context to {llm_path}...")
    with open(llm_path, "w") as f:
        json.dump(llm_output, f, indent=2)
    file_size_mb = os.path.getsize(llm_path) / (1024 * 1024)
    print(f"  Size: {file_size_mb:.1f} MB")

    # Compact LLM output (fits within ~50K tokens for most LLMs)
    compact_segments = [compact_segment(s) for s in sampled_segments]
    compact_output = {
        "metadata": {
            "dataset_name": dataset_name,
            "target_variable": _tgt,
            "target_description": _tgt_desc,
            "time_resolution": _time_res,
            "input_length": input_length,
            "output_length": output_length,
            "total_segments": len(all_segments),
            "sampled_segments": len(compact_segments),
            "compact_key": {
                "lb": "lookback window stats",
                "fc": "forecast window stats (target only)",
                "xc": "cross-channel stats",
                "m": "mean", "s": "std", "sl": "slope",
                "ctx": "season", "fc_hr": "forecast start hour",
            },
            "channels": {
                col: CHANNEL_META[col]
                for col in COMPACT_CONTINUOUS_CHANNELS + COMPACT_DISCRETE_CHANNELS
                if col in CHANNEL_META
            },
        },
        "global_statistics": global_stats,
        "segments": compact_segments,
    }
    compact_path = os.path.join(output_dir, f"{dataset_name}_llm_compact{suffix}.json")
    print(f"Saving compact LLM context to {compact_path}...")
    with open(compact_path, "w") as f:
        json.dump(compact_output, f, indent=2)
    file_size_kb = os.path.getsize(compact_path) / 1024
    est_tokens = os.path.getsize(compact_path) // 4
    print(f"  Size: {file_size_kb:.0f} KB (~{est_tokens} tokens)")

    # Summary report
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"  Dataset:          {dataset_name}")
    print(f"  Training rows:    {len(train_df)}")
    print(f"  Total segments:   {len(all_segments)}")
    print(f"  Sampled for LLM:  {len(sampled_segments)}")
    print(f"  Full output:      {full_path}")
    print(f"  LLM context:      {llm_path}")

    return full_output, llm_output


def main():
    parser = argparse.ArgumentParser(description="Part A: Data Preprocessing for LLM-Guided Prototype Init")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset CSV (e.g., dataset/LOF/RE.csv)")
    parser.add_argument("--output_dir", type=str, default="./preprocessing/output",
                        help="Directory to save output JSON files")
    parser.add_argument("--input_length", type=int, default=384,
                        help="Lookback window size (default: 384 = 4 days)")
    parser.add_argument("--output_length", type=int, default=96,
                        help="Forecast window size (default: 96 = 1 day)")
    parser.add_argument("--stride", type=int, default=96,
                        help="Stride between segments (default: 96 = 1 day)")
    parser.add_argument("--max_segments", type=int, default=300,
                        help="Max segments for LLM context (default: 300)")
    parser.add_argument("--n_subwindows", type=int, default=4,
                        help="Forecast window split into this many equal sub-windows. "
                             "Each sub-window gets its own per-channel stats block "
                             "used by the CBM's forecast concept layers. Default 4 "
                             "(96 forecast steps -> 4 x 24-step sub-windows = 4 x 6h).")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "vali", "test"],
                        help="Which slice of the CSV to segment. Output filename "
                             "gains a _<split> suffix (no suffix for train). The "
                             "70/10/20 boundaries match dataloader/dataloader.py "
                             "(legacy chronological split). Global statistics are "
                             "ALWAYS computed on the train slice -- val/test data "
                             "never leaks into the LLM-facing distribution.")
    args = parser.parse_args()

    run_preprocessing(
        data_path=args.data_path,
        output_dir=args.output_dir,
        input_length=args.input_length,
        output_length=args.output_length,
        stride=args.stride,
        max_segments_for_llm=args.max_segments,
        n_subwindows=args.n_subwindows,
        split=args.split,
    )


if __name__ == "__main__":
    main()

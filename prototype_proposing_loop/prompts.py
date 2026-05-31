"""
Prompt templates for the prototype proposing loop (function-based).

The LLM is asked to return, per prototype, a Python function ``select_segments``
that takes a list of segment dicts and returns matching indices. The full
schema is given inline so the LLM knows how to access fields.

The schema and example prototype shown to the LLM are dataset-specific
(different feature channels, different calendar fields, different example
regime). Flip ``DATASET`` below to switch between the two supported schemas.
``INITIAL_PROMPT`` is built from the corresponding block at module-load time;
``REVISION_PROMPT`` is generic and doesn't need to change.
"""

# ---------------------------------------------------------------------------
# Dataset switch — flip to "AirQuality" to point the LLM proposer at the
# Riverside PM2.5 schema. Keep this in sync with the ``DATASET`` constant in
# ``dataloader/dataloader.py``.
# ---------------------------------------------------------------------------
DATASET = "LOF"


# ===========================================================================
# Per-dataset blocks
# ===========================================================================

LOF_BLOCK = {
    # Used in the opening line of INITIAL_PROMPT.
    "domain_description": (
        "domain expert in electricity load forecasting for a province in "
        "northern China"
    ),

    # Used in the "Your Task" section.
    "regime_label": "electricity-load",

    # Inline schema example shown under "## Segment Schema".
    "schema_example": """```python
{{
  "segment_id": int,
  "start_idx": int,
  "end_idx": int,
  "lookback_stats": {{
    # one entry per channel; continuous channels have:
    "Skin_temperature": {{
      "mean": float, "median": float, "std": float, "min": float,
      "max": float, "range": float, "skewness": float,
      "kurtosis": float, "slope": float, "volume": float
    }},
    # discrete channels have:
    "is_workday": {{
      "mode": int, "n_unique": int, "distribution": {{"0": float, "1": float}}
    }},
    # ... one such entry per channel listed above
  }},
  "forecast_stats": {{
    # same structure as lookback_stats, but over the {output_length}-step forecast window
  }},
  "cross_channel_stats": {{
    "load_corr_Skin_temperature": float,
    "is_uniform_day_type": bool,
    "uniform_day_type": "workday" | "non_workday",
    "contains_holiday": bool,
    "is_all_holiday": bool
  }},
  "time_context": {{
    "season": "spring" | "summer" | "autumn" | "winter",
    "forecast_start": {{
      "hour": int, "solar_month": int, "solar_day": int, "minute": int
    }},
    # also: lookback_start, lookback_end, forecast_end
  }}
}}
```

NOTE: the column literally called "month" in the dataset actually stores the
**hour of day** (0-23), not the calendar month. The calendar month is in
"solar_month".""",

    # One worked example shown under "## Output Format".
    "example_prototype": """  {{
    "prototype_id": 1,
    "name": "cold_winter_workday",
    "description": "Cold winter workday regime with sustained heating load.",
    "function": "def select_segments(segments):\\n    matched = []\\n    for i, seg in enumerate(segments):\\n        lb = seg.get('lookback_stats', {{}})\\n        temp = lb.get('Skin_temperature', {{}}).get('mean', 999)\\n        is_workday = lb.get('is_workday', {{}}).get('mode', 0)\\n        smonth = lb.get('solar_month', {{}}).get('mode', 0)\\n        if temp < 0 and is_workday == 1 and smonth in [12, 1, 2]:\\n            matched.append(i)\\n    return matched"
  }}""",
}


AIR_BLOCK = {
    "domain_description": (
        "domain expert in PM2.5 air-quality forecasting for Riverside, "
        "California. The dominant drivers of PM2.5 concentration in this "
        "region are local emissions plus meteorological conditions that "
        "either disperse or trap pollutants — wind speed, boundary-layer "
        "height (vertical mixing depth), precipitation (which scavenges "
        "particles), and temperature inversions"
    ),

    "regime_label": "PM2.5 air-quality",

    "schema_example": """```python
{{
  "segment_id": int,
  "start_idx": int,
  "end_idx": int,
  "lookback_stats": {{
    # one entry per channel; continuous channels have:
    "temp_outdoor": {{
      "mean": float, "median": float, "std": float, "min": float,
      "max": float, "range": float, "skewness": float,
      "kurtosis": float, "slope": float, "volume": float
    }},
    # discrete channels have:
    "is_weekday": {{
      "mode": int, "n_unique": int, "distribution": {{"0": float, "1": float}}
    }},
    # ... one such entry per channel listed above (temp_outdoor,
    # rel_humidity, wind_spd_res, solar_radiation, uv_radiation,
    # baro_pressure, sp, skt, blh, tp, pm25_poc_3, month, day, hour,
    # day_of_week, is_weekend, is_weekday).
  }},
  "forecast_stats": {{
    # same structure as lookback_stats, but over the {output_length}-step forecast window
  }},
  "cross_channel_stats": {{
    "target_corr_temp_outdoor":  float,
    "target_corr_rel_humidity":  float,
    "target_corr_wind_spd_res":  float,
    "target_corr_baro_pressure": float,
    "target_corr_blh":           float,
    "is_uniform_day_type": bool,
    "uniform_day_type": "weekday" | "weekend"
  }},
  "time_context": {{
    "season": "spring" | "summer" | "autumn" | "winter",
    "forecast_start": {{
      "month": int, "day": int, "hour": int
    }},
    # also: lookback_start, lookback_end, forecast_end
  }}
}}
```

NOTE: this dataset's columns are literal — "month" is the calendar month
(1-12), "hour" is the hour of day (0-23), "day_of_week" is 0=Monday … 6=Sunday.
There is no lunar calendar field and no "minute" field (the data is hourly).
There are no holiday flags; day type is captured by the mutually-exclusive
``is_weekday`` / ``is_weekend`` pair.

Key meteorological channels for PM2.5:
  - ``wind_spd_res`` (m/s): low wind → poor horizontal dispersion → buildup.
  - ``blh`` (m, ERA5 boundary layer height): low BLH → shallow vertical
    mixing → trapped pollutants near surface. Often paired with low wind.
  - ``temp_outdoor`` (°F): cold mornings often coincide with surface
    inversions; hot summer afternoons can drive secondary aerosol formation.
  - ``rel_humidity`` (%): high humidity → hygroscopic particle growth.
  - ``tp`` (m): non-zero precipitation washes out aerosols (negative effect
    on PM2.5).
  - ``baro_pressure`` (hPa): high pressure → stagnant conditions, often
    correlated with elevated PM2.5.""",

    "example_prototype": """  {{
    "prototype_id": 1,
    "name": "stagnant_winter_morning",
    "description": "Winter early-morning weekday with low wind speed and shallow boundary layer trapping pollutants near the surface.",
    "function": "def select_segments(segments):\\n    matched = []\\n    for i, seg in enumerate(segments):\\n        lb = seg.get('lookback_stats', {{}})\\n        wind = lb.get('wind_spd_res', {{}}).get('mean', 999)\\n        blh = lb.get('blh', {{}}).get('mean', 9999)\\n        month = lb.get('month', {{}}).get('mode', 0)\\n        hour = lb.get('hour', {{}}).get('mode', 12)\\n        is_wkday = lb.get('is_weekday', {{}}).get('mode', 0)\\n        if wind < 2.0 and blh < 300 and month in [12, 1, 2] and hour in [5, 6, 7, 8, 9] and is_wkday == 1:\\n            matched.append(i)\\n    return matched"
  }}""",
}


# ---------------------------------------------------------------------------
# Resolve the active block at module load.
# ---------------------------------------------------------------------------
if DATASET == "LOF":
    _BLOCK = LOF_BLOCK
elif DATASET == "AirQuality":
    _BLOCK = AIR_BLOCK
else:
    raise ValueError(
        f"Unknown DATASET={DATASET!r} in prototype_proposing_loop.prompts. "
        f"Expected 'LOF' or 'AirQuality'."
    )


# ===========================================================================
# INITIAL_PROMPT — same skeleton across datasets, dataset-specific fragments
# are baked in at module-load time. ``{placeholders}`` left in the final
# string are filled at runtime by ``llm_proposer.format_initial_prompt``.
# ===========================================================================

INITIAL_PROMPT = f"""You are a {_BLOCK['domain_description']}. You are helping design a prototype-based forecasting model.

## Task Context

- Dataset: {{dataset_name}}
- Target: {{target_variable}} ({{target_description}})
- Time resolution: {{time_resolution}}
- Lookback window: {{input_length}} steps
- Forecast window: {{output_length}} steps
- Total training segments: {{total_segments}}

## Available Feature Channels

{{channel_descriptions}}

## Global Statistics (over all training segments)

{{global_statistics}}

## Cluster Summaries (training segments grouped into {{n_clusters}} natural regimes)

The training segments cluster into {{n_clusters}} natural groups by K-means on
per-segment summary vectors. Each cluster's "signature" tells you the
dominant value of each categorical channel and the deviation (in std-units)
of each continuous channel from the global mean. Use these as a guide for
choosing realistic thresholds.

{{cluster_summaries}}

## Segment Schema

Each segment is a dict with this structure:

{_BLOCK['schema_example']}

## Your Task

Design exactly {{n_prototypes}} prototypes that together cover the major
{_BLOCK['regime_label']} regimes in this dataset. The {{n_prototypes}} prototypes must
be MUTUALLY DISTINCT and COLLECTIVELY EXHAUSTIVE.

For each prototype, write a Python function that filters a list of segments
to those matching the prototype's regime.

Each function MUST satisfy ALL of these constraints:

1. Be named exactly `select_segments`
2. Take one argument `segments: list[dict]`
3. Return a list of integer indices (into `segments`) of matching segments
4. Use only standard Python — no `import`, no I/O, no attribute access on
   modules. Operate on the dict structure shown above.
5. Be deterministic and pure (no randomness, no global state).
6. Be defensive about missing keys: use `seg.get("lookback_stats", {{{{}}}}).get("CHANNEL", {{{{}}}}).get("STAT", default)`
   so the function never KeyErrors on a malformed segment.

## Output Format

Return a JSON array of {{n_prototypes}} objects with this exact shape:

```json
[
{_BLOCK['example_prototype']}
]
```

Important:
- Each prototype should match roughly 3-15% of total segments. Use the
  cluster summaries and global percentiles to choose realistic thresholds.
- Function code must be a single string with `\\n` line separators (so that
  the JSON stays valid). Keep functions short and readable; long control
  flow is unnecessary.
- The function bodies should reference real channel names and stat keys
  exactly as shown in the schema above.

Return ONLY the JSON array, no other text.
"""


# ===========================================================================
# REVISION_PROMPT — dataset-agnostic; reused unchanged across both schemas.
# ===========================================================================

REVISION_PROMPT = """You previously proposed prototypes whose Python `select_segments` functions
were validated against the full segment bank. Some failed the quality rubric.

## Current Prototypes

{current_prototypes}

## Validation Report

{feedback}

## Your Task

Revise the prototypes flagged as REVISION NEEDED so they pass every listed
failure mode. Keep each prototype's conceptual identity (don't change what
regime it represents); only adjust the function body — typically by
loosening or tightening thresholds, dropping over-restrictive predicates,
adding a discriminative one, or replacing one channel with a related one.

The revised functions must follow the same constraints as before:

1. Named `select_segments`, takes `segments: list[dict]`, returns list[int].
2. No imports, only standard Python.
3. Deterministic and pure.
4. Defensive `.get(...)` access for every nested field.

Return the COMPLETE list of all prototypes (revised + unchanged) as a JSON
array in the exact same format as before. No other text.
"""

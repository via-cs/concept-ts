"""
Prompt template for the 1 + N concept-bottleneck CBM. The LLM is asked
to return TWO sets of concepts in one JSON object:

  lookback_concepts  : K_lb atomic predicates over a full segment dict.
                       Used by the single LOOKBACK concept bottleneck.

  forecast_concepts  : K_fc atomic predicates over a SUB-WINDOW dict
                       (one of N=n_subwindows). Shared across the N
                       forecast concept bottlenecks -- the rules define
                       WHAT regimes exist, the per-sub-window activations
                       capture WHEN each regime fires.

Schema reminder for the LLM
---------------------------

Lookback rules see the standard segment dict:

    seg = {
      "lookback_stats": {channel -> {mean, std, slope, ...} | {mode, ...}},
      "forecast_stats": {channel -> ...},
      "cross_channel_stats": {...},
      "time_context": {lookback_start{...}, lookback_end{...},
                       forecast_start{...}, forecast_end{...}, season}
    }

Forecast rules see a SUB-WINDOW dict (one of the four blocks emitted by
``preprocessing.data_preprocessing.build_subwindow_blocks``):

    sub_seg = {
      "subwindow_idx": int (0..N-1),
      "stats":         {channel -> {mean, std, slope, ...} | {mode, ...}},
      "time_context":  {subwindow_idx, start{...}, end{...}, season},
      "cross_channel_stats": {...}  # inherited from the parent segment
    }
"""

DATASET = "LOF"


LOF_BLOCK = {
    "domain_description": (
        "domain expert in electricity load forecasting for a province in "
        "northern China"
    ),
    "regime_label": "electricity-load",

    "lookback_schema_example": """```python
{{
  "lookback_stats": {{
    "Skin_temperature": {{
      "mean": float, "median": float, "std": float, "min": float,
      "max": float, "range": float, "skewness": float,
      "kurtosis": float, "slope": float, "volume": float
    }},
    "is_workday": {{
      "mode": int, "n_unique": int, "distribution": {{"0": float, "1": float}}
    }},
    # ... 23 channels total, same continuous/discrete split as schema docs
  }},
  "forecast_stats": {{ ... same shape, computed over the {output_length}-step forecast window ... }},
  "cross_channel_stats": {{
    "load_corr_Skin_temperature": float,
    "load_corr_Relative_humidity": float,
    "load_corr_Surface_net_solar_radiation": float,
    "is_uniform_day_type": bool,
    "uniform_day_type": "workday" | "non_workday",
    "contains_holiday": bool,
    "is_all_holiday": bool
  }},
  "time_context": {{
    "lookback_start": {{"solar_month": int, "solar_day": int, "hour": int, "minute": int}},
    "lookback_end":   {{...}},
    "forecast_start": {{...}},
    "forecast_end":   {{...}},
    "season": "spring" | "summer" | "autumn" | "winter"
  }}
}}
```""",

    "subwindow_schema_example": """```python
{{
  "subwindow_idx": int (0..N-1, where N = {n_subwindows}),
  "stats": {{
    "Skin_temperature": {{
      "mean": float, "median": float, "std": float, "min": float,
      "max": float, "range": float, "skewness": float,
      "kurtosis": float, "slope": float, "volume": float
    }},
    "is_workday": {{
      "mode": int, "n_unique": int, "distribution": {{...}}
    }},
    # ... same 23 channels, stats computed over this sub-window's {subwindow_length} timesteps
  }},
  "time_context": {{
    "subwindow_idx": int,
    "start": {{"solar_month": int, "solar_day": int, "hour": int, "minute": int}},
    "end":   {{...}},
    "season": "spring" | "summer" | "autumn" | "winter"
  }},
  "cross_channel_stats": {{ ... same as the parent segment ... }}
}}
```

NOTE: with output_length = {output_length} and n_subwindows = {n_subwindows},
each sub-window covers {subwindow_length} consecutive timesteps. At 15-min
resolution that is {subwindow_hours} hours per sub-window.""",

    "example_lookback_concepts": """[
  {{
    "id": 0,
    "name": "is_history_cold",
    "description": "Lookback skin temperature mean below 0 deg C (heating regime).",
    "category": "weather_temperature",
    "function": "def matches(seg):\\n    t = seg.get('lookback_stats', {{}}).get('Skin_temperature', {{}}).get('mean')\\n    return t is not None and t < 0"
  }},
  {{
    "id": 1,
    "name": "is_history_load_high",
    "description": "Lookback load mean above 3500 MW (high-demand regime).",
    "category": "load_regime",
    "function": "def matches(seg):\\n    l = seg.get('lookback_stats', {{}}).get('load', {{}}).get('mean')\\n    return l is not None and l > 3500"
  }},
  {{
    "id": 2,
    "name": "is_history_workday",
    "description": "Lookback spans only workday timesteps.",
    "category": "calendar",
    "function": "def matches(seg):\\n    return seg.get('cross_channel_stats', {{}}).get('uniform_day_type') == 'workday'"
  }}
]""",

    "example_forecast_concepts": """[
  {{
    "id": 0,
    "name": "is_temperature_rising",
    "description": "Skin temperature trending upward inside this sub-window.",
    "category": "weather_dynamics",
    "function": "def matches(sub_seg):\\n    sl = sub_seg.get('stats', {{}}).get('Skin_temperature', {{}}).get('slope')\\n    return sl is not None and sl > 0.05"
  }},
  {{
    "id": 1,
    "name": "is_evening_block",
    "description": "Sub-window starts in the evening peak hours (16-21).",
    "category": "time_of_day",
    "function": "def matches(sub_seg):\\n    h = sub_seg.get('time_context', {{}}).get('start', {{}}).get('hour')\\n    return h is not None and 16 <= h <= 21"
  }},
  {{
    "id": 2,
    "name": "is_humid_block",
    "description": "Sub-window mean relative humidity above 70 percent.",
    "category": "weather_other",
    "function": "def matches(sub_seg):\\n    rh = sub_seg.get('stats', {{}}).get('Relative_humidity', {{}}).get('mean')\\n    return rh is not None and rh > 70"
  }}
]""",
}


if DATASET == "LOF":
    _BLOCK = LOF_BLOCK
else:
    raise ValueError(f"Unknown DATASET={DATASET!r} in CBM.concept_proposer.prompts.")


INITIAL_PROMPT = f"""You are a {_BLOCK['domain_description']}. You are helping design a CONCEPT BOTTLENECK MODEL for interpretable {_BLOCK['regime_label']} forecasting.

The model uses **1 + N concept bottlenecks**:

  * 1 LOOKBACK bottleneck: K_lb concepts that summarise the history window.
  * N FORECAST bottlenecks (one per forecast sub-window): K_fc shared concepts
    applied at each sub-window. Same vocabulary, different activations per
    sub-window.

You will propose BOTH concept sets in a single JSON response.

## Task Context

- Dataset: {{dataset_name}}
- Target: {{target_variable}} ({{target_description}})
- Time resolution: {{time_resolution}}
- Lookback window: {{input_length}} steps
- Forecast window: {{output_length}} steps split into {{n_subwindows}} sub-windows of {{subwindow_length}} steps each
- Total training segments: {{total_segments}}

## Available Feature Channels

{{channel_descriptions}}

## Global Statistics (across all training segments)

{{global_statistics}}

## Cluster Summaries ({{n_clusters}} natural regimes via K-means on segment summary vectors)

Use these to pick thresholds that actually split the data into the regimes
you see below.

{{cluster_summaries}}

## Schema 1 — segment dict (for LOOKBACK concepts)

A lookback concept function `def matches(seg) -> bool` receives this dict:

{_BLOCK['lookback_schema_example']}

## Schema 2 — sub-window dict (for FORECAST concepts)

A forecast concept function `def matches(sub_seg) -> bool` receives this dict
(one per sub-window, the same rule is applied to all {{n_subwindows}} sub-windows):

{_BLOCK['subwindow_schema_example']}

## Your Task

Propose:

  * Exactly **{{n_lookback_concepts}} LOOKBACK concepts**. These probe the HISTORY:
    things like "lookback temperature was cold", "lookback load was high",
    "lookback fully on workdays". Operate on `seg['lookback_stats']`,
    `seg['cross_channel_stats']`, `seg['time_context']` (especially
    `lookback_start`/`lookback_end`).

  * Exactly **{{n_forecast_concepts}} FORECAST concepts**. These probe ONE sub-window
    of the forecast at a time: things like "temperature rising in this block",
    "this is the evening peak block", "this block is humid". Operate on
    `sub_seg['stats']`, `sub_seg['time_context']` (especially `start.hour`,
    `season`), and `sub_seg['cross_channel_stats']`.

Hard requirements for EVERY function (both sets):

1. Named exactly ``matches``. Single positional arg.
2. **ATOMIC**: the body is ONE return statement evaluating a single boolean
   expression (helper variables to look up nested keys are fine). No loops,
   no if/else, no compound multi-axis predicates.
3. Defensive ``.get(...)`` access throughout -- missing keys must return
   ``False``, never raise.
4. No imports. Standard Python only.
5. Aim for **positive_rate in [5%, 60%]** across the {{total_segments}} segments
   (for forecast rules this is computed across all segment*sub-window pairs).
6. **Diversity** across categories: ``season``, ``time_of_day``, ``calendar``,
   ``weather_temperature``, ``weather_other``, ``weather_dynamics``,
   ``load_regime``, ``cross_channel``, ``misc``.
7. **No compounds**. ``is_history_cold`` is good; ``is_history_cold_and_workday``
   is bad (the CBM combines them at the embedding level).

## Output Format

Return a single JSON object with this exact shape:

```json
{{{{
  "lookback_concepts": [...{_BLOCK['example_lookback_concepts']}...],
  "forecast_concepts": [...{_BLOCK['example_forecast_concepts']}...]
}}}}
```

IDs in each array run 0..K-1 independently (lookback ids start at 0, forecast
ids also start at 0).

Return ONLY the JSON object. No prose before or after.
"""

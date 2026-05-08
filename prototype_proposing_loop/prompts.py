"""
Prompt templates for the prototype proposing loop (function-based).

The LLM is asked to return, per prototype, a Python function `select_segments`
that takes a list of segment dicts and returns matching indices. The full
schema is given inline so the LLM knows how to access fields.
"""


INITIAL_PROMPT = """You are a domain expert in electricity load forecasting for a province in northern China. You are helping design a prototype-based forecasting model.

## Task Context

- Dataset: {dataset_name}
- Target: {target_variable} ({target_description})
- Time resolution: {time_resolution}
- Lookback window: {input_length} steps
- Forecast window: {output_length} steps
- Total training segments: {total_segments}

## Available Feature Channels

{channel_descriptions}

## Global Statistics (over all training segments)

{global_statistics}

## Cluster Summaries (training segments grouped into {n_clusters} natural regimes)

The training segments cluster into {n_clusters} natural groups by K-means on
per-segment summary vectors. Each cluster's "signature" tells you the
dominant value of each categorical channel and the deviation (in std-units)
of each continuous channel from the global mean. Use these as a guide for
choosing realistic thresholds.

{cluster_summaries}

## Segment Schema

Each segment is a dict with this structure:

```python
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
"solar_month".

## Your Task

Design exactly {n_prototypes} prototypes that together cover the major
electricity-load regimes in this dataset. The {n_prototypes} prototypes must
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
6. Be defensive about missing keys: use `seg.get("lookback_stats", {{}}).get("CHANNEL", {{}}).get("STAT", default)`
   so the function never KeyErrors on a malformed segment.

## Output Format

Return a JSON array of {n_prototypes} objects with this exact shape:

```json
[
  {{
    "prototype_id": 1,
    "name": "cold_winter_workday",
    "description": "Cold winter workday regime with sustained heating load.",
    "function": "def select_segments(segments):\\n    matched = []\\n    for i, seg in enumerate(segments):\\n        lb = seg.get('lookback_stats', {{}})\\n        temp = lb.get('Skin_temperature', {{}}).get('mean', 999)\\n        is_workday = lb.get('is_workday', {{}}).get('mode', 0)\\n        smonth = lb.get('solar_month', {{}}).get('mode', 0)\\n        if temp < 0 and is_workday == 1 and smonth in [12, 1, 2]:\\n            matched.append(i)\\n    return matched"
  }}
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

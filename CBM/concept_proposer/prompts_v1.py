# ============================================================================
# FROZEN v1 PROMPT -- DO NOT MODIFY
# ============================================================================
# This file preserves the exact prompt text that generated the concept sets
# used for the BigData 2026 paper. The corresponding concept-JSON artifacts
# (concepts_validated_YYYYMMDD_*.json) were produced by importing THIS file.
#
# For active development, new experiments, or any post-submission runs,
# import ``prompts.py`` (v2) instead. v2 contains documentation fixes and
# clarifications; behavior on PRSA/hourly runs is essentially identical
# but LLM outputs will differ in wording.
#
# Summary of v1 -> v2 changes (see prompts.py for the full change log):
#   * Corrected architecture description ("1 + N" -> mode-specific counts).
#   * Added LOOKBACK-SUB-WINDOW to the "MAY reference PM2.5" allowlist.
#   * Removed forecast_stats and forecast_start / forecast_end from the
#     LOOKBACK schema examples (present in the dict but never intended
#     to be used by lookback concepts).
#   * Extended the delta_half trend-rule guidance to LOOKBACK-SUB-WINDOW.
#   * Schema 2 header now lists LOOKBACK-SUB-WINDOW alongside FORECAST
#     and FORECAST_GLOBAL.
#   * PRSA TARGET LEAKAGE warning clarifies that only forecast-side
#     concepts are banned from PM2.5, not LOOKBACK-SUB-WINDOW.
#   * Various minor wording clarifications.
#
# All of the above are documentation improvements. They do not fix
# functional bugs that would have affected the concepts stored in the
# JSON files backing the paper.
# ============================================================================

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

# Schema id from the single source of truth (dataset_config), selected by
# PROTO_DATASET (default "PRSA" → unchanged). Robust import so it works whether
# imported as a package or via propose_concepts.py's sibling-path trick.
try:
    from dataset_config import D as _DSCFG
except ModuleNotFoundError:                       # pragma: no cover
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))))
    from dataset_config import D as _DSCFG
DATASET = _DSCFG["name"]


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
      "kurtosis": float, "slope": float, "delta_half": float, "volume": float
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
      "kurtosis": float, "slope": float, "delta_half": float, "volume": float
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

    "example_lookback_sub_window_concepts": """[
  {{
    "id": 0,
    "name": "is_lookback_morning_rush_block",
    "description": "Lookback sub-window starts in the morning rush hours (7-10).",
    "category": "time_of_day",
    "function": "def matches(sub_seg):\\n    h = sub_seg.get('time_context', {{}}).get('start', {{}}).get('hour')\\n    return h is not None and 7 <= h <= 10"
  }},
  {{
    "id": 1,
    "name": "is_lookback_high_load_block",
    "description": "Lookback sub-window load mean above 3500 MW -- a high-demand chunk of recent history. This concept CAN reference load because the lookback is OBSERVED (unlike forecast concepts, which must not reference the target).",
    "category": "load_regime",
    "function": "def matches(sub_seg):\\n    l = sub_seg.get('stats', {{}}).get('load', {{}}).get('mean')\\n    return l is not None and l > 3500"
  }},
  {{
    "id": 2,
    "name": "is_lookback_cold_block",
    "description": "Lookback sub-window skin temperature mean below 0 deg C -- a cold, heating-driven chunk of recent history.",
    "category": "weather_temperature",
    "function": "def matches(sub_seg):\\n    t = sub_seg.get('stats', {{}}).get('Skin_temperature', {{}}).get('mean')\\n    return t is not None and t < 0"
  }},
  {{
    "id": 3,
    "name": "is_lookback_humid_block",
    "description": "Lookback sub-window mean relative humidity above 70 percent.",
    "category": "weather_other",
    "function": "def matches(sub_seg):\\n    rh = sub_seg.get('stats', {{}}).get('Relative_humidity', {{}}).get('mean')\\n    return rh is not None and rh > 70"
  }}
]""",

    "example_forecast_global_concepts": """[
  {{
    "id": 0,
    "name": "is_forecast_has_hot_hour",
    "description": "Maximum skin temperature over the WHOLE forecast exceeds 30 deg C -- the horizon contains at least one hot hour (cooling-driven demand). References a covariate, NOT the target load.",
    "category": "aggregate",
    "function": "def matches(sub_seg):\\n    m = sub_seg.get('stats', {{}}).get('Skin_temperature', {{}}).get('max')\\n    return m is not None and m > 30"
  }},
  {{
    "id": 1,
    "name": "is_forecast_temperature_rising",
    "description": "Skin temperature in the SECOND HALF of the forecast is meaningfully higher than the FIRST HALF. Uses delta_half (robust to the daily cycle) rather than OLS slope (which averages to ~0 on cyclical data).",
    "category": "trend",
    "function": "def matches(sub_seg):\\n    d = sub_seg.get('stats', {{}}).get('Skin_temperature', {{}}).get('delta_half')\\n    return d is not None and d > 1.0"
  }},
  {{
    "id": 2,
    "name": "is_forecast_uniformly_humid",
    "description": "Relative humidity across the whole forecast stays both high (mean > 70) AND uniform (std < 10) -- a persistently humid regime.",
    "category": "persistence",
    "function": "def matches(sub_seg):\\n    rh = sub_seg.get('stats', {{}}).get('Relative_humidity', {{}})\\n    m, s = rh.get('mean'), rh.get('std')\\n    return m is not None and s is not None and m > 70 and s < 10"
  }}
]""",
}


PRSA_BLOCK = {
    "domain_description": (
        "domain expert in urban air-quality forecasting (PM2.5) for Beijing. "
        "The dominant drivers of PM2.5 concentration in this region are local "
        "emissions plus meteorological conditions that either disperse or trap "
        "pollutants -- low wind speed (no horizontal dispersion), high pressure "
        "and inversions (no vertical mixing), high humidity (hygroscopic particle "
        "growth), precipitation (wet scavenging), and chemical interactions with "
        "co-emitted pollutants (PM10, SO2, NO2, CO, O3)"
    ),
    "regime_label": "PM2.5 air-quality",

    "lookback_schema_example": """```python
{{
  "lookback_stats": {{
    "TEMP": {{
      "mean": float, "median": float, "std": float, "min": float,
      "max": float, "range": float, "skewness": float,
      "kurtosis": float, "slope": float, "delta_half": float, "volume": float,
      # Frequency-domain features (ONLY present on lookback_stats for
      # continuous channels; sub-windows and forecast_stats do NOT have them).
      "fft_dom_period_h":  float,   # period (hours) of the strongest non-DC frequency
      "fft_dom_amp":       float,   # amplitude at that dominant period
      "fft_spec_entropy":  float,   # 0..1: 0 = strongly periodic, 1 = noisy / flat spectrum
      "fft_lowfreq_ratio": float,   # 0..1: fraction of energy in the low-frequency half
                                    #       (trend-vs-noise indicator)
      "fft_amp_p24h":      float,   # amplitude at the 24-hour period bin (daily cycle)
      "fft_amp_p12h":      float,   # amplitude at 12 hours (half-day / twice-daily peaks)
      "fft_amp_p8h":       float,   # amplitude at 8 hours
      "fft_amp_p6h":       float    # amplitude at 6 hours
    }},
    "hour": {{
      "mode": int, "n_unique": int, "distribution": {{"0": float, "1": float, ...}}
    }},
    # 15 channels total:
    #   discrete: year (0-4), month (1-12), day (1-31), hour (0-23)
    #   continuous: PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM, PM2.5
    # Every continuous channel has BOTH the standard stats and the fft_* features.
  }},
  "forecast_stats": {{ ... same continuous + discrete shape but WITHOUT fft_* fields ... }},
  "cross_channel_stats": {{
    "target_corr_PM10":  float,   # PM10 is the most-correlated pollutant
    "target_corr_WSPM":  float,   # wind speed (typically negative: wind disperses PM)
    "target_corr_TEMP":  float,
    "target_corr_DEWP":  float,   # dew point ~ humidity proxy
    "target_corr_PRES":  float,
    "target_corr_RAIN":  float
  }},
  "time_context": {{
    "lookback_start": {{"month": int, "day": int, "hour": int}},
    "lookback_end":   {{...}},
    "forecast_start": {{...}},
    "forecast_end":   {{...}},
    "season": "spring" | "summer" | "autumn" | "winter"
  }}
}}
```

NOTES:
- This dataset has no weekday/holiday flags. Day-type rules will not work.
- Wind direction (``wd``) was dropped during cleaning, so only wind SPEED (WSPM) is available.
- **Frequency stats**: the dataset is sampled hourly, the lookback covers
  {input_length} samples = {input_length} hours. The smallest period the FFT
  can resolve is 2h (Nyquist); the longest is the full window {input_length}h.
  ``fft_amp_p24h`` will be substantially larger than the other harmonics when
  a channel has a strong daily diurnal cycle (typical for TEMP, NO2, traffic-
  driven pollutants, less so for PM2.5 which depends more on weather regime).
  A high ``fft_dom_amp`` with ``fft_dom_period_h`` near 24 means "the signal
  is strongly daily-periodic". A high ``fft_spec_entropy`` (close to 1) means
  "no clear periodicity, looks like noise or a slow trend".""",

    "subwindow_schema_example": """```python
{{
  "subwindow_idx": int (0..N-1, where N = {n_subwindows}),
  "stats": {{
    "TEMP": {{
      "mean": float, "median": float, "std": float, "min": float,
      "max": float, "range": float, "skewness": float,
      "kurtosis": float, "slope": float, "delta_half": float, "volume": float
    }},
    "hour": {{
      "mode": int, "n_unique": int, "distribution": {{...}}
    }},
    # 14 USABLE channels here: year, month, day, hour, PM10, SO2, NO2, CO, O3,
    # TEMP, PRES, DEWP, RAIN, WSPM. PM2.5 IS THE PREDICTION TARGET and is
    # UNKNOWN in the forecast window -- its stats are masked/unavailable, so a
    # 'PM2.5' key here is NOT a valid input. Do NOT reference PM2.5.
  }},
  "time_context": {{
    "subwindow_idx": int,
    "start": {{"month": int, "day": int, "hour": int}},
    "end":   {{...}},
    "season": "spring" | "summer" | "autumn" | "winter"
  }},
  "cross_channel_stats": {{ ... same as the parent segment ... }}
}}
```

NOTE: with output_length = {output_length} and n_subwindows = {n_subwindows},
each sub-window covers {subwindow_length} consecutive timesteps. At 1-hour
resolution that is {subwindow_hours} hours per sub-window.

*** TREND-DIRECTION RULES — USE delta_half, NOT slope, ON SUB-WINDOWS ***
For any concept of the form "is X rising / falling in this block" or
"is X trending up over the forecast", USE THE ``delta_half`` STAT, not
``slope``.  Reason: PRSA sub-windows are {subwindow_hours} hours each.
If this is a full daily cycle (24h) the OLS slope of cyclical data like
TEMP/DEWP/wind/pollutants is systematically near zero (the sin/cos
oscillation averages out of the linear-fit coefficient) -- a ``slope > 0.1``
rule will basically never fire and the concept's AUROC will be terrible.
``delta_half`` is mean(second half) - mean(first half); it captures direction
even on cyclic-with-drift data. Use it as:
    ``stats['TEMP']['delta_half'] > 0.5``         -- "warmer in second half"
    ``stats['DEWP']['delta_half'] > 1.0``         -- "humidity rising in block"
    ``stats['WSPM']['delta_half'] < -0.3``        -- "wind falling in block"
``slope`` is still fine for LOOKBACK concepts (96-hour windows have several
days of drift dominating any one cycle) and for FORECAST-GLOBAL aggregate
concepts on multi-day horizons. Use ``slope`` only when the window length
is much longer than the dominant daily cycle.

*** TARGET LEAKAGE WARNING ***
PM2.5 is what the model PREDICTS. In the FORECAST window (every sub-window AND
the whole-forecast view used by global concepts) PM2.5 is unavailable to the
model -- it is masked out of the inputs. A forecast or forecast-global concept
that reads ``stats['PM2.5']`` would be defining a concept on the answer itself,
which is circular and forbidden. FORECAST-SUB-WINDOW and FORECAST-GLOBAL
concepts must be built ONLY from the known future drivers (meteorology: TEMP,
PRES, DEWP, RAIN, WSPM; other pollutants: PM10, SO2, NO2, CO, O3; and calendar).
ONLY lookback concepts may use PM2.5, because the history is observed.""",

    "example_lookback_concepts": """[
  {{
    "id": 0,
    "name": "is_history_low_wind",
    "description": "Lookback wind speed mean below 1.5 m/s -- stagnant air, pollutants accumulate.",
    "category": "weather_dynamics",
    "function": "def matches(seg):\\n    w = seg.get('lookback_stats', {{}}).get('WSPM', {{}}).get('mean')\\n    return w is not None and w < 1.5"
  }},
  {{
    "id": 1,
    "name": "is_history_winter_heating",
    "description": "Lookback spans winter (heating season -- elevated PM2.5 baseline).",
    "category": "season",
    "function": "def matches(seg):\\n    return seg.get('time_context', {{}}).get('season') == 'winter'"
  }},
  {{
    "id": 2,
    "name": "is_history_pm10_correlated",
    "description": "Strong positive correlation between PM2.5 and PM10 in the lookback (typical of dust / combustion sources).",
    "category": "cross_channel",
    "function": "def matches(seg):\\n    c = seg.get('cross_channel_stats', {{}}).get('target_corr_PM10')\\n    return c is not None and c > 0.7"
  }},
  {{
    "id": 3,
    "name": "is_history_strong_daily_temp_cycle",
    "description": "Lookback temperature has a strong 24-hour periodicity (clear-sky diurnal cycle -- typical of dry, stable weather without sustained cloud cover).",
    "category": "frequency",
    "function": "def matches(seg):\\n    a = seg.get('lookback_stats', {{}}).get('TEMP', {{}}).get('fft_amp_p24h')\\n    return a is not None and a > 60"
  }},
  {{
    "id": 4,
    "name": "is_history_pm25_noisy_spectrum",
    "description": "Lookback PM2.5 spectrum is broadband (high entropy) -- no single dominant period, suggesting episodic / event-driven pollution rather than a clean diurnal pattern.",
    "category": "frequency",
    "function": "def matches(seg):\\n    e = seg.get('lookback_stats', {{}}).get('PM2.5', {{}}).get('fft_spec_entropy')\\n    return e is not None and e > 0.85"
  }}
]""",

    "example_forecast_concepts": """[
  {{
    "id": 0,
    "name": "is_calm_block",
    "description": "Sub-window wind speed mean below 1.5 m/s -- no horizontal dispersion.",
    "category": "weather_dynamics",
    "function": "def matches(sub_seg):\\n    w = sub_seg.get('stats', {{}}).get('WSPM', {{}}).get('mean')\\n    return w is not None and w < 1.5"
  }},
  {{
    "id": 1,
    "name": "is_rush_hour_block",
    "description": "Sub-window starts in the morning rush hour (7-10).",
    "category": "time_of_day",
    "function": "def matches(sub_seg):\\n    h = sub_seg.get('time_context', {{}}).get('start', {{}}).get('hour')\\n    return h is not None and 7 <= h <= 10"
  }},
  {{
    "id": 2,
    "name": "is_humid_block",
    "description": "Sub-window dew point above 10 deg C (high absolute humidity -- particle growth).",
    "category": "weather_other",
    "function": "def matches(sub_seg):\\n    d = sub_seg.get('stats', {{}}).get('DEWP', {{}}).get('mean')\\n    return d is not None and d > 10"
  }}
]""",

    "example_lookback_sub_window_concepts": """[
  {{
    "id": 0,
    "name": "is_lookback_morning_rush_block",
    "description": "Lookback sub-window starts in the morning rush hour (7-10).",
    "category": "time_of_day",
    "function": "def matches(sub_seg):\\n    h = sub_seg.get('time_context', {{}}).get('start', {{}}).get('hour')\\n    return h is not None and 7 <= h <= 10"
  }},
  {{
    "id": 1,
    "name": "is_lookback_humid_block",
    "description": "Lookback sub-window dew point mean above 10 deg C (high absolute humidity in this recent chunk of history).",
    "category": "weather_other",
    "function": "def matches(sub_seg):\\n    d = sub_seg.get('stats', {{}}).get('DEWP', {{}}).get('mean')\\n    return d is not None and d > 10"
  }},
  {{
    "id": 2,
    "name": "is_lookback_pm25_spike_block",
    "description": "Lookback sub-window PM2.5 max above 100 ug/m3 -- a pollution spike in this recent 6-hour history chunk. This concept CAN reference PM2.5 because the lookback is observed (unlike forecast sub-window concepts which cannot).",
    "category": "pollutant_chemistry",
    "function": "def matches(sub_seg):\\n    m = sub_seg.get('stats', {{}}).get('PM2.5', {{}}).get('max')\\n    return m is not None and m > 100"
  }},
  {{
    "id": 3,
    "name": "is_lookback_calm_block",
    "description": "Lookback sub-window wind speed mean below 1.5 m/s -- stagnant air in this recent history chunk.",
    "category": "weather_dynamics",
    "function": "def matches(sub_seg):\\n    w = sub_seg.get('stats', {{}}).get('WSPM', {{}}).get('mean')\\n    return w is not None and w < 1.5"
  }}
]""",

    "example_forecast_global_concepts": """[
  {{
    "id": 0,
    "name": "is_forecast_has_stagnant_hour",
    "description": "Minimum wind speed over the WHOLE forecast drops below 0.5 m/s -- the forecast contains at least one near-calm hour, conditions in which airborne particles accumulate.",
    "category": "aggregate",
    "function": "def matches(sub_seg):\\n    m = sub_seg.get('stats', {{}}).get('WSPM', {{}}).get('min')\\n    return m is not None and m < 0.5"
  }},
  {{
    "id": 1,
    "name": "is_forecast_humidity_rising",
    "description": "Dew point in the SECOND HALF of the forecast is meaningfully higher than the FIRST HALF -- humidity (hence particle-growth potential) rises across the horizon. Uses delta_half (robust to daily cycle) rather than OLS slope (which averages to ~0 on cyclical data).",
    "category": "trend",
    "function": "def matches(sub_seg):\\n    d = sub_seg.get('stats', {{}}).get('DEWP', {{}}).get('delta_half')\\n    return d is not None and d > 1.0"
  }},
  {{
    "id": 2,
    "name": "is_forecast_uniformly_calm",
    "description": "Wind speed across the whole forecast stays both low (mean < 1.5) AND uniform (std < 0.5) -- prolonged stagnation regime.",
    "category": "persistence",
    "function": "def matches(sub_seg):\\n    w = sub_seg.get('stats', {{}}).get('WSPM', {{}})\\n    m, s = w.get('mean'), w.get('std')\\n    return m is not None and s is not None and m < 1.5 and s < 0.5"
  }}
]""",
}


if DATASET == "LOF":
    _BLOCK = LOF_BLOCK
elif DATASET == "PRSA":
    _BLOCK = PRSA_BLOCK
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

## Schema 2 — sub-window dict (for FORECAST and FORECAST_GLOBAL concepts)

A forecast concept function `def matches(sub_seg) -> bool` receives this dict.
Per-sub-window forecast concepts get one such dict per sub-window (rule applied
{{n_subwindows}} times). Global-forecast concepts get ONE such dict per segment
whose ``stats`` and ``time_context`` cover the WHOLE {{output_length}}-step
forecast window (i.e. ``stats`` is the segment's full ``forecast_stats`` and
``time_context.start`` / ``end`` are the segment's ``forecast_start`` / ``end``):

{_BLOCK['subwindow_schema_example']}

## Your Task

============================================================================
*** CRITICAL CONSTRAINT — READ THIS BEFORE PROPOSING ANY CONCEPT ***
============================================================================

PM2.5 IS THE PREDICTION TARGET. The forecast window's PM2.5 values are what
the model has to predict — they are MASKED / UNAVAILABLE at inference time.

  - LOOKBACK concepts MAY reference PM2.5 (the history is observed).
  - FORECAST-SUB-WINDOW concepts MUST NOT REFERENCE PM2.5 IN ANY WAY.
  - FORECAST-GLOBAL concepts MUST NOT REFERENCE PM2.5 IN ANY WAY.

Forecast-side concepts must be built EXCLUSIVELY from CONTEXT FEATURES:
meteorology (TEMP, PRES, DEWP, RAIN, WSPM), other pollutants (PM10, SO2,
NO2, CO, O3), and calendar (year, month, day, hour, season). NOTHING ELSE.

A forecast-side rule that touches PM2.5 — directly OR indirectly — defines
the concept on the answer itself. It is circular, it is data leakage, and
it will be REJECTED.

What "REFERENCING PM2.5" means in this prompt (all forbidden on forecast side):
  * Any string ``'PM2.5'`` appearing inside the function body
  * Any stats lookup keyed on PM2.5: ``stats['PM2.5']``, ``stats.get('PM2.5')``
  * Any FFT / mean / std / min / max / slope / median / quantile of PM2.5
  * Any "is_high_pollution_episode" / "is_PM_spike" / "is_pm25_rising" / etc.
    where the rule's TRUE meaning depends on PM2.5 magnitude
  * Any cross_channel field that you intend to interpret as a PM2.5 signal
    (note: ``cross_channel_stats.target_corr_*`` is computed from LOOKBACK
    PM2.5 only — it is OK to use on the LOOKBACK side, but on the forecast
    side these stats are the SAME lookback-derived values, not forecast PM2.5;
    using them in a forecast/global concept is permitted ONLY when you
    legitimately mean "lookback PM2.5 correlated with X". If you find
    yourself reaching for ``target_corr_*`` in a forecast-side concept,
    please reconsider whether the concept actually belongs in the LOOKBACK
    set instead.)

FORBIDDEN forecast/global concept examples (do NOT propose anything like these):
  * "is_forecast_pollution_episode" — would read PM2.5 max
  * "is_pm25_rising_block"          — slope of PM2.5
  * "is_high_pm25_block"            — mean PM2.5
  * "is_forecast_pm_spike"          — PM2.5 threshold
  * "is_pollutant_buildup"          — usually code reads PM2.5

PERMITTED forecast/global concept examples (covariates only):
  * "is_calm_block"                 — wind speed mean low
  * "is_humid_block"                — dew point mean high
  * "is_rush_hour_block"            — start hour in [7, 10]
  * "is_forecast_uniformly_calm"    — wind low + uniform across whole horizon
  * "is_forecast_humidity_rising"   — dew point slope > 0 over whole horizon
  * "is_low_wind_AND_high_pm10_carryover" — wind speed mean low (FUTURE) and
    PM10 mean high (FUTURE other-pollutant carryover from continuing emissions)

Before you write each forecast-side or global concept, ASK YOURSELF:
"Does this rule's truth value depend on a PM2.5 value the model would not
have at test time?" If yes (or unsure), REPLACE IT with a covariate-only rule.

============================================================================

Propose FOUR concept sets:

  * Exactly **{{n_lookback_concepts}} LOOKBACK concepts**. These probe the HISTORY:
    things like "lookback temperature was cold", "lookback load was high",
    "lookback fully on workdays". Operate on `seg['lookback_stats']`,
    `seg['cross_channel_stats']`, `seg['time_context']` (especially
    `lookback_start`/`lookback_end`).
    LOOKBACK concepts MAY use ``lookback_stats['PM2.5']`` because the history
    is observed.

  * Exactly **{{n_lookback_sub_window_concepts}} LOOKBACK-SUB-WINDOW concepts**.
    These probe ONE sub-window of the LOOKBACK history at a time (each rule
    fires {{n_lookback_subwindows}} times per segment, once per lookback
    sub-window): things like "this lookback sub-window is calm", "this
    lookback sub-window had a PM2.5 spike", "this lookback sub-window is
    morning rush hour". Operate on ``sub_seg['stats']`` and
    ``sub_seg['time_context']`` (especially ``start.hour``, ``season``).
    These are LOCAL, time-resolved descriptors of the RECENT HISTORY.
    LOOKBACK-SUB-WINDOW concepts MAY use ``stats['PM2.5']`` because the
    lookback is observed. They're intended for the ``experiment_mode=
    'lookback_only'`` training path (no future features available), where
    all forecasting signal has to come from characterising recent history
    chunks.

  * Exactly **{{n_forecast_concepts}} FORECAST-SUB-WINDOW concepts**. These probe
    ONE sub-window of the forecast at a time (each rule fires {{n_subwindows}}
    times per segment, once per sub-window): things like "temperature rising
    in this block", "this is the evening peak block", "this block is humid".
    Operate on `sub_seg['stats']`, `sub_seg['time_context']` (especially
    `start.hour`, `season`), and `sub_seg['cross_channel_stats']`.
    These are LOCAL, time-resolved descriptors.

    *** ABSOLUTE RULE — NO PM2.5 IN FORECAST-SUB-WINDOW CONCEPTS ***
    DO NOT reference ``stats['PM2.5']`` in any way — no mean, max, min, std,
    slope, fft, threshold, comparison. PM2.5 is the prediction target and is
    masked in the forecast window. A forecast-sub-window concept that touches
    PM2.5 is leakage. Use ONLY context features: meteorology (TEMP, PRES,
    DEWP, RAIN, WSPM), other pollutants (PM10, SO2, NO2, CO, O3), calendar.

  * Exactly **{{n_forecast_global_concepts}} FORECAST-GLOBAL concepts**. These probe
    the WHOLE forecast window as a single unit (each rule fires ONCE per
    segment). They should capture properties that DO NOT decompose into
    sub-window concepts -- aggregates, thresholds spanning the whole horizon,
    persistence across the forecast, cross-block comparisons.

    *** ABSOLUTE RULE — NO PM2.5 IN FORECAST-GLOBAL CONCEPTS ***
    Same prohibition as forecast-sub-window: do NOT reference ``stats['PM2.5']``
    in any way. Built from KNOWN future drivers ONLY — never from PM2.5 (the
    masked target). Use ONLY meteorology (TEMP, PRES, DEWP, RAIN, WSPM),
    other pollutants (PM10, SO2, NO2, CO, O3), and calendar.

    Good examples:
      - "min wind speed over the forecast drops below 0.5 m/s" (aggregate threshold)
      - "dew point slope across the WHOLE forecast is positive" (whole-horizon trend)
      - "wind stays low AND uniform across the whole forecast" (persistence)
      - "temperature range across the forecast exceeds 15 deg C" (aggregate span)
    Operate on the same ``sub_seg`` API, but the ``stats`` here cover the full
    forecast window so e.g. ``stats['WSPM']['min']`` is the min wind over all
    {{output_length}} forecast steps. Bad examples (avoid these -- they duplicate
    the sub-window set):
      - "forecast is in winter" (use a sub-window concept; it'll fire on every sub-window of a winter segment)
      - "forecast has at least one humid hour" (better as sub-window "is_humid_block")
      - "temperature is rising during the forecast" (better as sub-window dynamics)

Hard requirements for EVERY function (all three sets):

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
   ``pollutant_chemistry``, ``cross_channel``, ``regime``,
   ``frequency`` (FFT/periodicity-based, **lookback only**),
   ``aggregate`` / ``threshold`` / ``persistence`` / ``trend`` (good fits for
   the FORECAST-GLOBAL set), ``misc``.
7. **No compounds**. ``is_history_cold`` is good; ``is_history_cold_and_workday``
   is bad (the CBM combines them at the embedding level).
8. **NO TARGET LEAKAGE — ABSOLUTE PROHIBITION (most-violated rule, see top)**.
   PM2.5 is the forecast target. It is OBSERVED in the lookback but
   UNKNOWN (masked) in the forecast window. Therefore:
     - LOOKBACK concepts MAY use ``lookback_stats['PM2.5']``.
     - FORECAST-SUB-WINDOW and FORECAST-GLOBAL concepts MUST NOT reference
       ``stats['PM2.5']`` in any way (no mean/max/min/std/slope/median/fft/
       threshold/comparison of PM2.5). A forecast-side rule that touches PM2.5
       defines a concept on the answer itself and will be REJECTED. Use ONLY
       context features (meteorology, other pollutants, calendar) for any
       FORECAST-SUB-WINDOW or FORECAST-GLOBAL concept.
     - Also reject any forecast-side concept whose NAME or DESCRIPTION mentions
       PM2.5, "PM25", "pollution episode", "pm spike", "particulate spike",
       "AQI", etc. — even if the code seems not to read PM2.5 — because such
       a concept is implicitly trying to detect the target, and almost always
       either the code is wrong or the description is misleading. Use covariate
       names in the concept name (``is_calm_block``, not ``is_pm25_block``).

9. **MANDATORY SELF-CHECK (do this before producing JSON)**. After drafting
   the lists, scan every entry in ``forecast_concepts`` and
   ``forecast_global_concepts``:
     a. Does the ``function`` body string contain ``"PM2.5"`` anywhere?
        If YES → DELETE this concept and propose a covariate-only replacement.
     b. Does the ``name`` or ``description`` contain ``pm25`` / ``PM2.5`` /
        ``pollution episode`` / ``particulate``?
        If YES → DELETE this concept and propose a covariate-only replacement.
     c. Does the rule's TRUTH VALUE actually depend on PM2.5 magnitude (even
        if read via a wrapper, fetched via cross-channel, etc.)?
        If YES → DELETE this concept and propose a covariate-only replacement.
   Only after BOTH forecast lists pass this self-check should you emit JSON.

## Output Format

Return a single JSON object with this exact shape:

```json
{{{{
  "lookback_concepts":            [...{_BLOCK['example_lookback_concepts']}...],
  "lookback_sub_window_concepts": [...{_BLOCK['example_lookback_sub_window_concepts']}...],
  "forecast_concepts":            [...{_BLOCK['example_forecast_concepts']}...],
  "forecast_global_concepts":     [...{_BLOCK['example_forecast_global_concepts']}...]
}}}}
```

IDs in each array run 0..K-1 independently (lookback ids start at 0,
lookback-sub-window ids start at 0, forecast-sub-window ids start at 0,
forecast-global ids start at 0).

The trainer uses either the forecast side (in ``experiment_mode='future_aware'``)
OR the lookback-sub-window side (in ``experiment_mode='lookback_only'``);
the LOOKBACK (global) set is always used. Emitting all four in one shot means
one LLM run supports both experiment modes.

Return ONLY the JSON object. No prose before or after.
"""

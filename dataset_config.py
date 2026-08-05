"""
Single source of truth for every dataset-schema-dependent constant.

The active schema is chosen ONCE, at import time, from the ``PROTO_DATASET``
environment variable (default ``"PRSA"``). Because the default reproduces the
current Beijing-PM2.5 setup exactly, any process that does NOT set the env var
— the CBM new-model runs, all existing PM2.5 pipelines — resolves to precisely
the previous values and is unchanged. To run on LOF/RE, launch with a shell
prefix, e.g.::

    PROTO_DATASET=LOF python -u CBM/train_cbm.py ...

Every schema-dependent module imports ``D`` (the active config dict) or
``NAME`` from here instead of hardcoding ``"PM2.5"``, column indices, etc.

Fields in ``D``
---------------
    name                        schema id ("PRSA" | "LOF" | "AirQuality")
    target                      target column NAME (last column of the CSV)
    rows_per_day                samples per day (24 @1h, 96 @15min)
    input_dim                   #channels the window tensor carries (incl. target)
    unit                        target unit label for reports ("µg/m³" | "MW")
    time_resolution_hours       hours per sample (1.0 | 0.25)
    base_year, step_minutes     for reconstructing timestamps in case studies
    cat_cols                    categorical/calendar column NAMES (comparison loaders)
    class_feature_indices       categorical column INDICES (channel embedders)
    class_feature_num_classes   per-column #categories (0 = continuous), full length
    tod_channels                [(col_idx, multiplier), ...] → time-of-day index
    tod_period                  base period of the time-of-day sinusoid
    tod_max                     clamp max of the time-of-day index (rows_per_day-1)
    hour_channel                primary hour column index (also the enable flag)
    channel_meta                {col_name -> {type, description}} for preprocessing/LLM
    case_feats                  {display_name -> col_idx} continuous drivers to plot
    domain_block                which prompt block to use ("PRSA_BLOCK" | "LOF_BLOCK")
"""
import os

NAME = os.environ.get("PROTO_DATASET", "PRSA")


# ---------------------------------------------------------------------------
# PRSA — 15-column Beijing PM2.5 schema (target "PM2.5", last column).
#   layout: 0 year, 1 month, 2 day, 3 hour, 4 PM10, 5 SO2, 6 NO2, 7 CO, 8 O3,
#           9 TEMP, 10 PRES, 11 DEWP, 12 RAIN, 13 WSPM, 14 PM2.5(target)
# ---------------------------------------------------------------------------
_PRSA = dict(
    name="PRSA",
    target="PM2.5",
    rows_per_day=24,
    input_dim=15,
    unit="µg/m³",
    time_resolution_hours=1.0,
    base_year=2013,
    step_minutes=60,
    cat_cols=("year", "month", "day", "hour"),
    class_feature_indices=[0, 1, 2, 3],
    class_feature_num_classes=[
        5, 13, 32, 24,     # 0-3   year, month, day, hour
        0, 0, 0, 0, 0,     # 4-8   PM10, SO2, NO2, CO, O3
        0, 0, 0, 0, 0,     # 9-13  TEMP, PRES, DEWP, RAIN, WSPM
        0,                  # 14    PM2.5 (target)
    ],
    # PRSA has a single hour column (0-23), 24-hour sinusoid.
    tod_channels=[(3, 1.0)],
    tod_period=24.0,
    tod_max=23,
    hour_channel=3,
    channel_meta={
        "year":  {"type": "discrete", "description": "Year index 0..4 (mapped from raw 2013..2017 during cleaning)"},
        "month": {"type": "discrete", "description": "Calendar month (1-12)"},
        "day":   {"type": "discrete", "description": "Day of month (1-31)"},
        "hour":  {"type": "discrete", "description": "Hour of day (0-23)"},
        "PM10":  {"type": "continuous", "description": "PM10 concentration (μg/m³)"},
        "SO2":   {"type": "continuous", "description": "SO2 concentration (μg/m³)"},
        "NO2":   {"type": "continuous", "description": "NO2 concentration (μg/m³)"},
        "CO":    {"type": "continuous", "description": "CO concentration (μg/m³)"},
        "O3":    {"type": "continuous", "description": "O3 concentration (μg/m³)"},
        "TEMP":  {"type": "continuous", "description": "Surface temperature (°C)"},
        "PRES":  {"type": "continuous", "description": "Surface atmospheric pressure (hPa)"},
        "DEWP":  {"type": "continuous", "description": "Dew-point temperature (°C); higher = more humid air"},
        "RAIN":  {"type": "continuous", "description": "Hourly precipitation (mm)"},
        "WSPM":  {"type": "continuous", "description": "Wind speed (m/s) — low wind suppresses pollutant dispersion"},
        "PM2.5": {"type": "continuous", "description": "PM2.5 concentration (target, μg/m³)"},
    },
    case_feats={"TEMP": 9, "PRES": 10, "DEWP": 11, "RAIN": 12, "WSPM": 13},
    domain_block="PRSA_BLOCK",
)


# ---------------------------------------------------------------------------
# LOF — 23-column electricity-load schema (target "load", last column).
#   layout: 0 lunar_year, 1 lunar_month, 2 lunar_day, 3 solar_year,
#           4 solar_month, 5 solar_day, 6 month(=hour 0-23), 7 minute(0/15/30/45),
#           8 Skin_temperature, 9 Surface_pressure, 10 Surface_sensible_heat_flux,
#           11 Total_cloud_cover, 12 Surface_net_solar_radiation,
#           13 Total_precipitation_4h_ahead, 14 10_metre_wind, 15 Relative_humidity,
#           16 is_special_workday, 17 is_special_holiday, 18 is_holiday,
#           19 is_workday, 20 day_of_week, 21 pricing, 22 load(target)
# ---------------------------------------------------------------------------
_LOF = dict(
    name="LOF",
    target="load",
    rows_per_day=96,
    input_dim=23,
    unit="MW",
    time_resolution_hours=0.25,
    base_year=None,          # solar_year is an index (2..5), not a calendar year
    step_minutes=15,
    cat_cols=(
        "lunar_year", "lunar_month", "lunar_day", "solar_year", "solar_month",
        "solar_day", "month", "minute", "is_special_workday", "is_special_holiday",
        "is_holiday", "is_workday", "day_of_week",
    ),
    class_feature_indices=[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20],
    class_feature_num_classes=[
        6, 13, 32, 6, 13,      # 0-4
        32, 25, 61, 0, 0,      # 5-9
        0, 0, 0, 0, 0,         # 10-14
        0, 2, 2, 2, 2,         # 15-19
        7, 0, 0,               # 20-22
    ],
    # LOF is 15-min: combined time-of-day index = hour*4 + minute/15 -> 0..95,
    # with a 96-step base period (option c).
    tod_channels=[(6, 4.0), (7, 1.0 / 15.0)],
    tod_period=96.0,
    tod_max=95,
    hour_channel=6,          # primary hour column (also enables the augmentation)
    channel_meta={
        "lunar_year":  {"type": "discrete", "description": "Year in Chinese lunar calendar (1-5, cyclical)"},
        "lunar_month": {"type": "discrete", "description": "Month in Chinese lunar calendar (1-12)"},
        "lunar_day":   {"type": "discrete", "description": "Day in Chinese lunar calendar (1-30)"},
        "solar_year":  {"type": "discrete", "description": "Solar calendar year index (2-5)"},
        "solar_month": {"type": "discrete", "description": "Month of year (1-12)"},
        "solar_day":   {"type": "discrete", "description": "Day of month (1-31)"},
        "month":       {"type": "discrete", "description": "Hour of day (0-23, despite column name 'month')"},
        "minute":      {"type": "discrete", "description": "Minute within hour (0, 15, 30, 45)"},
        "is_special_workday": {"type": "discrete", "description": "Special working day (0/1)"},
        "is_special_holiday": {"type": "discrete", "description": "Special holiday (0/1)"},
        "is_holiday":  {"type": "discrete", "description": "Holiday (0/1)"},
        "is_workday":  {"type": "discrete", "description": "Workday (0/1)"},
        "day_of_week": {"type": "discrete", "description": "Day of week (0=Monday .. 6=Sunday)"},
        "pricing":     {"type": "discrete", "description": "Electricity pricing tier (0.1, 0.3, 1.0, 1.7, 2.0)"},
        "Skin_temperature":             {"type": "continuous", "description": "Surface skin temperature (°C)"},
        "Surface_pressure":             {"type": "continuous", "description": "Surface atmospheric pressure (kPa)"},
        "Surface_sensible_heat_flux":   {"type": "continuous", "description": "Surface sensible heat flux (W/m²)"},
        "Total_cloud_cover":            {"type": "continuous", "description": "Total cloud cover"},
        "Surface_net_solar_radiation":  {"type": "continuous", "description": "Net solar radiation at surface (W/m²)"},
        "Total_precipitation_4h_ahead": {"type": "continuous", "description": "Forecasted total precipitation 4h ahead (mm)"},
        "10_metre_wind":                {"type": "continuous", "description": "Wind speed at 10m height (m/s)"},
        "Relative_humidity":            {"type": "continuous", "description": "Relative humidity (%)"},
        "load":        {"type": "continuous", "description": "Electricity load (target variable, MW)"},
    },
    case_feats={
        "Skin_temp": 8, "Sfc_press": 9, "Cloud": 11,
        "Wind": 14, "Rel_hum": 15,
    },
    domain_block="LOF_BLOCK",
)


# ---------------------------------------------------------------------------
# AirQuality — 17-column Riverside schema (target "pm25_poc_3"). Kept so the
# channel embedders' existing AirQuality branch still resolves; not wired into
# preprocessing (which only ever supported LOF/PRSA).
#   layout: ... 10 month, 11 day, 12 hour, 13 day_of_week, 14 is_weekend,
#           15 is_weekday, 16 pm25_poc_3(target)
# ---------------------------------------------------------------------------
_AIRQUALITY = dict(
    name="AirQuality",
    target="pm25_poc_3",
    rows_per_day=24,
    input_dim=17,
    unit="µg/m³",
    time_resolution_hours=1.0,
    base_year=None,
    step_minutes=60,
    cat_cols=("month", "day", "hour", "day_of_week", "is_weekend", "is_weekday"),
    class_feature_indices=[10, 11, 12, 13, 14, 15],
    class_feature_num_classes=[
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        13, 32, 24, 7, 2,
        2, 0,
    ],
    tod_channels=[(12, 1.0)],
    tod_period=24.0,
    tod_max=23,
    hour_channel=12,
    channel_meta={},         # preprocessing does not support AirQuality
    case_feats={},
    domain_block="PRSA_BLOCK",
)


_ALL = {"PRSA": _PRSA, "LOF": _LOF, "AirQuality": _AIRQUALITY}

if NAME not in _ALL:
    raise ValueError(
        f"PROTO_DATASET={NAME!r} not recognized; expected one of {list(_ALL)}."
    )

D = _ALL[NAME]


def get(key):
    """Convenience accessor: dataset_config.get('target')."""
    return D[key]

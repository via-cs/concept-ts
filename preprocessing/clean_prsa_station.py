"""
Clean a raw Beijing PRSA station CSV into the ProtoLLM training format
(identical to ``dataset/PRSA_Data/Aotizhongxin.csv``).

Recipe (verified to reproduce Aotizhongxin.csv byte-for-byte):
  1. Drop the ``No``, ``wd`` (wind direction), and ``station`` columns.
  2. Remap ``year`` 2013..2017 -> index 0..4.
  3. Move the target ``PM2.5`` to the LAST column.
  4. Fill every data column's missing values (NA) by LINEAR interpolation
     (``limit_direction="both"`` so leading/trailing gaps use the nearest value).
     Calendar columns (year/month/day/hour) have no missing values.
  5. No rounding — full float precision is preserved.

Usage
-----
    python preprocessing/clean_prsa_station.py \\
        --raw   dataset/PRSA_Data/PRSA_Data_Tiantan_20130301-20170228.csv \\
        --out   dataset/PRSA_Data/Tiantan.csv

    # or batch several at once (output name = station, e.g. Tiantan.csv):
    python preprocessing/clean_prsa_station.py \\
        --raw dataset/PRSA_Data/PRSA_Data_Tiantan_20130301-20170228.csv \\
              dataset/PRSA_Data/PRSA_Data_Dingling_20130301-20170228.csv \\
        --out_dir dataset/PRSA_Data
"""
import argparse
import os

import pandas as pd

ORDER = ["year", "month", "day", "hour", "PM10", "SO2", "NO2", "CO", "O3",
         "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "PM2.5"]
DATACOLS = [c for c in ORDER if c not in ("year", "month", "day", "hour")]
DROP = ["No", "wd", "station"]


def clean_station(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path).drop(columns=DROP)
    df["year"] = df["year"] - 2013
    df = df[ORDER]                                             # PM2.5 -> last
    df[DATACOLS] = df[DATACOLS].interpolate(method="linear",
                                            limit_direction="both")
    return df


def _station_name(raw_path: str) -> str:
    base = os.path.basename(raw_path)
    # "PRSA_Data_Tiantan_20130301-20170228.csv" -> "Tiantan"
    if base.startswith("PRSA_Data_"):
        return base[len("PRSA_Data_"):].split("_")[0]
    return os.path.splitext(base)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", nargs="+", required=True, help="Raw PRSA station CSV(s).")
    p.add_argument("--out", default=None, help="Output path (only if a single --raw).")
    p.add_argument("--out_dir", default=None,
                   help="Directory for outputs; each file named <Station>.csv.")
    args = p.parse_args()

    if args.out and len(args.raw) > 1:
        p.error("--out takes a single --raw; use --out_dir for multiple.")

    for raw in args.raw:
        df = clean_station(raw)
        if args.out:
            out = args.out
        else:
            out_dir = args.out_dir or os.path.dirname(raw)
            out = os.path.join(out_dir, f"{_station_name(raw)}.csv")
        df.to_csv(out, index=False)
        na = int(pd.read_csv(raw).drop(columns=DROP).isna().sum().sum())
        print(f"  {os.path.basename(raw)} -> {out}  "
              f"({len(df):,} rows, {na:,} NA cells interpolated)")


if __name__ == "__main__":
    main()

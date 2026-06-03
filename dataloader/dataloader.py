import os
import random
import numpy as np
import pandas as pd
import torch

from torch.utils import data
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Train / val / test layout.
#
# We dedicate the final year of the CSV to evaluation, and interleave it into
# val / test so the model is scored across every season:
#
#   1. Training set     = rows [0 : N - ROWS_PER_YEAR)        — everything
#                                                               before the
#                                                               last year.
#   2. Last-year region = rows [N - ROWS_PER_YEAR : N)        — last 35040
#                                                               rows.
#   3. Last year is chopped into NUM_CHUNKS equal bi-weekly chunks
#      (1460 rows ≈ 15.2 days each). A round-robin assignment
#      ("V T T V T T …" with stride VAL_STRIDE) sends every 3rd chunk to
#      validation and the rest to test, so VAL covers 8/24 = 33.3 % and TEST
#      covers 16/24 = 66.7 % of the last year, distributed across all seasons.
#   4. Within each chunk we only emit windows whose entire
#      (input_length + output_length) frame lies inside that chunk, so no val
#      window's input overlaps a test row and vice-versa.
#   5. StandardScaler is fit on the training output only (everything before
#      the last year) — val/test stats never leak into the scaler.
#
# To revert to the original 70/10/20 chronological split, set USE_LAST_YEAR_SPLIT = False.
# ---------------------------------------------------------------------------

USE_LAST_YEAR_SPLIT = False

# ---------------------------------------------------------------------------
# Dataset switch — flip this single constant to point the dataloader at a
# different schema/resolution. TARGET_COLUMN and ROWS_PER_DAY below are
# derived automatically.
#
#   "LOF"        — 15-min resolution, target column "load"
#                  (96 rows/day, ~35 040 rows/year).
#   "AirQuality" — 1-hour resolution, target column "pm25_poc_3"
#                  (24 rows/day, ~8 760 rows/year). The CSV's
#                  ``timestamp_utc`` column is auto-dropped since it's
#                  non-numeric.
# ---------------------------------------------------------------------------
DATASET = "LOF"

if DATASET == "LOF":
    TARGET_COLUMN = "load"
    ROWS_PER_DAY  = 96               # 15-min resolution: 96 rows / day
elif DATASET == "AirQuality":
    TARGET_COLUMN = "pm25_poc_3"
    ROWS_PER_DAY  = 24               # 1-hour resolution: 24 rows / day
else:
    raise ValueError(
        f"Unknown DATASET={DATASET!r}. Expected 'LOF' or 'AirQuality'."
    )

ROWS_PER_YEAR = 365 * ROWS_PER_DAY
NUM_CHUNKS    = 24               # 24 bi-weekly chunks across the last year
VAL_STRIDE    = 3                # every VAL_STRIDE-th chunk is a val chunk


class LOF_Dataset(data.Dataset):
    def __init__(self, input_length, output_length, train, in_dim, data_path):
        self.in_dim = in_dim
        self.output_length = output_length
        self.input_length = input_length
        self.length = input_length + output_length
        self.data_path = data_path
        self.train = train

        # Generate output mask
        mask = [1] * self.length
        for i in range(self.output_length):
            mask[self.length - 1 - i] = 0
        self.mask = torch.Tensor(mask).reshape(self.length, 1)

        # Sparse window indexing — used for val/test when USE_LAST_YEAR_SPLIT
        # is on. For dense splits (train / no-split) this stays None and
        # __getitem__ falls back to direct index → start_id.
        self.valid_starts = None

        # Load data
        self.get_data(train, data_path)

    def __getitem__(self, index):
        if self.valid_starts is not None:
            start_id = self.valid_starts[index]
        else:
            start_id = index
        mid_id = start_id + self.length - self.output_length
        end_id = start_id + self.length

        # Build input feature + target mask
        input_feature = torch.Tensor(self.input[start_id:end_id])
        input_ans = torch.Tensor(self.output[start_id:end_id]) * self.mask
        input_data = torch.concat([input_feature, input_ans], dim=-1)

        # Ground truth for last output_length frames
        ans = torch.Tensor(self.output[mid_id:end_id])

        return (
            input_data.reshape(self.length, self.in_dim).detach(),
            ans.reshape(self.output_length, 1).detach(),
        )

    def get_data(self, train, data_path):
        dataset = pd.read_csv(data_path, encoding="utf-8")

        # Extract the target column (its name depends on the DATASET switch
        # at the top of this file).
        if TARGET_COLUMN not in dataset.columns:
            raise KeyError(
                f"Target column {TARGET_COLUMN!r} (required by DATASET="
                f"{DATASET!r}) not found in {data_path}. Available "
                f"columns: {list(dataset.columns)}"
            )
        self.output = dataset[TARGET_COLUMN].values.reshape(-1, 1).tolist()
        dataset = dataset.drop(TARGET_COLUMN, axis=1)

        # Drop any non-numeric columns (e.g. ``timestamp_utc`` in the
        # AirQuality CSV) — they can't be tensorised by __getitem__ later.
        # For the LOF schema all remaining columns are already numeric, so
        # this is a no-op there.
        dataset = dataset.select_dtypes(include=[np.number])

        self.input = dataset.values.tolist()

        N = len(self.input)

        if train in ["train", "test", "vali"] and USE_LAST_YEAR_SPLIT:
            # --------------- new: bi-weekly interleaved last-year split ----
            assert N >= ROWS_PER_YEAR + self.length, (
                f"Dataset ({N} rows) too short to carve off a {ROWS_PER_YEAR}-"
                f"row evaluation year plus a {self.length}-row window."
            )
            last_year_start = N - ROWS_PER_YEAR
            train_end = last_year_start
            chunk_size = ROWS_PER_YEAR // NUM_CHUNKS    # 1460 rows

            val_chunks  = [i for i in range(NUM_CHUNKS) if i % VAL_STRIDE == 0]
            test_chunks = [i for i in range(NUM_CHUNKS) if i % VAL_STRIDE != 0]

            self.ans_type = StandardScaler()
            self.ans_type.fit_transform(self.output[:train_end])

            if train == "train":
                # Dense indexing into the pre-last-year block.
                self.input  = self.input[:train_end]
                self.output = self.output[:train_end]
                # self.valid_starts stays None → __getitem__ uses index directly.
            else:
                # Keep self.input / self.output as the full series so each
                # window uses continuous real-time rows, then enumerate the
                # windows that fit entirely inside one assigned chunk.
                chunks = val_chunks if train == "vali" else test_chunks
                starts = []
                for ci in chunks:
                    chunk_lo = last_year_start + ci * chunk_size
                    chunk_hi = chunk_lo + chunk_size
                    last_valid = chunk_hi - self.length      # inclusive
                    if last_valid >= chunk_lo:
                        starts.extend(range(chunk_lo, last_valid + 1))
                self.valid_starts = starts

            self.output = self.ans_type.transform(self.output)
            return
        # ------------------ legacy 70/10/20 chronological split -----------
        if train in ["train", "test", "vali"]:
            train_end = int(0.7 * N)
            val_end   = int(0.8 * N)
        else:
            train_end = N
            val_end   = None

        self.ans_type = StandardScaler()
        self.ans_type.fit_transform(self.output[:train_end])

        if train == "train":
            self.input  = self.input[:train_end]
            self.output = self.output[:train_end]
        elif train == "vali":
            self.input  = self.input[train_end:val_end]
            self.output = self.output[train_end:val_end]
        elif train == "test":
            self.input  = self.input[val_end:]
            self.output = self.output[val_end:]

        self.output = self.ans_type.transform(self.output)

    def inver_data(self, inputs):
        return inputs * self.ans_type.scale_[0] + self.ans_type.mean_[0]

    def __len__(self):
        if self.valid_starts is not None:
            return len(self.valid_starts)
        return len(self.input) - self.length + 1

    

    



                    
                    

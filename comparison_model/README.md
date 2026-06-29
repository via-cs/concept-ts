# comparison_model — TiDE baseline (fair, known-future-covariate)

A TiDE (Das et al., 2023) baseline adapted for a **fair** comparison against
ProtoLLM's Concept Bottleneck Model on the Beijing PRSA Aotizhongxin PM2.5 task.

## Why this is a fair comparison

ProtoLLM's CBM forecasts PM2.5 using **known-future covariates** — the
meteorology and other pollutants over the forecast window are given as inputs
(only PM2.5 itself is masked). To compare like-for-like, this TiDE consumes the
same information:

- **Inputs:** the 14 exogenous channels (`year, month, day, hour, PM10, SO2,
  NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM`) over the **whole** window
  (lookback + forecast), plus the lookback PM2.5.
- **Never an input:** PM2.5 over the forecast window (no target leakage).
- **Same split & windows** as `dataloader/dataloader.py` (legacy chronological
  70/10/20, windows fully inside each slice, stride 1).
- **Same target normalization** as ProtoLLM (`StandardScaler` fit on the train
  PM2.5), so the z-scored MSE/MAE live in the same space and the raw µg/m³
  numbers (`MAE = z·scale`, `MSE = z·scale²`, `RMSE = √MSE`) line up directly.
- **Same model-selection rule:** early stopping on validation MAE.

The upstream `TiDE-main` only fed calendar time-features as covariates and
flattened the lookback covariates as a static attribute — that is a different,
weaker information regime, which is why the model here is adapted.

## Files

- `tide_model.py` — TiDE adapted to known-future covariates (batch-size-safe).
- `tide_dataset.py` — PRSA dataset matching ProtoLLM's split / scaling / windows.
- `train_tide.py` — training + evaluation, reporting z-scored **and** raw µg/m³
  MSE/MAE/RMSE in the same format as `CBM/train_cbm.py`.

## Run

Train + test (defaults match ProtoLLM's current 96→96 setup):

```bash
PYTHONPATH=. python -u comparison_model/train_tide.py \
    --dataset ./dataset/PRSA_data/Aotizhongxin.csv \
    --lookback_len 96 --pred_len 96 \
    --epoch 30 --batch_size 16 --lr 3e-4 --patience 5 \
    --hidden_size 256 --num_encoder_layers 2 --num_decoder_layers 2 \
    --decoder_output_dim 8 --temporal_decoder_hidden 128 \
    --feat_size 4 --drop_prob 0.3 \
    --checkpoint_dir ./checkpoint/tide
```

Evaluate an existing checkpoint only:

```bash
PYTHONPATH=. python -u comparison_model/train_tide.py \
    --dataset ./dataset/PRSA_data/Aotizhongxin.csv \
    --lookback_len 96 --pred_len 96 \
    --checkpoint_dir ./checkpoint/tide --eval_only
```

Compare the printed `test MAE (RAW µg/m³)` / `test MSE (RAW µg/m³)` lines
against the same lines from `CBM/train_cbm.py`. Keep `--lookback_len` and
`--pred_len` equal to ProtoLLM's `--input_length` / `--output_length`.

# ConceptTS: LLM-Guided Concept Bottlenecks for Interpretable Multivariate Time-Series Forecasting

This repository contains the official implementation for the **BigData2026** paper: **"ConceptTS: LLM-Guided Concept Bottlenecks for Interpretable Multivariate Time-Series Forecasting"**.

---

## 📖 Overview

**ConceptTS** is a multivariate forecasting model that adapts Concept Bottleneck Models (CBM) to provide human-understandable interpretability without scarificing prediction accuracy, guided by LLM which automatically does concept proposal and labeling.


## 🛠️ Installation

To set up the environment and install all dependencies, run:

```bash
pip install -r requirements.txt
```

## 🚀 Usage
1. To run the data preprocessing and statistical extraction, use this command:

```bash
python preprocessing/data_preprocessing.py \
    --data_path ./dataset/PRSA_Data/Aotizhongxin.csv \
    --output_dir ./preprocessing/output/Aotizhongxin \
    --input_length 96 --output_length 96 \
    --stride 6 --n_subwindows 24
```

n_subwindows denotes how many sub-windows you want to have for the forecast horizon; stride controls the stride between segments that are used to do statistical extraction (lower the stride, higher the concept proposal quality, but longer the data preprocessing time).

2. To run the LLM concept proposal pipeline, import your Claude API:

```bash
os.environ['ANTHROPIC_API_KEY'] = "sk-ant-api..."
```

Then run the concept proposal pipeline:
```bash
python CBM/concept_proposer/propose_concepts.py \
    --n_lookback_concepts 25 \
    --n_forecast_concepts 50 \
    --n_forecast_global_concepts 25 \
    --n_clusters 20 \
    --segments_path preprocessing/output/Aotizhongxin/Aotizhongxin_full_segments.json \
    --output_dir CBM/output
```

The script will list all the concepts being proposed for each concept category. At the end, it will show the filename of the propoesd concepts set. For example:
```Code Snippet
Wrote validated concepts -> CBM/output/concepts_validated_20260818_064924.json
```

3. Copy the concept json file name and run the forecaster training script. A few key points: n_subwindows should be exactly the same as in the concept proposal pipeline, residual_dim controls the dimension of residual connection (set it to 0 if you want to disable residual connection).
```bash
python CBM/train_cbm.py \
    --dataset            ./dataset/PRSA_Data/Aotizhongxin.csv \
    --concepts_path      ./CBM/output/concepts_validated_20260818_064924.json \
    --segments_path      ./preprocessing/output/Aotizhongxin/Aotizhongxin_full_segments.json \
    --val_segments_path  ./preprocessing/output/Aotizhongxin/Aotizhongxin_full_segments_vali.json \
    --test_segments_path ./preprocessing/output/Aotizhongxin/Aotizhongxin_full_segments_test.json \
    --epoch 100 --batch_size 16 \
    --input_length 96 --output_length 96 \
    --input_dim 15 --output_dim 1 \
    --num_layer 4 --dropout 0.5 --embed_dim 128 \
    --n_subwindows 24 --concept_emb_dim 16 \
    --static_concept_emb \
    --residual_dim 0 --alpha_residual_reg 0.01 \
    --alpha_forecast 1.0 --alpha_lookback 1.0 --alpha_forecast_concept 1.0 \
    --lr 3e-4 --weight_decay 1e-4 --patience 5
```

## ✍️ Citation
If you find our work or code useful in your research, please consider citing:

```Code Snippet
@article{
}
```

#!/bin/bash
# Reference invocation. For Colab, prefer the inline `!python ...` cell shown
# in the docs over running this script. All params are written out explicitly
# (no shell variables) so the command is self-describing.

cd "$(dirname "$0")/.."
PYTHONPATH="$(pwd):$PYTHONPATH" \
CUDA_VISIBLE_DEVICES=0 \
python -u ./interpretability/extract_top_k.py \
    --dataset ./dataset/LOF/RE.csv \
    --encoder_checkpoint ./checkpoint/embedding_learning/RE_OL96_IL384_L4_D0.5_EMB128_BDP0.1.pth \
    --prototype_checkpoint ./checkpoint/prototype_learning/RE_OL96_IL384_PN12_SR0.4.pth \
    --top_k 10 \
    --layer -1 \
    --similarity dot \
    --output_dir ./interpretability/output/RE_K10_deepest_dot \
    --num_layer 4 \
    --dropout 0.5 \
    --embed_dim 128 \
    --inverse_scale_target

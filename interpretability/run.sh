#!/bin/bash
# Find the top-K training segments most similar to each learned prototype.
# Run this AFTER scripts/train.sh has produced encoder + prototype checkpoints.
#
# To replicate the original ProtoTS paper's setup, scripts/train.sh should be
# run with no LLM-init flags (i.e. don't pass --init_embedding / --init_value /
# --anchor_mask). That gives you Kaiming-random prototype_embedding +
# torch.randn prototype_value at layer 0, then full backprop through training.

cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)

if [[ ":$PYTHONPATH:" != *":$ROOT_DIR:"* ]]; then
    export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
fi

export CUDA_VISIBLE_DEVICES=0

# ---- match these to the run that produced the checkpoints -------------------
DATASET_NAME="RE"             # RE, EA, PC, YC
NUM_LAYER=4
DROPOUT=0.5
EMBED_DIM=128
PROTOTYPE_NUM=12              # whatever you used in prototype_learning.py
SPLIT_RATIO=0.4

TOP_K=10                      # how many similar segments per prototype
LAYER=-1                      # -1 = deepest layer in the checkpoint
SIMILARITY="dot"              # dot | cosine | l2
# ----------------------------------------------------------------------------

ENCODER_CKPT="./checkpoint/embedding_learning/${DATASET_NAME}_OL96_IL384_L${NUM_LAYER}_D${DROPOUT}_EMB${EMBED_DIM}_BDP0.1.pth"
PROTO_CKPT="./checkpoint/prototype_learning/${DATASET_NAME}_OL96_IL384_PN${PROTOTYPE_NUM}_SR${SPLIT_RATIO}.pth"

OUT_DIR="./interpretability/output/${DATASET_NAME}_K${TOP_K}_L${LAYER}_${SIMILARITY}"

python -u ./interpretability/extract_top_k.py \
    --dataset "./dataset/LOF/${DATASET_NAME}.csv" \
    --encoder_checkpoint "${ENCODER_CKPT}" \
    --prototype_checkpoint "${PROTO_CKPT}" \
    --top_k ${TOP_K} \
    --layer ${LAYER} \
    --similarity ${SIMILARITY} \
    --output_dir "${OUT_DIR}" \
    --num_layer ${NUM_LAYER} \
    --dropout ${DROPOUT} \
    --embed_dim ${EMBED_DIM} \
    --inverse_scale     # remove this flag to plot in scaled (z-score) space

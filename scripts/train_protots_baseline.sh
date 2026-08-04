#!/usr/bin/env bash
#
# Faithful ORIGINAL ProtoTS baseline on the LOF/RE dataset, run inside ProtoLLM.
#
# The runtime switch is two environment variables, both exported below:
#   PROTO_ENCODER=mixer   -> original ProtoTS TSMixer encoder + patch_embedding
#   PROTO_DATASET=LOF     -> LOF schema (target "load", 96 rows/day, 23 cols)
#
# Because these default to "conv" / "PRSA" when UNSET, the CBM new model and
# every other ProtoLLM run is completely unaffected — this switch only applies
# to processes launched from this script.
#
# Two experiments (differ only in the entropy-regularizer weight):
#   1. default  --entropy_weight 1e-3   (original ProtoTS)
#   2. x10      --entropy_weight 1e-2
# Checkpoints are tagged with both the encoder (_mixer) and the weight (_EW…),
# so the two runs never overwrite each other, and neither collides with the
# existing conv checkpoints. After each run, score_distribution.py emits the
# per-prototype top-1 win count + mean activation score.
#
# Usage:  bash scripts/train_protots_baseline.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# ---- the runtime switch (scoped to this script's processes only) ----
export PROTO_ENCODER=mixer
export PROTO_DATASET=LOF

DATASET_CSV="./dataset/LOF/RE.csv"
BATCH=16; NUM_LAYER=4; DROPOUT=0.5; EMBED=128; BDP=0.1
INPUT_LEN=384; OUTPUT_LEN=96; INPUT_DIM=23; OUTPUT_DIM=1
EPOCHS_EMBED=40; EPOCHS_PROTO=60
PN=12; PLN=3; PSN=2; SR=0.4; TOPK=3; DE_STEP=5

EMB_DIR="./checkpoint/embedding_learning"
PROTO_DIR="./checkpoint/prototype_learning"
EMB_CKPT="${EMB_DIR}/RE_OL${OUTPUT_LEN}_IL${INPUT_LEN}_L${NUM_LAYER}_D${DROPOUT}_EMB${EMBED}_BDP${BDP}_mixer.pth"

# =====================================================================
# Stage 1 — embedding pretraining (shared by both experiments)
# =====================================================================
echo "### Stage 1: embedding pretraining (mixer encoder, LOF/RE) ###"
python -u ./exp/embedding_learning.py \
    --dataset "${DATASET_CSV}" \
    --epoch ${EPOCHS_EMBED} --batch_size ${BATCH} \
    --num_layer ${NUM_LAYER} --dropout ${DROPOUT} --embed_dim ${EMBED}

# =====================================================================
# Stage 2 + eval — one pass per entropy weight
# =====================================================================
for EW in 0.001 0.01; do
    echo "### Stage 2: prototype learning  (entropy_weight=${EW}) ###"
    python -u ./exp/prototype_learning.py \
        --dataset "${DATASET_CSV}" \
        --epoch ${EPOCHS_PROTO} --batch_size ${BATCH} \
        --num_layer ${NUM_LAYER} --dropout ${DROPOUT} --embed_dim ${EMBED} \
        --prototype_num ${PN} --prototype_layer_num ${PLN} \
        --prototype_split_num ${PSN} --split_ratio ${SR} \
        --top_k_activate ${TOPK} --de_step ${DE_STEP} \
        --entropy_weight ${EW} \
        --pretrain_checkpoint "${EMB_CKPT}"

    MODEL_CKPT="${PROTO_DIR}/RE_OL${OUTPUT_LEN}_IL${INPUT_LEN}_L${NUM_LAYER}_D${DROPOUT}_EMB${EMBED}_mixer_EW${EW}.pth"
    PROTO_CKPT="${PROTO_DIR}/RE_OL${OUTPUT_LEN}_IL${INPUT_LEN}_PN${PN}_SR${SR}_mixer_EW${EW}.pth"

    echo "### Post-hoc eval: top-1 win count + mean activation (EW=${EW}) ###"
    python -u ./experiments/score_distribution.py \
        --dataset "${DATASET_CSV}" \
        --pretrain_checkpoint  "${MODEL_CKPT}" \
        --prototype_checkpoint "${PROTO_CKPT}" \
        --input_length ${INPUT_LEN} --output_length ${OUTPUT_LEN} \
        --input_dim ${INPUT_DIM} --output_dim ${OUTPUT_DIM} \
        --embed_dim ${EMBED} --num_layer ${NUM_LAYER} --dropout ${DROPOUT} \
        --bottle_dim_p ${BDP} \
        --prototype_num ${PN} --prototype_split_num ${PSN} \
        --top_k_activate ${TOPK} --split_ratio ${SR} \
        --batch_size ${BATCH} \
        --tag "protoTS_RE_EW${EW}"
done

echo "### Done. Score-distribution figures/CSVs in ./experiments/output/ ###"

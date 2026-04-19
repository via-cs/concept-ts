cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)

if [[ ":$PYTHONPATH:" != *":$ROOT_DIR:"* ]]; then
    export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
fi

export CUDA_VISIBLE_DEVICES=0

DATASET_NAME="PC"       # RE, EA, PC, YC
BATCH_SIZE=16
NUM_LAYER=4
DROPOUT=0.5
EMBED_DIM=128
EPOCHS_EMBED=40
EPOCHS_PROTO=60

LOG_ROOT_DIR="./log"
JOB_TAG="best_config_bs${BATCH_SIZE}_nl${NUM_LAYER}_dr${DROPOUT}_ed${EMBED_DIM}"
LOG_DIR="${LOG_ROOT_DIR}/${DATASET_NAME}/${JOB_TAG}"
mkdir -p "${LOG_DIR}"

EMBED_LOG="${LOG_DIR}/embedding_train.log"
PROTO_LOG="${LOG_DIR}/prototype_train.log"


nohup python -u ./exp/embedding_learning.py \
    --dataset "./dataset/LOF/${DATASET_NAME}.csv" \
    --epoch ${EPOCHS_EMBED} \
    --batch_size ${BATCH_SIZE} \
    --num_layer ${NUM_LAYER} \
    --dropout ${DROPOUT} \
    --embed_dim ${EMBED_DIM} \
    > "${EMBED_LOG}" 2>&1

nohup python -u ./exp/prototype_learning.py \
    --dataset "./dataset/LOF/${DATASET_NAME}.csv" \
    --epoch ${EPOCHS_PROTO} \
    --batch_size ${BATCH_SIZE} \
    --num_layer ${NUM_LAYER} \
    --dropout ${DROPOUT} \
    --embed_dim ${EMBED_DIM} \
    --prototype_num 12 \
    --prototype_layer_num 3 \
    --prototype_split_num 2 \
    --split_ratio 0.4 \
    --de_step 5 \
    --pretrain_checkpoint "./checkpoint/embedding_learning/${DATASET_NAME}_OL96_IL384_L${NUM_LAYER}_D${DROPOUT}_EMB${EMBED_DIM}_BDP0.1.pth" \
    > "${PROTO_LOG}" 2>&1

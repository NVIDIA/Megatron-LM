#!/bin/bash
NAME=$1

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
CHECKPOINT_DIR="${OUTPUT}/${NAME}"
LOG_DIR="${CHECKPOINT_DIR}/logs"
mkdir -p ${LOG_DIR}
TENSORBOARD_DIR="${CHECKPOINT_DIR}/tensorboard"
mkdir -p ${TENSORBOARD_DIR}

# Get the data blend
DATA_BLEND="1.0 ${DATA}/rprenger/maps/corpus-04-12-23/corpus-04-12-23.text_document"
#BPE_DIR="${DATA}/rprenger/bpe"

options=" \
    --tokenizer-type NullTokenizer \
		--vocab-size 786 \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 256 \
    --train-samples 192000000 \
    --lr-decay-samples 166400000 \
    --lr-warmup-samples 162761 \
    --lr 3.0e-4 \
    --min-lr 3.0e-5 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 50 \
    --eval-interval 2000 \
    --data-path ${DATA_BLEND} \
    --save-interval 10000 \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.02 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --DDP-impl local \
    --bf16 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --checkpoint-activations "

run_cmd="cd $DIR && export CUDA_DEVICE_MAX_CONNECTIONS=1 && python pretrain_gpt.py ${options}"
#gitlab-master.nvidia.com/dl/dgx/pytorch:21.12-py3-devel
srun -l \
     --container-image "nvcr.io/nvidia/pytorch:22.04-py3" \
     --container-mounts "${CHECKPOINT_DIR}:${CHECKPOINT_DIR},${DATA}:${DATA},$DIR:$DIR" \
     --output=${LOG_DIR}/%x_%j_$DATETIME.log sh -c "${run_cmd}"#
set +x

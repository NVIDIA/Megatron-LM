#!/bin/bash

# Use: ./train.sh <data-path> <tokenizer-path>

MODEL_SCALE="800M" # or "8B"

case "${MODEL_SCALE}" in
    "800M")
        TENSOR_MODEL_PARALLEL_SIZE=1
        NUM_LAYERS=48
        HIDDEN_SIZE=1024
        NUM_ATTENTION_HEADS=16
        GLOBAL_BATCH_SIZE=32
        ;;
    "8B")
        TENSOR_MODEL_PARALLEL_SIZE=4
        NUM_LAYERS=56
        HIDDEN_SIZE=4096
        NUM_ATTENTION_HEADS=32
        GLOBAL_BATCH_SIZE=8
        ;;
    *)
        echo "Invalid version specified"
        exit 1
        ;;
esac

DATA_PATH=$1
TOKENIZER_PATH=$2

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4

CHECKPOINT_DIR="./checkpoints"
DATACACHE_DIR="./data-cache"
TENSORBOARD_DIR="./tensorboard"

mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

export TRITON_CACHE_DIR="./triton-cache/"
export TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"

SEQ_LEN=4096
TRAIN_SAMPLES=73242188  # 300B tokens / 4096
LR_WARMUP_SAMPLES=50000
LR_DECAY_SAMPLES=73192188 # TRAIN_SAMPLES - LR_WARMUP_SAMPLES

options=" \
       --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE} \
       --sequence-parallel \
       --pipeline-model-parallel-size 1 \
       --use-distributed-optimizer \
       --overlap-param-gather \
       --overlap-grad-reduce \
       --untie-embeddings-and-output-weights \
       --init-method-std 0.02 \
       --position-embedding-type none \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_ATTENTION_HEADS} \
       --group-query-attention \
       --num-query-groups 8 \
       --hybrid-attention-ratio 0.08 \
       --hybrid-mlp-ratio 0.5 \
       --seq-length ${SEQ_LEN} \
       --max-position-embeddings ${SEQ_LEN} \
       --train-samples ${TRAIN_SAMPLES} \
       --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
       --lr-decay-samples ${LR_DECAY_SAMPLES} \
       --save ${CHECKPOINT_DIR} \
       --load ${CHECKPOINT_DIR} \
       --data-path ${DATA_PATH} \
       --data-cache-path ${DATACACHE_DIR} \
       --split 99,1,0 \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model ${TOKENIZER_PATH} \
       --distributed-backend nccl \
       --micro-batch-size 4 \
       --global-batch-size ${GLOBAL_BATCH_SIZE} \
       --lr 2.5e-4 \
       --min-lr 2.5e-5 \
       --lr-decay-style cosine \
       --weight-decay 0.1 \
       --clip-grad 1.0 \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --disable-bias-linear \
       --normalization RMSNorm \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 10 \
       --save-interval 2000 \
       --eval-interval 2000 \
       --eval-iters 32 \
       --bf16 \
       --use-mcore-models \
       --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
       --no-create-attention-mask-in-dataloader \
       --tensorboard-dir ${TENSORBOARD_DIR}"

torchrun --nproc_per_node 8 ../../pretrain_mamba.py ${options}

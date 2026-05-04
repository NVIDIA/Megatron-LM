#!/bin/bash

# DeepSeek-V4 hybrid model: Mamba + CSA / HCA / DSA + dense MLP pretraining example.
#
# Hybrid pattern symbols:
#   M  - Mamba
#   C  - DSv4 Compressed-Sparse Attention (CSA, ratio = 4 by default)
#   H  - DSv4 Heavily-Compressed Attention (HCA, ratio = 128 by default)
#   D  - DeepSeek Sparse Attention (DSA, MLA + indexer)
#   -  - dense MLP
#   E  - MoE
#
# Usage:
#   ./dsv4_hybrid-1n.sh <checkpoint_path> <tensorboard_dir> <data_cache_dir> \
#                               <tokenizer_model> <data_blend_path>

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$1       # <Specify path>
TENSORBOARD_LOGS_PATH=$2 # <Specify path>
DATACACHE_PATH=$3        # <Specify path>
TOKENIZER_MODEL=$4       # <Specify path to file>
DATA_BLEND_PATH=$5       # <Specify path to data blend json>

SEQ_LEN=8192
TRAIN_SAMPLES=36621094
LR_WARMUP_SAMPLES=1024000
LR_DECAY_SAMPLES=36621094
LR_WSD_DECAY_SAMPLES=5493165

# 16-layer pattern: mamba bulk + interleaved CSA / HCA / DSA + dense MLP.
HYBRID_PATTERN="M-MCM-MHMDM-MHM-"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

HYBRID_MODEL_ARGS=(
    --hybrid-layer-pattern $HYBRID_PATTERN
    --hidden-size 2048
    --num-attention-heads 16
    --num-query-groups 8
    --ffn-hidden-size 8192
    --kv-channels 128
    --mamba-num-heads 64
    --mamba-head-dim 64
    --mamba-state-dim 128
    --mamba-num-groups 8
    --seq-length $SEQ_LEN
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type none
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --squared-relu
    --init-method-std 0.0198
)

# CSA / HCA / DSA arguments (apply to C, H, and D layers in the pattern).
DSV4_HYBRID_ATTN_ARGS=(
    --rope-type rope
    --q-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --csa-window-size 128
    --csa-compress-ratio-for-c 4
    --csa-compress-ratio-for-h 128
    --o-groups 4
    --o-lora-rank 128
    --dsa-indexer-n-heads 64
    --dsa-indexer-head-dim 128
    --dsa-indexer-topk 64
    --dsa-indexer-loss-coeff 0.0
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
    --train-samples $TRAIN_SAMPLES
    --lr 1.4e-3
    --min-lr 1.4e-5
    --lr-decay-style WSD
    --lr-warmup-samples $LR_WARMUP_SAMPLES
    --lr-decay-samples $LR_DECAY_SAMPLES
    --lr-wsd-decay-style minus_sqrt
    --lr-wsd-decay-samples $LR_WSD_DECAY_SAMPLES
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --bf16
    --override-opt_param-scheduler
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

DATA_ARGS=(
    --per-split-data-args-path $DATA_BLEND_PATH
    --data-cache-path $DATACACHE_PATH
    --tokenizer-type TikTokenizer
    --tokenizer-model $TOKENIZER_MODEL
    --tiktoken-pattern v2
    --num-workers 1
    --num-dataset-builder-threads 4
    --no-create-attention-mask-in-dataloader
    --no-mmap-bin-files
)

CHECKPOINT_ARGS=(
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --ckpt-format torch_dist
    --ckpt-fully-parallel-save
    --ckpt-fully-parallel-load
    --ckpt-assume-constant-structure
    --no-load-rng
    --save-interval 12500
    --save-retain-interval 100000
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --log-throughput
    --log-progress
    --log-params-norm
    --log-num-zeros-in-grad
    --log-memory-interval 1000
    --logging-level 20
    --eval-interval 1000
    --eval-iters 14
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
)

MISC_ARGS=(
    --seed 1234
    --rerun-mode disabled
    --attention-backend flash
    --disable-gloo-process-groups
    --use-mcore-models
    --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec
    --distributed-timeout-minutes 30
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_hybrid.py \
    ${HYBRID_MODEL_ARGS[@]} \
    ${DSV4_HYBRID_ATTN_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${CHECKPOINT_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${MISC_ARGS[@]}

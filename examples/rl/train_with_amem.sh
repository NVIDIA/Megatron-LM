#!/bin/bash

# Example script for running RL training with AMem NCCL plugin
# This demonstrates how to enable AMem for memory-efficient RL training

set -e

# ============================================================================
# AMem NCCL Plugin Configuration
# ============================================================================

# Path to AMem installation (modify this to your installation path)
AMEM_PATH="${AMEM_PATH:-/path/to/asystem-amem}"

# Required: Enable NCCL CUMEM
export NCCL_CUMEM_ENABLE=1

# Required: Enable AMem plugin
export AMEM_ENABLE=1

# Group ID for this training job (use different IDs for different process groups)
# For example: 100 for training, 200 for inference if running on shared GPUs
export AMEM_GROUPID=100

# Log level: 3=INFO, 4=DEBUG, 5=VERBOSE
export GMM_LOG=3

## TODO: expose AMEM_NCCL_OFFLOAD_FREE_TAG as a CLI option if needed

# Path to AMem-enabled NCCL library
export AMEM_NCCL_LIB_PATH="${AMEM_PATH}/third_party/nccl/build/lib/libnccl.so.2"
export LD_LIBRARY_PATH="${AMEM_PATH}/third_party/nccl/build/lib:${LD_LIBRARY_PATH}"

echo "=== AMem NCCL Plugin Configuration ==="
echo "NCCL_CUMEM_ENABLE: ${NCCL_CUMEM_ENABLE}"
echo "AMEM_ENABLE: ${AMEM_ENABLE}"
echo "AMEM_GROUPID: ${AMEM_GROUPID}"
echo "GMM_LOG: ${GMM_LOG}"
echo "AMEM_NCCL_LIB_PATH: ${AMEM_NCCL_LIB_PATH}"
echo "======================================"

# ============================================================================
# Training Configuration
# ============================================================================

# Model configuration
TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1

# RL configuration
GRPO_PROMPTS_PER_STEP=32
GRPO_GROUP_SIZE=2
GRPO_ITERATIONS=2
GRPO_KL_BETA=0.001

# Training configuration
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8
SEQ_LENGTH=2048

# Data paths (modify these to your data paths)
DATA_PATH="${DATA_PATH:-/path/to/your/data}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/path/to/tokenizer}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/path/to/checkpoints}"

# RL environment config
RL_ENV_CONFIG="${RL_ENV_CONFIG:-/path/to/rl_env_config.yaml}"

# ============================================================================
# Distributed Training Setup
# ============================================================================

WORLD_SIZE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6000

echo "=== Distributed Training Setup ==="
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "NNODES: ${NNODES}"
echo "=================================="

# ============================================================================
# Run Training with AMem Enabled
# ============================================================================

torchrun \
    --nproc_per_node=${WORLD_SIZE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    pretrain_gpt.py \
    --perform-rl-step \
    --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE} \
    --pipeline-model-parallel-size ${PIPELINE_MODEL_PARALLEL_SIZE} \
    --context-parallel-size ${CONTEXT_PARALLEL_SIZE} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --lr 1.0e-5 \
    --min-lr 1.0e-6 \
    --lr-decay-style cosine \
    --train-iters 100000 \
    --lr-warmup-iters 1000 \
    --distributed-backend nccl \
    --data-path ${DATA_PATH} \
    --vocab-file ${TOKENIZER_PATH}/vocab.json \
    --merge-file ${TOKENIZER_PATH}/merges.txt \
    --save ${CHECKPOINT_PATH} \
    --load ${CHECKPOINT_PATH} \
    --save-interval 1000 \
    --eval-interval 100 \
    --eval-iters 10 \
    --log-interval 10 \
    --tensorboard-dir ${CHECKPOINT_PATH}/tensorboard \
    --fp16 \
    --use-flash-attn \
    --sequence-parallel \
    --rl-amem-offload-during-rollout \
    --rl-amem-group-id ${AMEM_GROUPID} \
    --rl-amem-offload-during-rollout \
    --grpo-prompts-per-step ${GRPO_PROMPTS_PER_STEP} \
    --grpo-group-size ${GRPO_GROUP_SIZE} \
    --grpo-iterations ${GRPO_ITERATIONS} \
    --grpo-kl-beta ${GRPO_KL_BETA} \
    --grpo-default-temperature 1.0 \
    --grpo-default-top-p 0.9 \
    --langrl-env-config ${RL_ENV_CONFIG} \
    --langrl-inference-server-type inplace_megatron \
    --rl-offload-optimizer-during-inference \
    --rl-offload-kv-cache-during-training \
    --rl-reset-cuda-graphs \
    --rl-use-sequence-packing \
    --rl-sequence-packing-bin-size 8192 \
    --rl-sequence-packing-algo round-robin

echo "=== Training Complete ==="

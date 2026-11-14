#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Run once per node, inside the prepared Megatron dev container.
set -euo pipefail

PUZZLE_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$PUZZLE_REPO_ROOT"
export PYTHONPATH="$PUZZLE_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

: "${DATA_PATH:?Set DATA_PATH to the indexed dataset prefix (without .bin/.idx)}"
: "${VOCAB_FILE:?Set VOCAB_FILE to the GPT-2 vocab.json path}"
: "${MERGE_FILE:?Set MERGE_FILE to the GPT-2 merges.txt path}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a shared directory for checkpoints and logs}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the first training node}"

NNODES=${NNODES:-${SLURM_NNODES:-4}}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:?Set NODE_RANK or launch through srun}}
MASTER_PORT=${MASTER_PORT:-29500}
if (( NNODES * GPUS_PER_NODE != 16 )); then
    echo "This example uses 16 GB200 GPUs with EP16; see README.md." >&2
    exit 1
fi

CHECKPOINT_PATH=${CHECKPOINT_PATH:-$OUTPUT_DIR/checkpoints}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-$OUTPUT_DIR/data_cache}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-$OUTPUT_DIR/tensorboard}
mkdir -p "$CHECKPOINT_PATH" "$DATA_CACHE_PATH" "$TENSORBOARD_PATH"

# Communication settings from the full-size GB200 functional run.
# All 16 ranks must be allocated within the same GB200 NVLink domain.
export NCCL_GRAPH_REGISTER=0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_HIGH_PRIORITY=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=16
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NVLINK_DOMAIN_SIZE=72
export USE_MNNVL=1
export NVTE_BWD_LAYERNORM_SM_MARGIN=20
export NVTE_FWD_LAYERNORM_SM_MARGIN=20
export NVTE_NORM_BWD_USE_CUDNN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

TRAINING_ARGS=(
    # Architecture dimensions and the config list are defined in puzzle.py.
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 16
    --expert-tensor-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --check-for-large-grads

    --micro-batch-size 1
    --global-batch-size 32
    --train-iters 20
    --seq-length 2048
    --max-position-embeddings 2048
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --manual-gc
    --manual-gc-interval 100
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl native
    --attention-backend fused
    --te-rng-tracker

    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend hybridep
    --moe-flex-dispatcher-num-sms 16
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 1.0e-4
    --use-fused-weighted-squared-relu
    --mtp-loss-scaling-factor 0.3

    --data-path "$DATA_PATH"
    --vocab-file "$VOCAB_FILE"
    --merge-file "$MERGE_FILE"
    --data-cache-path "$DATA_CACHE_PATH"
    --split 949,50,1
    --dataloader-type single
    --num-workers 8
    --no-create-attention-mask-in-dataloader

    # FP32 master-parameter semantics/moments; keep BF16 parameter remainders.
    --bf16
    --grad-reduce-in-bf16
    --use-precision-aware-optimizer
    --main-grads-dtype fp32
    --main-params-dtype fp32
    --exp-avg-dtype fp32
    --exp-avg-sq-dtype fp32
    --optimizer adam
    --lr 1.6e-3
    --min-lr 1.6e-5
    --lr-decay-style cosine
    --lr-decay-iters 39735
    --lr-warmup-iters 333
    --lr-warmup-init 0.0
    --lr-wsd-decay-style minus_sqrt
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1.0e-8
    --weight-decay 0.1
    --clip-grad 1.0
    --override-opt-param-scheduler
    --attention-dropout 0.0
    --hidden-dropout 0.0

    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH"
    --save-interval 10
    --ckpt-format torch_dist
    --ckpt-assume-constant-structure
    --dist-ckpt-strictness log_all
    --eval-interval 500
    --eval-iters 32
    --log-interval 1
    --log-num-zeros-in-grad
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --tensorboard-dir "$TENSORBOARD_PATH"
    --timing-log-level 0
    --seed 1234
)

exec uv run --no-sync python -m torch.distributed.run \
    --nnodes "$NNODES" \
    --nproc-per-node "$GPUS_PER_NODE" \
    --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    examples/hybrid/puzzle/puzzle.py "${TRAINING_ARGS[@]}" "$@"

#!/bin/bash
# Det-vs-nondet per-NVTX-range perf breakdown for pretrain_gpt.py.
# Usage: bash tests/performance_tests/shell_test_utils/determinism/perf_breakdown.sh /tmp/leaderboards /tmp/logs
# CUDA_DEVICE_MAX_CONNECTIONS=1 is required for TP>1 on pre-Blackwell.
set -euo pipefail

OUT="${1:?usage: $0 LEADERBOARD_DIR LOG_DIR}"
LOG_DIR="${2:?usage: $0 LEADERBOARD_DIR LOG_DIR}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export LOG_DIR

# Clear stale per-rank logs from prior runs (torchrun never overwrites).
rm -rf "$LOG_DIR/torchrun-det" "$LOG_DIR/torchrun-nondet"

bash "$SCRIPT_DIR/run_nsys_breakdown.sh" "$OUT" -- \
  bash -c '
    set -euo pipefail
    uv run --no-sync python -m torch.distributed.run \
      --log-dir "$LOG_DIR/torchrun-$DETERMINISM_PERF_MODE" \
      --tee "0:3,7:3" \
      --redirects "3" \
      --nproc_per_node 8 \
      pretrain_gpt.py \
        --num-layers 4 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 256 \
        --max-position-embeddings 256 \
        --micro-batch-size 2 \
        --global-batch-size 16 \
        --train-iters 8 \
        --lr 1e-4 \
        --lr-decay-style constant \
        --lr-decay-iters 100 \
        --min-lr 1e-5 \
        --weight-decay 0 \
        --clip-grad 1.0 \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 1 \
        --distributed-backend nccl \
        --tokenizer-type NullTokenizer \
        --vocab-size 256 \
        --mock-data \
        --split 1,0,0 \
        --transformer-impl transformer_engine \
        --use-mcore-models \
        --no-gradient-accumulation-fusion \
        --bf16 \
        --log-interval 1 \
        --eval-iters 0 \
        --eval-interval 10000 \
        --no-load-optim \
        --no-load-rng \
        $([ "$DETERMINISM_PERF_MODE" = det ] && echo --deterministic-mode) \
        --profile \
        --nvtx-ranges \
        --profile-step-start 5 \
        --profile-step-end 7
  '

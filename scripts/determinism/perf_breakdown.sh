#!/bin/bash
# Det-vs-nondet per-NVTX-range perf breakdown for pretrain_gpt.py.
#
# Wraps ``pretrain_gpt.py`` twice (nondet baseline + det) under nsys via the
# generic ``run_nsys_breakdown.sh`` driver, then joins the two CSVs into a
# side-by-side leaderboard. This script holds the pretrain command so the
# CI recipe yaml stays small and the experiment is reproducible locally:
#
#   bash scripts/determinism/perf_breakdown.sh /tmp/leaderboards /tmp/logs
#
# CUDA_DEVICE_MAX_CONNECTIONS=1 is Megatron's pre-existing requirement for
# TP>1 on pre-Blackwell (asserted at arguments.py:1321 before
# --deterministic-mode takes effect).
set -euo pipefail

OUT="${1:?usage: $0 LEADERBOARD_DIR LOG_DIR}"
LOG_DIR="${2:?usage: $0 LEADERBOARD_DIR LOG_DIR}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export LOG_DIR

bash "$SCRIPT_DIR/run_nsys_breakdown.sh" "$OUT" -- \
  bash -c '
    set -euo pipefail
    uv run --no-sync python -m torch.distributed.run \
      --log-dir "$LOG_DIR/torchrun-$DETERMINISM_PERF_MODE" \
      --tee "0:3,7:3" \
      --redirects "3" \
      --nproc_per_node 8 \
      pretrain_gpt.py \
        --num-layers 2 \
        --hidden-size 512 \
        --num-attention-heads 4 \
        --seq-length 128 \
        --max-position-embeddings 128 \
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
        --profile-step-start 4 \
        --profile-step-end 7
  '

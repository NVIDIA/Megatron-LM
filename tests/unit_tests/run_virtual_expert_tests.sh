#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Focused regression budget: four minutes including torchrun/pytest startup.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
case "$GPUS_PER_NODE" in
    4|8) ;;
    *) echo "Virtual-expert tests require GPUS_PER_NODE=4 or 8" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS=1
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export NVTE_CPU_OFFLOAD_V1=1
export NVTE_USE_CUTLASS_GROUPED_GEMM=0
export HYBRID_EP_CACHE_DIR="${HYBRID_EP_CACHE_DIR:-${TMPDIR:-/tmp}/megatron-virtual-expert-hybridep}"
mkdir -p "${HYBRID_EP_CACHE_DIR}"

TEST_MODULE=(-m pytest)
if [[ "${VIRTUAL_EXPERT_TEST_COVERAGE:-0}" == "1" ]]; then
    TEST_MODULE=(-m coverage run --data-file=.coverage.unit_tests --source=megatron/core -m pytest)
fi

SECONDS=0
trap 'echo "Virtual-expert suite wall time: ${SECONDS}s"' EXIT
timeout --signal=TERM --kill-after=10s 240s \
    uv run --no-sync python -m torch.distributed.run --standalone --nproc-per-node="$GPUS_PER_NODE" \
    "${TEST_MODULE[@]}" -q -o addopts= --disable-warnings --durations=10 \
    tests/unit_tests/virtual_experts \
    "$@"

if [[ "${VIRTUAL_EXPERT_TEST_COVERAGE:-0}" == "1" ]]; then
    uv run --no-sync coverage combine -q
fi

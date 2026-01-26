#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# Helper script to run common refit benchmark configurations
# Usage: ./run_refit_benchmarks.sh [benchmark_number]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark_refit.py"

# Common settings
NUM_WARMUP=3
NUM_ITERATIONS=20

# Base model config - adjust as needed
BASE_ARGS="
    --num-layers 4 \
    --hidden-size 1024 \
    --num-attention-heads 8 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 1 \
    --num-benchmark-warmup ${NUM_WARMUP} \
    --num-benchmark-iterations ${NUM_ITERATIONS}
"

# Function to run a benchmark
run_benchmark() {
    local name=$1
    local ngpus=$2
    shift 2

    echo ""
    echo "========================================================================"
    echo "Running benchmark: ${name}"
    echo "========================================================================"
    echo ""

    python ${BENCHMARK_SCRIPT} ${BASE_ARGS} "$@"

    echo ""
    echo "Benchmark ${name} completed."
    echo ""
}

# Benchmark configurations
benchmark_tp2_to_tp1_collocated() {
    run_benchmark "TP2→TP1 Collocated (NCCL)" 2 \
        --tensor-model-parallel-size 2 \
        --rl-inference-tensor-model-parallel-size 1 \
        --refit-mode collocated \
        --refit-method nccl
}

benchmark_tp2_to_tp1_non_collocated() {
    run_benchmark "TP2→TP1 Non-Collocated (NCCL)" 3 \
        --tensor-model-parallel-size 2 \
        --rl-inference-tensor-model-parallel-size 1 \
        --refit-mode non-collocated \
        --refit-method nccl
}

benchmark_tp1_to_tp2_collocated() {
    run_benchmark "TP1→TP2 Collocated (NCCL)" 2 \
        --tensor-model-parallel-size 1 \
        --rl-inference-tensor-model-parallel-size 2 \
        --refit-mode collocated \
        --refit-method nccl
}

benchmark_tp2pp2_to_tp4pp1() {
    run_benchmark "TP2,PP2→TP4,PP1 Collocated (NCCL)" 4 \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 2 \
        --rl-inference-tensor-model-parallel-size 4 \
        --rl-inference-pipeline-model-parallel-size 1 \
        --refit-mode collocated \
        --refit-method nccl \
        --num-layers 8
}

benchmark_tp1pp2_to_tp2pp1() {
    run_benchmark "TP1,PP2→TP2,PP1 Collocated (NCCL)" 2 \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 2 \
        --rl-inference-tensor-model-parallel-size 2 \
        --rl-inference-pipeline-model-parallel-size 1 \
        --refit-mode collocated \
        --refit-method nccl
}

benchmark_ep2_to_ep1_moe() {
    run_benchmark "EP2→EP1 MoE Collocated (NCCL)" 2 \
        --tensor-model-parallel-size 1 \
        --expert-model-parallel-size 2 \
        --rl-inference-tensor-model-parallel-size 1 \
        --rl-inference-expert-model-parallel-size 1 \
        --num-experts 8 \
        --refit-mode collocated \
        --refit-method nccl \
        --ffn-hidden-size 4096
}

benchmark_ep2_to_ep4_moe() {
    run_benchmark "EP2→EP4 MoE Collocated (NCCL)" 4 \
        --tensor-model-parallel-size 1 \
        --expert-model-parallel-size 2 \
        --rl-inference-tensor-model-parallel-size 1 \
        --rl-inference-expert-model-parallel-size 4 \
        --num-experts 8 \
        --refit-mode collocated \
        --refit-method nccl \
        --ffn-hidden-size 4096
}

benchmark_nvshmem_comparison() {
    echo ""
    echo "========================================================================"
    echo "Backend Comparison: NCCL vs NVSHMEM"
    echo "========================================================================"

    run_benchmark "TP2→TP1 Collocated (NCCL)" 2 \
        --tensor-model-parallel-size 2 \
        --rl-inference-tensor-model-parallel-size 1 \
        --refit-mode collocated \
        --refit-method nccl

    run_benchmark "TP2→TP1 Collocated (NVSHMEM)" 2 \
        --tensor-model-parallel-size 2 \
        --rl-inference-tensor-model-parallel-size 1 \
        --refit-mode collocated \
        --refit-method nvshmem
}

benchmark_large_model() {
    run_benchmark "Large Model: TP2,PP2→TP4,PP1" 4 \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 2 \
        --rl-inference-tensor-model-parallel-size 4 \
        --rl-inference-pipeline-model-parallel-size 1 \
        --refit-mode collocated \
        --refit-method nccl \
        --num-layers 32 \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --num-benchmark-warmup 5 \
        --num-benchmark-iterations 50
}

benchmark_all_basic() {
    echo ""
    echo "========================================================================"
    echo "Running all basic benchmarks"
    echo "========================================================================"

    benchmark_tp2_to_tp1_collocated
    benchmark_tp1_to_tp2_collocated
    benchmark_tp2_to_tp1_non_collocated
}

# Main menu
show_menu() {
    echo ""
    echo "Refit Benchmark Suite"
    echo "====================="
    echo ""
    echo "Available benchmarks:"
    echo "  1) TP2→TP1 Collocated (2 GPUs)"
    echo "  2) TP2→TP1 Non-Collocated (3 GPUs)"
    echo "  3) TP1→TP2 Collocated (2 GPUs)"
    echo "  4) TP2,PP2→TP4,PP1 Collocated (4 GPUs)"
    echo "  5) TP1,PP2→TP2,PP1 Collocated (2 GPUs)"
    echo "  6) EP2→EP1 MoE Collocated (2 GPUs)"
    echo "  7) EP2→EP4 MoE Collocated (4 GPUs)"
    echo "  8) Backend Comparison: NCCL vs NVSHMEM (2 GPUs)"
    echo "  9) Large Model Benchmark (4 GPUs)"
    echo " 10) All Basic Benchmarks (2-3 GPUs each)"
    echo ""
    echo "Usage: $0 [benchmark_number]"
    echo "Example: $0 1"
    echo ""
}

# Parse arguments
if [ $# -eq 0 ]; then
    show_menu
    exit 0
fi

case "$1" in
    1) benchmark_tp2_to_tp1_collocated ;;
    2) benchmark_tp2_to_tp1_non_collocated ;;
    3) benchmark_tp1_to_tp2_collocated ;;
    4) benchmark_tp2pp2_to_tp4pp1 ;;
    5) benchmark_tp1pp2_to_tp2pp1 ;;
    6) benchmark_ep2_to_ep1_moe ;;
    7) benchmark_ep2_to_ep4_moe ;;
    8) benchmark_nvshmem_comparison ;;
    9) benchmark_large_model ;;
    10) benchmark_all_basic ;;
    *)
        echo "Error: Invalid benchmark number: $1"
        show_menu
        exit 1
        ;;
esac

echo ""
echo "All benchmarks completed successfully!"
echo ""

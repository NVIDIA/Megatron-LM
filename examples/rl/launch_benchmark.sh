#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# Convenience launcher for refit benchmarks with pre-configured scenarios
# This wraps benchmark_refit_sbatch.sh with sensible defaults and common configurations
#
# Usage: ./launch_benchmark.sh <scenario> [mode] [backend]
#
# Scenarios:
#   tp2-tp1       : TP2 → TP1 (2 GPUs)
#   tp2-tp4       : TP2 → TP4 (4 GPUs)
#   tp1-tp2       : TP1 → TP2 (2 GPUs)
#   tp2pp2-tp4pp1 : TP2,PP2 → TP4,PP1 (4 GPUs)
#   tp1pp2-tp2pp1 : TP1,PP2 → TP2,PP1 (2 GPUs)
#   ep2-ep1       : EP2 → EP1 MoE (2 GPUs)
#   ep2-ep4       : EP2 → EP4 MoE (4 GPUs)
#   large         : Large model TP2,PP2 → TP4,PP1 (4 GPUs)
#
# Mode:
#   interactive   : Run in interactive mode (requires salloc)
#   batch         : Submit batch job (default)
#
# Backend:
#   nccl          : Use NCCL backend (default)
#   nvshmem       : Use NVSHMEM backend
#   gloo          : Use Gloo backend (debug only)
#
# Examples:
#   # Interactive with NCCL
#   ./launch_benchmark.sh tp2-tp1 interactive nccl
#
#   # Batch with NVSHMEM
#   ./launch_benchmark.sh tp2-tp1 batch nvshmem
#
#   # Batch with default backend (NCCL)
#   ./launch_benchmark.sh large batch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/benchmark_refit_sbatch.sh"

# Default values
MODE="${2:-batch}"
BACKEND="${3:-nccl}"

# Base model configuration
BASE_ARGS="
    --num-layers 4
    --hidden-size 1024
    --num-attention-heads 8
    --seq-length 512
    --max-position-embeddings 512
    --micro-batch-size 1
    --num-benchmark-warmup 3
    --num-benchmark-iterations 20
    --refit-method $BACKEND
"

# Function to show usage
show_usage() {
    echo ""
    echo "Refit Benchmark Launcher"
    echo "========================"
    echo ""
    echo "Usage: $0 <scenario> [mode] [backend]"
    echo ""
    echo "Scenarios:"
    echo "  tp2-tp1        TP2 → TP1 (2 GPUs)"
    echo "  tp2-tp4        TP2 → TP4 (4 GPUs)"
    echo "  tp1-tp2        TP1 → TP2 (2 GPUs)"
    echo "  tp2pp2-tp4pp1  TP2,PP2 → TP4,PP1 (4 GPUs)"
    echo "  tp1pp2-tp2pp1  TP1,PP2 → TP2,PP1 (2 GPUs)"
    echo "  ep2-ep1        EP2 → EP1 MoE (2 GPUs)"
    echo "  ep2-ep4        EP2 → EP4 MoE (4 GPUs)"
    echo "  large          Large model (4 GPUs)"
    echo ""
    echo "Mode (optional, default: batch):"
    echo "  interactive    Run in interactive mode"
    echo "  batch          Submit batch job"
    echo ""
    echo "Backend (optional, default: nccl):"
    echo "  nccl           Use NCCL backend"
    echo "  nvshmem        Use NVSHMEM backend"
    echo "  gloo           Use Gloo backend"
    echo ""
    echo "Examples:"
    echo "  $0 tp2-tp1 interactive"
    echo "  $0 large batch nvshmem"
    echo ""
}

# Function to launch benchmark
launch() {
    local ngpus=$1
    local nodes=${2:-1}
    shift 2
    local extra_args="$@"

    if [ "$MODE" == "interactive" ]; then
        echo "Requesting interactive allocation with ${ngpus} GPUs on ${nodes} node(s)..."
        salloc --nodes=${nodes} --gpus-per-node=${ngpus} --partition=interactive --time=00:30:00 \
            bash -c "cd $SCRIPT_DIR && ./benchmark_refit_sbatch.sh ${BASE_ARGS} ${extra_args}"
    elif [ "$MODE" == "batch" ]; then
        echo "Submitting batch job with ${ngpus} GPUs on ${nodes} node(s)..."
        sbatch --nodes=${nodes} --gpus-per-node=${ngpus} \
            ${SBATCH_SCRIPT} ${BASE_ARGS} ${extra_args}
    else
        echo "Error: Invalid mode '$MODE'. Use 'interactive' or 'batch'"
        exit 1
    fi
}

# Parse scenario
SCENARIO="${1:-}"

if [ -z "$SCENARIO" ]; then
    show_usage
    exit 1
fi

case "$SCENARIO" in
    tp2-tp1)
        echo "Scenario: TP2 → TP1 Collocated"
        launch 2 1 \
            --tensor-model-parallel-size 2 \
            --rl-inference-tensor-model-parallel-size 1 \
            --refit-mode collocated
        ;;

    tp2-tp4)
        echo "Scenario: TP2 → TP4 Collocated"
        launch 4 1 \
            --tensor-model-parallel-size 2 \
            --rl-inference-tensor-model-parallel-size 4 \
            --refit-mode collocated
        ;;

    tp1-tp2)
        echo "Scenario: TP1 → TP2 Collocated"
        launch 2 1 \
            --tensor-model-parallel-size 1 \
            --rl-inference-tensor-model-parallel-size 2 \
            --refit-mode collocated
        ;;

    tp2pp2-tp4pp1)
        echo "Scenario: TP2,PP2 → TP4,PP1 Collocated"
        launch 4 1 \
            --tensor-model-parallel-size 2 \
            --pipeline-model-parallel-size 2 \
            --rl-inference-tensor-model-parallel-size 4 \
            --rl-inference-pipeline-model-parallel-size 1 \
            --refit-mode collocated \
            --num-layers 8
        ;;

    tp1pp2-tp2pp1)
        echo "Scenario: TP1,PP2 → TP2,PP1 Collocated"
        launch 2 1 \
            --tensor-model-parallel-size 1 \
            --pipeline-model-parallel-size 2 \
            --rl-inference-tensor-model-parallel-size 2 \
            --rl-inference-pipeline-model-parallel-size 1 \
            --refit-mode collocated \
            --num-layers 8
        ;;

    ep2-ep1)
        echo "Scenario: EP2 → EP1 MoE Collocated"
        launch 2 1 \
            --tensor-model-parallel-size 1 \
            --expert-model-parallel-size 2 \
            --rl-inference-tensor-model-parallel-size 1 \
            --rl-inference-expert-model-parallel-size 1 \
            --num-experts 8 \
            --refit-mode collocated \
            --ffn-hidden-size 4096
        ;;

    ep2-ep4)
        echo "Scenario: EP2 → EP4 MoE Collocated"
        launch 4 1 \
            --tensor-model-parallel-size 1 \
            --expert-model-parallel-size 2 \
            --rl-inference-tensor-model-parallel-size 1 \
            --rl-inference-expert-model-parallel-size 4 \
            --num-experts 8 \
            --refit-mode collocated \
            --ffn-hidden-size 4096
        ;;

    large)
        echo "Scenario: Large Model TP2,PP2 → TP4,PP1"
        launch 4 1 \
            --tensor-model-parallel-size 2 \
            --pipeline-model-parallel-size 2 \
            --rl-inference-tensor-model-parallel-size 4 \
            --rl-inference-pipeline-model-parallel-size 1 \
            --refit-mode collocated \
            --num-layers 16 \
            --hidden-size 4096 \
            --num-attention-heads 32 \
            --seq-length 2048 \
            --max-position-embeddings 2048 \
            --num-benchmark-warmup 5 \
            --num-benchmark-iterations 50
        ;;

    -h|--help|help)
        show_usage
        exit 0
        ;;

    *)
        echo "Error: Unknown scenario '$SCENARIO'"
        show_usage
        exit 1
        ;;
esac

echo ""
echo "Benchmark launched successfully!"
echo ""

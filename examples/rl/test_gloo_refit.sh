#!/bin/bash
# Quick test script for GLOO refit in collocated mode with a small model
# This validates basic refit works before testing NVSHMEM/non-collocated

set -e

cat << 'EOF'
================================================================================
GLOO/Collocated Refit Quick Test
================================================================================

This script tests GLOO-based refit in collocated mode with a small model:
  ✓ Collocated refit logic is correct
  ✓ Basic infrastructure is functioning
  ✓ Model weight swapping works

Small Model Configuration:
  - 4 layers (vs 61 for Kimi K2)
  - 1024 hidden size (vs 7168 for Kimi K2)
  - 8 attention heads (vs 64 for Kimi K2)
  - 16 experts (vs 384 for Kimi K2)

Test Scenario:
  Source:  TP=4, EP=2 (8 GPUs for training)
  Target:  TP=2, EP=4 (8 GPUs for inference)
  Total:   8 GPUs used (collocated - models share same GPUs)
  Mode:    collocated (shared GPU set)
  Method:  GLOO

Expected runtime: ~1-2 minutes

Note: This uses all 8 GPUs in collocated mode.
================================================================================
EOF

# Check if we're in an interactive allocation
if [ -z "$SLURM_JOB_ID" ]; then
    echo ""
    echo "ERROR: Not in a SLURM allocation!"
    echo ""
    echo "Please request an interactive allocation first:"
    echo "  salloc --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --time=30:00 --partition=interactive"
    echo ""
    echo "Then run this script again:"
    echo "  ./test_gloo_refit.sh"
    exit 1
fi

echo ""
echo "Running in allocation: Job ID $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs available: $SLURM_GPUS_ON_NODE"
echo ""

# Determine script paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark_refit.py"
SBATCH_WRAPPER="${SCRIPT_DIR}/benchmark_refit_sbatch.sh"

# Check if benchmark script exists
if [ ! -f "$BENCHMARK_SCRIPT" ]; then
    echo "ERROR: benchmark_refit.py not found at: $BENCHMARK_SCRIPT"
    exit 1
fi

echo "Starting GLOO/collocated refit test..."
echo ""

# Small model parameters
NUM_LAYERS=4
HIDDEN_SIZE=1024
NUM_ATTENTION_HEADS=8
SEQ_LENGTH=512
MAX_POSITION_EMBEDDINGS=512
MICRO_BATCH_SIZE=1

# MoE parameters (small)
NUM_EXPERTS=16
MOE_ROUTER_TOPK=2
MOE_INTERMEDIATE_SIZE=512
FFN_HIDDEN_SIZE=2688

# Parallelism config - collocated mode with 8 GPUs
SRC_TP=4
SRC_EP=2
DST_TP=2
DST_EP=4

echo "Configuration:"
echo "  Model: ${NUM_LAYERS} layers, ${HIDDEN_SIZE} hidden, ${NUM_ATTENTION_HEADS} heads"
echo "  MoE: ${NUM_EXPERTS} experts, ${MOE_ROUTER_TOPK} active per token"
echo "  Source: TP=${SRC_TP}, EP=${SRC_EP} (${SRC_TP}x${SRC_EP}=$((SRC_TP * SRC_EP)) GPUs)"
echo "  Target: TP=${DST_TP}, EP=${DST_EP} (${DST_TP}x${DST_EP}=$((DST_TP * DST_EP)) GPUs)"
echo "  Mode: collocated (models share GPUs)"
echo ""

# Run the benchmark via the wrapper script
"${SBATCH_WRAPPER}" \
    --tensor-model-parallel-size ${SRC_TP} \
    --expert-model-parallel-size ${SRC_EP} \
    --expert-tensor-parallel-size 1 \
    --rl-inference-tensor-model-parallel-size ${DST_TP} \
    --rl-inference-expert-model-parallel-size ${DST_EP} \
    --rl-inference-expert-tensor-model-parallel-size 1 \
    --refit-mode collocated \
    --refit-method gloo \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --num-experts ${NUM_EXPERTS} \
    --moe-router-topk ${MOE_ROUTER_TOPK} \
    --moe-shared-expert-intermediate-size ${MOE_INTERMEDIATE_SIZE} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --disable-bias-linear \
    --num-benchmark-warmup 2 \
    --num-benchmark-iterations 3

exit_code=$?

echo ""
echo "=================================================================================="
if [ $exit_code -eq 0 ]; then
    echo "✓ GLOO/collocated refit test PASSED!"
    echo ""
    echo "Basic refit is working. Next steps:"
    echo "  1. Test non-collocated mode: modify test_nvshmem_refit.sh to use gloo first"
    echo "  2. Once working, switch to NVSHMEM"
else
    echo "✗ GLOO/collocated refit test FAILED with exit code: ${exit_code}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check logs for detailed error messages"
    echo "  2. This is the simplest configuration - if this fails, there's a basic issue"
fi
echo "=================================================================================="

exit $exit_code

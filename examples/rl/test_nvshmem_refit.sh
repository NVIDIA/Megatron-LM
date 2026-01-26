#!/bin/bash
# Quick test script for NVSHMEM refit with a small model
# This validates NVSHMEM works before running large-scale benchmarks

set -e

cat << 'EOF'
================================================================================
NVSHMEM Refit Quick Test
================================================================================

This script tests NVSHMEM-based refit with a small model to validate:
  ✓ NVSHMEM is available and working
  ✓ Non-collocated refit logic is correct
  ✓ GPU-to-GPU cross-node communication works
  ✓ Benchmark infrastructure is functioning

Small Model Configuration:
  - 4 layers (vs 61 for Kimi K2)
  - 1024 hidden size (vs 7168 for Kimi K2)
  - 8 attention heads (vs 64 for Kimi K2)
  - 16 experts (vs 384 for Kimi K2)

Test Scenario:
  Source:  TP=2, EP=2 (4 GPUs for training)
  Target:  TP=2, EP=1 (2 GPUs for inference)
  Total:   6 GPUs needed (2 GPUs will be idle on 8-GPU node)
  Mode:    non-collocated (separate GPU sets)
  Method:  NVSHMEM

Expected runtime: ~1-2 minutes

Note: This uses 6 out of 8 available GPUs. Ranks 6-7 will be idle.
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
    echo "  ./test_nvshmem_refit.sh"
    exit 1
fi

echo ""
echo "Running in allocation: Job ID ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs available: ${SLURM_GPUS_ON_NODE}"
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

echo "Starting NVSHMEM refit test..."
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
FFN_HIDDEN_SIZE=2688  # ~2.5x hidden_size

# Parallelism config
SRC_TP=2
SRC_EP=2
DST_TP=2
DST_EP=1

echo "Configuration:"
echo "  Model: ${NUM_LAYERS} layers, ${HIDDEN_SIZE} hidden, ${NUM_ATTENTION_HEADS} heads"
echo "  MoE: ${NUM_EXPERTS} experts, ${MOE_ROUTER_TOPK} active per token"
echo "  Source: TP=${SRC_TP}, EP=${SRC_EP} (${SRC_TP}×${SRC_EP}=$((SRC_TP * SRC_EP)) GPUs)"
echo "  Target: TP=${DST_TP}, EP=${DST_EP} (${DST_TP}×${DST_EP}=$((DST_TP * DST_EP)) GPUs)"
echo "  Total GPUs needed: $((SRC_TP * SRC_EP + DST_TP * DST_EP))"
echo ""

# Run the benchmark via the wrapper script
"${SBATCH_WRAPPER}" \
    --tensor-model-parallel-size ${SRC_TP} \
    --expert-model-parallel-size ${SRC_EP} \
    --expert-tensor-parallel-size 1 \
    --rl-inference-tensor-model-parallel-size ${DST_TP} \
    --rl-inference-expert-model-parallel-size ${DST_EP} \
    --rl-inference-expert-tensor-model-parallel-size 1 \
    --refit-mode non-collocated \
    --refit-method nvshmem \
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
    echo "✓ NVSHMEM refit test PASSED!"
    echo ""
    echo "NVSHMEM is working correctly. You can now run the full Kimi K2 benchmark:"
    echo "  sbatch benchmark_refit_kimi_k2.sh"
else
    echo "✗ NVSHMEM refit test FAILED with exit code: ${exit_code}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check if NVSHMEM is available:"
    echo "     python -c 'import nvshmem' 2>/dev/null && echo 'NVSHMEM installed' || echo 'NVSHMEM missing'"
    echo ""
    echo "  2. Check logs for detailed error messages"
    echo ""
    echo "  3. Try with NCCL instead:"
    echo "     Use --refit-method nccl instead of nvshmem"
fi
echo "=================================================================================="

exit $exit_code

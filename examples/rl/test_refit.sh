#!/bin/bash
# Quick test script for refit (collocated mode like RL training loop)

set -e

cat << 'EOF'
================================================================================
Refit Quick Test (Collocated Mode)
================================================================================

Tests refit in collocated mode (same as RL training loop):
  ✓ Both models on same 8 GPUs
  ✓ Different parallelism configs
  ✓ NCCL backend

Model Configuration:
  - 4 layers, 1024 hidden, 8 heads
  - 16 MoE experts

Test Scenario:
  Source:  TP=4, EP=2 (8 GPUs for training)
  Target:  TP=2, EP=4 (8 GPUs for inference)  
  Mode:    collocated (shared GPU set)
  Backend: NCCL

Expected runtime: ~1-2 minutes
================================================================================
EOF

if [ -z "$SLURM_JOB_ID" ]; then
    echo ""
    echo "ERROR: Not in a SLURM allocation!"
    echo ""
    echo "Please request an interactive allocation first:"
    echo "  salloc --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --time=30:00 --partition=interactive"
    echo ""
    echo "Then run this script again:"
    echo "  ./test_refit.sh"
    exit 1
fi

echo ""
echo "Running in allocation: Job ID $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs available: $SLURM_GPUS_ON_NODE"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_WRAPPER="${SCRIPT_DIR}/benchmark_refit_sbatch.sh"

if [ ! -f "$SBATCH_WRAPPER" ]; then
    echo "ERROR: benchmark_refit_sbatch.sh not found"
    exit 1
fi

echo "Starting refit test..."
echo ""

"${SBATCH_WRAPPER}" \
    --tensor-model-parallel-size 4 \
    --expert-model-parallel-size 2 \
    --expert-tensor-parallel-size 1 \
    --rl-inference-tensor-model-parallel-size 2 \
    --rl-inference-expert-model-parallel-size 4 \
    --rl-inference-expert-tensor-model-parallel-size 1 \
    --refit-mode collocated \
    --refit-method nccl \
    --num-layers 4 \
    --hidden-size 1024 \
    --num-attention-heads 8 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 1 \
    --num-experts 16 \
    --moe-router-topk 2 \
    --moe-shared-expert-intermediate-size 512 \
    --ffn-hidden-size 2688 \
    --disable-bias-linear \
    --num-benchmark-warmup 2 \
    --num-benchmark-iterations 3

exit_code=$?

echo ""
echo "=================================================================================="
if [ $exit_code -eq 0 ]; then
    echo "✓ Refit test PASSED!"
    echo ""
    echo "Ready for Kimi K2 benchmark:"
    echo "  sbatch benchmark_refit_kimi_k2.sh"
else
    echo "✗ Refit test FAILED with exit code: ${exit_code}"
fi
echo "=================================================================================="

exit $exit_code

#!/bin/bash
# Test NVSHMEM with COLLOCATED mode first (simpler)
set -e

cat << 'EOF'
================================================================================
NVSHMEM Collocated Mode Test
================================================================================

Testing NVSHMEM in the simpler collocated mode:
  ✓ Both models share the same 8 GPUs
  ✓ Tests if NVSHMEM backend works at all
  ✓ Simpler than non-collocated

Source:  TP=4, EP=2 (8 GPUs)
Target:  TP=2, EP=4 (8 GPUs)
Mode:    collocated (shared GPU set)
Method:  NVSHMEM

Expected runtime: ~1-2 minutes
================================================================================
EOF

if [ -z "$SLURM_JOB_ID" ]; then
    echo "ERROR: Not in a SLURM allocation!"
    exit 1
fi

echo ""
echo "Running in allocation: Job ID $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs available: $SLURM_GPUS_ON_NODE"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_WRAPPER="${SCRIPT_DIR}/benchmark_refit_sbatch.sh"

echo "Starting NVSHMEM collocated test..."
echo ""

"${SBATCH_WRAPPER}" \
    --tensor-model-parallel-size 4 \
    --expert-model-parallel-size 2 \
    --expert-tensor-parallel-size 1 \
    --rl-inference-tensor-model-parallel-size 2 \
    --rl-inference-expert-model-parallel-size 4 \
    --rl-inference-expert-tensor-model-parallel-size 1 \
    --refit-mode collocated \
    --refit-method nvshmem \
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
    echo "✓ NVSHMEM collocated test PASSED!"
    echo ""
    echo "NVSHMEM backend works. Now try non-collocated mode."
else
    echo "✗ NVSHMEM collocated test FAILED with exit code: ${exit_code}"
    echo ""
    echo "Even collocated mode fails - there's a fundamental issue with NVSHMEM or collocated mode itself."
fi
echo "=================================================================================="

exit $exit_code

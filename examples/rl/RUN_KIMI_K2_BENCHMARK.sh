#!/bin/bash
# Quick launcher for Kimi K2 refit benchmark
# This is a convenience wrapper around the main sbatch script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================================"
echo "Kimi K2 Refit Benchmark Launcher"
echo "============================================================================"
echo ""
echo "This will submit a benchmark job to test refit performance on the"
echo "Kimi K2 model (1.04T parameters, 384 experts) with realistic high-"
echo "parallelism configurations."
echo ""
echo "Resource Requirements:"
echo "  - Nodes: 16 (configurable in benchmark_refit_kimi_k2.sh)"
echo "  - GPUs: 128 total (16 nodes × 8 GPUs)"
echo "  - Time: 2 hours"
echo "  - Partition: batch"
echo ""
echo "Test Scenarios (7 total):"
echo "  1-5: Non-collocated (NVSHMEM) - separate GPU sets"
echo "  6-7: Collocated (NCCL) - shared GPU sets"
echo ""
echo "Results will be saved to:"
echo "  ${SCRIPT_DIR}/benchmark_results/kimi_k2/logs/"
echo ""
echo "============================================================================"
echo ""

# Check if script exists
if [ ! -f "${SCRIPT_DIR}/benchmark_refit_kimi_k2.sh" ]; then
    echo "Error: benchmark_refit_kimi_k2.sh not found in ${SCRIPT_DIR}"
    exit 1
fi

# Ask for confirmation
read -p "Submit benchmark job? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Submit job
echo "Submitting job..."
JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/benchmark_refit_kimi_k2.sh")

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Job submitted successfully!"
    echo "  Job ID: ${JOB_ID}"
    echo ""
    echo "Monitor progress:"
    echo "  squeue -j ${JOB_ID}"
    echo ""
    echo "View logs (once started):"
    echo "  tail -f ${SCRIPT_DIR}/benchmark_results/kimi_k2/logs/kimi_k2_*_${JOB_ID}_summary.txt"
    echo ""
    echo "Cancel job:"
    echo "  scancel ${JOB_ID}"
    echo ""
else
    echo ""
    echo "✗ Job submission failed!"
    echo "  Check your SLURM configuration and resource availability."
    exit 1
fi

#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Quick test script for NVSHMEM fix validation
# Tests on 2 nodes (16 GPUs) before running full 16-node benchmark
#
#SBATCH --job-name=test-nvshmem-fix
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:15:00
#SBATCH --partition=batch
#SBATCH --account=llmservice_fm_text
#SBATCH --exclusive

set -e

# ============================================================================
# Configuration
# ============================================================================

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NVSHMEM environment variables
# Buffer allocation: 4 buffers × 64MB = 256MB per PE per refit call
# With warmup + iterations, need enough for multiple allocations
# Set symmetric size to 1GB per PE for safety
export NVSHMEM_SYMMETRIC_SIZE=1073741824  # 1GB per PE
export NVSHMEM_DEBUG=WARN
export NVSHMEM_DEBUG_SUBSYS=INIT,BOOTSTRAP
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_BOOTSTRAP_TIMEOUT=300
export NVSHMEM_BOOTSTRAP_MAX_RETRIES=10

# Determine script paths
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_FPATH=$(realpath "$0")
    SCRIPT_DIR=$(dirname "$SCRIPT_FPATH")
fi
BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark_refit.py"

# Find megatron-rl directory
MEGATRON_RL_DIR=$(dirname $(dirname "$SCRIPT_DIR"))
export PYTHONPATH="${MEGATRON_RL_DIR}:${PYTHONPATH:-}"

echo "============================================================================"
echo "NVSHMEM FIX VALIDATION TEST (2 nodes, 16 GPUs)"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: ${SLURM_GPUS_ON_NODE}"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_ON_NODE))"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "BENCHMARK_SCRIPT: $BENCHMARK_SCRIPT"
echo "MEGATRON_RL_DIR: $MEGATRON_RL_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo "============================================================================"

# Container image
if [ -z "${CONTAINER_IMAGE}" ]; then
    SQSH_PATH="/lustre/fsw/portfolios/adlr/users/wdykas/images/nvshmem+adlr+megatron-rl+260113.sqsh"
    if [ -f $SQSH_PATH ]; then
        CONTAINER_IMAGE=$SQSH_PATH
    else
        CONTAINER_IMAGE="/lustre/fsw/portfolios/adlr/users/wdykas/images/nvshmem+adlr+megatron-rl+260113.sqsh"
    fi
fi

# Setup output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="/lustre/fsw/portfolios/adlr/users/wdykas/benchmark_results/kimi_k2"
fi
mkdir -p "${OUTPUT_DIR}/logs"
LOG_DIR="${OUTPUT_DIR}/logs"

# Generate run ID
DATETIME=$(date +'%y%m%d_%H%M%S')
RUN_ID="test_nvshmem_fix_${DATETIME}_${SLURM_JOB_ID}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# ============================================================================
# SMALL Model Configuration FOR QUICK TESTING
# ============================================================================
# Even smaller than the 16-node test for faster iteration

NUM_LAYERS=8                       # Reduced for speed
HIDDEN_SIZE=2048                   # Reduced for speed
NUM_ATTENTION_HEADS=16             # Reduced for speed
NUM_KEY_VALUE_HEADS=16             # Reduced for speed
SEQ_LENGTH=1024                    # Reduced for speed
MAX_POSITION_EMBEDDINGS=131072

# RoPE parameters
ROTARY_BASE=10000

# MoE configuration
NUM_EXPERTS=64                     # Reduced for speed
MOE_ROUTER_TOPK=2                  # Reduced for speed
MOE_FFN_HIDDEN_SIZE=512            # Reduced for speed
MOE_SHARED_EXPERT_SIZE=512

# Vocabulary
VOCAB_SIZE=32000                   # Reduced for speed

# Other parameters
FFN_HIDDEN_SIZE=5504               # Reduced for speed
MICRO_BATCH_SIZE=1

# ============================================================================
# Test Configurations
# ============================================================================
# Using 2 nodes (16 GPUs): 8 source + 8 destination

CONFIGS=(
    # Test 1: GLOO baseline (should work)
    "4:2:2:4:non-collocated:gloo:TP4_EP2_to_TP2_EP4_8plus8_mla_gloo_bf16"

    # Test 2: NVSHMEM (testing the fix)
    "4:2:2:4:non-collocated:nvshmem:TP4_EP2_to_TP2_EP4_8plus8_mla_nvshmem_bf16"
)

NUM_BENCHMARK_WARMUP=1
NUM_BENCHMARK_ITERATIONS=1  # Just 1 iteration for quick validation

# ============================================================================
# Run Tests
# ============================================================================

MOUNTS="/home:/home,/lustre:/lustre"
echo "Container: $CONTAINER_IMAGE"
echo ""

# Save job info
if [ -n "$SLURM_JOB_ID" ]; then
    scontrol show job $SLURM_JOB_ID | tee "${LOG_DIR}/job_info_${RUN_ID}.log"
fi

# Create results summary file
SUMMARY_FILE="${LOG_DIR}/${RUN_ID}_summary.txt"
echo "NVSHMEM Fix Validation Test" > "$SUMMARY_FILE"
echo "============================" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "Job ID: ${SLURM_JOB_ID}" >> "$SUMMARY_FILE"
echo "Nodes: ${SLURM_NNODES}" >> "$SUMMARY_FILE"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_ON_NODE))" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Model Configuration:" >> "$SUMMARY_FILE"
echo "  Layers: $NUM_LAYERS" >> "$SUMMARY_FILE"
echo "  Hidden size: $HIDDEN_SIZE" >> "$SUMMARY_FILE"
echo "  Attention heads: $NUM_ATTENTION_HEADS" >> "$SUMMARY_FILE"
echo "  MoE experts: $NUM_EXPERTS" >> "$SUMMARY_FILE"
echo "  Active experts: $MOE_ROUTER_TOPK" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Results:" >> "$SUMMARY_FILE"
echo "--------" >> "$SUMMARY_FILE"

# Run benchmarks for each configuration
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r SRC_TP SRC_EP DST_TP DST_EP REFIT_MODE REFIT_METHOD DESCRIPTION <<< "$config"

    echo ""
    echo "============================================================================"
    echo "Running test: ${DESCRIPTION}"
    echo "  Source: TP=${SRC_TP}, EP=${SRC_EP} ($((SRC_TP * SRC_EP)) GPUs)"
    echo "  Target: TP=${DST_TP}, EP=${DST_EP} ($((DST_TP * DST_EP)) GPUs)"
    echo "  Mode: ${REFIT_MODE}, Method: ${REFIT_METHOD}"
    echo "============================================================================"

    CONFIG_LOG="${LOG_DIR}/${RUN_ID}_${DESCRIPTION}.log"

    # Build command arguments
    ARGS=(
        --bf16
        --use-mcore-models
        --normalization RMSNorm
        --swiglu
        --untie-embeddings-and-output-weights
        --disable-bias-linear
        --position-embedding-type rope
        --use-rotary-position-embeddings
        --rotary-base $ROTARY_BASE
        --multi-latent-attention
        --q-lora-rank 512
        --kv-lora-rank 512
        --qk-head-dim 128
        --qk-pos-emb-head-dim 64
        --v-head-dim 128
        --tensor-model-parallel-size $SRC_TP
        --expert-model-parallel-size $SRC_EP
        --expert-tensor-parallel-size 1
        --rl-inference-tensor-model-parallel-size $DST_TP
        --rl-inference-expert-model-parallel-size $DST_EP
        --rl-inference-expert-tensor-model-parallel-size 1
        --refit-mode $REFIT_MODE
        --refit-method $REFIT_METHOD
        --num-layers $NUM_LAYERS
        --hidden-size $HIDDEN_SIZE
        --num-attention-heads $NUM_ATTENTION_HEADS
        --seq-length $SEQ_LENGTH
        --max-position-embeddings $MAX_POSITION_EMBEDDINGS
        --micro-batch-size $MICRO_BATCH_SIZE
        --num-benchmark-warmup $NUM_BENCHMARK_WARMUP
        --num-benchmark-iterations $NUM_BENCHMARK_ITERATIONS
        --vocab-size $VOCAB_SIZE
        --num-experts $NUM_EXPERTS
        --moe-router-topk $MOE_ROUTER_TOPK
        --moe-ffn-hidden-size $MOE_FFN_HIDDEN_SIZE
        --moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT_SIZE
        --ffn-hidden-size $FFN_HIDDEN_SIZE
    )

    echo "Command arguments: ${ARGS[@]}"
    echo ""

    # Run with srun
    srun -l \
        --verbose \
        --container-image "${CONTAINER_IMAGE}" \
        --container-mounts "$MOUNTS" \
        --output="${CONFIG_LOG}" \
        bash -c "export PYTHONPATH='${PYTHONPATH}'; export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'; python -u '${BENCHMARK_SCRIPT}' ${ARGS[*]}"

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ Test completed successfully"

        if [[ "$REFIT_METHOD" == "gloo" ]]; then
            echo "  ✓ GLOO baseline works"
        elif [[ "$REFIT_METHOD" == "nvshmem" ]]; then
            echo "  ✓✓ NVSHMEM FIX VERIFIED!"
            echo "  → Safe to run full 16-node benchmark"
        fi

        # Extract results from log
        echo "" >> "$SUMMARY_FILE"
        echo "Config: ${DESCRIPTION}" >> "$SUMMARY_FILE"
        echo "  Source: TP=${SRC_TP}, EP=${SRC_EP} ($((SRC_TP * SRC_EP)) GPUs)" >> "$SUMMARY_FILE"
        echo "  Target: TP=${DST_TP}, EP=${DST_EP} ($((DST_TP * DST_EP)) GPUs)" >> "$SUMMARY_FILE"
        echo "  Mode: ${REFIT_MODE}, Method: ${REFIT_METHOD}" >> "$SUMMARY_FILE"
        grep -A 2 "^Mean refit time:" "$CONFIG_LOG" >> "$SUMMARY_FILE" || echo "  Results not found in log" >> "$SUMMARY_FILE"
    else
        echo "✗ Test failed with exit code: $exit_code"

        if [[ "$REFIT_METHOD" == "gloo" ]]; then
            echo "  ✗ GLOO baseline failed - check model config"
        elif [[ "$REFIT_METHOD" == "nvshmem" ]]; then
            echo "  ✗ NVSHMEM still has issues"
            echo "  → Check log: $CONFIG_LOG"
        fi

        echo "" >> "$SUMMARY_FILE"
        echo "Config: ${DESCRIPTION}" >> "$SUMMARY_FILE"
        echo "  FAILED (exit code: $exit_code)" >> "$SUMMARY_FILE"

        # If NVSHMEM fails, don't continue - we need to fix it first
        if [[ "$REFIT_METHOD" == "nvshmem" ]]; then
            echo ""
            echo "NVSHMEM test failed - stopping here to debug"
            echo "Check log: $CONFIG_LOG"
            break
        fi
    fi

    echo "Log saved to: $CONFIG_LOG"
done

echo ""
echo "============================================================================"
echo "Test complete!"
echo "============================================================================"
echo "Summary file: $SUMMARY_FILE"
echo ""

cat "$SUMMARY_FILE"

echo ""
echo "============================================================================"
if grep -q "TP4_EP2_to_TP2_EP4_8plus8_mla_nvshmem_bf16" "$SUMMARY_FILE" && ! grep -A 2 "TP4_EP2_to_TP2_EP4_8plus8_mla_nvshmem_bf16" "$SUMMARY_FILE" | grep -q "FAILED"; then
    echo "✓✓ NVSHMEM FIX VERIFIED!"
    echo "   → Barrier fix resolved the deadlock"
    echo "   → Safe to run full 16-node (128 GPU) benchmark"
    echo ""
    echo "Next step: sbatch benchmark_refit_kimi_k2.sh"
else
    echo "✗ NVSHMEM test did not complete successfully"
    echo "   → Check the log for details"
fi
echo "============================================================================"
echo ""

exit 0

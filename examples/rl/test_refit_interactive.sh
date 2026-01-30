#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Single-node interactive test script for refit validation
# Run this on an interactive node with 8 GPUs
#
# Usage:
#   # Get an interactive node first:
#   srun --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --time=00:30:00 --pty bash
#
#   # Then run this script:
#   bash test_refit_interactive.sh [gloo|nvshmem|nvshmem-dsatur|nvshmem-greedy|nccl|all]
#

set -e

# ============================================================================
# Parse arguments
# ============================================================================
REFIT_METHOD="${1:-gloo}"  # Default to gloo for safety

if [[ "$REFIT_METHOD" == "all" ]]; then
    METHODS=("gloo" "nccl" "nvshmem-dsatur" "nvshmem-greedy")
elif [[ "$REFIT_METHOD" == "nvshmem" ]]; then
    # When user specifies "nvshmem", test both algorithms
    METHODS=("nvshmem-dsatur" "nvshmem-greedy")
else
    METHODS=("$REFIT_METHOD")
fi

# ============================================================================
# Configuration
# ============================================================================

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NVSHMEM environment variables (only used if nvshmem is selected)
export NVSHMEM_SYMMETRIC_SIZE=1073741824  # 1GB per PE
export NVSHMEM_DEBUG=WARN
export NVSHMEM_DISABLE_CUDA_VMM=1

# Determine script paths
SCRIPT_FPATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_FPATH")
BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark_refit.py"

# Find megatron-rl directory
MEGATRON_RL_DIR=$(dirname $(dirname "$SCRIPT_DIR"))
export PYTHONPATH="${MEGATRON_RL_DIR}:${PYTHONPATH:-}"

# Check GPU count
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "============================================================================"
echo "REFIT INTERACTIVE TEST (Single Node)"
echo "============================================================================"
echo "Detected GPUs: $NUM_GPUS"
echo "Test method(s): ${METHODS[*]}"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "BENCHMARK_SCRIPT: $BENCHMARK_SCRIPT"
echo "PYTHONPATH: $PYTHONPATH"
echo "============================================================================"

if [ "$NUM_GPUS" -lt 8 ]; then
    echo "WARNING: Less than 8 GPUs detected. Adjusting configuration..."
fi

# ============================================================================
# SMALL Model Configuration for Quick Testing
# ============================================================================

NUM_LAYERS=4                       # Minimal for fast testing
HIDDEN_SIZE=1024                   # Small for speed
NUM_ATTENTION_HEADS=8              
NUM_KEY_VALUE_HEADS=8              
SEQ_LENGTH=512                     
MAX_POSITION_EMBEDDINGS=8192

# RoPE parameters
ROTARY_BASE=10000

# MoE configuration
NUM_EXPERTS=8                      # Small for single node
MOE_ROUTER_TOPK=2                  
MOE_FFN_HIDDEN_SIZE=256            
MOE_SHARED_EXPERT_SIZE=256

# Vocabulary
VOCAB_SIZE=32000                   

# Other parameters
FFN_HIDDEN_SIZE=2752               
MICRO_BATCH_SIZE=1

NUM_BENCHMARK_WARMUP=1
NUM_BENCHMARK_ITERATIONS=2

# ============================================================================
# Test Configurations for 8 GPUs
# ============================================================================
# Single node with 8 GPUs: 4 source + 4 destination (non-collocated)
# Or 8 source = 8 destination (collocated)

# Adjust based on GPU count
if [ "$NUM_GPUS" -ge 8 ]; then
    # 8 GPU configs
    SRC_TP=2
    SRC_EP=2  # 4 GPUs for source
    DST_TP=1
    DST_EP=4  # 4 GPUs for dest
    REFIT_MODE="non-collocated"
    
    # Alternative: collocated mode (all 8 GPUs have both models)
    # SRC_TP=2
    # SRC_EP=4  # 8 GPUs
    # DST_TP=4
    # DST_EP=2  # 8 GPUs  
    # REFIT_MODE="collocated"
elif [ "$NUM_GPUS" -ge 4 ]; then
    # 4 GPU configs (collocated only)
    SRC_TP=2
    SRC_EP=2
    DST_TP=2
    DST_EP=2
    REFIT_MODE="collocated"
else
    echo "ERROR: Need at least 4 GPUs for testing"
    exit 1
fi

echo ""
echo "Test configuration:"
echo "  Source: TP=${SRC_TP}, EP=${SRC_EP} ($((SRC_TP * SRC_EP)) GPUs)"
echo "  Target: TP=${DST_TP}, EP=${DST_EP} ($((DST_TP * DST_EP)) GPUs)"
echo "  Mode: ${REFIT_MODE}"
echo ""

# ============================================================================
# Output directory
# ============================================================================
OUTPUT_DIR="${SCRIPT_DIR}/test_results"
mkdir -p "$OUTPUT_DIR"
DATETIME=$(date +'%y%m%d_%H%M%S')

# ============================================================================
# Run Tests
# ============================================================================

for method in "${METHODS[@]}"; do
    echo ""
    echo "============================================================================"
    echo "Testing: ${method}"
    echo "============================================================================"

    LOG_FILE="${OUTPUT_DIR}/test_${DATETIME}_${method}.log"

    # Parse method and scheduling algorithm
    if [[ "$method" == "nvshmem-"* ]]; then
        # Extract algorithm from method name (e.g., "nvshmem-dsatur" -> "dsatur")
        REFIT_METHOD_ACTUAL="nvshmem"
        SCHEDULING_ALGO="${method#nvshmem-}"
    else
        REFIT_METHOD_ACTUAL="$method"
        SCHEDULING_ALGO="dsatur"  # Default for nvshmem
    fi

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
        --tensor-model-parallel-size $SRC_TP
        --expert-model-parallel-size $SRC_EP
        --expert-tensor-parallel-size 1
        --rl-inference-tensor-model-parallel-size $DST_TP
        --rl-inference-expert-model-parallel-size $DST_EP
        --rl-inference-expert-tensor-model-parallel-size 1
        --refit-mode $REFIT_MODE
        --refit-method $REFIT_METHOD_ACTUAL
        --nvshmem-scheduling-algorithm $SCHEDULING_ALGO
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
    
    echo "Running with torchrun on $NUM_GPUS GPUs..."
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Run with torchrun
    set +e  # Don't exit on error, we want to report it
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        "$BENCHMARK_SCRIPT" "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
    
    exit_code=${PIPESTATUS[0]}
    set -e
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        echo "✓ ${method} test PASSED"
        # Show scheduling results
        grep -E "Schedule built.*iterations|DSatur result|Greedy scheduling" "$LOG_FILE" | head -3 || true
        # Show timing results
        grep -E "(Iteration [0-9]+/[0-9]+:|Mean refit time|Throughput)" "$LOG_FILE" | tail -5 || true
    else
        echo "✗ ${method} test FAILED (exit code: $exit_code)"
        echo "  Check log: $LOG_FILE"
    fi
    echo ""
done

echo "============================================================================"
echo "Test complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================================"

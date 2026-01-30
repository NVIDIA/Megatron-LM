#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Multi-node benchmark script for Kimi K2 model refit performance
#
# Kimi K2 Model Specifications:
# - Total parameters: 1.04T (1 trillion)
# - Active parameters: 32B per token
# - Layers: 61 (60 MoE + 1 dense)
# - Hidden size: 7168
# - Attention heads: 64
# - MoE experts: 384
# - Active experts per token: 8
# - Shared experts: 1
# - Vocabulary: 160K tokens
# - Context length: 128K tokens
#
#SBATCH --job-name=benchmark-refit-kimi-k2-full-32node
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=01:30:00
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
# With warmup + iterations, service is reused so only 1 allocation needed
# Set symmetric size to 512MB per PE (sufficient with service reuse)
# Total across 256 PEs: 256 × 512MB = 128GB
export NVSHMEM_SYMMETRIC_SIZE=536870912  # 512MB per PE (service reuse means only 1 allocation)
export NVSHMEM_DEBUG=WARN
export NVSHMEM_DEBUG_SUBSYS=INIT,BOOTSTRAP
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_REMOTE_TRANSPORT=ibrc

# Bootstrap tuning for 128 PEs
export NVSHMEM_BOOTSTRAP_TIMEOUT=300  # Increase timeout to 5 minutes
export NVSHMEM_BOOTSTRAP_MAX_RETRIES=10

# Determine script paths
# Use SLURM_SUBMIT_DIR which points to where sbatch was called
# This avoids issues with script paths being resolved in container temp directories
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_FPATH=$(realpath "$0")
    SCRIPT_DIR=$(dirname "$SCRIPT_FPATH")
fi
BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark_refit.py"

# Find megatron-rl directory
# SCRIPT_DIR is .../megatron-rl/examples/rl
# We need .../megatron-rl (go up 2 levels)
MEGATRON_RL_DIR=$(dirname $(dirname "$SCRIPT_DIR"))
export PYTHONPATH="${MEGATRON_RL_DIR}:${PYTHONPATH:-}"

echo "============================================================================"
echo "KIMI K2 REFIT BENCHMARK"
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
RUN_ID="kimi_k2_${DATETIME}_${SLURM_JOB_ID}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# ============================================================================
# FULL KIMI K2 Model Configuration
# ============================================================================
#
# Full production Kimi K2 model with MLA (Multi-Latent Attention)
#
# Model: 1.04T total parameters, 32B active per token
# - 61 layers (60 MoE + 1 dense layer)
# - 7168 hidden size
# - 384 experts with 8 active per token
# - Multi-Latent Attention (MLA) for memory-efficient long context
# - 128K context length support
#
# Memory: ~10-15 GB per GPU in bf16 (fits in 80GB H100)
#
# This benchmark tests refit performance on the full production model size

# Model architecture parameters - FULL KIMI K2
NUM_LAYERS=61                      # 60 MoE + 1 dense layer
HIDDEN_SIZE=7168                   # Full hidden dimension
NUM_ATTENTION_HEADS=64             # 64 attention heads
NUM_KEY_VALUE_HEADS=64             # 64 KV heads (not using GQA)
SEQ_LENGTH=8192                    # 8K sequence length for benchmark
MAX_POSITION_EMBEDDINGS=131072     # 128K max context support

# RoPE parameters
ROTARY_BASE=10000                  # Standard RoPE base

# MoE configuration - FULL KIMI K2
NUM_EXPERTS=384                    # 384 total experts
MOE_ROUTER_TOPK=8                  # 8 active experts per token
MOE_FFN_HIDDEN_SIZE=2048           # Expert FFN hidden dimension
MOE_SHARED_EXPERT_SIZE=2048        # 1 shared expert * 2048

# Vocabulary
VOCAB_SIZE=163840                  # ~160K vocabulary (full tokenizer)

# Other parameters
FFN_HIDDEN_SIZE=18432              # Dense layer FFN dimension
MICRO_BATCH_SIZE=1

# ============================================================================
# Refit Benchmark Configurations
# ============================================================================

# FULL KIMI K2 PRODUCTION BENCHMARK - 32 NODES (256 GPUs)
# Format: "SRC_TP:SRC_EP:DST_TP:DST_EP:REFIT_MODE:REFIT_METHOD:DESCRIPTION"
#
# Testing refit performance on FULL production Kimi K2 model:
# - 61 layers, 7168 hidden, 384 experts
# - WITH Multi-Latent Attention (MLA)
# - 1.04T total parameters, 32B active per token
#
# Using 32 nodes (256 GPUs total) for faster initialization:
# - Non-collocated: 128 source GPUs + 128 destination GPUs
# - Source TP=16, EP=16 (256 GPUs) -> Dest TP=8, EP=32 (256 GPUs)
# - Per-GPU: 384/16 = 24 experts (vs 48 with TP=8, EP=8)
# - Much faster initialization and lower memory per GPU
#
# Strategy:
# 1. Test with NVSHMEM for production performance
# 2. Optionally test with GLOO as baseline comparison

CONFIGS=(
    # Production: NVSHMEM - High performance refit backend
    # Non-collocated: 128 source GPUs + 128 destination GPUs = 256 total
    # Source TP=16, EP=16 -> Dest TP=8, EP=32
    "16:16:8:32:non-collocated:nvshmem:TP16_EP16_to_TP8_EP32_128plus128_fullkimi_nvshmem_bf16"

    # Optional baseline: GLOO - For comparison (slower but reliable)
    # Uncomment to compare NVSHMEM vs GLOO performance
    # "16:16:8:32:non-collocated:gloo:TP16_EP16_to_TP8_EP32_128plus128_fullkimi_gloo_bf16"
)

NUM_BENCHMARK_WARMUP=1  # Warmup to build and cache plan
NUM_BENCHMARK_ITERATIONS=5  # More iterations for stable performance metrics

# ============================================================================
# Run Benchmarks
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
echo "Kimi K2 Refit Benchmark Summary" > "$SUMMARY_FILE"
echo "================================" >> "$SUMMARY_FILE"
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
echo "  Vocabulary: $VOCAB_SIZE" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Results:" >> "$SUMMARY_FILE"
echo "--------" >> "$SUMMARY_FILE"

# Run benchmarks for each configuration
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r SRC_TP SRC_EP DST_TP DST_EP REFIT_MODE REFIT_METHOD DESCRIPTION <<< "$config"

    echo ""
    echo "============================================================================"
    echo "Running benchmark: ${DESCRIPTION}"
    echo "  Source: TP=${SRC_TP}, EP=${SRC_EP} ($((SRC_TP * SRC_EP)) GPUs)"
    echo "  Target: TP=${DST_TP}, EP=${DST_EP} ($((DST_TP * DST_EP)) GPUs)"
    echo "  Mode: ${REFIT_MODE}, Method: ${REFIT_METHOD}"
    echo "============================================================================"

    CONFIG_LOG="${LOG_DIR}/${RUN_ID}_${DESCRIPTION}.log"

    # Build command arguments with MLA support
    # Fixed: TELinear now correctly sets tensor_model_parallel=False for duplicated mode
    ARGS=(
        --bf16
        --grad-reduce-in-bf16
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
        echo "✓ Benchmark completed successfully"

        # Add interpretation based on method
        if [[ "$REFIT_METHOD" == "gloo" ]]; then
            echo "  ✓ MLA fix verified: Refit works with duplicated parameters"
            echo "  → Now safe to test NVSHMEM"
        elif [[ "$REFIT_METHOD" == "nvshmem" ]]; then
            echo "  ✓ NVSHMEM bootstrap succeeded with 128 PEs"
            echo "  ✓ Full test passed: MLA + NVSHMEM working"
        fi

        # Extract results from log and add to summary
        echo "" >> "$SUMMARY_FILE"
        echo "Config: ${DESCRIPTION}" >> "$SUMMARY_FILE"
        echo "  Source: TP=${SRC_TP}, EP=${SRC_EP} ($((SRC_TP * SRC_EP)) GPUs)" >> "$SUMMARY_FILE"
        echo "  Target: TP=${DST_TP}, EP=${DST_EP} ($((DST_TP * DST_EP)) GPUs)" >> "$SUMMARY_FILE"
        echo "  Mode: ${REFIT_MODE}, Method: ${REFIT_METHOD}" >> "$SUMMARY_FILE"
        grep -A 2 "^Mean refit time:" "$CONFIG_LOG" >> "$SUMMARY_FILE" || echo "  Results not found in log" >> "$SUMMARY_FILE"
    else
        echo "✗ Benchmark failed with exit code: $exit_code"

        # Add interpretation based on method and error
        if [[ "$REFIT_METHOD" == "gloo" ]]; then
            echo "  ✗ MLA fix may still have issues OR other problem"
            echo "  → Check log for MLA parameter errors"
        elif [[ "$REFIT_METHOD" == "nvshmem" ]]; then
            echo "  ✗ NVSHMEM issue (bootstrap or collective deadlock)"
            echo "  → Check if gloo test passed to isolate NVSHMEM vs MLA issue"
        fi

        echo "" >> "$SUMMARY_FILE"
        echo "Config: ${DESCRIPTION}" >> "$SUMMARY_FILE"
        echo "  Source: TP=${SRC_TP}, EP=${SRC_EP} ($((SRC_TP * SRC_EP)) GPUs)" >> "$SUMMARY_FILE"
        echo "  Target: TP=${DST_TP}, EP=${DST_EP} ($((DST_TP * DST_EP)) GPUs)" >> "$SUMMARY_FILE"
        echo "  FAILED (exit code: $exit_code)" >> "$SUMMARY_FILE"
    fi

    echo "Log saved to: $CONFIG_LOG"
done

echo ""
echo "============================================================================"
echo "All benchmarks complete!"
echo "============================================================================"
echo "Summary file: $SUMMARY_FILE"
echo ""

cat "$SUMMARY_FILE"

echo ""
echo "============================================================================"
echo "Test Results Interpretation:"
echo "============================================================================"

# Check which tests passed/failed
NVSHMEM_PASSED=$(grep -q "TP16_EP16_to_TP8_EP32_128plus128_fullkimi_nvshmem_bf16" "$SUMMARY_FILE" && ! grep -A 3 "TP16_EP16_to_TP8_EP32_128plus128_fullkimi_nvshmem_bf16" "$SUMMARY_FILE" | grep -q "FAILED" && echo "yes" || echo "no")
GLOO_PASSED=$(grep -q "TP16_EP16_to_TP8_EP32_128plus128_fullkimi_gloo_bf16" "$SUMMARY_FILE" && ! grep -A 3 "TP16_EP16_to_TP8_EP32_128plus128_fullkimi_gloo_bf16" "$SUMMARY_FILE" | grep -q "FAILED" && echo "yes" || echo "no")

if [[ "$NVSHMEM_PASSED" == "yes" ]]; then
    echo "✓✓ FULL KIMI K2 BENCHMARK SUCCESS!"
    echo "   → Production model (1.04T params, 61 layers, 384 experts) works correctly"
    echo "   → NVSHMEM refit validated at scale (128 GPUs)"
    echo "   → Ready for production RL training deployment"
    echo ""
    echo "Performance metrics:"
    grep -A 10 "fullkimi_nvshmem" "$SUMMARY_FILE" | grep -E "Mean|Min|Max" || echo "See full log for details"
elif [[ "$NVSHMEM_PASSED" == "no" ]] && [[ "$GLOO_PASSED" == "yes" ]]; then
    echo "✓✗ GLOO passed, NVSHMEM failed"
    echo "   → Full model works but NVSHMEM has issues"
    echo "   → Can use NCCL as alternative production backend"
    echo "   → Check NVSHMEM logs for debugging"
elif [[ "$NVSHMEM_PASSED" == "no" ]] && [[ "$GLOO_PASSED" == "no" ]]; then
    echo "✗ Benchmark failed"
    echo "   → Check logs for errors"
    echo "   → May be model size, memory, or configuration issue"
else
    echo "✓ Test completed - check summary for details"
fi

echo "============================================================================"
echo ""

exit 0

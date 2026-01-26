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
#SBATCH --job-name=benchmark-refit-kimi-k2
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
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

# Determine script paths
SCRIPT_FPATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_FPATH")
BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark_refit.py"

# Find megatron-rl directory
REPO_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))
MEGATRON_RL_DIR="${REPO_ROOT}/megatron-rl"
export PYTHONPATH="${MEGATRON_RL_DIR}:${PYTHONPATH:-}"

echo "============================================================================"
echo "KIMI K2 REFIT BENCHMARK"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: ${SLURM_GPUS_ON_NODE}"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_ON_NODE))"
echo "PYTHONPATH: $PYTHONPATH"
echo "============================================================================"

# Container image
if [ -z "${CONTAINER_IMAGE}" ]; then
    SQSH_PATH="/lustre/fsw/portfolios/llmservice/users/ksanthanam/images/adlr+megatron+mamba-dynamic-engine.sqsh"
    if [ -f $SQSH_PATH ]; then
        CONTAINER_IMAGE=$SQSH_PATH
    else
        CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/ksanthanam/images/adlr+megatron+mamba-dynamic-engine.sqsh"
    fi
fi

# Setup output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${SCRIPT_DIR}/benchmark_results/kimi_k2"
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
# Kimi K2 Model Configuration
# ============================================================================

# Model architecture parameters (matching Kimi K2 specs)
NUM_LAYERS=61
HIDDEN_SIZE=7168
NUM_ATTENTION_HEADS=64
SEQ_LENGTH=8192  # Using smaller seq length for benchmark, full model supports 128K
MAX_POSITION_EMBEDDINGS=131072  # 128K context

# MoE configuration
NUM_EXPERTS=384
MOE_ROUTER_TOPK=8
NUM_SHARED_EXPERTS=1
MOE_INTERMEDIATE_SIZE=2048

# Vocabulary
VOCAB_SIZE=160000

# Other parameters
FFN_HIDDEN_SIZE=18432  # Typically ~2.5-3x hidden_size for dense layers
MICRO_BATCH_SIZE=1

# ============================================================================
# Refit Benchmark Configurations
# ============================================================================

# Realistic high-parallelism configurations for 1T parameter MoE model
# Format: "SRC_TP:SRC_EP:DST_TP:DST_EP:REFIT_MODE:REFIT_METHOD:DESCRIPTION"
#
# With 128 GPUs (16 nodes × 8 GPUs), we can test:
# - Non-collocated: Separate GPU sets for training and inference models
# - Realistic training parallelism → lower inference parallelism
# - Both TP and EP refit behaviors

CONFIGS=(
    # === Collocated Scenarios (like RL training loop) ===
    # Both models share 128 GPUs - this is how refit is used in production

    # Scenario 1: TP/EP swap test
    "16:8:8:16:collocated:nccl:TP16_EP8_to_TP8_EP16_128_collocated"

    # Scenario 2: Test TP scaling
    "32:4:16:8:collocated:nccl:TP32_EP4_to_TP16_EP8_128_collocated"

    # Scenario 3: High TP to balanced
    "64:2:32:4:collocated:nccl:TP64_EP2_to_TP32_EP4_128_collocated"

    # Scenario 4: Test EP scaling
    "16:8:16:4:collocated:nccl:TP16_EP8_to_TP16_EP4_128_collocated"

    # Scenario 5: Balanced config
    "32:4:32:4:collocated:nccl:TP32_EP4_to_TP32_EP4_128_collocated"
)

NUM_BENCHMARK_WARMUP=3  # Extra warmup for NVSHMEM initialization
NUM_BENCHMARK_ITERATIONS=5

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

    # Build command arguments
    ARGS=(
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
        --num-experts $NUM_EXPERTS
        --moe-router-topk $MOE_ROUTER_TOPK
        --moe-shared-expert-intermediate-size $MOE_INTERMEDIATE_SIZE
        --ffn-hidden-size $FFN_HIDDEN_SIZE
        --disable-bias-linear
    )

    echo "Command arguments: ${ARGS[@]}"

    # Run with srun
    srun -l \
        --verbose \
        --container-image "${CONTAINER_IMAGE}" \
        --container-mounts "$MOUNTS" \
        --output="${CONFIG_LOG}" \
        sh -c "export PYTHONPATH=${PYTHONPATH}; python -u ${BENCHMARK_SCRIPT} ${ARGS[@]}"

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ Benchmark completed successfully"

        # Extract results from log and add to summary
        echo "" >> "$SUMMARY_FILE"
        echo "Config: ${DESCRIPTION}" >> "$SUMMARY_FILE"
        echo "  Source: TP=${SRC_TP}, EP=${SRC_EP} ($((SRC_TP * SRC_EP)) GPUs)" >> "$SUMMARY_FILE"
        echo "  Target: TP=${DST_TP}, EP=${DST_EP} ($((DST_TP * DST_EP)) GPUs)" >> "$SUMMARY_FILE"
        echo "  Mode: ${REFIT_MODE}, Method: ${REFIT_METHOD}" >> "$SUMMARY_FILE"
        grep -A 2 "^Mean refit time:" "$CONFIG_LOG" >> "$SUMMARY_FILE" || echo "  Results not found in log" >> "$SUMMARY_FILE"
    else
        echo "✗ Benchmark failed with exit code: $exit_code"
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

exit 0

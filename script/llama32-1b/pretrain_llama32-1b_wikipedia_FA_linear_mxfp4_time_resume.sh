#!/bin/bash

# =============================================================================
# Training Script for LLaMA 3.2 1B - Time-Resume Adaptive Quantization
# Script: pretrain_llama32-1b_wikipedia_FA_linear_mxfp4_time_resume.sh
# Features: Adaptive quantization with time-resume capability
# =============================================================================

# Set script metadata
SCRIPT_NAME="$(basename "$0")"
SCRIPT_VERSION="1.0.0"
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
START_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# =============================================================================
# Configuration Parameters
# =============================================================================

# Parse command line arguments
SCALING_CONTROL=${6:-"max_minus_1"}
CHECKPOINT_PATH=${1:-"checkpoints/llama32_1b/pretrain_llama32-1b_wikipedia_FA_linear_mxfp4_time_resume_${SCALING_CONTROL}"}
TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/llama32_1b_mxfp4_time_resume_${SCALING_CONTROL}"}
TOKENIZER_ARG=${3:-"model/llama3.2-1b"}
DATA_ARG=${4:-"dataset/wikipedia_processed/wikipedia_processed_text_document"}
DTYPE=${5:-"bf16"}

# Time-resume specific parameters
QUANT_LOSS_THRESHOLD=${7:-"0.1"}
QUANT_WINDOW_SIZE=${8:-"5"}
QUANT_CHECKPOINT_INTERVAL=${9:-"1"}
QUANT_FALLBACK_STRATEGY=${10:-"bf16"}
QUANT_RECOVERY_BUFFER=${11:-"2"}

# =============================================================================
# Environment Setup
# =============================================================================

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Starting training script: $SCRIPT_NAME"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Time-resume adaptive quantization enabled"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Configuration:"
echo "  - Scaling Control: $SCALING_CONTROL"
echo "  - Loss Threshold: $QUANT_LOSS_THRESHOLD"
echo "  - Window Size: $QUANT_WINDOW_SIZE"
echo "  - Checkpoint Interval: $QUANT_CHECKPOINT_INTERVAL"
echo "  - Fallback Strategy: $QUANT_FALLBACK_STRATEGY"
echo "  - Recovery Buffer: $QUANT_RECOVERY_BUFFER"

# Create directories if they don't exist
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# Set up logging paths
HOST_TENSORBOARD_LOGS_PATH="${TENSORBOARD_LOGS_PATH}_logs"
mkdir -p "$HOST_TENSORBOARD_LOGS_PATH"

# =============================================================================
# Training Execution
# =============================================================================

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Starting training execution..."

# Execute the training script with time-resume parameters
bash examples/llama/train_llama32_1b_h100_fp8.sh \
    "$CHECKPOINT_PATH" \
    "$TENSORBOARD_LOGS_PATH" \
    "$TOKENIZER_ARG" \
    "$DATA_ARG" \
    "$DTYPE" \
    --scaling-control "$SCALING_CONTROL" \
    --time-resume \
    --quant-loss-threshold "$QUANT_LOSS_THRESHOLD" \
    --quant-window-size "$QUANT_WINDOW_SIZE" \
    --quant-checkpoint-interval "$QUANT_CHECKPOINT_INTERVAL" \
    --quant-fallback-strategy "$QUANT_FALLBACK_STRATEGY" \
    --quant-recovery-buffer "$QUANT_RECOVERY_BUFFER" \
    2>&1 | tee "${HOST_TENSORBOARD_LOGS_PATH}/training_pretrain_llama32-1b_wikipedia_FA_linear_mxfp4_time_resume_${SCALING_CONTROL}_$(date +'%y-%m-%d_%H-%M-%S').log"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

# =============================================================================
# Finalization
# =============================================================================

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
END_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Training completed"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Start time: $START_TIME"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] End time: $END_TIME"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Exit code: $TRAINING_EXIT_CODE"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] Training completed successfully"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Checkpoint saved to: $CHECKPOINT_PATH"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Logs saved to: $HOST_TENSORBOARD_LOGS_PATH"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Check logs for details: $HOST_TENSORBOARD_LOGS_PATH"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Script finished: $SCRIPT_NAME"

exit $TRAINING_EXIT_CODE

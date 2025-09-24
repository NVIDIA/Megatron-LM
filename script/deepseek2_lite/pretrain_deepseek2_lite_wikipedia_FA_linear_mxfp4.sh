#!/bin/bash

# =============================================================================
# Training Script for DEEPSEEK2_LITE - Updated with new pattern
# Script: pretrain_deepseek2_lite_wikipedia_FA_linear_mxfp4.sh
# Quantization Type: mxfp4
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
CHECKPOINT_PATH=${1:-"checkpoints/deepseek2_lite/pretrain_deepseek2_lite_wikipedia_FA_linear_mxfp4"}
TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/deepseek2_lite_mxfp4"}
TOKENIZER_ARG=${3:-"model/deepseek2_lite"}
DATA_ARG=${4:-"dataset/wikipedia_processed/wikipedia_processed_text_document"}
DTYPE=${5:-"bf16"}

# =============================================================================
# Environment Setup
# =============================================================================

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Starting training script: $SCRIPT_NAME"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Checkpoint Path: $CHECKPOINT_PATH"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] TensorBoard Path: $TENSORBOARD_LOGS_PATH"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Tokenizer Path: $TOKENIZER_ARG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Data Path: $DATA_ARG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Data Type: $DTYPE"

# Export tensorboard logs path
export HOST_TENSORBOARD_LOGS_PATH="$TENSORBOARD_LOGS_PATH"

# Create directories if they don't exist
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# =============================================================================
# Quantization Type Modification
# =============================================================================

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Modifying quantization types to mxfp4..."

# Modify linear layer quantization
sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'mxfp4'/" \
    megatron/core/tensor_parallel/layers.py

# Modify attention quantization
sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'mxfp4'/" \
    megatron/core/transformer/dot_product_attention.py

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] Quantization type modifications completed"

# =============================================================================
# Training Execution
# =============================================================================

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Starting training execution..."

# Execute the training script with timestamped logging
bash examples/deepseek2_lite/train_deepseek2_lite_h100_fp8.sh \
    "$CHECKPOINT_PATH" \
    "$TENSORBOARD_LOGS_PATH" \
    "$TOKENIZER_ARG" \
    "$DATA_ARG" \
    "$DTYPE" \
    2>&1 | tee "${HOST_TENSORBOARD_LOGS_PATH}/training_pretrain_deepseek2_lite_wikipedia_FA_linear_mxfp4_$(date +'%y-%m-%d_%H-%M-%S').log"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

# =============================================================================
# Finalization
# =============================================================================

if [[ $TRAINING_EXIT_CODE -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] Training completed successfully"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Checkpoint saved to: $CHECKPOINT_PATH"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] TensorBoard logs saved to: $TENSORBOARD_LOGS_PATH"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] Training failed with exit code: $TRAINING_EXIT_CODE"
fi

exit $TRAINING_EXIT_CODE

#!/bin/bash

# =============================================================================
# Training Script for LLaMA 3.2 1B - Updated with new pattern
# Script: pretrain_llama32-1b_dolma_FA_mxfp8.sh
# Quantization Type: mxfp8
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
CHECKPOINT_PATH=${1:-"checkpoints/llama32_1b/pretrain_llama32-1b_dolma_FA_mxfp8"}
TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/llama32_1b_mxfp8"}
TOKENIZER_ARG=${3:-"model/llama3.2-1b"}
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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Modifying quantization types to mxfp8..."

# Modify linear layer quantization
sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'mxfp8'/" \
    megatron/core/tensor_parallel/layers.py

# Modify attention quantization
sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'mxfp8'/" \
    megatron/core/transformer/dot_product_attention.py

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] Quantization type modifications completed"

# =============================================================================
# Training Execution
# =============================================================================

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Starting training execution..."

# Execute the training script with timestamped logging
bash examples/llama/train_llama32_1b_h100_fp8.sh \
    "$CHECKPOINT_PATH" \
    "$TENSORBOARD_LOGS_PATH" \
    "$TOKENIZER_ARG" \
    "$DATA_ARG" \
    "$DTYPE" \
    2>&1 | tee "${HOST_TENSORBOARD_LOGS_PATH}/training_pretrain_llama32-1b_dolma_FA_mxfp8_$(date +'%y-%m-%d_%H-%M-%S').log"

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

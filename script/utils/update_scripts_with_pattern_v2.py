#!/usr/bin/env python3
"""
Script to update all training scripts with the new pattern:
1. Export HOST_TENSORBOARD_LOGS_PATH
2. Use sed commands to modify quantization types
3. Execute training with timestamped logging

This version handles llama31-8b and deepseek2_lite directories.
"""

import os
import re
import shutil
from pathlib import Path

def backup_file(file_path):
    """Create a backup of the original file."""
    backup_path = f"{file_path}.backup_pattern_update_v2"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"Created backup: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")

def extract_quant_type_from_filename(filename):
    """Extract quantization type from filename."""
    filename_str = str(filename)
    if "hifp8" in filename_str:
        return "hifp8"
    elif "mxfp8" in filename_str:
        return "mxfp8"
    elif "mxfp4" in filename_str:
        return "mxfp4"
    else:
        return "bf16"  # Default

def extract_model_name_from_filename(filename):
    """Extract model name from filename."""
    filename_str = str(filename)
    if "llama31-8b" in filename_str:
        return "llama31-8b"
    elif "deepseek2_lite" in filename_str:
        return "deepseek2_lite"
    else:
        return "unknown"

def get_training_script_path(model_name):
    """Get the appropriate training script path based on model name."""
    if model_name == "llama31-8b":
        return "examples/llama/train_llama3_8b_h100_fp8.sh"  # 注意是 llama3 不是 llama31
    elif model_name == "deepseek2_lite":
        return "examples/deepseek2_lite/train_deepseek2_lite_h100_fp8.sh"  # 使用新创建的 deepseek2_lite 脚本
    else:
        return "examples/llama/train_llama32_1b_h100_fp8.sh"  # Default

def update_script_with_pattern(script_path):
    """Update a script with the new pattern."""
    print(f"Updating script: {script_path}")
    
    # Create backup
    backup_file(script_path)
    
    # Extract information from filename
    quant_type = extract_quant_type_from_filename(script_path)
    model_name = extract_model_name_from_filename(script_path)
    training_script = get_training_script_path(model_name)
    
    # Create new script content following the pattern
    script_name = os.path.basename(str(script_path))
    new_content = f'''#!/bin/bash

# =============================================================================
# Training Script for {model_name.upper()} - Updated with new pattern
# Script: {script_name}
# Quantization Type: {quant_type}
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
CHECKPOINT_PATH=${{1:-"checkpoints/{model_name}/{script_name.replace('.sh', '')}"}}
TENSORBOARD_LOGS_PATH=${{2:-"tensorboard_logs/{model_name}_{quant_type}"}}
TOKENIZER_ARG=${{3:-"model/{model_name}"}}
DATA_ARG=${{4:-"dataset/wikipedia_processed/wikipedia_processed_text_document"}}
DTYPE=${{5:-"bf16"}}

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
# ============================================================================='''

    # Add quantization modification if not bf16
    if quant_type != "bf16":
        new_content += f'''

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Modifying quantization types to {quant_type}..."

# Modify linear layer quantization
sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'{quant_type}'/" \\
    megatron/core/tensor_parallel/layers.py

# Modify attention quantization
sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'{quant_type}'/" \\
    megatron/core/transformer/dot_product_attention.py

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] Quantization type modifications completed"'''
    else:
        new_content += '''

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] BF16 training - no quantization modification needed"'''

    # Add training execution
    new_content += f'''

# =============================================================================
# Training Execution
# =============================================================================

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Starting training execution..."

# Execute the training script with timestamped logging
bash {training_script} \\
    "$CHECKPOINT_PATH" \\
    "$TENSORBOARD_LOGS_PATH" \\
    "$TOKENIZER_ARG" \\
    "$DATA_ARG" \\
    "$DTYPE" \\
    2>&1 | tee "${{HOST_TENSORBOARD_LOGS_PATH}}/training_{script_name.replace('.sh', '')}_$(date +'%y-%m-%d_%H-%M-%S').log"

TRAINING_EXIT_CODE=${{PIPESTATUS[0]}}

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
'''

    # Write the new content
    with open(str(script_path), 'w') as f:
        f.write(new_content)
    
    # Make the script executable
    os.chmod(str(script_path), 0o755)
    
    print(f"Updated script: {script_path}")

def update_directory_scripts(directory_path):
    """Update all scripts in a directory."""
    if not directory_path.exists():
        print(f"Error: Directory {directory_path} does not exist")
        return 0
    
    # Find all .sh files in the directory
    script_files = list(directory_path.glob("*.sh"))
    
    # Filter out backup files
    script_files = [f for f in script_files if not f.name.endswith('.backup')]
    
    print(f"\nFound {len(script_files)} scripts to update in {directory_path.name}:")
    for script_file in script_files:
        print(f"  - {script_file.name}")
    
    # Update each script
    updated_count = 0
    for script_file in script_files:
        try:
            update_script_with_pattern(script_file)
            updated_count += 1
        except Exception as e:
            print(f"Error updating {script_file}: {e}")
    
    print(f"\n✅ Updated {updated_count} scripts in {directory_path.name}")
    return updated_count

def main():
    """Main function to update all scripts."""
    script_dir = Path(__file__).parent
    
    # Directories to update
    directories_to_update = [
        script_dir / "llama31-8b",
        script_dir / "deepseek2_lite"
    ]
    
    total_updated = 0
    
    for directory in directories_to_update:
        if directory.exists():
            updated_count = update_directory_scripts(directory)
            total_updated += updated_count
        else:
            print(f"Warning: Directory {directory} does not exist, skipping...")
    
    print(f"\n🎉 Total updated {total_updated} scripts across all directories!")
    print("\nThe updated scripts now include:")
    print("  1. HOST_TENSORBOARD_LOGS_PATH export")
    print("  2. sed commands to modify quantization types")
    print("  3. Timestamped logging with tee command")
    print("  4. Model-specific training script paths")
    
    return 0

if __name__ == "__main__":
    exit(main())

#!/bin/bash
# =============================================================================
# Update Training Scripts with Auto-detected Quantization Type
# 
# This script updates all training scripts in llama32-1b and llama31-8b directories
# to automatically set the correct quantization type based on the script filename.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================================================="
echo "Updating Training Scripts with Auto-detected Quantization Type"
echo "=============================================================================="
echo "Project root: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"

# Directories to process
TARGET_DIRS=(
    "script/llama32-1b"
    "script/llama31-8b"
)

# Function to extract quantization type from filename
extract_quant_type() {
    local filename="$1"
    
    # Extract quantization type from filename
    if [[ "$filename" =~ _mxfp8[._-] ]]; then
        echo "mxfp8"
    elif [[ "$filename" =~ _mxfp4[._-] ]]; then
        echo "mxfp4"
    elif [[ "$filename" =~ _hifp8[._-] ]]; then
        echo "hifp8"
    elif [[ "$filename" =~ _bf16[._-] ]]; then
        echo "bf16"
    else
        echo "unknown"
    fi
}

# Function to update script
update_script() {
    local script_file="$1"
    local quant_type="$2"
    
    # Create backup
    cp "$script_file" "${script_file}.bak"
    
    # Check if script already has quantization modification section
    if grep -q "# Quantization Type Modification" "$script_file"; then
        # Update existing sed commands
        if [ "$quant_type" == "bf16" ]; then
            # For BF16, replace quantization section with a simple message
            sed -i '/# Quantization Type Modification/,/^echo.*Quantization type modifications completed/c\
# =============================================================================\
# Quantization Type Modification\
# =============================================================================\
\
echo "[$(date '"'"'+%Y-%m-%d %H:%M:%S'"'"')] [INFO] BF16 training - no quantization modification needed"' "$script_file"
        else
            # For quantized types, update sed commands
            sed -i "s|^\(echo.*Modifying quantization types to\) [^.]*\.\.\.|\\1 ${quant_type}...|" "$script_file"
            sed -i "/# Modify linear layer quantization/,/megatron\/core\/tensor_parallel\/layers.py/c\\
# Modify linear layer quantization\\
sed -i \"s/^\\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\\)'[^']*'/\\\\1'${quant_type}'/\" \\\\\\
    megatron/core/tensor_parallel/layers.py" "$script_file"
            
            sed -i "/# Modify attention quantization/,/megatron\/core\/transformer\/dot_product_attention.py/c\\
# Modify attention quantization\\
sed -i \"s/^\\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\\)'[^']*'/\\\\1'${quant_type}'/\" \\\\\\
    megatron/core/transformer/dot_product_attention.py" "$script_file"
        fi
    else
        # Add quantization modification section before training execution
        if [ "$quant_type" == "bf16" ]; then
            # For BF16, just add a comment
            sed -i '/# Training Execution/i\
# =============================================================================\
# Quantization Type Modification\
# =============================================================================\
\
echo "[$(date '"'"'+%Y-%m-%d %H:%M:%S'"'"')] [INFO] BF16 training - no quantization modification needed"\
\
' "$script_file"
        else
            # For quantized types, add sed commands
            sed -i '/# Training Execution/i\
# =============================================================================\
# Quantization Type Modification\
# =============================================================================\
\
echo "[$(date '"'"'+%Y-%m-%d %H:%M:%S'"'"')] [INFO] Modifying quantization types to '"${quant_type}"'..."\
\
# Modify linear layer quantization\
sed -i "s/^\\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\\)'"'"'[^'"'"']*'"'"'/\\1'"'"''"${quant_type}"''"'"'/" \\\\\
    megatron/core/tensor_parallel/layers.py\
\
# Modify attention quantization\
sed -i "s/^\\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\\)'"'"'[^'"'"']*'"'"'/\\1'"'"''"${quant_type}"''"'"'/" \\\\\
    megatron/core/transformer/dot_product_attention.py\
\
echo "[$(date '"'"'+%Y-%m-%d %H:%M:%S'"'"')] [SUCCESS] Quantization type modifications completed"\
\
' "$script_file"
        fi
    fi
    
    echo "  ✅ Updated: $script_file (quant_type: $quant_type)"
}

# Process each directory
TOTAL_UPDATED=0

for target_dir in "${TARGET_DIRS[@]}"; do
    if [ ! -d "$target_dir" ]; then
        echo "⚠️  Warning: Directory not found: $target_dir"
        continue
    fi
    
    echo "Processing directory: $target_dir"
    echo ""
    
    # Find all .sh files
    scripts=($(find "$target_dir" -maxdepth 1 -type f -name "*.sh" | sort))
    
    for script_file in "${scripts[@]}"; do
        filename=$(basename "$script_file")
        quant_type=$(extract_quant_type "$filename")
        
        if [ "$quant_type" == "unknown" ]; then
            echo "  ⚠️  Skipped: $filename (unknown quantization type)"
            continue
        fi
        
        update_script "$script_file" "$quant_type"
        ((TOTAL_UPDATED++))
    done
    
    echo ""
done

echo "=============================================================================="
echo "Summary"
echo "=============================================================================="
echo "Total scripts updated: $TOTAL_UPDATED"
echo ""
echo "Backup files created with .bak extension"
read -p "Remove backup files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    find script/llama32-1b script/llama31-8b -name "*.bak" -delete 2>/dev/null || true
    echo "✅ All backup files removed"
else
    echo "ℹ️  Backup files kept. Remove manually with: find script/ -name '*.bak' -delete"
fi

echo ""
echo "=============================================================================="
echo "Update complete!"
echo "=============================================================================="
echo ""
echo "Scripts now automatically set quantization type based on filename:"
echo "  - *mxfp8*.sh  → custom_quant_type='mxfp8'"
echo "  - *mxfp4*.sh  → custom_quant_type='mxfp4'"
echo "  - *hifp8*.sh  → custom_quant_type='hifp8'"
echo "  - *bf16*.sh   → No modification (BF16 baseline)"
echo ""


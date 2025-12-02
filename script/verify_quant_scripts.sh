#!/bin/bash
# =============================================================================
# Verify and Fix Quantization Type in Training Scripts
# 
# Checks if the quantization type in sed commands matches the filename
# and offers to fix mismatches
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================================================="
echo "Verifying Quantization Type in Training Scripts"
echo "=============================================================================="
echo ""

cd "$PROJECT_ROOT"

# Directories to check
TARGET_DIRS=(
    "script/llama32-1b"
    "script/llama31-8b"
)

# Function to extract quantization type from filename
extract_quant_from_filename() {
    local filename="$1"
    
    if [[ "$filename" =~ _mxfp8 ]]; then
        echo "mxfp8"
    elif [[ "$filename" =~ _mxfp4 ]]; then
        echo "mxfp4"
    elif [[ "$filename" =~ _hifp8 ]]; then
        echo "hifp8"
    elif [[ "$filename" =~ _bf16 ]]; then
        echo "bf16"
    else
        echo "unknown"
    fi
}

# Function to extract quantization type from sed command
extract_quant_from_sed() {
    local script_file="$1"
    
    # Extract from first sed command
    local quant_in_sed=$(grep -E "sed.*custom_quant_type.*=" "$script_file" | head -1 | sed -n "s/.*'\\([^']*\\)'.*/\\1/p")
    
    if [ -n "$quant_in_sed" ]; then
        echo "$quant_in_sed"
    else
        echo "none"
    fi
}

MISMATCHES=0
TOTAL_CHECKED=0

for target_dir in "${TARGET_DIRS[@]}"; do
    if [ ! -d "$target_dir" ]; then
        continue
    fi
    
    echo "Checking directory: $target_dir"
    echo "$(printf '=%.0s' {1..80})"
    
    # Find all .sh files
    scripts=($(find "$target_dir" -maxdepth 1 -type f -name "pretrain*.sh" | sort))
    
    for script_file in "${scripts[@]}"; do
        filename=$(basename "$script_file")
        quant_from_name=$(extract_quant_from_filename "$filename")
        quant_from_sed=$(extract_quant_from_sed "$script_file")
        
        ((TOTAL_CHECKED++))
        
        # Compare
        if [ "$quant_from_name" == "bf16" ]; then
            # BF16 scripts should not have sed commands or should have a comment
            if [ "$quant_from_sed" == "none" ]; then
                echo "  ✅ $filename: BF16 (no sed, correct)"
            elif grep -q "BF16 training - no quantization" "$script_file" 2>/dev/null; then
                echo "  ✅ $filename: BF16 (correct)"
            else
                echo "  ❌ $filename: Expected BF16 but has sed for '$quant_from_sed'"
                MISMATCHES=$((MISMATCHES + 1))
            fi
        elif [ "$quant_from_name" != "unknown" ]; then
            if [ "$quant_from_name" == "$quant_from_sed" ]; then
                echo "  ✅ $filename: $quant_from_name (correct)"
            else
                echo "  ❌ $filename: Expected '$quant_from_name' but sed has '$quant_from_sed'"
                MISMATCHES=$((MISMATCHES + 1))
            fi
        else
            echo "  ⚠️  $filename: Unknown quantization type"
        fi
    done
    
    echo ""
done

echo "=============================================================================="
echo "Summary"
echo "=============================================================================="
echo "Total scripts checked: $TOTAL_CHECKED"
echo "Mismatches found: $MISMATCHES"
echo ""

if [ $MISMATCHES -gt 0 ]; then
    echo "⚠️  Found $MISMATCHES mismatch(es)"
    echo ""
    echo "To fix all scripts automatically, run:"
    echo "  bash script/update_quant_scripts.sh"
    exit 1
else
    echo "✅ All scripts have correct quantization type settings"
    exit 0
fi


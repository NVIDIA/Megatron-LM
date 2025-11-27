#!/bin/bash
# =============================================================================
# Batch Test Script for Quantization Error Analysis
# 
# Tests quant_hif8 and _quantize_mx on all tensors in enhanced_tensor_logs/bf16
# Generates a table showing quantization errors for:
# - Forward/Backward passes
# - Tensor types: input, weight, query, key, value
# - Formats: MXFP8(E4M3), MXFP8(E5M2), HiFP8
#
# Usage:
#   ./test_dtype_batch.sh [TENSOR_DIR] [OUTPUT_DIR] [LAYER_FILTER]
#
# Examples:
#   # Test single layer (L1)
#   ./test_dtype_batch.sh enhanced_tensor_logs/bf16 ./results 1
#
#   # Calculate average across all layers (L1-L16)
#   ./test_dtype_batch.sh enhanced_tensor_logs/bf16 ./results all
#   ./test_dtype_batch.sh enhanced_tensor_logs/bf16 ./results 1-16
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

# Input directory
TENSOR_DIR="${1:-enhanced_tensor_logs/bf16}"

# Output directory for results
OUTPUT_DIR="${2:-./quant_test_results}"

# Layer filter (default: L1)
# If set to "all" or "1-16", will calculate average across all layers
LAYER_FILTER="${3:-1}"

# Calculate average across layers (if LAYER_FILTER is "all" or "1-16")
CALC_AVG=false
if [[ "$LAYER_FILTER" == "all" ]] || [[ "$LAYER_FILTER" == "1-16" ]]; then
    CALC_AVG=true
    LAYER_FILTER="all"
fi

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/test_dtype.py"

# Supported tensor types
TENSOR_TYPES=("input" "weight" "query" "key" "value")

# Supported formats
FORMATS=("hifp8" "fp8_e4m3" "fp8_e5m2")
FORMAT_NAMES=("HIFP" "MXFP8(E4M3)" "MXFP8(E5M2)")

# =============================================================================
# Validation
# =============================================================================

if [ ! -d "$TENSOR_DIR" ]; then
    echo "❌ Error: Tensor directory does not exist: $TENSOR_DIR"
    exit 1
fi

if [ ! -f "$TEST_SCRIPT" ]; then
    echo "❌ Error: Test script not found: $TEST_SCRIPT"
    exit 1
fi

# =============================================================================
# Helper Functions
# =============================================================================

# Extract tensor type from filename
get_tensor_type() {
    local filename="$1"
    local filename_lower=$(echo "$filename" | tr '[:upper:]' '[:lower:]')
    
    for ttype in "${TENSOR_TYPES[@]}"; do
        if [[ "$filename_lower" == *"_${ttype}.pt" ]] || \
           [[ "$filename_lower" == *"-${ttype}.pt" ]] || \
           [[ "$filename_lower" == *"_${ttype}_"* ]] || \
           [[ "$filename_lower" == *"-${ttype}-"* ]]; then
            echo "$ttype"
            return 0
        fi
    done
    echo ""
}

# Extract pass type (forward/backward) from filename
get_pass_type() {
    local filename="$1"
    local filename_lower=$(echo "$filename" | tr '[:upper:]' '[:lower:]')
    
    if [[ "$filename_lower" == *"_forward_"* ]]; then
        echo "forward"
    elif [[ "$filename_lower" == *"_backward_"* ]]; then
        echo "backward"
    else
        echo ""
    fi
}

# Extract layer number from filename
get_layer() {
    local filename="$1"
    if [[ "$filename" =~ _L([0-9]+)_ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

# Run quantization test and extract relative error
run_quantization_test() {
    local tensor_file="$1"
    local format="$2"
    local output_file="$OUTPUT_DIR/$(basename "$tensor_file" .pt)_${format}.txt"
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    # Run test and capture output
    if python3 "$TEST_SCRIPT" "$tensor_file" --format "$format" > "$output_file" 2>&1; then
        # Extract relative error from output (handle both "Relative Error: X.XX" format)
        local rel_error=$(grep -i "Relative Error:" "$output_file" | tail -1 | sed 's/.*Relative Error:[[:space:]]*\([0-9.eE+-]*\).*/\1/')
        if [ -n "$rel_error" ] && [[ "$rel_error" =~ ^[0-9.eE+-]+$ ]]; then
            echo "$rel_error"
        else
            echo "N/A"
        fi
    else
        echo "ERROR"
    fi
}

# Calculate average of numeric values (handles scientific notation)
calculate_average() {
    local values=("$@")
    local count=0
    local temp_file=$(mktemp)
    
    # Collect valid values
    for val in "${values[@]}"; do
        if [[ "$val" =~ ^[0-9.eE+-]+$ ]] && [[ "$val" != "N/A" ]] && [[ "$val" != "ERROR" ]]; then
            echo "$val" >> "$temp_file"
            ((count++))
        fi
    done
    
    if [ $count -eq 0 ]; then
        rm -f "$temp_file"
        echo "N/A"
        return
    fi
    
    # Use Python to calculate average (handles scientific notation properly)
    local avg=$(python3 -c "
import sys
values = []
with open('$temp_file', 'r') as f:
    for line in f:
        values.append(float(line.strip()))
if values:
    avg = sum(values) / len(values)
    print(f'{avg:.6e}')
else:
    print('N/A')
" 2>/dev/null)
    
    rm -f "$temp_file"
    echo "$avg"
}

# =============================================================================
# Main Processing
# =============================================================================

cd "$PROJECT_ROOT"

echo "=============================================================================="
echo "Batch Quantization Error Analysis"
echo "=============================================================================="
echo "Tensor directory: $TENSOR_DIR"
echo "Output directory: $OUTPUT_DIR"
if [ "$CALC_AVG" = true ]; then
    echo "Mode: Average across all layers (L1-L16)"
else
    echo "Layer filter: L${LAYER_FILTER}"
fi
echo "Test script: $TEST_SCRIPT"
echo "=============================================================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find all tensor files
TENSOR_FILES=($(find "$TENSOR_DIR" -type f -name "*.pt" | sort))

if [ ${#TENSOR_FILES[@]} -eq 0 ]; then
    echo "⚠️  Warning: No .pt tensor files found in $TENSOR_DIR"
    exit 0
fi

echo "Found ${#TENSOR_FILES[@]} tensor file(s)"
echo ""

# Organize files by pass type, tensor type, and layer
declare -A FORWARD_FILES
declare -A BACKWARD_FILES

# For average mode: organize by pass_type -> tensor_type -> layer
declare -A FORWARD_FILES_BY_LAYER
declare -A BACKWARD_FILES_BY_LAYER

for tensor_file in "${TENSOR_FILES[@]}"; do
    layer=$(get_layer "$(basename "$tensor_file")")
    
    # Skip if no layer found
    if [ -z "$layer" ]; then
        continue
    fi
    
    # Check layer filter
    if [ "$CALC_AVG" = false ] && [ "$layer" != "$LAYER_FILTER" ]; then
        continue
    fi
    
    # Only process layers 1-16 for average mode
    if [ "$CALC_AVG" = true ] && (( layer < 1 || layer > 16 )); then
        continue
    fi
    
    tensor_type=$(get_tensor_type "$(basename "$tensor_file")")
    pass_type=$(get_pass_type "$(basename "$tensor_file")")
    
    if [ -n "$tensor_type" ] && [ -n "$pass_type" ]; then
        if [ "$CALC_AVG" = true ]; then
            # Store files by layer for average calculation
            if [ "$pass_type" == "forward" ]; then
                FORWARD_FILES_BY_LAYER["${tensor_type}_L${layer}"]="$tensor_file"
            elif [ "$pass_type" == "backward" ]; then
                BACKWARD_FILES_BY_LAYER["${tensor_type}_L${layer}"]="$tensor_file"
            fi
        else
            # Single layer mode
            if [ "$pass_type" == "forward" ]; then
                FORWARD_FILES["$tensor_type"]="$tensor_file"
            elif [ "$pass_type" == "backward" ]; then
                BACKWARD_FILES["$tensor_type"]="$tensor_file"
            fi
        fi
    fi
done

# =============================================================================
# Generate Results Table
# =============================================================================

# Function to generate table for a pass type
generate_table() {
    local pass_type="$1"
    local -n files_ref="$2"
    
    echo "=============================================================================="
    echo "$(echo $pass_type | tr '[:lower:]' '[:upper:]') Pass"
    echo "=============================================================================="
    echo ""
    
    # Print header
    printf "%-20s |" "Format"
    for ttype in "${TENSOR_TYPES[@]}"; do
        printf " %-12s |" "$ttype"
    done
    echo ""
    echo "$(printf '=%.0s' {1..95})"
    
    # Process each format
    for i in "${!FORMATS[@]}"; do
        format="${FORMATS[$i]}"
        format_name="${FORMAT_NAMES[$i]}"
        
        printf "%-20s |" "$format_name"
        
        # Process each tensor type
        for ttype in "${TENSOR_TYPES[@]}"; do
            if [ -n "${files_ref[$ttype]}" ]; then
                tensor_file="${files_ref[$ttype]}"
                echo -n "Testing $ttype ($format)... " >&2
                result=$(run_quantization_test "$tensor_file" "$format")
                
                # Format result for display
                if [[ "$result" == "ERROR" ]] || [[ "$result" == "N/A" ]]; then
                    printf " %-12s |" "$result"
                else
                    # Format as scientific notation with 4 decimal places
                    printf " %12.4e |" "$result"
                fi
                echo "done" >&2
            else
                printf " %-12s |" "N/A"
            fi
        done
        echo ""
    done
    echo ""
}

# Function to generate average table
generate_average_table() {
    local pass_type="$1"
    local -n files_ref="$2"
    
    echo "=============================================================================="
    echo "$(echo $pass_type | tr '[:lower:]' '[:upper:]') Pass - Average Across Layers (L1-L16)"
    echo "=============================================================================="
    echo ""
    
    # Print header
    printf "%-20s |" "Format"
    for ttype in "${TENSOR_TYPES[@]}"; do
        printf " %-12s |" "$ttype"
    done
    echo ""
    echo "$(printf '=%.0s' {1..95})"
    
    # Process each format
    for i in "${!FORMATS[@]}"; do
        format="${FORMATS[$i]}"
        format_name="${FORMAT_NAMES[$i]}"
        
        printf "%-20s |" "$format_name"
        
        # Process each tensor type
        for ttype in "${TENSOR_TYPES[@]}"; do
            # Collect errors from all layers for this tensor type
            local layer_errors=()
            local found_any=false
            
            for layer in {1..16}; do
                local key="${ttype}_L${layer}"
                if [ -n "${files_ref[$key]}" ]; then
                    tensor_file="${files_ref[$key]}"
                    echo -n "Testing $ttype L${layer} ($format)... " >&2
                    result=$(run_quantization_test "$tensor_file" "$format")
                    echo "done" >&2
                    
                    if [[ "$result" =~ ^[0-9.eE+-]+$ ]] && [[ "$result" != "N/A" ]] && [[ "$result" != "ERROR" ]]; then
                        layer_errors+=("$result")
                        found_any=true
                    fi
                fi
            done
            
            # Calculate average
            if [ "$found_any" = true ] && [ ${#layer_errors[@]} -gt 0 ]; then
                avg_result=$(calculate_average "${layer_errors[@]}")
                if [[ "$avg_result" == "N/A" ]]; then
                    printf " %-12s |" "$avg_result"
                else
                    printf " %12.4e |" "$avg_result"
                fi
            else
                printf " %-12s |" "N/A"
            fi
        done
        echo ""
    done
    echo ""
}

# Generate tables
if [ "$CALC_AVG" = true ]; then
    # Average mode
    if [ ${#FORWARD_FILES_BY_LAYER[@]} -gt 0 ]; then
        generate_average_table "forward" FORWARD_FILES_BY_LAYER
    fi
    
    if [ ${#BACKWARD_FILES_BY_LAYER[@]} -gt 0 ]; then
        generate_average_table "backward" BACKWARD_FILES_BY_LAYER
    fi
else
    # Single layer mode
    if [ ${#FORWARD_FILES[@]} -gt 0 ]; then
        generate_table "forward" FORWARD_FILES
    fi
    
    if [ ${#BACKWARD_FILES[@]} -gt 0 ]; then
        generate_table "backward" BACKWARD_FILES
    fi
fi

# =============================================================================
# Summary
# =============================================================================

echo "=============================================================================="
echo "Summary"
echo "=============================================================================="
echo "Results saved to: $OUTPUT_DIR/"
echo "Total tensor files found: ${#TENSOR_FILES[@]}"
echo ""

if [ "$CALC_AVG" = true ]; then
    echo "Mode: Average across layers (L1-L16)"
    echo ""
    echo "Forward pass tensors found by layer:"
    for ttype in "${TENSOR_TYPES[@]}"; do
        local count=0
        for layer in {1..16}; do
            local key="${ttype}_L${layer}"
            if [ -n "${FORWARD_FILES_BY_LAYER[$key]}" ]; then
                ((count++))
            fi
        done
        if [ $count -gt 0 ]; then
            echo "  - $ttype: found in $count layer(s)"
        fi
    done
    echo ""
    echo "Backward pass tensors found by layer:"
    for ttype in "${TENSOR_TYPES[@]}"; do
        local count=0
        for layer in {1..16}; do
            local key="${ttype}_L${layer}"
            if [ -n "${BACKWARD_FILES_BY_LAYER[$key]}" ]; then
                ((count++))
            fi
        done
        if [ $count -gt 0 ]; then
            echo "  - $ttype: found in $count layer(s)"
        fi
    done
else
    echo "Mode: Single layer (L${LAYER_FILTER})"
    echo ""
    echo "Forward pass tensors found: ${#FORWARD_FILES[@]}"
    for ttype in "${TENSOR_TYPES[@]}"; do
        if [ -n "${FORWARD_FILES[$ttype]}" ]; then
            echo "  - $ttype: $(basename "${FORWARD_FILES[$ttype]}")"
        fi
    done
    echo ""
    echo "Backward pass tensors found: ${#BACKWARD_FILES[@]}"
    for ttype in "${TENSOR_TYPES[@]}"; do
        if [ -n "${BACKWARD_FILES[$ttype]}" ]; then
            echo "  - $ttype: $(basename "${BACKWARD_FILES[$ttype]}")"
        fi
    done
fi
echo ""


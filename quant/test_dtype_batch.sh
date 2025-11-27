#!/bin/bash
# =============================================================================
# Batch Test Script for Quantization Error Analysis
# 
# Tests quant_hif8 and _quantize_mx on all tensors in enhanced_tensor_logs/bf16
# Generates a table showing quantization errors for:
# - Forward/Backward passes
# - Tensor types: input, weight, query, key, value
# - Formats: MXFP8(E4M3), MXFP8(E5M2), HiFP8
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
LAYER_FILTER="${3:-1}"

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

# =============================================================================
# Main Processing
# =============================================================================

cd "$PROJECT_ROOT"

echo "=============================================================================="
echo "Batch Quantization Error Analysis"
echo "=============================================================================="
echo "Tensor directory: $TENSOR_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Layer filter: L${LAYER_FILTER}"
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

# Organize files by pass type and tensor type
declare -A FORWARD_FILES
declare -A BACKWARD_FILES

for tensor_file in "${TENSOR_FILES[@]}"; do
    # Check if file matches layer filter
    layer=$(get_layer "$(basename "$tensor_file")")
    if [ -z "$layer" ] || [ "$layer" != "$LAYER_FILTER" ]; then
        continue
    fi
    
    tensor_type=$(get_tensor_type "$(basename "$tensor_file")")
    pass_type=$(get_pass_type "$(basename "$tensor_file")")
    
    if [ -n "$tensor_type" ] && [ -n "$pass_type" ]; then
        if [ "$pass_type" == "forward" ]; then
            FORWARD_FILES["$tensor_type"]="$tensor_file"
        elif [ "$pass_type" == "backward" ]; then
            BACKWARD_FILES["$tensor_type"]="$tensor_file"
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

# Generate tables
if [ ${#FORWARD_FILES[@]} -gt 0 ]; then
    generate_table "forward" FORWARD_FILES
fi

if [ ${#BACKWARD_FILES[@]} -gt 0 ]; then
    generate_table "backward" BACKWARD_FILES
fi

# =============================================================================
# Summary
# =============================================================================

echo "=============================================================================="
echo "Summary"
echo "=============================================================================="
echo "Results saved to: $OUTPUT_DIR/"
echo "Total tensor files processed: ${#TENSOR_FILES[@]}"
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
echo ""


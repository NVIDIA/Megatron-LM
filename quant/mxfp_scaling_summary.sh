#!/bin/bash
# =============================================================================
# MXFP Scaling Summary Script
# 批量处理 enhanced_tensor_logs/bf16 目录下的所有 tensor 文件
# 对每个 tensor 进行 fp8_e4m3, fp8_e5m2, fp4_e2m1 三种格式的模拟量化分析
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Input directory containing BF16 tensors
INPUT_DIR="${1:-enhanced_tensor_logs/bf16}"

# Output base directory (default: ./draw/scaling_analysis/)
OUTPUT_BASE_DIR="${2:-./draw/scaling_analysis}"

# Element formats to test
ELEM_FORMATS=("fp8_e4m3" "fp8_e5m2" "fp4_e2m1")

# Script directory (where mxfp_scaling_test.py is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/mxfp_scaling_test.py"

# =============================================================================
# Validation
# =============================================================================

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Check if test script exists
if [ ! -f "$TEST_SCRIPT" ]; then
    echo "❌ Error: Test script not found: $TEST_SCRIPT"
    exit 1
fi

# =============================================================================
# Find all tensor files
# =============================================================================

echo "=============================================================================="
echo "MXFP Scaling Summary - Batch Processing"
echo "=============================================================================="
echo "Input directory: $INPUT_DIR"
echo "Output base directory: $OUTPUT_BASE_DIR"
echo "Element formats: ${ELEM_FORMATS[*]}"
echo "=============================================================================="
echo ""

# Find all .pt files in the input directory (recursively)
TENSOR_FILES=($(find "$INPUT_DIR" -type f -name "*.pt" | sort))

if [ ${#TENSOR_FILES[@]} -eq 0 ]; then
    echo "⚠️  Warning: No .pt tensor files found in $INPUT_DIR"
    exit 0
fi

TOTAL_TENSORS=${#TENSOR_FILES[@]}
TOTAL_TESTS=$((TOTAL_TENSORS * ${#ELEM_FORMATS[@]}))

echo "Found $TOTAL_TENSORS tensor file(s)"
echo "Will run $TOTAL_TESTS test(s) in total"
echo ""

# =============================================================================
# Process each tensor with each format
# =============================================================================

cd "$PROJECT_ROOT"

SUCCESSFUL_TESTS=0
FAILED_TESTS=0
FAILED_FILES=()

# Track progress
CURRENT_TEST=0

for tensor_file in "${TENSOR_FILES[@]}"; do
    tensor_name=$(basename "$tensor_file" .pt)
    tensor_rel_path=$(realpath --relative-to="$PROJECT_ROOT" "$tensor_file")
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Processing tensor: $tensor_name"
    echo "File: $tensor_rel_path"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    for elem_format in "${ELEM_FORMATS[@]}"; do
        CURRENT_TEST=$((CURRENT_TEST + 1))
        
        echo ""
        echo "[$CURRENT_TEST/$TOTAL_TESTS] Testing: $tensor_name with $elem_format"
        echo "──────────────────────────────────────────────────────────────────────"
        
        # Run the test script
        # Output will be saved to: ./draw/scaling_analysis/{elem_format}/{tensor_name}/
        if python3 "$TEST_SCRIPT" "$tensor_file" \
            --elem-format "$elem_format" \
            --output-dir "$OUTPUT_BASE_DIR/$elem_format/$tensor_name"; then
            
            SUCCESSFUL_TESTS=$((SUCCESSFUL_TESTS + 1))
            echo "✅ Success: $tensor_name ($elem_format)"
        else
            FAILED_TESTS=$((FAILED_TESTS + 1))
            FAILED_FILES+=("$tensor_name ($elem_format)")
            echo "❌ Failed: $tensor_name ($elem_format)"
        fi
        
        echo ""
    done
    
    echo ""
done

# =============================================================================
# Summary
# =============================================================================

echo "=============================================================================="
echo "FINAL SUMMARY"
echo "=============================================================================="
echo "Total tensors processed: $TOTAL_TENSORS"
echo "Total tests run: $TOTAL_TESTS"
echo "Successful: $SUCCESSFUL_TESTS"
echo "Failed: $FAILED_TESTS"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo "🎉 All tests completed successfully!"
    echo ""
    echo "Results saved to: $OUTPUT_BASE_DIR/"
    echo "  - fp8_e4m3/: FP8 E4M3 format results"
    echo "  - fp8_e5m2/: FP8 E5M2 format results"
    echo "  - fp4_e2m1/: FP4 E2M1 format results"
    echo ""
    echo "Each tensor has its own subdirectory with:"
    echo "  - Log files (.log)"
    echo "  - Plot images (.png)"
    echo "  - Detailed results (.txt)"
    exit 0
else
    echo "⚠️  Some tests failed. Failed tests:"
    for failed in "${FAILED_FILES[@]}"; do
        echo "  - $failed"
    done
    echo ""
    echo "Check individual log files for details."
    exit 1
fi


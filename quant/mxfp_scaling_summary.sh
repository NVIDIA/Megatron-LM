#!/bin/bash
# =============================================================================
# MXFP Scaling Summary Script (Multi-threaded)
# 批量处理 enhanced_tensor_logs/bf16 目录下的所有 tensor 文件
# 对每个 tensor 进行 fp8_e4m3, fp8_e5m2, fp4_e2m1 三种格式的模拟量化分析
#
# Usage:
#   ./mxfp_scaling_summary.sh [INPUT_DIR] [OUTPUT_DIR] [JOBS]
#
# Arguments:
#   INPUT_DIR    - Input directory containing BF16 tensors (default: enhanced_tensor_logs/bf16)
#   OUTPUT_DIR   - Output base directory (default: ./draw/scaling_analysis)
#   JOBS         - Number of parallel jobs (default: number of CPU cores)
#
# Example:
#   ./mxfp_scaling_summary.sh enhanced_tensor_logs/bf16 ./draw/scaling_analysis 8
# =============================================================================

# =============================================================================
# Configuration
# =============================================================================

# Parse command line arguments
INPUT_DIR="${1:-enhanced_tensor_logs/bf16}"
OUTPUT_BASE_DIR="${2:-./draw/scaling_analysis}"
JOBS="${3:-$(nproc 2>/dev/null || echo 4)}"  # Number of parallel jobs (default: CPU cores or 4)

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
echo "MXFP Scaling Summary - Batch Processing (Multi-threaded)"
echo "=============================================================================="
echo "Input directory: $INPUT_DIR"
echo "Output base directory: $OUTPUT_BASE_DIR"
echo "Element formats: ${ELEM_FORMATS[*]}"
echo "Parallel jobs: $JOBS"
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
# Process each tensor with each format (Parallel)
# =============================================================================

cd "$PROJECT_ROOT"

# Create temporary directory for job tracking
TMP_DIR=$(mktemp -d)
trap "rm -rf '$TMP_DIR'" EXIT

# Create output lock file for synchronized output
OUTPUT_LOCK="$TMP_DIR/output.lock"
touch "$OUTPUT_LOCK"

# Check if flock is available
HAS_FLOCK=$(command -v flock >/dev/null 2>&1 && echo "yes" || echo "no")

# Export variables for subprocesses
export PROJECT_ROOT TEST_SCRIPT OUTPUT_BASE_DIR TMP_DIR TOTAL_TESTS OUTPUT_LOCK HAS_FLOCK

# Build task list
TASK_ID=0
declare -a TASKS

for tensor_file in "${TENSOR_FILES[@]}"; do
    for elem_format in "${ELEM_FORMATS[@]}"; do
        TASK_ID=$((TASK_ID + 1))
        TASKS+=("$tensor_file|$elem_format|$TASK_ID")
    done
done

# Process tasks in parallel
echo "Starting parallel processing with $JOBS concurrent jobs..."
echo ""

SUCCESSFUL_TESTS=0
FAILED_TESTS=0
FAILED_FILES=()

# Process all tasks using job control
for task in "${TASKS[@]}"; do
    IFS='|' read -r tensor_file elem_format test_id <<< "$task"
    tensor_name=$(basename "$tensor_file" .pt)
    output_dir="$OUTPUT_BASE_DIR/$elem_format/$tensor_name"
    log_file="$TMP_DIR/test_${test_id}.log"
    result_file="$TMP_DIR/result_${test_id}.txt"
    
    # Wait if we've reached the job limit
    while [ $(jobs -r | wc -l) -ge $JOBS ]; do
        sleep 0.1
    done
    
    # Start background job
    (
        {
            echo "[$test_id/$TOTAL_TESTS] Testing: $tensor_name with $elem_format"
            echo "──────────────────────────────────────────────────────────────────────"
            
            # Run the test script
            if python3 "$TEST_SCRIPT" "$tensor_file" \
                --elem-format "$elem_format" \
                --output-dir "$output_dir" > "$log_file" 2>&1; then
                echo "SUCCESS|$tensor_name ($elem_format)" > "$result_file"
                echo "✅ Success: $tensor_name ($elem_format)"
            else
                echo "FAILED|$tensor_name ($elem_format)" > "$result_file"
                echo "❌ Failed: $tensor_name ($elem_format)"
            fi
            echo ""
        } | {
            # Use flock to synchronize output (if available)
            if [ "$HAS_FLOCK" = "yes" ]; then
                flock -x 200
                cat
            else
                # Fallback: simple output without locking
                cat
            fi
        } 200>"$OUTPUT_LOCK" 2>/dev/null || cat
    ) &
done

# Wait for all background jobs to complete
wait

# Collect results
for task in "${TASKS[@]}"; do
    IFS='|' read -r tensor_file elem_format test_id <<< "$task"
    result_file="$TMP_DIR/result_${test_id}.txt"
    
    if [ -f "$result_file" ]; then
        IFS='|' read -r status message <<< "$(cat "$result_file")"
        if [ "$status" = "SUCCESS" ]; then
            SUCCESSFUL_TESTS=$((SUCCESSFUL_TESTS + 1))
        else
            FAILED_TESTS=$((FAILED_TESTS + 1))
            FAILED_FILES+=("$message")
        fi
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        tensor_name=$(basename "$tensor_file" .pt)
        FAILED_FILES+=("$tensor_name ($elem_format)")
    fi
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


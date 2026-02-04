#!/bin/bash

set -e
set -x

# Parse command line arguments
usage() {
    echo "Usage: $0 --bucket BUCKET [--environment {lts|dev}] [--output-dir DIR]"
    echo ""
    echo "Available buckets:"
    echo "  - tests/unit_tests/data/"
    echo "  - tests/unit_tests/dist_checkpointing/*.py"
    echo "  - tests/unit_tests/dist_checkpointing/models/"
    echo "  - tests/unit_tests/transformer/*.py"
    echo "  - tests/unit_tests/transformer/moe"
    echo "  - tests/unit_tests/distributed/fsdp"
    echo "  - tests/unit_tests"
    echo ""
    echo "Examples:"
    echo "  $0 --bucket tests/unit_tests/data/"
    echo "  $0 --bucket tests/unit_tests/transformer/*.py --environment dev"
    exit 1
}

# Default values
ENVIRONMENT="lts"
OUT_DIR="output"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --help|-h)
        usage
        ;;
    --bucket)
        BUCKET="$2"
        shift 2
        ;;
    --environment)
        ENVIRONMENT="$2"
        shift 2
        ;;
    --output-dir)
        OUT_DIR="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        usage
        ;;
    esac
done

# Validate required arguments
if [[ -z "${BUCKET:-}" ]]; then
    echo "Error: --bucket is required"
    usage
fi

# Validate ENVIRONMENT
if [[ "$ENVIRONMENT" != "lts" && "$ENVIRONMENT" != "dev" ]]; then
    echo "Error: ENVIRONMENT must be either 'lts' or 'dev'"
    usage
fi

# Get script directory
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_PATH


NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
echo "Number of GPUs: $NUM_GPUS"

mkdir -p $OUT_DIR

# Build pytest markers
MARKER="(not flaky and not flaky_in_dev and not internal and not failing_on_rocm and not failing_on_upstream or test_on_rocm) and not experimental"

if [[ "$ENVIRONMENT" == "lts" ]]; then
    MARKER="$MARKER and not flaky"
fi

if [[ "$ENVIRONMENT" == "dev" ]]; then
    MARKER="$MARKER and not flaky_in_dev"
fi

if [[ "$HIP_ARCHITECTURES" == "gfx90a" ]]; then
    MARKER="$MARKER and not failing_on_rocm_mi250"
fi

# Extract test cases to ignore from YAML (all buckets except the one we want)
YAML_FILE="$SCRIPT_PATH/tests/test_utils/recipes/unit-tests.yaml"

if [[ ! -f "$YAML_FILE" ]]; then
    echo "Error: YAML file not found at $YAML_FILE"
    exit 1
fi

# Check if yq is available
if ! command -v yq &> /dev/null; then
    echo "Warning: yq is not installed. Will run all tests in bucket without exclusions."
    echo "Install yq with: pip install yq"
    IGNORE_TEST_CASES=""
else
    export BUCKET
    IGNORE_TEST_CASES=$(
        cat $YAML_FILE |
            yq eval '
        with(.products[].test_case; del(.[] | select(. == env(BUCKET)))) 
        | .products[].test_case[]
        ' |
            tr " " "\n"
    )
fi

# Build ignore arguments for pytest
IGNORE_ARGS=()
while IFS= read -r test_case; do
    if [[ -z "$test_case" ]]; then
        continue
    fi
    
    if [[ $test_case == *\** ]]; then
        # Handle wildcard patterns - expand them
        FILES=($(ls $test_case 2>/dev/null || echo ""))
        if [[ ${#FILES[@]} -gt 0 ]]; then
            echo "Ignoring files matching pattern: $test_case"
            for file in "${FILES[@]}"; do
                IGNORE_ARGS+=("--ignore=$file")
            done
        fi
    else
        echo "Ignoring: $test_case"
        IGNORE_ARGS+=("--ignore=$test_case")
    fi
done <<<"$IGNORE_TEST_CASES"

echo ""
echo "============================================"
echo "Running bucket: $BUCKET"
echo "Environment: $ENVIRONMENT"
echo "Markers: $MARKER"
echo "Ignore args count: ${#IGNORE_ARGS[@]}"
echo "============================================"
echo ""

# Verify bucket exists
if [[ "$BUCKET" == *\** ]]; then
    # Handle wildcard patterns - just check if pattern expands
    TEST_FILES=$(ls $BUCKET 2>/dev/null || echo "")
    if [[ -z "$TEST_FILES" ]]; then
        echo "Error: No files match pattern '$BUCKET'"
        exit 1
    fi
    echo "Bucket pattern matches files"
elif [[ -d "$BUCKET" ]]; then
    echo "Bucket is a directory: $BUCKET"
elif [[ -f "$BUCKET" ]]; then
    echo "Bucket is a file: $BUCKET"
else
    echo "Error: Bucket '$BUCKET' not found"
    exit 1
fi

# Generate unique report name from bucket path
BUCKET_NAME=$(echo "$BUCKET" | tr '/' '_' | tr '*' 'x' | sed 's/^_//' | sed 's/_$//')

# Special handling for FSDP tests - must run each test method in separate sessions
# Reason: FSDP tests can corrupt PyTorch's distributed state (process groups, mesh resources)
# which affects subsequent tests. Running each test method in its own torchrun session
# ensures each starts with fresh Python interpreter state.
if [[ "$BUCKET" == *"distributed/fsdp"* ]]; then
    echo "============================================"
    echo "FSDP bucket detected"
    echo "Running each test method in separate session"
    echo "to avoid process group state corruption"
    echo "============================================"
    echo ""
    
    # Discover all unique test method names dynamically from Python files
    # Using grep to find "def test_" patterns - more reliable than pytest --collect-only
    echo "Discovering test methods..."
    TEST_METHODS=$(find $BUCKET -name "test_*.py" -exec grep -h "^    def test_" {} \; 2>/dev/null | \
        sed 's/^    def //g' | \
        sed 's/(.*//g' | \
        sort -u)
    
    # Fallback: if no test methods found with indented pattern, try without indent
    if [[ -z "$TEST_METHODS" ]]; then
        TEST_METHODS=$(find $BUCKET -name "test_*.py" -exec grep -h "def test_" {} \; 2>/dev/null | \
            sed 's/.*def //g' | \
            sed 's/(.*//g' | \
            sort -u)
    fi
    
    if [[ -z "$TEST_METHODS" ]]; then
        echo "Error: No test methods found in $BUCKET"
        exit 1
    fi
    
    # Count total test methods
    TOTAL_METHODS=$(echo "$TEST_METHODS" | wc -l)
    echo "Found $TOTAL_METHODS unique test methods:"
    echo "$TEST_METHODS" | sed 's/^/  - /'
    echo ""
    
    # Run each test method in its own session
    SESSION=1
    FAILED_TESTS=()
    
    for TEST_METHOD in $TEST_METHODS; do
        echo "--------------------------------------------"
        echo "Session $SESSION/$TOTAL_METHODS: $TEST_METHOD"
        echo "--------------------------------------------"
        echo ""
        
        # Generate safe filename from test method name
        SAFE_NAME=$(echo "$TEST_METHOD" | tr -cd '[:alnum:]_')
        
        if torchrun --standalone --nproc_per_node=$NUM_GPUS -m pytest \
            --showlocals --tb=long -v -s -m "$MARKER" \
            --csv $OUT_DIR/test_report_${BUCKET_NAME}_${SAFE_NAME}.csv \
            ${IGNORE_ARGS[@]} \
            -k "$TEST_METHOD" \
            $BUCKET; then
            echo ""
            echo "✓ Session $SESSION passed: $TEST_METHOD"
        else
            echo ""
            echo "✗ Session $SESSION failed: $TEST_METHOD"
            FAILED_TESTS+=("$TEST_METHOD")
        fi
        
        echo ""
        SESSION=$((SESSION + 1))
    done
    
    # Summary
    echo "============================================"
    if [[ ${#FAILED_TESTS[@]} -eq 0 ]]; then
        echo "✓ All $TOTAL_METHODS FSDP test methods passed!"
    else
        echo "✗ ${#FAILED_TESTS[@]} test method(s) failed:"
        for FAILED in "${FAILED_TESTS[@]}"; do
            echo "  - $FAILED"
        done
        exit 1
    fi
    echo "============================================"
else
    # Standard bucket - run all tests in one session
    echo "Running pytest for bucket..."
    echo ""

    if torchrun --standalone --nproc_per_node=$NUM_GPUS -m pytest \
        --showlocals --tb=long -v -s -m "$MARKER" \
        --csv $OUT_DIR/test_report_${BUCKET_NAME}.csv \
        ${IGNORE_ARGS[@]} \
        $BUCKET; then
        
        echo ""
        echo "============================================"
        echo "✓ All tests in bucket passed successfully!"
        echo "============================================"
    else
        echo ""
        echo "============================================"
        echo "✗ Tests in bucket failed"
        echo "============================================"
        exit 1
    fi
fi

echo ""
echo "Done!"

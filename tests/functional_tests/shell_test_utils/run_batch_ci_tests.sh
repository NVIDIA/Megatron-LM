#!/bin/bash
#
# Script to submit batch jobs to run test scripts across different compute nodes
#
# Usage:
#   ./run_batch_ci_tests.sh <test_script> [num_jobs] [partition]
#
# Arguments:
#   test_script  - Path to test script in test_cases/ (required)
#   num_jobs     - Number of jobs to submit (default: 10)
#   partition    - Slurm partition to use (default: interactive)
#
# Examples:
#   ./run_batch_ci_tests.sh test_cases/moe/gpt_grpo_tp4tp2_pp1_ep4ep2_dp8_throughputtest.sh
#   ./run_batch_ci_tests.sh test_cases/gpt/gpt3_mcore_te_tp2_pp2.sh 5
#   ./run_batch_ci_tests.sh test_cases/bert/bert_mcore_tp2_pp2.sh 10 batch_block1
#
# To list available test scripts:
#   ./run_batch_ci_tests.sh --list
#   ./run_batch_ci_tests.sh --list moe      # List only moe tests
#   ./run_batch_ci_tests.sh --list gpt      # List only gpt tests
#

set -e

# Function to list available test scripts
list_tests() {
    local filter="${1:-}"
    echo "Available test scripts in test_cases/:"
    echo
    if [ -n "$filter" ]; then
        # List tests in specific subdirectory
        if [ -d "test_cases/$filter" ]; then
            find "test_cases/$filter" -name "*.sh" -type f | sort
        else
            echo "No test_cases/$filter directory found."
            echo "Available subdirectories:"
            ls -d test_cases/*/ 2>/dev/null | sed 's|test_cases/||g; s|/||g' | xargs -I {} echo "  {}"
            exit 1
        fi
    else
        # List all tests grouped by subdirectory
        for dir in test_cases/*/; do
            if [ -d "$dir" ]; then
                subdir=$(basename "$dir")
                echo "=== $subdir ==="
                find "$dir" -name "*.sh" -type f | sort | sed 's|^|  |'
                echo
            fi
        done
    fi
    exit 0
}

# Handle --list option
if [ "${1:-}" = "--list" ]; then
    list_tests "${2:-}"
fi

# Configuration (same as start_ci_interactive.sh)
export DATASET_DIR=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci
export TGT_IMAGE=gitlab-master.nvidia.com/adlr/megatron-lm/mcore_ci_dev:main
export ACCOUNT=llmservice_fm_text

# The test script to run inside the container (first argument, required)
TEST_SCRIPT="${1:-}"

if [ -z "$TEST_SCRIPT" ]; then
    echo "ERROR: Test script path is required"
    echo
    echo "Usage: $0 <test_script> [num_jobs] [partition]"
    echo
    echo "Run '$0 --list' to see available test scripts"
    exit 1
fi

# Number of jobs to submit (second argument, default 10)
NUM_JOBS=${2:-10}

# Partition (third argument, default to same as interactive - change if needed)
# Common batch partition names: batch, batch_block1, dgx_batch, etc.
export PARTITION=${3:-interactive}

# Verify test script exists
if [ ! -f "$TEST_SCRIPT" ]; then
    echo "ERROR: Test script not found: $TEST_SCRIPT"
    echo "Make sure you run this from the megatron-rl directory"
    echo
    echo "Run '$0 --list' to see available test scripts"
    exit 1
fi

# Extract test name from script path for job naming
# e.g., "test_cases/moe/gpt_grpo_tp4tp2_pp1_ep4ep2_dp8_throughputtest.sh" -> "gpt_grpo_tp4tp2_pp1_ep4ep2_dp8_throughputtest"
TEST_NAME=$(basename "$TEST_SCRIPT" .sh)

# Output directory for logs (include test name for clarity)
LOG_DIR="$(pwd)/batch_test_logs_${TEST_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Container mounts
CONTAINER_MOUNTS="$DATASET_DIR:/mnt/artifacts,$(pwd):/opt/megatron-lm"

echo "============================================="
echo "Batch CI Test Submission"
echo "============================================="
echo "Test Script:  $TEST_SCRIPT"
echo "Test Name:    $TEST_NAME"
echo "Partition:    $PARTITION"
echo "Account:      $ACCOUNT"
echo "Image:        $TGT_IMAGE"
echo "Dataset Dir:  $DATASET_DIR"
echo "Num Jobs:     $NUM_JOBS"
echo "Log Dir:      $LOG_DIR"
echo "============================================="
echo

# Submit jobs
# Truncate test name if too long for job name (max ~64 chars typically)
SHORT_TEST_NAME="${TEST_NAME:0:50}"

for i in $(seq 1 $NUM_JOBS); do
    JOB_NAME="${SHORT_TEST_NAME}_run_${i}"
    
    sbatch \
        --job-name="$JOB_NAME" \
        --partition="$PARTITION" \
        --account="$ACCOUNT" \
        --nodes=1 \
        --gpus-per-task=8 \
        --time=1:00:00 \
        --exclusive \
        --output="$LOG_DIR/${JOB_NAME}_%j.out" \
        --error="$LOG_DIR/${JOB_NAME}_%j.err" \
        --export=ALL \
        --wrap="srun \
            --container-image=$TGT_IMAGE \
            --container-workdir=/opt/megatron-lm \
            --container-mounts=$CONTAINER_MOUNTS \
            --no-container-mount-home \
            bash -c 'cd /opt/megatron-lm && time bash $TEST_SCRIPT'"
    
    echo "Submitted job $i: $JOB_NAME"
done

echo
echo "============================================="
echo "All $NUM_JOBS jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "Logs will be written to: $LOG_DIR"
echo "============================================="

# Create a helper script to check results
cat > "$LOG_DIR/check_results.sh" << 'CHECKEOF'
#!/bin/bash
# Check the results of all batch test runs

LOG_DIR="$(dirname "$0")"
echo "Checking results in: $LOG_DIR"
echo

total=0
passed=0
failed=0
pending=0

# Match any .out file that ends with _run_N_JOBID.out pattern
for outfile in "$LOG_DIR"/*_run_*.out; do
    if [ -f "$outfile" ]; then
        total=$((total + 1))
        jobname=$(basename "$outfile" .out)
        
        # Check if file is empty (job still running or not started)
        if [ ! -s "$outfile" ]; then
            echo "PENDING: $jobname (no output yet)"
            pending=$((pending + 1))
            continue
        fi
        
        # Check for success: look for "This test wrote results into" which indicates completion
        if grep -q "This test wrote results into" "$outfile" 2>/dev/null; then
            # Check for errors/failures
            if grep -Ei "FAILED|AssertionError|Exception:|Traceback" "$outfile" 2>/dev/null | grep -v "grep" > /dev/null; then
                echo "FAILED:  $jobname"
                failed=$((failed + 1))
            else
                # Extract timing info
                timing=$(grep -E "^real\s" "$outfile" 2>/dev/null | head -1 || echo "")
                echo "PASSED:  $jobname $timing"
                passed=$((passed + 1))
            fi
        else
            # Job might still be running or crashed early
            if grep -qi "error\|failed\|exception\|traceback" "$outfile" 2>/dev/null; then
                echo "FAILED:  $jobname (error in output)"
                failed=$((failed + 1))
            else
                echo "RUNNING: $jobname (incomplete output)"
                pending=$((pending + 1))
            fi
        fi
    fi
done

echo
echo "============================================="
echo "Summary:"
echo "  Passed:  $passed"
echo "  Failed:  $failed"
echo "  Pending: $pending"
echo "  Total:   $total"
echo "============================================="

if [ $failed -gt 0 ]; then
    exit 1
elif [ $pending -gt 0 ]; then
    exit 2
else
    exit 0
fi
CHECKEOF
chmod +x "$LOG_DIR/check_results.sh"

# Create a script to show node info for each job
cat > "$LOG_DIR/show_nodes.sh" << 'NODEEOF'
#!/bin/bash
# Show which node each job ran on

LOG_DIR="$(dirname "$0")"
echo "Node assignments for batch tests:"
echo

# Match any .out file that ends with _run_N_JOBID.out pattern
for outfile in "$LOG_DIR"/*_run_*.out; do
    if [ -f "$outfile" ]; then
        jobname=$(basename "$outfile" .out)
        jobid=$(echo "$outfile" | grep -oP '\d+(?=\.out)')
        
        # Try to get node from sacct or from output file
        node=$(sacct -j "$jobid" --format=NodeList --noheader 2>/dev/null | head -1 | tr -d ' ')
        if [ -z "$node" ]; then
            node="unknown"
        fi
        
        echo "$jobname (job $jobid): $node"
    fi
done
NODEEOF
chmod +x "$LOG_DIR/show_nodes.sh"

echo "After jobs complete:"
echo "  - Run '$LOG_DIR/check_results.sh' to check results"
echo "  - Run '$LOG_DIR/show_nodes.sh' to see which nodes were used"
echo
echo "To run other tests, use: $0 --list to see available test scripts"

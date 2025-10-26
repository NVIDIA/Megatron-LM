#!/bin/bash
set -euxo pipefail

# Parse command line arguments
usage() {
    echo "Usage: $0 --tag {latest|legacy} --environment {lts|dev} --bucket BUCKET [--unit-test-repeat N] [--unit-test-timeout N] --log-dir LOG_DIR"
    exit 1
}

# Get directory of this script
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_PATH/../../

# Default values
UNIT_TEST_REPEAT=1
UNIT_TEST_TIMEOUT=10
LOG_DIR=$(pwd)/logs

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --help)
        usage
        ;;
    --tag)
        TAG="$2"
        shift 2
        ;;
    --environment)
        ENVIRONMENT="$2"
        shift 2
        ;;
    --bucket)
        BUCKET="$2"
        shift 2
        ;;
    --unit-test-repeat)
        UNIT_TEST_REPEAT="$2"
        shift 2
        ;;
    --unit-test-timeout)
        UNIT_TEST_TIMEOUT="$2"
        shift 2
        ;;
    --log-dir)
        LOG_DIR="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        usage
        ;;
    esac
done

# Validate required arguments
if [[ -z "${TAG:-}" || -z "${ENVIRONMENT:-}" || -z "${BUCKET:-}" ]]; then
    echo "Error: Missing required arguments"
    usage
fi

# Validate TAG
if [[ "$TAG" != "latest" && "$TAG" != "legacy" ]]; then
    echo "Error: TAG must be either 'latest' or 'legacy'"
    usage
fi

# Validate ENVIRONMENT
if [[ "$ENVIRONMENT" != "lts" && "$ENVIRONMENT" != "dev" ]]; then
    echo "Error: ENVIRONMENT must be either 'dev' or 'dev'"
    usage
fi

# Validate LOG_DIR
if [[ -z "${LOG_DIR:-}" ]]; then
    echo "Error: LOG_DIR is required"
    usage
else
    mkdir -p $LOG_DIR
fi

# Set default timeout if not specified
if [[ "$UNIT_TEST_TIMEOUT" == "10" ]]; then
    UNIT_TEST_TIMEOUT=$((10 * UNIT_TEST_REPEAT))
fi

# Convert ENVIRONMENT to lowercase for internal use
ENVIRONMENT=$(echo "$ENVIRONMENT" | tr '[:upper:]' '[:lower:]')

if [[ "$TAG" == "latest" ]]; then
    TEST_PATH="/opt/megatron-lm"
else
    TEST_PATH="/opt/megatron-lm-legacy/"
fi

cd $TEST_PATH

MARKER=()
if [[ "$TAG" == "legacy" ]]; then
    MARKER+=("not internal")
fi

if [[ "$ENVIRONMENT" == "lts" ]]; then
    MARKER+=("not flaky")
fi

if [[ "$ENVIRONMENT" == "dev" ]]; then
    MARKER+=("not flaky_in_dev")
fi

MARKER_ARG=$(printf "%s" "${MARKER[0]}")
for element in "${MARKER[@]:1}"; do
    MARKER_ARG+=" and $element"
done

export BUCKET
IGNORE_ARGS=()
while IFS= read -r line; do
    [[ -n "$line" ]] && IGNORE_ARGS+=("$line")
done < <(python tests/unit_tests/find_test_cases.py "$BUCKET")

echo "------ARGUMENTS for SLURM ---"
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NUM_NODES=${NUM_NODES:-${SLURM_NNODES:-1}}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODE_RANK=${SLURM_NODEID:-${SLURM_NODEID:-0}}
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --node_rank $NODE_RANK
    --log-dir $LOG_DIR
    --tee "0:3"
    --redirects "3"
)

# Reduce memory usage by NCCL
export NCCL_MAX_NCHANNELS=1
export NCCL_NVLS_ENABLE=0
export ONE_LOGGER_JOB_CATEGORY=test

for i in $(seq $UNIT_TEST_REPEAT); do
    echo "Running prod test suite."
    CMD=$(echo uv run --no-sync python -m torch.distributed.run ${DISTRIBUTED_ARGS[@]} \
        -m coverage run \
        --data-file=.coverage.unit_tests \
        --source=megatron/core \
        -m pytest \
        -xvs \
        ${IGNORE_ARGS[@]} \
        -m "'not experimental and ${MARKER_ARG}'" $(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||'))
    eval "$CMD"

    if [[ "$TAG" == "latest" ]]; then
        CMD=$(echo uv run --no-sync python -m torch.distributed.run ${DISTRIBUTED_ARGS[@]} -m pytest \
            -xvs \
            --experimental \
             ${IGNORE_ARGS[@]} \
            -m "'experimental and ${MARKER_ARG}'" $(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||'))

        eval "$CMD"
    fi

done

coverage combine -q
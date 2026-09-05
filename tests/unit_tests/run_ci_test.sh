#!/bin/bash
set -euxo pipefail

# Parse command line arguments
usage() {
    echo "Usage: $0 --tag {latest|legacy} --environment {lts|dev} --bucket BUCKET [--platform {h100|gb200}] [--unit-test-repeat N] [--unit-test-timeout N] --log-dir LOG_DIR"
    exit 1
}

# Get directory of this script
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_PATH/../../"

# Default values
UNIT_TEST_REPEAT=1
UNIT_TEST_TIMEOUT=10
LOG_DIR=$(pwd)/logs
PLATFORM=h100

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
    --platform)
        PLATFORM="$2"
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
    mkdir -p "$LOG_DIR"
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

cd "$TEST_PATH"

MARKER=()
if [[ "$PLATFORM" == "gb200" ]]; then
    MARKER+=("launch_on_gb200")
fi

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
TEST_TARGETS=()
PYTEST_GUARD_ARGS=()

load_full_bucket() {
    IGNORE_ARGS=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && IGNORE_ARGS+=("$line")
    done < <(python tests/unit_tests/find_test_cases.py "$BUCKET" "$PLATFORM")
    TEST_TARGETS=("${BUCKET%/\*\*/\*.py}")
}

load_selected_files() {
    local decoded_files
    if ! decoded_files=$(python - "$UNIT_TEST_FILES_B64" <<'PY'
import base64
import binascii
import json
import sys
from pathlib import Path, PurePosixPath

try:
    payload = base64.b64decode(sys.argv[1], altchars=b"-_", validate=True)
    files = json.loads(payload)
except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError) as error:
    raise SystemExit(f"Invalid selective test payload: {error}") from error

if not isinstance(files, list) or not files:
    raise SystemExit("Selective test payload must be a non-empty JSON list")
if not all(isinstance(value, str) for value in files):
    raise SystemExit("Selective test payload entries must be paths")
if len(files) != len(set(files)):
    raise SystemExit("Selective test payload contains duplicate paths")

repo_root = Path.cwd().resolve()
unit_root = (repo_root / "tests/unit_tests").resolve()
for value in files:
    if "\n" in value or "\r" in value:
        raise SystemExit(f"Invalid selective test path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"Unsafe selective test path: {value!r}")
    candidate = (repo_root / path).resolve()
    try:
        candidate.relative_to(unit_root)
    except ValueError as error:
        raise SystemExit(f"Selective test path is outside tests/unit_tests: {value!r}") from error
    if not candidate.is_file() or not candidate.name.startswith("test_") or candidate.suffix != ".py":
        raise SystemExit(f"Selective test path is not a test file: {value!r}")
    print(value)
PY
    ); then
        return 1
    fi
    mapfile -t TEST_TARGETS <<< "$decoded_files"
    if [[ "${#TEST_TARGETS[@]}" -eq 0 ]]; then
        return 1
    fi
    IGNORE_ARGS=()
}

if [[ -n "${UNIT_TEST_FILES_B64:-}" ]]; then
    if load_selected_files; then
        PYTEST_GUARD_ARGS=(-p tests.unit_tests.selective_test_guard)
        SELECTIVE_STATE_DIR=$(mktemp -d /tmp/megatron-selective-tests.XXXXXX)
        export MCORE_SELECTED_TEST_SENTINEL="$SELECTIVE_STATE_DIR/collected"
        trap 'rm -rf "$SELECTIVE_STATE_DIR"' EXIT
        echo "Running ${#TEST_TARGETS[@]} selectively chosen test file(s) in bucket $BUCKET."
    else
        echo "::error::The selective test payload is invalid or stale; refusing a partial CI run."
        exit 1
    fi
else
    load_full_bucket
fi

echo "------ARGUMENTS for SLURM ---"
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_NODES=${NUM_NODES:-${SLURM_NNODES:-1}}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODE_RANK=${SLURM_NODEID:-${SLURM_NODEID:-0}}
DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
    --node_rank "$NODE_RANK"
    --log-dir "$LOG_DIR"
    --tee "0:3"
    --redirects "3"
)

export ONE_LOGGER_JOB_CATEGORY=test

# Run a pytest command. On marker-driven platforms a bucket can legitimately
# contain no matching tests; treat pytest's "no tests collected" (exit 5) as a
# pass there instead of aborting the job under `set -e`.
run_test_cmd() {
    local rc=0
    set +e
    "$@"
    rc=$?
    set -e
    if [[ "$rc" -eq 5 && "$PLATFORM" == "gb200" ]]; then
        echo "No tests collected for this bucket on $PLATFORM (pytest exit 5) — treating as pass."
        return 0
    fi
    return "$rc"
}

for _iteration in $(seq "$UNIT_TEST_REPEAT"); do
    echo "Running prod test suite."
    run_test_cmd uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
        -m coverage run \
        --data-file=.coverage.unit_tests \
        --source=megatron/core \
        -m pytest \
        "${PYTEST_GUARD_ARGS[@]}" \
        -vs \
        "${IGNORE_ARGS[@]}" \
        -m "not experimental and ${MARKER_ARG}" \
        "${TEST_TARGETS[@]}"

    if [[ "$TAG" == "latest" ]]; then
        run_test_cmd uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
            -m pytest \
            "${PYTEST_GUARD_ARGS[@]}" \
            -vs \
            --experimental \
            "${IGNORE_ARGS[@]}" \
            -m "experimental and ${MARKER_ARG}" \
            "${TEST_TARGETS[@]}"
    fi
done

if [[ -n "${UNIT_TEST_FILES_B64:-}" ]] && [[ ! -f "$MCORE_SELECTED_TEST_SENTINEL" ]]; then
    echo "::error::No selectively chosen tests survived collection and marker filtering."
    exit 1
fi

coverage combine -q

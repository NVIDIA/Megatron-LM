#!/bin/bash
set -euxo pipefail

# Parse command line arguments
usage() {
    echo "Usage: $0 --tag {latest|legacy} --environment {lts|dev} --bucket BUCKET [--platform {h100|gb200}] [--unit-test-repeat N] [--unit-test-timeout N] --log-dir LOG_DIR"
    exit 1
}

# Get directory of this script
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_PATH/../../

# Default values
UNIT_TEST_REPEAT=1
UNIT_TEST_TIMEOUT=10
LOG_DIR=$(pwd)/logs
PLATFORM=h100
UNIT_TESTMON_MODE=${UNIT_TESTMON_MODE:-full}
UNIT_TESTMON_CACHE_DIR=${UNIT_TESTMON_CACHE_DIR:-assets_dir/testmon}

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

if [[ "$UNIT_TESTMON_MODE" != "full" && "$UNIT_TESTMON_MODE" != "enforce" && "$UNIT_TESTMON_MODE" != "preselected" && "$UNIT_TESTMON_MODE" != "baseline" ]]; then
    echo "Error: invalid Testmon mode: $UNIT_TESTMON_MODE"
    usage
fi
if [[ "$UNIT_TESTMON_MODE" != "full" && ( "$TAG" != "latest" || "$ENVIRONMENT" != "dev" ) ]]; then
    echo "Testmon is only enabled for the latest dev suite; running the full bucket."
    UNIT_TESTMON_MODE=full
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

if [[ -n "${UNIT_TEST_ROOT:-}" ]]; then
    # Standalone compatibility launchers must run their own checkout.
    TEST_PATH="$UNIT_TEST_ROOT"
elif [[ "$TAG" == "latest" ]]; then
    TEST_PATH="/opt/megatron-lm"
else
    TEST_PATH="/opt/megatron-lm-legacy/"
fi

cd $TEST_PATH

# Load bucket-specific launch behavior before discovery and distributed setup.
UNIT_BUCKET_LAUNCHER=
case "$BUCKET" in
tests/unit_tests/inference/test_batch_invariant_fp8.py)
    UNIT_BUCKET_LAUNCHER=tests/test_utils/run_batch_invariant_fp8.sh
    ;;
tests/unit_tests/inference/engines/test_disagg_coordinator_pairwise.py|tests/unit_tests/inference/test_disagg_transfer_pairwise.py)
    UNIT_BUCKET_LAUNCHER=tests/test_utils/run_disagg_pairwise.sh
    ;;
esac
if [[ -n "$UNIT_BUCKET_LAUNCHER" ]]; then
    source "$UNIT_BUCKET_LAUNCHER"
    [[ "$UNIT_TEST_REPEAT" =~ ^[1-9][0-9]*$ ]]
    unit_bucket_setup
    LOG_DIR="$(cd "$LOG_DIR" && pwd)"
    UNIT_LAUNCH=0
fi

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
while IFS= read -r line; do
    [[ -n "$line" ]] && IGNORE_ARGS+=("$line")
done < <(python tests/unit_tests/find_test_cases.py "$BUCKET" "$PLATFORM")

echo "------ARGUMENTS for SLURM ---"
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
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

export ONE_LOGGER_JOB_CATEGORY=test

# Run a pytest command. On marker-driven platforms a bucket can legitimately
# contain no matching tests; treat pytest's "no tests collected" (exit 5) as a
# pass there instead of aborting the job under `set -e`.
run_test_cmd() {
    local rc=0
    "$@" || rc=$?
    if [[ "$rc" -eq 5 && "$PLATFORM" == "gb200" ]]; then
        echo "No tests collected for this bucket on $PLATFORM (pytest exit 5) — treating as pass."
        return 0
    fi
    return "$rc"
}

# Build worker arguments before adding a bucket's torchrun launch settings.
run_unit_worker() {
    local mode="$1" phase="$2" profile="$3" append="$4"
    shift 4
    local -a worker command
    if [[ "$mode" == baseline || "$mode" == select ]]; then
        worker=(tests/unit_tests/testmon_selector.py --mode "$mode"
            --cache-dir "$UNIT_TESTMON_CACHE_DIR" --phase "$phase")
        [[ "$append" != true ]] || worker+=(--append)
        worker+=(-- "$@")
    elif [[ "$phase" == prod ]]; then
        worker=(-m coverage run --data-file=.coverage.unit_tests --source=megatron/core
            -m pytest "$@")
    else
        worker=(-m pytest "$@")
    fi

    command=(uv run --no-sync python -m torch.distributed.run)
    if [[ -z "$profile" ]]; then
        "${command[@]}" "${DISTRIBUTED_ARGS[@]}" "${worker[@]}"
        return $?
    fi

    UNIT_LAUNCH=$((UNIT_LAUNCH + 1))
    local launch_dir="$LOG_DIR/$profile/$phase-$UNIT_LAUNCH"
    mkdir -p "$launch_dir" || return
    command+=(--standalone --nnodes=1 --nproc-per-node="$GPUS_PER_NODE" --max-restarts=0
        --log-dir "$launch_dir" --redirects=3 --tee=3)
    if [[ "$mode" == select ]]; then
        "${command[@]}" "${worker[@]}"
        return $?
    fi

    export UNIT_TEST_JUNIT_DIR="$launch_dir/results"
    rm -rf -- "$UNIT_TEST_JUNIT_DIR" || return
    mkdir -p "$UNIT_TEST_JUNIT_DIR" || return
    local rc=0
    # Expand RANK in each worker and retain argv boundaries through the shell.
    "${command[@]}" --no-python bash -c \
        'exec uv run --no-sync python "$@" --junitxml="$UNIT_TEST_JUNIT_DIR/rank-$RANK.xml"' \
        bash "${worker[@]}" || rc=$?
    [[ "$rc" -eq 0 || "$rc" -eq 5 ]] || return "$rc"
    validate_unit_results "$mode" "$phase"
}

validate_unit_results() {
    uv run --no-sync python - "$UNIT_TEST_JUNIT_DIR" "$GPUS_PER_NODE" "$1" "$2" <<'PYTHON'
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

root, ranks, mode, phase = sys.argv[1:]
for rank in range(int(ranks)):
    report = Path(root) / f"rank-{rank}.xml"
    assert report.is_file(), f"Rank {rank} is missing its report: {report}"
    cases = ET.parse(report).findall(".//testcase")
    if mode in ("full", "baseline") and phase == "prod":
        assert cases, f"Rank {rank} executed no tests: {report}"
    assert all(case.find(tag) is None for case in cases for tag in ("skipped", "failure", "error")), (
        f"Rank {rank} has skipped or failed tests: {report}"
    )
PYTHON
}

run_unit_phase() {
    local mode="$1" phase="$2"
    shift 2
    if [[ -n "$UNIT_BUCKET_LAUNCHER" ]]; then
        local rc=0
        unit_bucket_run "$mode" "$phase" "$@" || rc=$?
        if [[ "$mode" == baseline && "$rc" -ne 0 ]]; then
            rm -f -- "$UNIT_TESTMON_CACHE_DIR/$phase/metadata.json"
        fi
        return "$rc"
    else
        run_unit_worker "$mode" "$phase" "" false "$@"
    fi
}

# Keep this path identical to exhaustive CI. The installed Testmon plugin is
# inactive unless one of the selective paths below passes --testmon.
run_full_tests() {
    local target
    target=$(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||')
    for i in $(seq $UNIT_TEST_REPEAT); do
        echo "Running prod test suite."
        run_test_cmd run_unit_phase full prod \
            -vs "${IGNORE_ARGS[@]}" -m "not experimental and ${MARKER_ARG}" "$target"
        if [[ "$TAG" == latest ]]; then
            run_test_cmd run_unit_phase full experimental \
                -vs --experimental "${IGNORE_ARGS[@]}" -m "experimental and ${MARKER_ARG}" "$target"
        fi
    done
    coverage combine -q
}

write_testmon_summary() {
    local result="$1"
    local selected_count="${2:-}"
    local mandatory_count="${3:-}"
    mkdir -p "$UNIT_TESTMON_CACHE_DIR"
    {
        echo "### Unit Testmon"
        echo
        echo "- Mode: \`$UNIT_TESTMON_MODE\`"
        echo "- Bucket: \`$BUCKET\`"
        echo "- Result: $result"
        if [[ -n "$selected_count" ]]; then
            echo "- Selected pytest targets across phases: \`$selected_count\`"
        fi
        if [[ -n "$mandatory_count" ]]; then
            echo "- Mandatory test files (changed source mappings): \`$mandatory_count\`"
        fi
    } > "$UNIT_TESTMON_CACHE_DIR/summary.md"
}

run_testmon_phase() {
    local mode="$1"
    local phase="$2"
    shift 2
    if [[ "${UNIT_TESTMON_INSTALLED:-false}" != "true" ]]; then
        uv pip install --python /opt/venv/bin/python --no-deps "pytest-testmon==2.2.0" || return
        UNIT_TESTMON_INSTALLED=true
    fi
    run_unit_phase "$mode" "$phase" "$@"
}

run_baseline_tests() {
    local cache_root="${UNIT_TESTMON_CACHE_DIR:?}"
    local target
    target=$(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||')
    rm -rf -- "$cache_root/prod" "$cache_root/experimental" "$cache_root/.testmon-work"
    rm -f -- "$cache_root/summary.md"
    mkdir -p "$cache_root"

    run_testmon_phase baseline prod \
        -vs "${IGNORE_ARGS[@]}" -m "not experimental and ${MARKER_ARG}" "$target"
    run_testmon_phase baseline experimental \
        -vs --experimental "${IGNORE_ARGS[@]}" -m "experimental and ${MARKER_ARG}" "$target"
    write_testmon_summary "baseline produced"
}

merge_rank_selections() {
    local phase="$1"
    local output="$UNIT_TESTMON_CACHE_DIR/.testmon-work/$phase/selected-tests"
    local expected_ranks=$((NUM_NODES * GPUS_PER_NODE))
    local -a rank_selections

    shopt -s nullglob
    rank_selections=("$UNIT_TESTMON_CACHE_DIR/.testmon-work/$phase"/rank-*/selected-tests)
    shopt -u nullglob
    if [[ "${#rank_selections[@]}" -ne "$expected_ranks" ]]; then
        echo "Expected $expected_ranks Testmon selections for $phase, found ${#rank_selections[@]}."
        return 1
    fi
    cat "${rank_selections[@]}" | LC_ALL=C sort -u > "$output"
}

# Unit tests mapped to changed source directories in
# tests/unit_tests/testmon_mandatory_tests.yaml always run on top of the
# Testmon selection. The host writes the PR's changed-file list next to the baseline.
apply_mandatory_tests() {
    local phase="$1"
    uv run --no-sync python tests/unit_tests/testmon_mandatory.py \
        --changed-files "$UNIT_TESTMON_CACHE_DIR/changed-files" \
        --bucket "$BUCKET" \
        --platform "dgx_$PLATFORM" \
        --selection "$UNIT_TESTMON_CACHE_DIR/.testmon-work/$phase/selected-tests" \
        "${IGNORE_ARGS[@]}"
}

run_selected_phase() {
    local phase="$1"
    local selection_file="$UNIT_TESTMON_CACHE_DIR/.testmon-work/$phase/selected-tests"
    local -a selected_tests=()

    while IFS= read -r nodeid; do
        [[ -n "$nodeid" ]] && selected_tests+=("$nodeid")
    done < "$selection_file"
    if [[ "${#selected_tests[@]}" -eq 0 ]]; then
        echo "Testmon selected no $phase tests."
        return 0
    fi

    # Mandatory files can contain no tests for this phase.
    local rc=0
    if [[ "$phase" == prod ]]; then
        run_unit_phase selected "$phase" \
            -vs "${IGNORE_ARGS[@]}" -m "not experimental and ${MARKER_ARG}" \
            "${selected_tests[@]}" || rc=$?
    else
        run_unit_phase selected "$phase" \
            -vs --experimental "${IGNORE_ARGS[@]}" -m "experimental and ${MARKER_ARG}" \
            "${selected_tests[@]}" || rc=$?
    fi
    if [[ "$rc" -eq 5 ]]; then
        echo "No $phase tests collected from the selection (pytest exit 5) — treating as pass."
        return 0
    fi
    return "$rc"
}

run_enforced_tests() {
    local target prod_count experimental_count mandatory_count
    target=$(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||')
    if [[ "${UNIT_TESTMON_MODE:-enforce}" == "preselected" ]]; then
        # Consume the CPU decision without running selection again on GPU.
        if ! uv run --no-sync python tests/unit_tests/testmon_selector.py prepare \
                --plan "$UNIT_TESTMON_CACHE_DIR/plan.json" --cache-dir "$UNIT_TESTMON_CACHE_DIR" \
                --bucket "$BUCKET" --platform "dgx_$PLATFORM" \
                --source-sha "$(git rev-parse HEAD)"; then
            write_testmon_summary "CPU selection plan is unavailable or invalid"
            return 1
        fi
    else
        rm -rf -- "$UNIT_TESTMON_CACHE_DIR/.testmon-work"
        if ! run_testmon_phase select prod \
            -vs "${IGNORE_ARGS[@]}" -m "not experimental and ${MARKER_ARG}" "$target" \
            || ! merge_rank_selections prod \
            || ! apply_mandatory_tests prod \
            || ! run_testmon_phase select experimental \
                -vs --experimental "${IGNORE_ARGS[@]}" -m "experimental and ${MARKER_ARG}" "$target" \
            || ! merge_rank_selections experimental \
            || ! apply_mandatory_tests experimental; then
            write_testmon_summary "full fallback: Testmon selection failed"
            run_full_tests
            return
        fi
    fi

    prod_count=$(wc -l < "$UNIT_TESTMON_CACHE_DIR/.testmon-work/prod/selected-tests")
    experimental_count=$(wc -l < "$UNIT_TESTMON_CACHE_DIR/.testmon-work/experimental/selected-tests")
    mandatory_count=$(wc -l < "$UNIT_TESTMON_CACHE_DIR/.testmon-work/prod/mandatory-tests")
    for i in $(seq "$UNIT_TEST_REPEAT"); do
        run_selected_phase prod
        run_selected_phase experimental
    done
    if ! compgen -G '.coverage.unit_tests*' > /dev/null; then
        # Keep the downstream coverage aggregation valid when Testmon selects
        # only experimental tests (or no tests) for this bucket.
        uv run --no-sync python -c '
from pathlib import Path
from coverage import CoverageData

data = CoverageData(basename=".coverage.unit_tests")
data.add_lines({str(Path("megatron/core/__init__.py").resolve()): []})
data.write()
'
    fi
    coverage combine -q
    write_testmon_summary "selective tests passed" "$((prod_count + experimental_count))" "$mandatory_count"
}

case "$UNIT_TESTMON_MODE" in
full)
    run_full_tests
    ;;
baseline)
    run_baseline_tests
    ;;
enforce|preselected)
    run_enforced_tests
    ;;
esac

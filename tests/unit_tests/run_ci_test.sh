#!/bin/bash
set -euxo pipefail

usage() {
    echo "Usage: $0 --tag {latest|legacy} --environment {lts|dev} --bucket BUCKET [--platform {h100|gb200}] [--unit-test-repeat N] [--unit-test-timeout N] [--unit-testmon-mode {full|enforce|baseline}] --log-dir LOG_DIR"
    exit 1
}

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_PATH/../../"

UNIT_TEST_REPEAT=1
UNIT_TEST_TIMEOUT=10
LOG_DIR=$(pwd)/logs
PLATFORM=h100
UNIT_TESTMON_MODE=full
UNIT_TESTMON_CACHE_DIR=assets_dir/testmon

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
    --unit-testmon-mode)
        UNIT_TESTMON_MODE="$2"
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

if [[ -z "${TAG:-}" || -z "${ENVIRONMENT:-}" || -z "${BUCKET:-}" ]]; then
    echo "Error: Missing required arguments"
    usage
fi
if [[ "$TAG" != "latest" && "$TAG" != "legacy" ]]; then
    echo "Error: TAG must be either 'latest' or 'legacy'"
    usage
fi
if [[ "$ENVIRONMENT" != "lts" && "$ENVIRONMENT" != "dev" ]]; then
    echo "Error: ENVIRONMENT must be either 'lts' or 'dev'"
    usage
fi
if [[ "$UNIT_TESTMON_MODE" != "full" && "$UNIT_TESTMON_MODE" != "enforce" && "$UNIT_TESTMON_MODE" != "baseline" ]]; then
    echo "Error: invalid Testmon mode: $UNIT_TESTMON_MODE"
    usage
fi
if [[ "$UNIT_TESTMON_MODE" != "full" && ( "$TAG" != "latest" || "$ENVIRONMENT" != "dev" ) ]]; then
    echo "Testmon is only enabled for the latest dev suite; running the full bucket."
    UNIT_TESTMON_MODE=full
fi

mkdir -p "$LOG_DIR"
if [[ "$UNIT_TEST_TIMEOUT" == "10" ]]; then
    UNIT_TEST_TIMEOUT=$((10 * UNIT_TEST_REPEAT))
fi
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

run_test_cmd() {
    local cmd="$1"
    local rc=0
    set +e
    eval "$cmd"
    rc=$?
    set -e
    if [[ "$rc" -eq 5 && "$PLATFORM" == "gb200" ]]; then
        echo "No tests collected for this bucket on $PLATFORM (pytest exit 5) — treating as pass."
        return 0
    fi
    return "$rc"
}

# Keep this path identical to exhaustive CI. Testmon is installed and invoked
# only by the baseline/enforce paths below.
run_full_tests() {
    for i in $(seq "$UNIT_TEST_REPEAT"); do
        echo "Running prod test suite."
        CMD=$(echo uv run --no-sync python -m torch.distributed.run ${DISTRIBUTED_ARGS[@]} \
            -m coverage run \
            --data-file=.coverage.unit_tests \
            --source=megatron/core \
            -m pytest \
            -vs \
            ${IGNORE_ARGS[@]} \
            -m "'not experimental and ${MARKER_ARG}'" $(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||'))
        run_test_cmd "$CMD"

        if [[ "$TAG" == "latest" ]]; then
            CMD=$(echo uv run --no-sync python -m torch.distributed.run ${DISTRIBUTED_ARGS[@]} -m pytest \
                -vs \
                --experimental \
                ${IGNORE_ARGS[@]} \
                -m "'experimental and ${MARKER_ARG}'" $(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||'))
            run_test_cmd "$CMD"
        fi
    done
    coverage combine -q
}

install_testmon() {
    uv sync --locked --only-group testmon --inexact --no-install-project
}

write_testmon_summary() {
    local result="$1"
    local selected_count="${2:-}"
    mkdir -p "$UNIT_TESTMON_CACHE_DIR"
    {
        echo "### Unit Testmon"
        echo
        echo "- Mode: \`$UNIT_TESTMON_MODE\`"
        echo "- Bucket: \`$BUCKET\`"
        echo "- Result: $result"
        if [[ -n "$selected_count" ]]; then
            echo "- Selected tests: \`$selected_count\`"
        fi
    } > "$UNIT_TESTMON_CACHE_DIR/summary.md"
}

run_testmon_phase() {
    local mode="$1"
    local phase="$2"
    local with_coverage="$3"
    shift 3
    local -a command=(uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}")
    if [[ "$with_coverage" == "true" ]]; then
        command+=(
            -m coverage run
            --data-file=.coverage.unit_tests
            --source=megatron/core
        )
    fi
    command+=(
        tests/unit_tests/testmon_selector.py
        --mode "$mode"
        --cache-dir "$UNIT_TESTMON_CACHE_DIR"
        --phase "$phase"
        -- "$@"
    )
    "${command[@]}"
}

run_full_fallback() {
    local previous_addopts="${PYTEST_ADDOPTS-}"
    export PYTEST_ADDOPTS="${previous_addopts:+$previous_addopts }-p no:testmon -p no:pytest-testmon"
    run_full_tests
    if [[ -n "$previous_addopts" ]]; then
        export PYTEST_ADDOPTS="$previous_addopts"
    else
        unset PYTEST_ADDOPTS
    fi
}

run_baseline_tests() {
    local cache_root="${UNIT_TESTMON_CACHE_DIR:?}"
    local target
    target=$(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||')
    rm -rf -- "$cache_root/prod" "$cache_root/experimental" "$cache_root/.testmon-work"
    rm -f -- "$cache_root/summary.md"
    mkdir -p "$cache_root"

    install_testmon
    run_testmon_phase baseline prod false \
        -vs "${IGNORE_ARGS[@]}" -m "not experimental and ${MARKER_ARG}" "$target"
    run_testmon_phase baseline experimental false \
        -vs --experimental "${IGNORE_ARGS[@]}" -m "experimental and ${MARKER_ARG}" "$target"
    write_testmon_summary "baseline produced"
}

run_always_tests() {
    local -a test_files=()
    if [[ "$BUCKET" == "tests/unit_tests/**/*.py" ]]; then
        test_files+=(tests/unit_tests/test_basic.py)
    fi
    if [[ "$PLATFORM" == "h100" && "$BUCKET" == "tests/unit_tests/inference/**/*.py" ]]; then
        test_files+=(tests/unit_tests/inference/test_data_parallel_inference_coordinator.py)
    fi
    [[ "${#test_files[@]}" -gt 0 ]] || return 0

    uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
        -m coverage run \
        --data-file=.coverage.unit_tests \
        --source=megatron/core \
        -m pytest \
        -p no:testmon \
        -p no:pytest-testmon \
        -vs "${test_files[@]}"
}

selected_count() {
    local count_file="$UNIT_TESTMON_CACHE_DIR/.testmon-work/$1/rank-0/selected-count"
    local count
    [[ -f "$count_file" ]] || return 1
    count=$(< "$count_file")
    [[ "$count" =~ ^[0-9]+$ ]] || return 1
    echo "$count"
}

selected_test_failed() {
    local exit_code_file
    for exit_code_file in "$UNIT_TESTMON_CACHE_DIR/.testmon-work/$1"/rank-*/pytest-exit-code; do
        [[ -f "$exit_code_file" ]] || continue
        [[ "$(< "$exit_code_file")" == "1" ]] && return 0
    done
    return 1
}

run_enforced_tests() {
    local target phase=prod prod_count=0 experimental_count=0 rc=0
    target=$(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||')

    if ! install_testmon; then
        write_testmon_summary "full fallback: Testmon installation failed"
        run_full_fallback
        return
    fi

    set +e
    for i in $(seq "$UNIT_TEST_REPEAT"); do
        phase=prod
        run_testmon_phase enforce prod true \
            -vs "${IGNORE_ARGS[@]}" -m "not experimental and ${MARKER_ARG}" "$target"
        rc=$?
        if [[ "$rc" -ne 0 ]]; then
            break
        fi
        prod_count=$(selected_count prod)
        rc=$?
        if [[ "$rc" -ne 0 ]]; then
            break
        fi
        if [[ "$prod_count" -eq 0 ]]; then
            rm -f -- .coverage.unit_tests*
        fi

        phase=experimental
        run_testmon_phase enforce experimental false \
            -vs --experimental "${IGNORE_ARGS[@]}" -m "experimental and ${MARKER_ARG}" "$target"
        rc=$?
        if [[ "$rc" -ne 0 ]]; then
            break
        fi
        experimental_count=$(selected_count experimental)
        rc=$?
        if [[ "$rc" -ne 0 ]]; then
            break
        fi
    done
    set -e

    if [[ "$rc" -ne 0 ]]; then
        if selected_test_failed "$phase"; then
            write_testmon_summary "selective tests failed"
            return "$rc"
        fi
        write_testmon_summary "full fallback: Testmon execution failed"
        run_full_fallback
        return
    fi

    run_always_tests
    if compgen -G '.coverage.unit_tests*' > /dev/null; then
        coverage combine -q
    fi
    write_testmon_summary "selective tests passed" "$((prod_count + experimental_count))"
}

case "$UNIT_TESTMON_MODE" in
full)
    run_full_tests
    ;;
baseline)
    run_baseline_tests
    ;;
enforce)
    run_enforced_tests
    ;;
esac

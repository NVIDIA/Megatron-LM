#!/bin/bash
set -euxo pipefail

usage() {
    echo "Usage: $0 --tag {latest|legacy} --environment {lts|dev} --bucket BUCKET [--platform {h100|gb200}] [--unit-test-repeat N] [--unit-test-timeout N] --log-dir LOG_DIR"
    exit 1
}

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_PATH/../../"

UNIT_TEST_REPEAT=1
UNIT_TEST_TIMEOUT=10
LOG_DIR=$(pwd)/logs
PLATFORM=h100
UNIT_TESTMON_MODE=${UNIT_TESTMON_MODE:-full}
UNIT_TESTMON_CACHE_DIR=${UNIT_TESTMON_CACHE_DIR:-assets_dir/testmon}

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
if [[ "$UNIT_TESTMON_MODE" != "full" && "$UNIT_TESTMON_MODE" != "enforce" && "$UNIT_TESTMON_MODE" != "baseline" && "$UNIT_TESTMON_MODE" != "bootstrap" ]]; then
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

# Keep this path identical to exhaustive CI. The installed Testmon plugin is
# inactive unless one of the selective paths below passes --testmon.
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
            echo "- Selected files: \`$selected_count\`"
        fi
    } > "$UNIT_TESTMON_CACHE_DIR/summary.md"
}

run_testmon_phase() {
    local mode="$1"
    local phase="$2"
    shift 2
    local -a command=(uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}")
    command+=(
        tests/unit_tests/testmon_selector.py
        --mode "$mode"
        --cache-dir "$UNIT_TESTMON_CACHE_DIR"
        --phase "$phase"
        -- "$@"
    )
    "${command[@]}"
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

run_bootstrap_tests() {
    # Keep the existing exhaustive path for its normal coverage artifact, then
    # make a second one-time pass to record Testmon's per-test dependencies.
    run_full_tests
    run_baseline_tests
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

run_selected_phase() {
    local phase="$1"
    local selection_file="$UNIT_TESTMON_CACHE_DIR/.testmon-work/$phase/selected-tests"
    local -a selected_files=()
    local -a command=(uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}")

    while IFS= read -r path; do
        [[ -n "$path" ]] && selected_files+=("$path")
    done < "$selection_file"
    if [[ "${#selected_files[@]}" -eq 0 ]]; then
        echo "Testmon selected no $phase tests."
        return 0
    fi

    if [[ "$phase" == "prod" ]]; then
        command+=(
            -m coverage run
            --data-file=.coverage.unit_tests
            --source=megatron/core
            -m pytest
            -vs
            "${IGNORE_ARGS[@]}"
            -m "not experimental and ${MARKER_ARG}"
            "${selected_files[@]}"
        )
    else
        command+=(
            -m pytest
            -vs
            --experimental
            "${IGNORE_ARGS[@]}"
            -m "experimental and ${MARKER_ARG}"
            "${selected_files[@]}"
        )
    fi
    "${command[@]}"
}

run_enforced_tests() {
    local target prod_count experimental_count
    target=$(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||')
    rm -rf -- "$UNIT_TESTMON_CACHE_DIR/.testmon-work"

    if ! run_testmon_phase select prod \
        -vs "${IGNORE_ARGS[@]}" -m "not experimental and ${MARKER_ARG}" "$target" \
        || ! merge_rank_selections prod \
        || ! run_testmon_phase select experimental \
            -vs --experimental "${IGNORE_ARGS[@]}" -m "experimental and ${MARKER_ARG}" "$target" \
        || ! merge_rank_selections experimental; then
        write_testmon_summary "full fallback: Testmon selection failed"
        run_full_tests
        return
    fi

    prod_count=$(wc -l < "$UNIT_TESTMON_CACHE_DIR/.testmon-work/prod/selected-tests")
    experimental_count=$(wc -l < "$UNIT_TESTMON_CACHE_DIR/.testmon-work/experimental/selected-tests")
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
    write_testmon_summary "selective tests passed" "$((prod_count + experimental_count))"
}

case "$UNIT_TESTMON_MODE" in
full)
    run_full_tests
    ;;
baseline)
    run_baseline_tests
    ;;
bootstrap)
    run_bootstrap_tests
    ;;
enforce)
    run_enforced_tests
    ;;
esac

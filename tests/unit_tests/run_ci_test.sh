#!/bin/bash
set -euxo pipefail

usage() {
    echo "Usage: $0 --tag {latest|legacy} --environment {lts|dev} --bucket BUCKET [--platform {h100|gb200}] [--unit-test-repeat N] [--unit-test-timeout N] [--unit-testmon-mode {full|enforce|baseline}] [--unit-testmon-cache-dir DIR] [--unit-testmon-base-sha SHA] [--unit-testmon-config-hash HASH] --log-dir LOG_DIR"
    exit 1
}

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_PATH/../../"

UNIT_TEST_REPEAT=1
UNIT_TEST_TIMEOUT=10
LOG_DIR=$(pwd)/logs
PLATFORM=h100
UNIT_TESTMON_MODE=full
UNIT_TESTMON_CACHE_DIR=
UNIT_TESTMON_BASE_SHA=
UNIT_TESTMON_CONFIG_HASH=

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
    --unit-testmon-cache-dir)
        UNIT_TESTMON_CACHE_DIR="$2"
        shift 2
        ;;
    --unit-testmon-base-sha)
        UNIT_TESTMON_BASE_SHA="$2"
        shift 2
        ;;
    --unit-testmon-config-hash)
        UNIT_TESTMON_CONFIG_HASH="$2"
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
if [[ "$UNIT_TESTMON_MODE" != "full" && -z "$UNIT_TESTMON_CACHE_DIR" ]]; then
    echo "Error: --unit-testmon-cache-dir is required in $UNIT_TESTMON_MODE mode"
    usage
fi
if [[ "$UNIT_TESTMON_MODE" == "enforce" && ( -z "$UNIT_TESTMON_BASE_SHA" || -z "$UNIT_TESTMON_CONFIG_HASH" ) ]]; then
    echo "Incomplete Testmon identity; running the full bucket."
    UNIT_TESTMON_MODE=full
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
TESTMON_WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
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

run_testmon_phase() {
    local mode="$1"
    local phase="$2"
    shift 2
    uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
        tests/unit_tests/testmon_selector.py run \
        --mode "$mode" \
        --cache-dir "$UNIT_TESTMON_CACHE_DIR" \
        --phase "$phase" \
        -- "$@"
}

write_testmon_summary() {
    local result="$1"
    local duration="$2"
    local selected_count="${3:-0}"
    local cache_age="${4:-unknown}"
    local selection_ratio="${5:-unknown}"
    mkdir -p "$UNIT_TESTMON_CACHE_DIR"
    {
        echo "### Unit Testmon"
        echo
        echo "- Mode: \`$UNIT_TESTMON_MODE\`"
        echo "- Platform: \`$PLATFORM\`"
        echo "- Bucket: \`$BUCKET\`"
        echo "- Result: $result"
        echo "- Cache age: \`$cache_age\`"
        echo "- Selected files: \`$selected_count\`"
        echo "- Selection ratio: \`$selection_ratio\`"
        echo "- Selector duration: \`${duration}s\`"
    } > "$UNIT_TESTMON_CACHE_DIR/summary.md"
}

run_baseline_tests() {
    local target
    target=$(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||')
    install_testmon
    run_testmon_phase baseline prod \
        -vs "${IGNORE_ARGS[@]}" -m "not experimental and ${MARKER_ARG}" "$target"
    run_testmon_phase baseline experimental \
        -vs --experimental "${IGNORE_ARGS[@]}" -m "experimental and ${MARKER_ARG}" "$target"
    write_testmon_summary "baseline produced" 0 0
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

selective_command() {
    local rc=0
    set +e
    "$@"
    rc=$?
    set -e
    if [[ "$rc" -eq 5 ]]; then
        echo "No tests matched this selected phase; treating it as a pass."
        return 0
    fi
    return "$rc"
}

run_enforced_tests() {
    local started=$SECONDS
    local target head_sha selection_output selection_metrics cache_age selection_ratio rc
    local -a identity selected_files
    target=$(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||')
    head_sha=$(git rev-parse HEAD)
    identity=(
        --repo-root .
        --cache-dir "$UNIT_TESTMON_CACHE_DIR"
        --metadata "$UNIT_TESTMON_CACHE_DIR/metadata.json"
        --platform "$PLATFORM"
        --world-size "$TESTMON_WORLD_SIZE"
        --bucket "$BUCKET"
        --config-hash "$UNIT_TESTMON_CONFIG_HASH"
        --base-sha "$UNIT_TESTMON_BASE_SHA"
        --head-sha "$head_sha"
    )

    set +e
    uv run --no-sync python tests/unit_tests/testmon_selector.py select "${identity[@]}" --validate-only
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
        write_testmon_summary "full fallback: baseline or change validation failed" "$((SECONDS - started))"
        run_full_fallback
        return
    fi

    set +e
    install_testmon
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
        write_testmon_summary "full fallback: Testmon installation failed" "$((SECONDS - started))"
        run_full_fallback
        return
    fi

    set +e
    run_testmon_phase select prod \
        -vs "${IGNORE_ARGS[@]}" -m "not experimental and ${MARKER_ARG}" "$target"
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
        run_testmon_phase select experimental \
            -vs --experimental "${IGNORE_ARGS[@]}" -m "experimental and ${MARKER_ARG}" "$target"
        rc=$?
    fi
    set -e
    if [[ "$rc" -ne 0 ]]; then
        write_testmon_summary "full fallback: Testmon collection failed" "$((SECONDS - started))"
        run_full_fallback
        return
    fi

    set +e
    selection_output=$(uv run --no-sync python tests/unit_tests/testmon_selector.py select \
        "${identity[@]}" --output "$UNIT_TESTMON_CACHE_DIR/selected.json")
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
        write_testmon_summary "full fallback: unsafe selection" "$((SECONDS - started))"
        run_full_fallback
        return
    fi

    selected_files=()
    while IFS= read -r selected; do
        [[ -n "$selected" ]] && selected_files+=("$selected")
    done <<< "$selection_output"
    selection_metrics=$(uv run --no-sync python -c \
        'import datetime,json,sys; s=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); t=datetime.datetime.fromisoformat(m["producer_time"].replace("Z", "+00:00")); print("{:.1f}h\t{:.1%}".format((datetime.datetime.now(datetime.timezone.utc)-t).total_seconds()/3600, s["selection_ratio"]))' \
        "$UNIT_TESTMON_CACHE_DIR/selected.json" "$UNIT_TESTMON_CACHE_DIR/metadata.json" 2>/dev/null || true)
    IFS=$'\t' read -r cache_age selection_ratio <<< "$selection_metrics"
    write_testmon_summary "selected" "$((SECONDS - started))" "${#selected_files[@]}" \
        "${cache_age:-unknown}" "${selection_ratio:-unknown}"
    if [[ ${#selected_files[@]} -gt 0 ]]; then
        {
            echo
            echo "Selected test files:"
            echo
            for selected in "${selected_files[@]}"; do
                echo "- \`$selected\`"
            done
        } >> "$UNIT_TESTMON_CACHE_DIR/summary.md"
    fi
    if [[ ${#selected_files[@]} -eq 0 ]]; then
        echo "Testmon selected no files for this bucket."
        return
    fi

    for i in $(seq "$UNIT_TEST_REPEAT"); do
        selective_command uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
            -m coverage run --data-file=.coverage.unit_tests --source=megatron/core \
            -m pytest -p no:testmon -p no:pytest-testmon -vs "${IGNORE_ARGS[@]}" \
            -m "not experimental and ${MARKER_ARG}" "${selected_files[@]}"
        selective_command uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
            -m pytest -p no:testmon -p no:pytest-testmon -vs --experimental "${IGNORE_ARGS[@]}" \
            -m "experimental and ${MARKER_ARG}" "${selected_files[@]}"
    done
    if compgen -G '.coverage.unit_tests*' > /dev/null; then
        coverage combine -q
    fi
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

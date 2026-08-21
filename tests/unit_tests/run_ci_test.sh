#!/bin/bash
set -euxo pipefail

# Parse command line arguments
usage() {
    echo "Usage: $0 --tag {latest|legacy} --environment {lts|dev} --bucket BUCKET [--platform {h100|gb200}] [--unit-test-repeat N] [--unit-test-timeout N] [--unit-testmon-mode {full|enforce|baseline}] [--unit-testmon-cache-dir DIR] [--unit-testmon-selected-manifest FILE] --log-dir LOG_DIR"
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
UNIT_TESTMON_MODE=full
UNIT_TESTMON_CACHE_DIR=
UNIT_TESTMON_SELECTED_MANIFEST=

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
    --unit-testmon-mode)
        UNIT_TESTMON_MODE="$2"
        shift 2
        ;;
    --unit-testmon-cache-dir)
        UNIT_TESTMON_CACHE_DIR="$2"
        shift 2
        ;;
    --unit-testmon-selected-manifest)
        UNIT_TESTMON_SELECTED_MANIFEST="$2"
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

# Validate Testmon mode. The default preserves the pre-Testmon execution path.
if [[ "$UNIT_TESTMON_MODE" != "full" && "$UNIT_TESTMON_MODE" != "enforce" && "$UNIT_TESTMON_MODE" != "baseline" ]]; then
    echo "Error: UNIT_TESTMON_MODE must be one of 'full', 'enforce', or 'baseline'"
    usage
fi

if [[ "$UNIT_TESTMON_MODE" != "full" && -z "$UNIT_TESTMON_CACHE_DIR" ]]; then
    echo "Error: --unit-testmon-cache-dir is required in $UNIT_TESTMON_MODE mode"
    usage
fi

if [[ "$UNIT_TESTMON_MODE" == "enforce" && ( "$TAG" != "latest" || "$ENVIRONMENT" != "dev" ) ]]; then
    echo "Testmon enforcement is only valid for the latest dev suite; running the full suite."
    UNIT_TESTMON_MODE=full
fi

if [[ "$UNIT_TESTMON_MODE" == "baseline" && ( "$TAG" != "latest" || "$ENVIRONMENT" != "dev" ) ]]; then
    echo "Error: Testmon baselines must use the latest dev suite"
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
    for i in $(seq $UNIT_TEST_REPEAT); do
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

run_full_fallback() {
    local previous_pytest_addopts="${PYTEST_ADDOPTS-}"
    export PYTEST_ADDOPTS="${previous_pytest_addopts:+$previous_pytest_addopts }-p no:pytest-testmon"
    run_full_tests
    if [[ -n "$previous_pytest_addopts" ]]; then
        export PYTEST_ADDOPTS="$previous_pytest_addopts"
    else
        unset PYTEST_ADDOPTS
    fi
}

install_testmon_dependency() {
    # Keep Testmon out of the exhaustive/full environment. ``--inexact`` adds
    # the pinned group without removing the unit-test environment prepared by
    # the action, while ``--no-install-project`` avoids rebuilding Megatron.
    uv sync --locked --only-group testmon --inexact --no-install-project
}

run_testmon_phase() {
    local wrapper_mode="$1"
    local phase="$2"
    shift 2

    uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
        tests/unit_tests/testmon_selected_plugin.py run \
        --mode "$wrapper_mode" \
        --cache-dir "$UNIT_TESTMON_CACHE_DIR" \
        --phase "$phase" \
        -- "$@"
}

write_baseline_summary() {
    local summary="$UNIT_TESTMON_CACHE_DIR/summary.md"
    mkdir -p "$UNIT_TESTMON_CACHE_DIR"
    {
        echo "### Unit Testmon"
        echo
        echo "- Mode: baseline"
        echo "- Platform: \`$PLATFORM\`"
        echo "- Bucket: \`$BUCKET\`"
        echo "- World size: \`$TESTMON_WORLD_SIZE\`"
        echo "- Phases: \`prod\`, \`experimental\`"
    } > "$summary"
}

write_enforce_summary() {
    local duration="$1"
    local reason="$2"
    shift 2
    local eligible_count selection_percentage
    local -a selected_files=()
    if [[ -z "$reason" ]]; then
        eligible_count="$1"
        selection_percentage="$2"
        shift 2
        selected_files=("$@")
    fi
    local summary="$UNIT_TESTMON_CACHE_DIR/summary.md"
    mkdir -p "$UNIT_TESTMON_CACHE_DIR"
    {
        echo "### Unit Testmon"
        echo
        echo "- Mode: enforce"
        echo "- Platform: \`$PLATFORM\`"
        echo "- Bucket: \`$BUCKET\`"
        echo "- Selector duration: \`${duration}s\`"
        if [[ -n "$reason" ]]; then
            echo "- Fallback: $reason"
        else
            echo "- Selected files: \`${#selected_files[@]}/$eligible_count\`"
            echo "- Selection ratio: \`$selection_percentage\`"
            if [[ ${#selected_files[@]} -gt 0 ]]; then
                echo
                echo "Selected test files:"
                echo
                for test_file in "${selected_files[@]}"; do
                    echo "- \`$test_file\`"
                done
            fi
        fi
    } > "$summary"
}

run_baseline_tests() {
    local test_target
    test_target=$(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||')

    install_testmon_dependency

    # Each child rank removes only its own database before collecting a fresh
    # baseline. The shared manifest builder checkpoints and validates them
    # after the container exits.
    run_testmon_phase baseline prod \
        -vs \
        "${IGNORE_ARGS[@]}" \
        -m "not experimental and ${MARKER_ARG}" \
        "$test_target"

    if [[ "$TAG" == "latest" ]]; then
        run_testmon_phase baseline experimental \
            -vs \
            --experimental \
            "${IGNORE_ARGS[@]}" \
            -m "experimental and ${MARKER_ARG}" \
            "$test_target"
    fi

    uv run --no-sync python tests/unit_tests/testmon_selected_plugin.py verify-baseline \
        --cache-dir "$UNIT_TESTMON_CACHE_DIR" \
        --world-size "$TESTMON_WORLD_SIZE"

    write_baseline_summary
}

run_enforced_tests() {
    local test_target selection_start selection_duration validation_output validation_rc
    local selection_rc manifest_rc eligible_count selection_percentage
    local -a selected_files=()
    test_target=$(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||')
    selection_start=$SECONDS

    if [[ -z "$UNIT_TESTMON_SELECTED_MANIFEST" ]]; then
        UNIT_TESTMON_SELECTED_MANIFEST="$UNIT_TESTMON_CACHE_DIR/selected.json"
    fi

    # The selector's manifest validation covers schema/topology identities,
    # database presence, SHA-256 checksums, and SQLite integrity. Any doubt is
    # a local full-bucket fallback, never a partial test run.
    set +e
    local -a validation_args=(
        uv run --no-sync python tests/unit_tests/testmon/tooling.py validate-manifest
        --cache-dir "$UNIT_TESTMON_CACHE_DIR"
        --manifest "$UNIT_TESTMON_CACHE_DIR/manifest.json"
        --index-record "$UNIT_TESTMON_CACHE_DIR/expected-index-record.json"
        --platform "$PLATFORM"
        --world-size "$TESTMON_WORLD_SIZE"
        --bucket "$BUCKET"
    )
    if [[ -n "${UNIT_TESTMON_ENVIRONMENT_HASH:-}" ]]; then
        validation_args+=(--environment-hash "$UNIT_TESTMON_ENVIRONMENT_HASH")
    fi
    validation_output=$("${validation_args[@]}" 2>&1)
    validation_rc=$?
    set -e
    if [[ "$validation_rc" -ne 0 ]]; then
        selection_duration=$((SECONDS - selection_start))
        echo "Testmon baseline validation failed; running the full bucket: $validation_output"
        write_enforce_summary "$selection_duration" "baseline validation failed"
        run_full_fallback
        return
    fi

    set +e
    install_testmon_dependency
    selection_rc=$?
    set -e
    if [[ "$selection_rc" -ne 0 ]]; then
        selection_duration=$((SECONDS - selection_start))
        echo "Testmon dependency installation failed; running the full bucket."
        write_enforce_summary "$selection_duration" "Testmon dependency installation failed"
        run_full_fallback
        return
    fi

    set +e
    run_testmon_phase select prod \
        -vs \
        "${IGNORE_ARGS[@]}" \
        -m "not experimental and ${MARKER_ARG}" \
        "$test_target"
    selection_rc=$?
    if [[ "$selection_rc" -eq 0 && "$TAG" == "latest" ]]; then
        run_testmon_phase select experimental \
            -vs \
            --experimental \
            "${IGNORE_ARGS[@]}" \
            -m "experimental and ${MARKER_ARG}" \
            "$test_target"
        selection_rc=$?
    fi
    set -e
    if [[ "$selection_rc" -ne 0 ]]; then
        selection_duration=$((SECONDS - selection_start))
        echo "Testmon selection failed; running the full bucket."
        write_enforce_summary "$selection_duration" "rank selection failed"
        run_full_fallback
        return
    fi

    local -a union_args=(
        uv run --no-sync python tests/unit_tests/testmon/tooling.py union-selection
        --cache-dir "$UNIT_TESTMON_CACHE_DIR"
        --bucket "$BUCKET"
        --platform "$PLATFORM"
        --world-size "$TESTMON_WORLD_SIZE"
        --manifest "$UNIT_TESTMON_CACHE_DIR/manifest.json"
        --output "$UNIT_TESTMON_SELECTED_MANIFEST"
    )
    if [[ -f "$UNIT_TESTMON_CACHE_DIR/direct-tests.json" ]]; then
        union_args+=(--direct-tests-json "$UNIT_TESTMON_CACHE_DIR/direct-tests.json")
    fi

    set +e
    "${union_args[@]}"
    selection_rc=$?
    set -e
    if [[ "$selection_rc" -ne 0 ]]; then
        selection_duration=$((SECONDS - selection_start))
        echo "Testmon rank union failed; running the full bucket."
        write_enforce_summary "$selection_duration" "selection union or bucket validation failed"
        run_full_fallback
        return
    fi

    local manifest_lines
    local -a manifest_values=()
    manifest_lines=$(mktemp)
    set +e
    uv run --no-sync python tests/unit_tests/testmon_selected_plugin.py selected-files \
        --manifest "$UNIT_TESTMON_SELECTED_MANIFEST" \
        --repo-root . \
        --include-summary-metrics > "$manifest_lines"
    manifest_rc=$?
    set -e
    if [[ "$manifest_rc" -eq 0 ]]; then
        mapfile -t manifest_values < "$manifest_lines"
        if [[ ${#manifest_values[@]} -lt 2 ]]; then
            manifest_rc=1
        else
            eligible_count="${manifest_values[0]}"
            selection_percentage="${manifest_values[1]}"
            selected_files=("${manifest_values[@]:2}")
        fi
    fi
    rm -f "$manifest_lines"
    if [[ "$manifest_rc" -ne 0 ]]; then
        selection_duration=$((SECONDS - selection_start))
        echo "Testmon selected-file manifest is unsafe; running the full bucket."
        write_enforce_summary "$selection_duration" "selected-file manifest validation failed"
        run_full_fallback
        return
    fi

    selection_duration=$((SECONDS - selection_start))
    write_enforce_summary \
        "$selection_duration" "" "$eligible_count" "$selection_percentage" "${selected_files[@]}"

    if [[ ${#selected_files[@]} -eq 0 ]]; then
        echo "Testmon selected no tests for this bucket; treating it as a successful no-op."
        return
    fi

    # Testmon is deliberately blocked during execution. All ranks receive the
    # same sorted file list, production retains Coverage.py, and experimental
    # retains ordinary pytest.
    for i in $(seq $UNIT_TEST_REPEAT); do
        echo "Running selected prod test suite."
        uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
            -m coverage run \
            --data-file=.coverage.unit_tests \
            --source=megatron/core \
            -m pytest \
            -p no:pytest-testmon \
            -vs \
            "${IGNORE_ARGS[@]}" \
            -m "not experimental and ${MARKER_ARG}" \
            "${selected_files[@]}"

        if [[ "$TAG" == "latest" ]]; then
            echo "Running selected experimental test suite."
            uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
                -m pytest \
                -p no:pytest-testmon \
                -vs \
                --experimental \
                "${IGNORE_ARGS[@]}" \
                -m "experimental and ${MARKER_ARG}" \
                "${selected_files[@]}"
        fi
    done

    coverage combine -q
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

#!/bin/bash
set -euxo pipefail

# Parse command line arguments
usage() {
    echo "Usage: $0 --tag {latest|legacy} --environment {lts|dev} --bucket BUCKET [--platform {h100|gb200}] [--unit-test-repeat N] [--unit-test-timeout N] [--large] --log-dir LOG_DIR"
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
LARGE=false
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
    --large)
        LARGE=true
        shift
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
if [[ "$LARGE" == "true" && "$UNIT_TESTMON_MODE" != "full" ]]; then
    echo "Large unit tests run every case; running the full bucket."
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

# These buckets need fixed world sizes and isolated precision processes, but
# share the same Testmon baseline/selection lifecycle as every other bucket.
UNIT_PROFILES=()
UNIT_PYTEST_ARGS=()
UNIT_RESULTS_DIR=
case "$BUCKET" in
tests/unit_tests/inference/test_batch_invariant_fp8.py)
    GPUS_PER_NODE=1
    export CUBLASLT_WORKSPACE_SIZE=0
    case "$PLATFORM" in
        h100) UNIT_PROFILES=(delayed blockwise) ;;
        gb200) UNIT_PROFILES=(mxfp8) ;;
        *) echo "Unsupported FP8 platform: $PLATFORM" >&2; exit 1 ;;
    esac
    ;;
tests/unit_tests/inference/engines/test_disagg_coordinator_pairwise.py)
    GPUS_PER_NODE=4
    UNIT_PROFILES=(default)
    UNIT_PYTEST_ARGS=(--experimental)
    if ! uv run --no-sync python -c 'from nixl import _api'; then
        if [[ "$UNIT_TESTMON_MODE" == full ]]; then
            echo "NIXL is unavailable; skipping $BUCKET"
            exit 0
        fi
        echo "NIXL must be importable to record or select this Testmon bucket." >&2
        exit 1
    fi
    ;;
tests/unit_tests/inference/test_disagg_transfer_pairwise.py)
    GPUS_PER_NODE=6
    UNIT_PROFILES=(default)
    UNIT_PYTEST_ARGS=(--experimental)
    ;;
esac
if (( ${#UNIT_PROFILES[@]} )); then
    UNIT_RESULTS_DIR="$(cd "$LOG_DIR" && pwd)/unit-results"
    rm -rf -- "$UNIT_RESULTS_DIR"
    mkdir -p "$UNIT_RESULTS_DIR"
    UNIT_LAUNCH=0
    UNIT_RESULTS_MODE="$UNIT_TESTMON_MODE"
fi

MARKER=()
if [[ "$LARGE" == "true" ]]; then
    # Tests marked large need several GiB of device memory each. Only --large
    # runs them, and every other launch deselects them below.
    MARKER+=("large" "not experimental")
fi
if [[ "$PLATFORM" == "gb200" ]]; then
    MARKER+=("launch_on_gb200")
elif [[ "$LARGE" == "true" ]]; then
    # A large test that needs GB200 memory carries launch_on_gb200; here it
    # could only skip.
    MARKER+=("not launch_on_gb200")
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

if [[ "$LARGE" != "true" ]]; then
    MARKER+=("not large")
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

# Keep profile data in Testmon's native environments within each phase database.
# Appending the second profile preserves the first; CPU selection already reads
# all environments. Each torchrun still gets a fresh interpreter and rank logs.
run_profiled_command() {
    local phase="$1" mode="$2"
    shift 2
    local -a command=("$@") profile_args wrapper_args
    local profile selection output rank_dir rc first=true
    local launcher_length=$((6 + ${#DISTRIBUTED_ARGS[@]}))
    for profile in "${UNIT_PROFILES[@]}"; do
        profile_args=("${UNIT_PYTEST_ARGS[@]}")
        wrapper_args=()
        if [[ "$profile" != default ]]; then
            selection="$profile"
            [[ "$profile" != delayed ]] || selection='not mxfp8 and not blockwise'
            profile_args+=(-k "$selection")
            if [[ "$mode" == baseline || "$mode" == select ]]; then
                profile_args+=("--testmon-env=$profile")
            fi
        fi
        if [[ "$mode" == baseline && "$first" == false ]]; then
            wrapper_args=(--append)
        fi
        first=false
        if [[ "$mode" == select ]]; then
            "${command[@]}" "${profile_args[@]}" || return $?
            for output in "$UNIT_TESTMON_CACHE_DIR/.testmon-work/$phase"/rank-*/selected-tests; do
                mv -- "$output" "$output.$profile" || return
            done
        else
            UNIT_LAUNCH=$((UNIT_LAUNCH + 1))
            export UNIT_TEST_JUNIT_DIR="$UNIT_RESULTS_DIR/$profile/$phase-$UNIT_LAUNCH"
            mkdir -p "$UNIT_TEST_JUNIT_DIR"
            # Expand RANK inside the worker, including when it runs the Testmon wrapper.
            rc=0
            "${command[@]:0:launcher_length}" --no-python bash -c \
                'exec uv run --no-sync python "$@" --junitxml="$UNIT_TEST_JUNIT_DIR/rank-$RANK.xml"' bash \
                "${command[@]:launcher_length:1}" "${wrapper_args[@]}" \
                "${command[@]:launcher_length+1}" "${profile_args[@]}" || rc=$?
            [[ "$rc" -eq 0 || "$rc" -eq 5 ]] || return "$rc"
        fi
    done
    if [[ "$mode" == select ]]; then
        for rank_dir in "$UNIT_TESTMON_CACHE_DIR/.testmon-work/$phase"/rank-*; do
            cat "$rank_dir"/selected-tests.* | LC_ALL=C sort -u > "$rank_dir/selected-tests" || return
        done
    fi
}

validate_profile_results() {
    uv run --no-sync python - "$UNIT_RESULTS_DIR" "$GPUS_PER_NODE" "$UNIT_RESULTS_MODE" "${UNIT_PROFILES[@]}" <<'PY'
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

root, ranks, mode, *profiles = sys.argv[1:]
for profile in profiles:
    launches = list((Path(root) / profile).iterdir()) if (Path(root) / profile).exists() else []
    if mode in ("full", "baseline"):
        assert launches, f"{profile}: executed no tests"
    for launch in launches:
        for rank in range(int(ranks)):
            report = launch / f"rank-{rank}.xml"
            assert report.is_file(), f"{profile}: rank {rank} is missing its report in {launch.name}"
            cases = ET.parse(report).findall(".//testcase")
            if mode in ("full", "baseline") and launch.name.startswith("prod-"):
                assert cases, f"{profile}: rank {rank} executed no tests in {launch.name}"
            assert all(case.find(tag) is None for case in cases for tag in ("skipped", "failure", "error")), (
                f"{profile}: rank {rank} has skipped or failed tests in {launch.name}"
            )
PY
}

# Ranks of a large test file's cases; every other file's cases run on one GPU.
large_test_ranks() {
    case "$1" in
    tests/unit_tests/inference/test_variable_collectives.py) echo 2 ;; # cross-rank collectives
    *) echo 1 ;;
    esac
}

# --large: run each case marked large under the bucket in its own torchrun, so
# that one case's allocation or sticky CUDA error cannot reach the next.
run_large_tests() {
    local target file case_id rc i iteration
    local -a files cases failed=()
    # validate_profile_results reads these: each case is a profile of its own.
    local -a UNIT_PROFILES
    local GPUS_PER_NODE UNIT_RESULTS_MODE=full
    target=$(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||')
    # Search the whole bucket: IGNORE_ARGS skips the child buckets of
    # unit-tests.yaml, which never runs large tests.
    mapfile -t files < <(grep -rlE --include='test_*.py' 'mark\.large\b' "$target" | LC_ALL=C sort)
    if (( ${#files[@]} == 0 )); then
        # Nothing in this bucket needs the large lane; a marked file whose cases
        # all fail to collect still fails below.
        echo "No test file under $target is marked large; nothing to run."
        return 0
    fi
    rc=0
    uv run --no-sync python -m pytest --collect-only -q -p no:cacheprovider -m "$MARKER_ARG" \
        "${files[@]}" > "$LOG_DIR/collected.txt" || rc=$?
    # Exit 5 means nothing matched; the check below names the files.
    if (( rc != 0 && rc != 5 )); then
        cat "$LOG_DIR/collected.txt"
        return "$rc"
    fi
    for file in "${files[@]}"; do
        if ! grep -qF -- "$file::" "$LOG_DIR/collected.txt"; then
            echo "$file is marked large, but none of its cases matches -m '$MARKER_ARG'." >&2
            return 1
        fi
    done
    mapfile -t cases < <(grep -E '^tests/.+::' "$LOG_DIR/collected.txt")

    UNIT_RESULTS_DIR="$(cd "$LOG_DIR" && pwd)/unit-results"
    rm -rf -- "$UNIT_RESULTS_DIR"
    mkdir -p "$UNIT_RESULTS_DIR"
    echo "Running ${#cases[@]} large cases from ${#files[@]} files, one process each."
    for i in "${!cases[@]}"; do
        case_id=${cases[$i]}
        UNIT_PROFILES=("$i")
        GPUS_PER_NODE=$(large_test_ranks "${case_id%%::*}")
        echo "Large case $i on $GPUS_PER_NODE rank(s): $case_id"
        rc=0
        for iteration in $(seq "$UNIT_TEST_REPEAT"); do
            export UNIT_TEST_JUNIT_DIR="$UNIT_RESULTS_DIR/$i/prod-$iteration"
            mkdir -p "$UNIT_TEST_JUNIT_DIR"
            # RANK expands inside each worker. A hung case must not use up the job.
            # shellcheck disable=SC2016
            timeout --kill-after=60 900 uv run --no-sync python -m torch.distributed.run \
                --standalone --nnodes=1 --nproc-per-node="$GPUS_PER_NODE" --max-restarts=0 \
                --log-dir "$LOG_DIR/$i" --redirects=3 --tee=3 --no-python bash -c \
                'exec uv run --no-sync python -m pytest -vs "$@" --junitxml="$UNIT_TEST_JUNIT_DIR/rank-$RANK.xml"' \
                bash -m "$MARKER_ARG" "$case_id" || rc=$?
            (( rc == 0 )) || break
        done
        # The FP8 and disaggregation buckets' check: each rank's report holds a
        # test, and none was skipped or failed.
        if (( rc != 0 )); then
            failed+=("$case_id (exit $rc)")
        elif ! validate_profile_results; then
            failed+=("$case_id (a rank ran no test, or one skipped)")
        fi
    done

    echo "${#cases[@]} large cases ran; ${#failed[@]} failed."
    if (( ${#failed[@]} )); then
        printf 'FAILED %s\n' "${failed[@]}"
        return 1
    fi
}

# Keep this path identical to exhaustive CI. The installed Testmon plugin is
# inactive unless one of the selective paths below passes --testmon.
run_full_tests() {
    UNIT_RESULTS_MODE=full
    local phase target
    local -a command
    target=$(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||')
    for i in $(seq $UNIT_TEST_REPEAT); do
        for phase in prod experimental; do
            [[ "$phase" != experimental || "$TAG" == latest ]] || continue
            command=(uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}")
            if [[ "$phase" == prod ]]; then
                command+=(-m coverage run --data-file=.coverage.unit_tests --source=megatron/core
                    -m pytest -vs "${IGNORE_ARGS[@]}" -m "not experimental and ${MARKER_ARG}" "$target")
            else
                command+=(-m pytest -vs --experimental "${IGNORE_ARGS[@]}"
                    -m "experimental and ${MARKER_ARG}" "$target")
            fi
            if (( ${#UNIT_PROFILES[@]} )); then
                run_profiled_command "$phase" full "${command[@]}"
            else
                run_test_cmd "${command[@]}"
            fi
        done
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
    local -a command=(uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}")
    command+=(
        tests/unit_tests/testmon_selector.py
        --mode "$mode"
        --cache-dir "$UNIT_TESTMON_CACHE_DIR"
        --phase "$phase"
        -- "$@"
    )
    if [[ -n "${UNIT_RESULTS_DIR:-}" ]]; then
        run_profiled_command "$phase" "$mode" "${command[@]}"
    else
        "${command[@]}"
    fi
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
    local -a command=(uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}")

    while IFS= read -r nodeid; do
        [[ -n "$nodeid" ]] && selected_tests+=("$nodeid")
    done < "$selection_file"
    if [[ "${#selected_tests[@]}" -eq 0 ]]; then
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
            "${selected_tests[@]}"
        )
    else
        command+=(
            -m pytest
            -vs
            --experimental
            "${IGNORE_ARGS[@]}"
            -m "experimental and ${MARKER_ARG}"
            "${selected_tests[@]}"
        )
    fi
    # Mandatory files are passed whole, so a phase's marker filter can
    # legitimately deselect everything in them (pytest exit 5).
    local rc=0
    if [[ -n "${UNIT_RESULTS_DIR:-}" ]]; then
        run_profiled_command "$phase" selected "${command[@]}" || rc=$?
    else
        "${command[@]}" || rc=$?
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

if [[ "$LARGE" == "true" ]]; then
    run_large_tests
    exit
fi

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

if (( ${#UNIT_PROFILES[@]} )); then
    validate_profile_results
fi

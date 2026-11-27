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

if [[ "$TAG" == "latest" ]]; then
    TEST_PATH="/opt/megatron-lm"
else
    TEST_PATH="/opt/megatron-lm-legacy/"
fi

cd $TEST_PATH

# Profiles are launches a bucket needs apart from its main launch at GPUS_PER_NODE ranks: a
# fresh process per FP8 recipe, or a fixed world size per file. They run the tests in
# ISOLATED_DIR, which the main launch leaves out, in the same job and the same Testmon
# baseline/selection lifecycle.
UNIT_PROFILES=()
UNIT_RESULTS_DIR=
UNIT_TARGETS=()
MAIN_ARGS=()
ISOLATED_DIR=tests/unit_tests/inference/isolated

# profile_spec PROFILE: set PROFILE_WORLD, PROFILE_FILE (a file or directory), PROFILE_SELECT
# (-k), PROFILE_ARGS and PROFILE_ENV for one profile of the isolated inference tests.
# PROFILE_PACKED profiles fill the runner with concurrent shards in a full run; PROFILE_SKIPS_OK
# profiles may report skipped tests.
profile_spec() {
    PROFILE_SELECT=
    PROFILE_ARGS=()
    PROFILE_ENV=()
    PROFILE_PACKED=false
    PROFILE_SKIPS_OK=false
    case "$1" in
    delayed | blockwise | mxfp8)
        PROFILE_WORLD=1
        PROFILE_FILE=$ISOLATED_DIR/test_batch_invariant_fp8.py
        PROFILE_SELECT=$1
        [[ "$1" != delayed ]] || PROFILE_SELECT='not mxfp8 and not blockwise'
        PROFILE_ENV=(CUBLASLT_WORKSPACE_SIZE=0)
        ;;
    pairs)
        # Two-DP-replica coordinator routing, one-prefill/one-decode disaggregation pairs and
        # the two-worker NIXL handoff, which skips without NIXL.
        PROFILE_WORLD=2
        PROFILE_FILE=$ISOLATED_DIR/two_rank
        PROFILE_PACKED=true
        PROFILE_SKIPS_OK=true
        ;;
    transfer)
        PROFILE_WORLD=6
        PROFILE_FILE=$ISOLATED_DIR/test_disagg_transfer_pairwise.py
        PROFILE_ARGS=(--experimental)
        ;;
    coordinator)
        PROFILE_WORLD=4
        PROFILE_FILE=$ISOLATED_DIR/test_disagg_coordinator_pairwise.py
        PROFILE_ARGS=(--experimental)
        ;;
    *)
        echo "Unknown unit-test profile: $1" >&2
        return 1
        ;;
    esac
}

# set_unit_profiles: for the inference buckets, set UNIT_PROFILES to the profiles this platform
# runs, and keep ISOLATED_DIR out of the main launch (MAIN_ARGS) or, for the isolated bucket,
# launch at the widest profile's world size (GPUS_PER_NODE).
set_unit_profiles() {
    local profile widest=1
    local -a profiles=()
    [[ "$BUCKET" == "tests/unit_tests/inference/**/*.py" \
        || "$BUCKET" == "$ISOLATED_DIR/**/*.py" ]] || return 0
    if [[ ! -d "$ISOLATED_DIR" ]]; then
        # The legacy suite (an older release's tests, run by this script) predates ISOLATED_DIR;
        # its main launch runs all of its inference tests, as that release did.
        echo "This suite has no $ISOLATED_DIR; $BUCKET runs without profiles."
        return 0
    fi
    # The inference job also runs the inference tests that need their own launches: a fresh
    # one-rank process per FP8 recipe, six ranks for the KV transfer pairs, two for the
    # routed-coordinator and disaggregation pairs, and four for the disaggregation coordinator.
    # The isolated bucket runs only these. UNIT_TEST_PROFILES runs a subset (e.g. "delayed
    # blockwise").
    case "$PLATFORM" in
        h100) UNIT_PROFILES=(delayed blockwise transfer pairs coordinator) ;;
        gb200) UNIT_PROFILES=(mxfp8) ;;
        *) echo "Unsupported platform for $BUCKET: $PLATFORM" >&2; exit 1 ;;
    esac
    if [[ -n "${UNIT_TEST_PROFILES:-}" ]]; then
        read -r -a UNIT_PROFILES <<< "$UNIT_TEST_PROFILES"
    fi
    if [[ " ${UNIT_PROFILES[*]} " == *" coordinator "* ]] \
        && ! uv run --no-sync python -c 'from nixl import _api'; then
        # The coordinator tests need NIXL; without it, run the other profiles.
        echo "NIXL is unavailable; skipping the coordinator profile."
        for profile in "${UNIT_PROFILES[@]}"; do
            [[ "$profile" == coordinator ]] || profiles+=("$profile")
        done
        UNIT_PROFILES=("${profiles[@]}")
    fi
    # Testmon merges one selection per rank of the widest launch.
    for profile in "${UNIT_PROFILES[@]}"; do
        profile_spec "$profile"
        (( PROFILE_WORLD <= widest )) || widest=$PROFILE_WORLD
    done
    if [[ "$BUCKET" == "$ISOLATED_DIR/**/*.py" ]]; then
        if (( ${#UNIT_PROFILES[@]} == 0 )); then
            echo "No profile of $BUCKET can run here."
            exit 0
        fi
        GPUS_PER_NODE=$widest
    else
        MAIN_ARGS=(--ignore="$ISOLATED_DIR")
        if (( widest > ${GPUS_PER_NODE:-8} )); then
            echo "A profile of $BUCKET needs $widest ranks, more than its main launch." >&2
            exit 1
        fi
    fi
}

set_unit_profiles
if (( ${#UNIT_PROFILES[@]} )); then
    UNIT_RESULTS_DIR="$(cd "$LOG_DIR" && pwd)/unit-results"
    rm -rf -- "$UNIT_RESULTS_DIR"
    mkdir -p "$UNIT_RESULTS_DIR"
    UNIT_LAUNCH=0
    UNIT_RESULTS_MODE="$UNIT_TESTMON_MODE"
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

# Packed shards. The runner has NODE_GPUS GPUs, and every rank runs every selected test. A
# bucket that needs fewer ranks than the runner has leaves the other GPUs idle, and a test that
# needs one rank repeats the same work on every rank. A full run, and the run of the tests
# Testmon selected, therefore pack independent shards onto the runner instead:
#   - a bucket whose GPUS_PER_NODE divides NODE_GPUS runs NODE_GPUS / GPUS_PER_NODE
#     concurrent shards, each with its own GPUS_PER_NODE GPUs and torchrun;
#   - in a bucket that uses every GPU, tests marked `single_rank` first run as NODE_GPUS
#     concurrent one-GPU shards, and the full-width launch that follows excludes them.
# Each shard runs every UNIT_TEST_SHARD-th selected test (tests/unit_tests/conftest.py), so
# the shards together run each test once, at the world size it needs. Testmon's baseline and
# selection keep one launch at GPUS_PER_NODE ranks. UNIT_TEST_PACK=0 turns packing off, for the
# main launch and the profiles (run_profiled_command).
UNIT_TEST_PACK=${UNIT_TEST_PACK:-1}
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=, read -r -a NODE_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t NODE_DEVICES < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
fi
NODE_GPUS=${#NODE_DEVICES[@]}

# run_sharded WORLD SHARDS PYTEST_ARGS...: SHARDS concurrent torchruns of WORLD ranks each.
run_sharded() {
    local world="$1" shards="$2"
    shift 2
    local -a pids=()
    local shard devices rc=0 shard_rc
    for ((shard = 0; shard < shards; shard++)); do
        devices=$(IFS=,; echo "${NODE_DEVICES[*]:shard*world:world}")
        # --standalone binds each shard's rendezvous store to a free port the OS picks; fixed
        # ports near MASTER_PORT can fall in the ephemeral range and collide with live sockets.
        CUDA_VISIBLE_DEVICES="$devices" UNIT_TEST_SHARD="$shard/$shards" \
            uv run --no-sync python -m torch.distributed.run \
            --standalone --nproc_per_node "$world" \
            --log-dir "$LOG_DIR/shard-$world-$shard" --tee "0:3" --redirects "3" \
            -m coverage run --data-file=.coverage.unit_tests --source=megatron/core \
            -m pytest "$@" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        shard_rc=0
        wait "$pid" || shard_rc=$?
        # A shard can receive no tests when a pass selects fewer tests than shards.
        [[ "$shard_rc" -eq 0 || "$shard_rc" -eq 5 ]] || rc="$shard_rc"
    done
    return "$rc"
}

# uses_mark MARK TARGET IGNORE_ARGS...: a file under TARGET that none of IGNORE_ARGS (--ignore
# of a file or directory) removes applies pytest.mark.MARK.
uses_mark() {
    local mark="$1" target="$2" file ignore
    shift 2
    while IFS= read -r file; do
        for ignore; do
            ignore="${ignore#--ignore=}"
            [[ "$file" != "$ignore" && "$file" != "$ignore"/* ]] || continue 2
        done
        return 0
    done < <(grep -rlsFw --include='*.py' "pytest.mark.$mark" "$target")
    return 1
}

# main_uses_mark MARK TARGET: a file the main launch collects under TARGET applies the mark.
main_uses_mark() {
    uses_mark "$1" "$2" "${IGNORE_ARGS[@]}" "${MAIN_ARGS[@]}"
}

# launch_profile PHASE MODE PROFILE DEVICES APPEND LAUNCH SHARD LAUNCH_ARGS...: one profile's
# launch. LAUNCH_ARGS is what follows the torchrun arguments (a module or script and its options),
# without test targets. The targets are the profile's file or directory, or the node ids of
# UNIT_TARGETS in it. DEVICES, when set, confines a concurrent launch to those GPUs; SHARD
# ("<index>/<count>"), when set, runs that share of the profile's tests.
launch_profile() {
    local phase="$1" mode="$2" profile="$3" devices="$4" append="$5" launch="$6" shard="$7"
    shift 7
    local -a targets=() args=() dist=() wrapper=() env_args=()
    local target output junit rc=0
    profile_spec "$profile"
    if [[ "$mode" == selected ]]; then
        for target in "${UNIT_TARGETS[@]}"; do
            if [[ "$target" == "$PROFILE_FILE" || "$target" == "$PROFILE_FILE"/* \
                || "$target" == "$PROFILE_FILE::"* ]]; then
                targets+=("$target")
            fi
        done
        (( ${#targets[@]} )) || return 0
    else
        targets=("$PROFILE_FILE")
    fi
    args=("${PROFILE_ARGS[@]}")
    [[ -z "$PROFILE_SELECT" ]] || args+=(-k "$PROFILE_SELECT")
    if [[ "$mode" == baseline || "$mode" == select ]]; then
        args+=("--testmon-env=$profile")
    fi
    [[ "$append" != true ]] || wrapper=(--append)
    if [[ -n "$devices" ]]; then
        env_args=(CUDA_VISIBLE_DEVICES="$devices")
        dist=(--standalone --nproc_per_node "$PROFILE_WORLD")
    else
        dist=(--nproc_per_node "$PROFILE_WORLD" --nnodes 1 --master_addr "$MASTER_ADDR"
            --master_port "$MASTER_PORT" --node_rank 0)
    fi
    dist+=(--log-dir "$LOG_DIR/profile-$profile-$launch${shard:+-${shard%/*}}" --tee "0:3"
        --redirects "3")
    env_args+=("${PROFILE_ENV[@]}")
    [[ -z "$shard" ]] || env_args+=(UNIT_TEST_SHARD="$shard")
    if [[ "$mode" == select ]]; then
        env "${env_args[@]}" uv run --no-sync python -m torch.distributed.run "${dist[@]}" \
            "$1" "${wrapper[@]}" "${@:2}" "${args[@]}" "${targets[@]}" || return
        for output in "$UNIT_TESTMON_CACHE_DIR/.testmon-work/$phase"/rank-*/selected-tests; do
            mv -- "$output" "$output.$profile" || return
        done
        return 0
    fi
    junit="$UNIT_RESULTS_DIR/$profile/$phase-$launch${shard:+-${shard%/*}}"
    mkdir -p "$junit"
    # Expand RANK inside the worker, including when it runs the Testmon wrapper.
    env "${env_args[@]}" UNIT_TEST_JUNIT_DIR="$junit" \
        uv run --no-sync python -m torch.distributed.run "${dist[@]}" --no-python bash -c \
        'exec uv run --no-sync python "$@" --junitxml="$UNIT_TEST_JUNIT_DIR/rank-$RANK.xml"' bash \
        "$1" "${wrapper[@]}" "${@:2}" "${args[@]}" "${targets[@]}" || rc=$?
    [[ "$rc" -eq 0 || "$rc" -eq 5 ]] || return "$rc"
}

# run_profiled_command PHASE MODE APPEND LAUNCH_ARGS...: run every profile (see launch_profile).
# Profile data stays in Testmon's native environments within each phase database: Testmon's
# baseline and selection run the profiles one at a time, each baseline appending to the phase
# database once it holds one (APPEND=true when the main launch recorded it), and CPU selection
# already reads all environments. A full run, and the run of the selected tests, start the
# profiles that fit on the runner together unless packing is off (UNIT_TEST_PACK=0); each
# torchrun still gets a fresh interpreter and its own rank logs.
run_profiled_command() {
    local phase="$1" mode="$2" append="$3" pack=false
    shift 3
    local profile rank_dir pid devices used=0 rc=0
    local -a pids=()
    if [[ ( "$mode" == full || "$mode" == selected ) && "$UNIT_TEST_PACK" == 1 \
        && "$NUM_NODES" -eq 1 ]]; then
        pack=true
    fi
    for profile in "${UNIT_PROFILES[@]}"; do
        profile_spec "$profile"
        UNIT_LAUNCH=$((UNIT_LAUNCH + 1))
        if [[ "$pack" == true && "$PROFILE_PACKED" == true ]] \
            && (( NODE_GPUS > PROFILE_WORLD && NODE_GPUS % PROFILE_WORLD == 0 )); then
            # A packed profile takes the whole runner: finish the wave in flight, then start
            # one shard per PROFILE_WORLD GPUs.
            for pid in "${pids[@]}"; do
                wait "$pid" || rc=$?
            done
            [[ "$rc" -eq 0 ]] || return "$rc"
            pids=()
            used=0
            local shards=$((NODE_GPUS / PROFILE_WORLD)) shard
            for ((shard = 0; shard < shards; shard++)); do
                devices=$(IFS=,; echo "${NODE_DEVICES[*]:shard*PROFILE_WORLD:PROFILE_WORLD}")
                launch_profile "$phase" "$mode" "$profile" "$devices" false "$UNIT_LAUNCH" \
                    "$shard/$shards" "$@" &
                pids+=("$!")
            done
            for pid in "${pids[@]}"; do
                wait "$pid" || rc=$?
            done
            [[ "$rc" -eq 0 ]] || return "$rc"
            pids=()
        elif [[ "$pack" == true ]] && (( PROFILE_WORLD <= NODE_GPUS )); then
            if (( used + PROFILE_WORLD > NODE_GPUS )); then
                for pid in "${pids[@]}"; do
                    wait "$pid" || rc=$?
                done
                [[ "$rc" -eq 0 ]] || return "$rc"
                pids=()
                used=0
            fi
            devices=$(IFS=,; echo "${NODE_DEVICES[*]:used:PROFILE_WORLD}")
            used=$((used + PROFILE_WORLD))
            launch_profile "$phase" "$mode" "$profile" "$devices" false "$UNIT_LAUNCH" "" "$@" &
            pids+=("$!")
        else
            launch_profile "$phase" "$mode" "$profile" "" "$append" "$UNIT_LAUNCH" "" "$@" || return
            [[ "$mode" != baseline ]] || append=true
        fi
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || rc=$?
    done
    [[ "$rc" -eq 0 ]] || return "$rc"
    if [[ "$mode" == select ]]; then
        for rank_dir in "$UNIT_TESTMON_CACHE_DIR/.testmon-work/$phase"/rank-*; do
            cat "$rank_dir"/selected-tests.* | LC_ALL=C sort -u > "$rank_dir/selected-tests" || return
        done
    fi
}

validate_profile_results() {
    local profile
    local -a worlds=()
    for profile in "${UNIT_PROFILES[@]}"; do
        profile_spec "$profile"
        worlds+=("$profile:$PROFILE_WORLD:$PROFILE_SKIPS_OK")
    done
    uv run --no-sync python - "$UNIT_RESULTS_DIR" "$UNIT_RESULTS_MODE" "${worlds[@]}" <<'PY'
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

root, mode, *worlds = sys.argv[1:]
for profile, ranks, skips_ok in (item.split(":") for item in worlds):
    launches = list((Path(root) / profile).iterdir()) if (Path(root) / profile).exists() else []
    if mode in ("full", "baseline"):
        assert launches, f"{profile}: executed no tests"
    banned = ("failure", "error") if skips_ok == "true" else ("skipped", "failure", "error")
    for launch in launches:
        for rank in range(int(ranks)):
            report = launch / f"rank-{rank}.xml"
            assert report.is_file(), f"{profile}: rank {rank} is missing its report in {launch.name}"
            cases = ET.parse(report).findall(".//testcase")
            if mode in ("full", "baseline") and launch.name.startswith("prod-"):
                assert cases, f"{profile}: rank {rank} executed no tests in {launch.name}"
            assert all(case.find(tag) is None for case in cases for tag in banned), (
                f"{profile}: rank {rank} has {' or '.join(banned)} tests in {launch.name}"
            )
PY
}

# Keep this path identical to exhaustive CI. The installed Testmon plugin is
# inactive unless one of the selective paths below passes --testmon.
# pack_plan TARGET: set PACK_SHARDS (concurrent shards of GPUS_PER_NODE GPUs) and
# PACK_SINGLE_RANK (whether the main launch's single_rank tests under TARGET run first as
# one-GPU shards).
pack_plan() {
    PACK_SHARDS=1
    PACK_SINGLE_RANK=false
    [[ "$UNIT_TEST_PACK" == 1 && "$NUM_NODES" -eq 1 ]] || return 0
    if (( GPUS_PER_NODE < NODE_GPUS && NODE_GPUS % GPUS_PER_NODE == 0 )); then
        PACK_SHARDS=$((NODE_GPUS / GPUS_PER_NODE))
    elif (( GPUS_PER_NODE > 1 && GPUS_PER_NODE == NODE_GPUS )) \
        && main_uses_mark single_rank "$1"; then
        PACK_SINGLE_RANK=true
    fi
}

# run_prod_tests ARGS...: the main launch's prod phase over ARGS (the bucket directory, or the
# node ids Testmon selected, and MAIN_ARGS), packed as pack_plan decided.
run_prod_tests() {
    local -a args=(-vs "${IGNORE_ARGS[@]}")
    local rc=0
    if (( PACK_SHARDS > 1 )); then
        echo "Running prod tests as $PACK_SHARDS packed shards of $GPUS_PER_NODE GPUs."
        run_sharded "$GPUS_PER_NODE" "$PACK_SHARDS" "${args[@]}" \
            -m "not experimental and ${MARKER_ARG}" "$@"
        return
    fi
    if [[ "$PACK_SINGLE_RANK" == true ]]; then
        echo "Running single_rank prod tests as $NODE_GPUS one-GPU shards."
        run_sharded 1 "$NODE_GPUS" "${args[@]}" \
            -m "not experimental and single_rank and ${MARKER_ARG}" "$@" || return
        args+=(-m "not experimental and not single_rank and ${MARKER_ARG}")
    else
        args+=(-m "not experimental and ${MARKER_ARG}")
    fi
    run_test_cmd uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
        -m coverage run --data-file=.coverage.unit_tests --source=megatron/core \
        -m pytest "${args[@]}" "$@" || rc=$?
    # The single_rank lane can take every test, which leaves the full-width launch nothing.
    [[ "$rc" -eq 0 || ( "$rc" -eq 5 && "$PACK_SINGLE_RANK" == true ) ]] || return "$rc"
}

# run_bucket PHASE MODE LAUNCH_ARGS...: one phase of the bucket over UNIT_TARGETS (the bucket
# directory, or the node ids Testmon selected). MODE is full, baseline, select or selected, and
# LAUNCH_ARGS is what follows the torchrun arguments, without test targets. The main launch
# takes the targets outside ISOLATED_DIR at GPUS_PER_NODE ranks (a prod phase of a full or
# selected run packed as pack_plan decided); then each profile takes its own.
run_bucket() {
    local phase="$1" mode="$2" target output rc=0 append=false profiles=true
    shift 2
    local -a main=()
    for target in "${UNIT_TARGETS[@]}"; do
        [[ "$target" == "$ISOLATED_DIR" || "$target" == "$ISOLATED_DIR"/* ]] || main+=("$target")
    done
    (( ${#UNIT_PROFILES[@]} )) || profiles=false
    if [[ "$mode" == full && "$phase" == experimental ]]; then
        # Without pytest.mark.experimental a launch selects nothing in this phase; skip it,
        # which otherwise starts every rank only to deselect every test.
        if (( ${#main[@]} )) && ! main_uses_mark experimental "${main[0]}"; then
            echo "No experimental tests in ${main[0]}; skipping the experimental launch."
            main=()
        fi
        if [[ "$profiles" == true ]] \
            && ! uses_mark experimental "$ISOLATED_DIR" "${IGNORE_ARGS[@]}"; then
            echo "No experimental tests in $ISOLATED_DIR; skipping the experimental profiles."
            profiles=false
        fi
    fi
    if (( ${#main[@]} )); then
        if [[ "$phase" == prod && ( "$mode" == full || "$mode" == selected ) ]]; then
            run_prod_tests "${MAIN_ARGS[@]}" "${main[@]}" || rc=$?
        else
            run_test_cmd uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
                "$@" "${MAIN_ARGS[@]}" "${main[@]}" || rc=$?
        fi
        # The phase's markers can deselect every selected test (pytest exit 5).
        [[ "$rc" -eq 0 || ( "$rc" -eq 5 && "$mode" == selected ) ]] || return "$rc"
        if [[ "$profiles" == true && "$mode" == select ]]; then
            # Keep the main launch's selections apart; run_profiled_command merges them all.
            for output in "$UNIT_TESTMON_CACHE_DIR/.testmon-work/$phase"/rank-*/selected-tests; do
                mv -- "$output" "$output.main" || return
            done
        fi
        [[ "$mode" != baseline ]] || append=true
    fi
    [[ "$profiles" == true ]] || return "$rc"
    run_profiled_command "$phase" "$mode" "$append" "$@"
}

run_full_tests() {
    UNIT_RESULTS_MODE=full
    local phase
    local -a launch
    UNIT_TARGETS=("$(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||')")
    for i in $(seq $UNIT_TEST_REPEAT); do
        for phase in prod experimental; do
            [[ "$phase" != experimental || "$TAG" == latest ]] || continue
            if [[ "$phase" == prod ]]; then
                launch=(-m coverage run --data-file=.coverage.unit_tests --source=megatron/core
                    -m pytest -vs "${IGNORE_ARGS[@]}" -m "not experimental and ${MARKER_ARG}")
            else
                launch=(-m pytest -vs --experimental "${IGNORE_ARGS[@]}"
                    -m "experimental and ${MARKER_ARG}")
            fi
            run_bucket "$phase" full "${launch[@]}"
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
    local -a launch=(
        tests/unit_tests/testmon_selector.py
        --mode "$mode"
        --cache-dir "$UNIT_TESTMON_CACHE_DIR"
        --phase "$phase"
        --
    )
    # The last pytest argument is the bucket directory.
    UNIT_TARGETS=("${@: -1}")
    run_bucket "$phase" "$mode" "${launch[@]}" "${@:1:$#-1}"
}

run_baseline_tests() {
    local cache_root="${UNIT_TESTMON_CACHE_DIR:?}"
    local target
    target=$(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||')
    rm -rf -- "$cache_root/prod" "$cache_root/experimental" "$cache_root/.testmon-work"
    rm -f -- "$cache_root/summary.md"
    mkdir -p "$cache_root"

    # Testmon pauses an outer `coverage run` tracer, so collect production
    # coverage through pytest-cov, which Testmon forwards recorded lines to.
    run_testmon_phase baseline prod \
        --cov=megatron/core --cov-config=pyproject.toml --cov-report= \
        -vs "${IGNORE_ARGS[@]}" -m "not experimental and ${MARKER_ARG}" "$target"
    run_testmon_phase baseline experimental \
        -vs --experimental "${IGNORE_ARGS[@]}" -m "experimental and ${MARKER_ARG}" "$target"
    # Platforms may collect no production tests for a bucket.
    if compgen -G '.coverage.unit_tests*' > /dev/null; then
        coverage combine -q
    fi
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
    local -a selected_tests=() launch

    while IFS= read -r nodeid; do
        [[ -n "$nodeid" ]] && selected_tests+=("$nodeid")
    done < "$selection_file"
    if [[ "${#selected_tests[@]}" -eq 0 ]]; then
        echo "Testmon selected no $phase tests."
        return 0
    fi

    if [[ "$phase" == "prod" ]]; then
        launch=(
            -m coverage run
            --data-file=.coverage.unit_tests
            --source=megatron/core
            -m pytest
            -vs
            "${IGNORE_ARGS[@]}"
            -m "not experimental and ${MARKER_ARG}"
        )
    else
        launch=(
            -m pytest
            -vs
            --experimental
            "${IGNORE_ARGS[@]}"
            -m "experimental and ${MARKER_ARG}"
        )
    fi
    # Mandatory files are passed whole, so a phase's marker filter can
    # legitimately deselect everything in them (pytest exit 5).
    local rc=0
    UNIT_TARGETS=("${selected_tests[@]}")
    run_bucket "$phase" selected "${launch[@]}" || rc=$?
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

pack_plan "$(echo "$BUCKET" | sed 's|/\*\*/\*\.py$||')"

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

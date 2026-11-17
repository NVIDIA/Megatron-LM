#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

unit_bucket_setup() {
    NUM_NODES=1
    GPUS_PER_NODE=1
    export CUBLASLT_WORKSPACE_SIZE=0
    case "$PLATFORM" in
        h100) UNIT_PROFILES=(delayed blockwise) ;;
        gb200) UNIT_PROFILES=(mxfp8) ;;
        *) echo "Unsupported FP8 platform: $PLATFORM" >&2; return 1 ;;
    esac
}

unit_bucket_run() {
    local mode="$1" phase="$2"
    shift 2
    local profile selection output rank_dir append=false
    local -a profile_args
    for profile in "${UNIT_PROFILES[@]}"; do
        selection="$profile"
        # Include admission checks once, alongside the delayed-scaling profile.
        [[ "$profile" != delayed ]] || selection='not mxfp8 and not blockwise'
        profile_args=("$@" -k "$selection")
        if [[ "$mode" == baseline || "$mode" == select ]]; then
            profile_args+=("--testmon-env=$profile")
        fi
        run_unit_worker "$mode" "$phase" "$profile" "$append" "${profile_args[@]}" || return $?
        [[ "$mode" != baseline ]] || append=true
        if [[ "$mode" == select ]]; then
            for output in "$UNIT_TESTMON_CACHE_DIR/.testmon-work/$phase"/rank-*/selected-tests; do
                mv -- "$output" "$output.$profile" || return
            done
        fi
    done
    if [[ "$mode" == select ]]; then
        for rank_dir in "$UNIT_TESTMON_CACHE_DIR/.testmon-work/$phase"/rank-*; do
            cat "$rank_dir"/selected-tests.* | LC_ALL=C sort -u > "$rank_dir/selected-tests" || return
        done
    fi
}

# Direct callers retain the positional interface and execute this checkout.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    UNIT_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    export UNIT_TEST_ROOT
    exec bash "$UNIT_TEST_ROOT/tests/unit_tests/run_ci_test.sh" \
        --tag latest --platform "$1" --environment "$2" --unit-test-repeat "$3" \
        --bucket tests/unit_tests/inference/test_batch_invariant_fp8.py --log-dir "$4"
fi

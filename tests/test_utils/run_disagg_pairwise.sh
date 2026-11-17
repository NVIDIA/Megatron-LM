#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

unit_bucket_setup() {
    NUM_NODES=1
    case "$BUCKET" in
        tests/unit_tests/inference/engines/test_disagg_coordinator_pairwise.py)
            GPUS_PER_NODE=4
            if ! uv run --no-sync python -c 'from nixl import _api'; then
                if [[ "$UNIT_TESTMON_MODE" == full ]]; then
                    echo "NIXL is unavailable; skipping $BUCKET"
                    exit 0
                fi
                echo "NIXL must be importable to record or select this Testmon bucket." >&2
                return 1
            fi
            ;;
        tests/unit_tests/inference/test_disagg_transfer_pairwise.py)
            GPUS_PER_NODE=6
            ;;
        *) echo "Unsupported disaggregation bucket: $BUCKET" >&2; return 1 ;;
    esac
}

unit_bucket_run() {
    local mode="$1" phase="$2"
    shift 2
    # Enable runtime features independently of the phase's pytest marker filter.
    run_unit_worker "$mode" "$phase" default false "$@" --experimental
}

# Direct callers retain the positional interface and execute this checkout.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    case "$3" in
        tests/unit_tests/inference/engines/test_disagg_coordinator_pairwise.py|tests/unit_tests/inference/test_disagg_transfer_pairwise.py) ;;
        *) echo "Unsupported disaggregation bucket: $3" >&2; exit 1 ;;
    esac
    UNIT_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    export UNIT_TEST_ROOT
    exec bash "$UNIT_TEST_ROOT/tests/unit_tests/run_ci_test.sh" \
        --tag latest --environment "$1" --unit-test-repeat "$2" --bucket "$3" --log-dir "$4"
fi

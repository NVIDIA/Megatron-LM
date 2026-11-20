#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail

# Retain the standalone entry point; CI recipes invoke the shared runner directly.
UNIT_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export UNIT_TEST_ROOT
case "$1" in
    h100) UNIT_TEST_PROFILES="delayed blockwise" ;;
    *) UNIT_TEST_PROFILES=mxfp8 ;;
esac
export UNIT_TEST_PROFILES
exec bash "$UNIT_TEST_ROOT/tests/unit_tests/run_ci_test.sh" \
    --tag latest --platform "$1" --environment "$2" --unit-test-repeat "$3" \
    --bucket "tests/unit_tests/inference/isolated/**/*.py" --log-dir "$4"

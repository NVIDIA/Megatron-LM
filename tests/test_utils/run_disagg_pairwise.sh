#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail

# Retain the standalone entry point; CI recipes invoke the shared runner directly.
UNIT_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export UNIT_TEST_ROOT
exec bash "$UNIT_TEST_ROOT/tests/unit_tests/run_ci_test.sh" \
    --tag latest --environment "$1" --unit-test-repeat "$2" --bucket "$3" --log-dir "$4"

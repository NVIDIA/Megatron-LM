#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

set -euo pipefail

TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "MLite validation summary"
  echo "overall=FAIL exit_code=1 reason=mandatory_dependency_missing dependency=python3"
  exit 1
fi

exec python3 "${TEST_ROOT}/_test_harness/runner.py" "$@"

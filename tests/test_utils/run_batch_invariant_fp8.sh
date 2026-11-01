#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail

PLATFORM=$1
ENVIRONMENT=$2
REPEAT=$3
LOG_DIR=$4
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

case "$PLATFORM" in
    h100) PROFILES=(delayed blockwise) ;;
    gb200) PROFILES=(mxfp8) ;;
    *) echo "Unsupported FP8 test platform: $PLATFORM" >&2; exit 1 ;;
esac
case "$ENVIRONMENT" in
    dev) MARKER="not experimental and not flaky_in_dev" ;;
    lts) MARKER="not experimental and not flaky" ;;
    *) echo "Unsupported test environment: $ENVIRONMENT" >&2; exit 1 ;;
esac
[[ "$REPEAT" =~ ^[1-9][0-9]*$ ]]
mkdir -p "$LOG_DIR"
LOG_DIR=$(cd "$LOG_DIR" && pwd)
export CUBLASLT_WORKSPACE_SIZE=0

for ((iteration = 1; iteration <= REPEAT; iteration++)); do
    for profile in "${PROFILES[@]}"; do
        selection=$profile
        if [[ "$profile" == delayed ]]; then
            # Include the admission checks added by descendant PRs exactly once.
            selection="not mxfp8 and not blockwise"
        fi
        result="$LOG_DIR/$iteration-$profile.xml"
        # TE and cuBLAS cache process state: each recipe gets a fresh interpreter.
        uv run --no-sync python -m torch.distributed.run \
            --standalone --nnodes=1 --nproc-per-node=1 --max-restarts=0 \
            --log-dir "$LOG_DIR/$iteration-$profile" --redirects=3 --tee=3 \
            -m coverage run --parallel-mode --source=megatron/core \
            -m pytest -vs tests/unit_tests/inference/test_batch_invariant_fp8.py \
            -k "$selection" -m "$MARKER" --junitxml="$result"
        # The suite normalizes empty collections to success; this dedicated job must run.
        uv run --no-sync python - "$result" <<'PY'
import sys
import xml.etree.ElementTree as ET

cases = ET.parse(sys.argv[1]).findall(".//testcase")
assert cases, "FP8 job collected no tests"
assert all(case.find(tag) is None for case in cases for tag in ("skipped", "failure", "error"))
PY
    done
done
uv run --no-sync python -m coverage combine -q

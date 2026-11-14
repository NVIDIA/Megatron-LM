#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail

ENVIRONMENT=$1
REPEAT=$2
LOG_DIR=$3
TEST_FILE=tests/unit_tests/determinism/kernels/test_dsa_batch_invariant.py
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

[[ "$ENVIRONMENT" == dev ]]
[[ "$REPEAT" =~ ^[1-9][0-9]*$ ]]
mkdir -p "$LOG_DIR"
LOG_DIR=$(cd "$LOG_DIR" && pwd)

# cuBLASLt caches its workspace at first use; TE attention selection is process-wide.
export CUBLASLT_WORKSPACE_SIZE=0
export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=0
export NVTE_FLASH_ATTN_V2=0 NVTE_FLASH_ATTN_V3=0 NVTE_FLASH_ATTN_V4=1

for ((iteration = 1; iteration <= REPEAT; iteration++)); do
    result="$LOG_DIR/$iteration-dsa-bi.xml"
    uv run --no-sync python -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node=1 --max-restarts=0 \
        --log-dir "$LOG_DIR/$iteration-dsa-bi" --redirects=3 --tee=3 \
        -m coverage run --parallel-mode --source=megatron/core \
        -m pytest -vs "$TEST_FILE" \
        -m "launch_on_gb200 and not experimental and not flaky_in_dev" \
        --junitxml="$result"
    uv run --no-sync python - "$result" <<'PY'
import sys
import xml.etree.ElementTree as ET

cases = ET.parse(sys.argv[1]).findall(".//testcase")
assert cases, "DSA BI job collected no tests"
assert all(case.find(tag) is None for case in cases for tag in ("skipped", "failure", "error"))
PY
done
uv run --no-sync python -m coverage combine -q

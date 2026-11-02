#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail

ENVIRONMENT=$1
REPEAT=$2
BUCKET=$3
LOG_DIR=$4
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

case "$BUCKET" in
    tests/unit_tests/inference/engines/test_disagg_coordinator_pairwise.py) RANKS=4 NIXL=1 ;;
    tests/unit_tests/inference/test_disagg_transfer_pairwise.py) RANKS=6 NIXL=0 ;;
    *) echo "Unsupported disaggregation bucket: $BUCKET" >&2; exit 1 ;;
esac
case "$ENVIRONMENT" in
    dev) MARKER="not flaky_in_dev" ;;
    lts) MARKER="not flaky" ;;
    *) echo "Unsupported test environment: $ENVIRONMENT" >&2; exit 1 ;;
esac
[[ "$REPEAT" =~ ^[1-9][0-9]*$ ]]
# The coordinator handoff is NIXL-only; an image without NIXL has nothing to run.
if ((NIXL)) && ! uv run --no-sync python -c "from nixl import _api" 2>/dev/null; then
    echo "NIXL is not installed; skipping $BUCKET"
    exit 0
fi
mkdir -p "$LOG_DIR"
LOG_DIR=$(cd "$LOG_DIR" && pwd)

for ((iteration = 1; iteration <= REPEAT; iteration++)); do
    result="$LOG_DIR/$iteration-results"
    mkdir -p "$result"
    # RANK is expanded inside each torchrun worker.
    # shellcheck disable=SC2016
    uv run --no-sync python -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node="$RANKS" --max-restarts=0 \
        --log-dir "$LOG_DIR/$iteration" --redirects=3 --tee=3 \
        --no-python bash -c '
            exec uv run --no-sync python -m coverage run --parallel-mode --source=megatron/core \
                -m pytest -vs --experimental "$2" -m "$3" --junitxml="$1/rank-$RANK.xml"
        ' bash "$result" "$BUCKET" "$MARKER"
    # Empty collection and wrong-world-size skips must not look like coverage.
    uv run --no-sync python - "$result" "$RANKS" <<'PY'
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

for rank in range(int(sys.argv[2])):
    cases = ET.parse(Path(sys.argv[1]) / f"rank-{rank}.xml").findall(".//testcase")
    assert cases, f"Disaggregation rank {rank} collected no tests"
    assert all(case.find(tag) is None for case in cases for tag in ("skipped", "failure", "error"))
PY
done
uv run --no-sync python -m coverage combine -q

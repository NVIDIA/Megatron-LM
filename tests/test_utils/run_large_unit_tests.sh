#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail

# Runs every unit test marked `large`, each case in its own process, so one case's
# allocation or sticky CUDA error cannot affect the next. Unit-test buckets skip these
# cases; the nightly recipe tests/test_utils/recipes/h100/unit-tests-large.yaml runs them.
ENVIRONMENT=$1
LOG_DIR=$2
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

case "$ENVIRONMENT" in
    dev) MARKER="large and not experimental and not flaky_in_dev" ;;
    lts) MARKER="large and not experimental and not flaky" ;;
    *) echo "Unsupported test environment: $ENVIRONMENT" >&2; exit 1 ;;
esac
mkdir -p "$LOG_DIR"
LOG_DIR=$(cd "$LOG_DIR" && pwd)

# Ranks for a case's file; every other case runs on one GPU.
ranks() {
    case "$1" in
        tests/unit_tests/inference/test_variable_collectives.py) echo 2 ;;  # cross-rank collectives
        *) echo 1 ;;
    esac
}

mapfile -t FILES < <(grep -rlE --include='test_*.py' 'mark\.large\b' tests/unit_tests | sort)
if ((${#FILES[@]} == 0)); then
    echo "No unit test is marked large."
    exit 0
fi
# Exit 5: every marked case is deselected (e.g. flaky_in_dev).
rc=0
uv run --no-sync python -m pytest --collect-only -q -p no:cacheprovider -m "$MARKER" \
    "${FILES[@]}" > "$LOG_DIR/collected.txt" || rc=$?
if ((rc != 0 && rc != 5)); then
    cat "$LOG_DIR/collected.txt"
    exit "$rc"
fi
mapfile -t CASES < <(grep -E '^tests/.+::' "$LOG_DIR/collected.txt")
echo "Running ${#CASES[@]} large cases from ${#FILES[@]} files, one process each."

FAILED=()
for i in "${!CASES[@]}"; do
    status=0
    # A hung case must not use up the lane's time limit.
    timeout --kill-after=60 900 uv run --no-sync python -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node="$(ranks "${CASES[$i]%%::*}")" \
        --max-restarts=0 --log-dir "$LOG_DIR/$i" --redirects=3 --tee=3 \
        -m pytest -vs "${CASES[$i]}" -m "$MARKER" || status=$?
    ((status == 0)) || FAILED+=("${CASES[$i]} (exit $status)")
done

echo "${#CASES[@]} large cases ran; ${#FAILED[@]} failed."
if ((${#FAILED[@]})); then
    printf 'FAILED %s\n' "${FAILED[@]}"
    exit 1
fi

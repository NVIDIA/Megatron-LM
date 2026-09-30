#!/usr/bin/env bash

set -euo pipefail

checkpoint_root=$(mktemp -d "${TMPDIR:-/tmp}/engram-reshard.XXXXXX")
test_file=tests/unit_tests/models/engram/test_checkpoint_reshard.py
trap 'rm -rf -- "$checkpoint_root"' EXIT

run_phase() {
    local source_ep=$1
    local phase=$2
    local world_size=$3
    ENGRAM_CHECKPOINT_DIR="$checkpoint_root" \
        ENGRAM_CHECKPOINT_PHASE="${phase}_ep${source_ep}" \
        python -m torch.distributed.run --standalone --nproc-per-node="$world_size" \
        -m pytest -q "$test_file"
}

run_phase 1 save 1
run_phase 1 load 4
run_phase 2 save 2
run_phase 2 load 4

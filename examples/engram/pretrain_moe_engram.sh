#!/bin/bash
# MoE + distributed Engram on the same GPT-2 FineWeb recipe as the baseline.
# 0.73A0.05B-0.8: 208 experts, memory ID 1 before Hybrid attention slot 2.
# Tables use row_a2a + RowSparseAdam.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
read -r -a HASH_TABLE_MIN_SIZES <<< "${ENGRAM_HASH_TABLE_MIN_SIZES:-205700 205700}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Engram-0.73A0.05B-0.8}"
exec bash "${SCRIPT_DIR}/run_moe_gpt2.sh" \
    --num-experts "${NUM_EXPERTS:-208}" \
    --engram-hash-table-min-sizes "${HASH_TABLE_MIN_SIZES[@]}" \
    --engram-max-ngram-size 3 \
    --engram-embedding-dim-per-ngram "${ENGRAM_EMBEDDING_DIM_PER_NGRAM:-320}" \
    --engram-num-hash-heads-per-ngram 8 \
    --engram-layer-ids 1 \
    --engram-target-layer-indices 2 \
    --engram-table-backend row_a2a \
    --engram-seed 0 \
    --engram-kernel-size 4 \
    "$@"

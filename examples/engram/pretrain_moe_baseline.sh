#!/bin/bash
# Dense-optimizer MoE baseline: 0.73A0.05B configuration, GPT-2 tokenizer, FineWeb-10B.
# No Engram tables. Compare against pretrain_moe_engram.sh.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "${SCRIPT_DIR}/run_moe_gpt2.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${ROOT}/experimental/lite:${PYTHONPATH:-}"

pytest experimental/lite/tests/unit "$@"

if [[ "${MLITE_RUN_SMOKE:-0}" == "1" ]]; then
  NPROC="${MLITE_SMOKE_NPROC:-${WORLD_SIZE:-1}}"
  if (( NPROC < 1 || NPROC > 8 )); then
    echo "MLITE smoke tests require 1 <= MLITE_SMOKE_NPROC <= 8, got ${NPROC}" >&2
    exit 2
  fi
  torchrun --standalone --nproc_per_node="${NPROC}" \
    -m pytest --mlite-smoke experimental/lite/tests/smoke "$@"
fi

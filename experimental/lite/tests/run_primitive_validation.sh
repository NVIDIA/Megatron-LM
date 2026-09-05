#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LITE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${LITE_ROOT}/../.." && pwd)"

export PYTHONPATH="${LITE_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
python -m pytest -q "${LITE_ROOT}/tests/unit" "$@"

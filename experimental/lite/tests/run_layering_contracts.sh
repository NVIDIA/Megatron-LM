#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/experimental/lite${PYTHONPATH:+:$PYTHONPATH}"
pytest -q experimental/lite/tests/unit/runtime/test_layering_contracts.py "$@"

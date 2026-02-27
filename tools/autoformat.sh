#!/bin/bash
set -euox pipefail

GIT_VERSION=$(git version | awk '{print $3}')
GIT_MAJOR=$(echo $GIT_VERSION | awk -F. '{print $1}')
GIT_MINOR=$(echo $GIT_VERSION | awk -F. '{print $2}')

if [[ $GIT_MAJOR -eq 2 && $GIT_MINOR -lt 31 ]]; then
    echo "Git version must be at least 2.31.0. Found $GIT_VERSION"
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
CHECK_ONLY=${CHECK_ONLY:-false}
SKIP_DOCS=${SKIP_DOCS:-false}

BASE_REF=${BASE_REF:-main}
git remote add autoformatter-remote "https://github.com/NVIDIA/Megatron-LM.git" || true
git fetch autoformatter-remote ${BASE_REF}
CHANGED_FILES=$(git diff --name-only --diff-filter=d --merge-base autoformatter-remote/${BASE_REF} megatron/core tests/ | grep '\.py$' || true)
ADDITIONAL_RUFF_FORMAT_ARGS=""
ADDITIONAL_RUFF_CHECK_ARGS=""

if [[ $CHECK_ONLY == true ]]; then
    ADDITIONAL_RUFF_FORMAT_ARGS="--check --diff"
    ADDITIONAL_RUFF_CHECK_ARGS="--no-fix"
else
    ADDITIONAL_RUFF_CHECK_ARGS="--fix"
fi

if [[ $SKIP_DOCS == true ]]; then
    ADDITIONAL_RUFF_CHECK_ARGS="$ADDITIONAL_RUFF_CHECK_ARGS --extend-per-file-ignores='**:D101,D103'"
fi

if [[ -n "$CHANGED_FILES" ]]; then
    ruff format $ADDITIONAL_RUFF_FORMAT_ARGS $CHANGED_FILES
    ruff check $ADDITIONAL_RUFF_CHECK_ARGS $CHANGED_FILES
    mypy --explicit-package-bases --follow-imports=skip $CHANGED_FILES || true
else
    echo Changeset is empty, all good.
fi

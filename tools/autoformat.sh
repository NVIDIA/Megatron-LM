#!/bin/bash
set -euox pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CHECK_ONLY=${CHECK_ONLY:-false}
CHANGED_FILES=$(git diff --name-only --diff-filter=d --merge-base origin/main megatron/core tests/ | grep '\.py$' || true)
ADDITIONAL_ARGS=""
ADDITIONAL_BLACK_ARGS=""

if [[ $CHECK_ONLY == true ]]; then
    ADDITIONAL_ARGS="--check"
    ADDITIONAL_BLACK_ARGS="--diff"
fi

if [[ -n "$CHANGED_FILES" ]]; then
    black --skip-magic-trailing-comma $ADDITIONAL_ARGS $ADDITIONAL_BLACK_ARGS --verbose $CHANGED_FILES
    isort $ADDITIONAL_ARGS $CHANGED_FILES
else
    echo Changeset is empty, all good.
fi

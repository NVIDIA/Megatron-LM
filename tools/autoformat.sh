#!/bin/bash
set -euox pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CHECK_ONLY=${CHECK_ONLY:-false}
CHANGED_FILES=$(git diff --name-only --diff-filter=d --merge-base origin/main megatron/core | grep '\.py$' || true)
ADDITIONAL_ARGS=""

if [[ $CHECK_ONLY == true ]]; then
    ADDITIONAL_ARGS="--check "
fi

# for now we just format core
if [[ -n "$CHANGED_FILES" ]]; then
    black $ADDITIONAL_ARGS --verbose --diff $CHANGED_FILES
    isort $ADDITIONAL_ARGS $CHANGED_FILES
else
    echo Changeset is empty, all good.
fi

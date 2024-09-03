#!/bin/bash
set -euox pipefail

GIT_VERSION=$(git version | awk '{print $3}')
GIT_MAJOR=$(echo $GIT_VERSION | awk -F. '{print $1}')
GIT_MINOR=$(echo $GIT_VERSION | awk -F. '{print $2}')

if [[ $GIT_MAJOR -eq 2 && $GIT_MINOR -lt 31 ]]; then
    echo "Git version must be at least 2.31.0. Found $GIT_VERSION"
    exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CHECK_ONLY=${CHECK_ONLY:-false}
SKIP_DOCS=${SKIP_DOCS:-false}

CHANGED_FILES=$(git diff --name-only --diff-filter=d --merge-base origin/main megatron/core tests/ | grep '\.py$' || true)
ADDITIONAL_ARGS=""
ADDITIONAL_BLACK_ARGS=""
ADDITIONAL_PYLINT_ARGS=""


if [[ $CHECK_ONLY == true ]]; then
    ADDITIONAL_ARGS="--check"
    ADDITIONAL_BLACK_ARGS="--diff"
fi

if [[ $SKIP_DOCS == true ]]; then
    ADDITIONAL_PYLINT_ARGS="--disable=C0115,C0116"
fi

if [[ -n "$CHANGED_FILES" ]]; then
    black --skip-magic-trailing-comma $ADDITIONAL_ARGS $ADDITIONAL_BLACK_ARGS --verbose $CHANGED_FILES
    isort $ADDITIONAL_ARGS $CHANGED_FILES
    pylint $ADDITIONAL_PYLINT_ARGS $CHANGED_FILES
else
    echo Changeset is empty, all good.
fi

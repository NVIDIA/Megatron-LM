#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

CHANGED_FILES=$(git diff --name-only origin/main | grep '^megatron/core' || true)

# for now we just format core


if [[ -n "$CHANGED_FILES" ]]; then
    black $CHANGED_FILES
    isort $CHANGED_FILES
fi

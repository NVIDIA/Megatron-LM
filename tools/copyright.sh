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

BASE_REF=${BASE_REF:-main}
git remote set-url origin "https://${GITLAB_ENDPOINT}/$CI_PROJECT_NAMESPACE/megatron-lm.git"
git fetch origin ${BASE_REF}
CHANGED_FILES=$(git diff --name-only --diff-filter=d --merge-base origin/${BASE_REF} megatron/core tests/ | grep '\.py$' || true)

if [[ -n "$CHANGED_FILES" ]]; then
   CMD="python ${SCRIPT_DIR}/check_copyright.py"

   # Add the files
   CMD="$CMD --from-year 2019 $CHANGED_FILES"

   # Run the check
   eval $CMD
fi
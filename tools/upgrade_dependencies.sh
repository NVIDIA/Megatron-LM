#!/bin/bash

set -eoxu pipefail

if [ -z "${GITLAB_ENDPOINT:-}" ]; then
    echo "GITLAB_ENDPOINT is not set. Please set the GITLAB_ENDPOINT environment variable to the gitlab endpoint of the Megatron-LM repository."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd $SCRIPT_DIR/..

docker run \
    --rm \
    -v $(pwd):/workdir/ \
    -w /workdir/ \
    $GITLAB_ENDPOINT/adlr/megatron-lm/mcore_ci_dev:main \
    uv lock --upgrade

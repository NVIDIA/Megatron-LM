#!/bin/bash

set -eoxu pipefail

if [ -z "${GITLAB_ENDPOINT:-}" ]; then
    echo "GITLAB_ENDPOINT is not set. Please set the GITLAB_ENDPOINT environment variable to the gitlab endpoint of the Megatron-LM repository."
    exit 1
fi

UPGRADE=false

for arg in "$@"; do
    case $arg in
        --upgrade) UPGRADE=true ;;
        --help)
            echo "Usage: $(basename $0) [--upgrade]"
            echo ""
            echo "Options:"
            echo "  --upgrade  Upgrade dependencies in addition to updating the lockfile"
            echo "  --help     Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  GITLAB_ENDPOINT  Hostname of the internal GitLab registry (no scheme, e.g. gitlab.example.com)"
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd $SCRIPT_DIR/..

UV_CMD="uv lock"
if [ "$UPGRADE" = true ]; then
    UV_CMD="uv lock --upgrade"
fi

docker run \
    --rm \
    -v $(pwd):/workdir/ \
    -w /workdir/ \
    $GITLAB_ENDPOINT/adlr/megatron-lm/mcore_ci_dev:main \
    $UV_CMD

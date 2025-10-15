#!/bin/bash
set -euxo pipefail

# Default values
MCORE_REPO="https://github.com/nvidia/megatron-lm.git"
MCORE_MR_COMMIT="main"
MCORE_BACKWARDS_COMMIT=""

# Parse command line arguments
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Clone and setup megatron-lm repositories for testing.

Options:
    --repo URL              Git repository URL (default: $MCORE_REPO)
    --backwards-commit COMMIT Commit hash or reference for the backwards compatibility test
    --help                  Show this help message

Example:
    $0 --repo $MCORE_REPO \\
       --backwards-commit core_r0.12.0
EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --repo)
        MCORE_REPO="$2"
        shift 2
        ;;
    --backwards-commit)
        MCORE_BACKWARDS_COMMIT="$2"
        shift 2
        ;;
    --help)
        usage
        ;;
    *)
        echo "Unknown option: $1"
        usage
        ;;
    esac
done

# Validate required arguments
if [[ -z "${MCORE_BACKWARDS_COMMIT:-}" ]]; then
    echo "Error: --backwards-commit is required"
    usage
fi

# Checkout backwards-ref
rm -rf megatron-lm-legacy
mkdir megatron-lm-legacy
pushd megatron-lm-legacy
git init
git remote add origin $MCORE_REPO
git fetch origin $MCORE_BACKWARDS_COMMIT
git checkout $MCORE_BACKWARDS_COMMIT
git rev-parse HEAD
rm -rf megatron
cp -a ../megatron-lm/megatron ./
popd

# Copy unit test script
cp megatron-lm/tests/unit_tests/run_ci_test.sh megatron-lm-legacy/tests/unit_tests/run_ci_test.sh

#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

# Initialize variables
REPO_URL="https://github.com/Dao-AILab/causal-conv1d.git"
REPO_REF="v1.2.2.post1"
OUTPUT_WHEEL_DIR="$(pwd)/wheels"
SCRIPT_DIR="$(dirname $(realpath $0))"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --repo-url)
        REPO_URL="$2"
        shift 2
        ;;
    --repo-ref)
        REPO_REF="$2"
        shift 2
        ;;
    --output-wheel-dir)
        OUTPUT_WHEEL_DIR="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: $0 --repo-url URL --repo-ref REF --output-wheel-dir DIR"
        exit 1
        ;;
    esac
done

# Check if required arguments are provided
if [ -z "$REPO_URL" ] || [ -z "$REPO_REF" ] || [ -z "$OUTPUT_WHEEL_DIR" ]; then
    echo "Error: --repo-url, --repo-ref, and --output-wheel-dir are required"
    echo "Usage: $0 --repo-url URL --repo-ref REF --output-wheel-dir DIR"
    exit 1
fi

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
echo "Working in temporary directory: ${TEMP_DIR}"
python3 -m venv "${TEMP_DIR}/venv" --system-site-packages
source "${TEMP_DIR}/venv/bin/activate"

# Ensure cleanup on script exit
trap 'rm -rf "${TEMP_DIR}"' EXIT

# Change to temporary directory
cd "${TEMP_DIR}"

# Initialize git repository
git init

# Perform git fetch with depth 1
git fetch "${REPO_URL}" "${REPO_REF}" --depth 1

git checkout FETCH_HEAD

# Fetch submodules
git submodule update --init --recursive

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_WHEEL_DIR}"

# Build the wheel using python -m build
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
pip3 wheel --no-cache-dir --no-deps -w "${OUTPUT_WHEEL_DIR}" .

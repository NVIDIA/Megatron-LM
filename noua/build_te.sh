#!/bin/bash
# Build TransformerEngine from source inside the container
# Usage: srun --container-image=... --container-mounts=... bash noua/build_te.sh

set -euo pipefail

TE_SRC=/fsx/nouamane/projects/TransformerEngine-260325
BUILD_LOG="${TE_SRC}/build_te.log"

echo "Building TE from: ${TE_SRC}"
echo "Build log: ${BUILD_LOG}"

cd "${TE_SRC}"

# Build in-place so .so files stay in the worktree
NVTE_FRAMEWORK=pytorch pip install -e . --no-build-isolation 2>&1 | tee "${BUILD_LOG}"

echo ""
echo "=== Built .so files ==="
find "${TE_SRC}" -name "*.so" -newer "${BUILD_LOG}" -o -name "*.so" -newer setup.py | head -20
echo ""
echo "Done. Check ${BUILD_LOG} for details."

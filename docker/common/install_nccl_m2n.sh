# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

#!/bin/bash

set -euxo pipefail

NCCL_EXTENSIONS_COMMIT="6a81bce7c9fe5874e8c852224fa47c077033da72"
PYTHON="${UV_PROJECT_ENVIRONMENT:-/opt/venv}/bin/python"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --NCCL_EXTENSIONS_COMMIT=*)
            NCCL_EXTENSIONS_COMMIT="${1#*=}"
            ;;
        --PYTHON=*)
            PYTHON="${1#*=}"
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [[ ! "${NCCL_EXTENSIONS_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "NCCL_EXTENSIONS_COMMIT must be a full 40-character Git SHA" >&2
    exit 1
fi

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
test -x "${CUDA_HOME}/bin/nvcc"
test -x "${PYTHON}"

WORK_DIR=$(mktemp -d)
trap 'rm -rf "${WORK_DIR}"' EXIT

NCCL_EXTENSIONS_DIR="${WORK_DIR}/nccl-extensions"
git init "${NCCL_EXTENSIONS_DIR}"
git -C "${NCCL_EXTENSIONS_DIR}" remote add origin https://github.com/NVIDIA/nccl-extensions.git
git -C "${NCCL_EXTENSIONS_DIR}" fetch --depth 1 origin "${NCCL_EXTENSIONS_COMMIT}"
git -C "${NCCL_EXTENSIONS_DIR}" checkout --detach FETCH_HEAD
test "$(git -C "${NCCL_EXTENSIONS_DIR}" rev-parse HEAD)" = "${NCCL_EXTENSIONS_COMMIT}"

# The Debian NCCL packages split headers and libraries across /usr/include and
# the multiarch library directory. M2N expects a single NCCL_HOME prefix.
if [[ -z "${NCCL_HOME:-}" ]]; then
    NCCL_HOME="${WORK_DIR}/nccl"
    ARCH_LIB=$(dpkg-architecture -qDEB_HOST_MULTIARCH)
    install -d "${NCCL_HOME}/include" "${NCCL_HOME}/lib"
    ln -s /usr/include/nccl.h "${NCCL_HOME}/include/nccl.h"
    ln -s /usr/include/nccl_device.h "${NCCL_HOME}/include/nccl_device.h"
    ln -s /usr/include/nccl_device "${NCCL_HOME}/include/nccl_device"
    ln -s "/usr/lib/${ARCH_LIB}/libnccl.so" "${NCCL_HOME}/lib/libnccl.so"
fi

test -f "${NCCL_HOME}/include/nccl.h"
test -f "${NCCL_HOME}/include/nccl_device.h"
test -d "${NCCL_HOME}/include/nccl_device"

BUILD_DIR="${WORK_DIR}/build"
NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100" \
    make -C "${NCCL_EXTENSIONS_DIR}/nccl_m2n" build \
        NCCL_HOME="${NCCL_HOME}" \
        CUDA_HOME="${CUDA_HOME}" \
        BUILDDIR="${BUILD_DIR}"

install -Dm755 \
    "${BUILD_DIR}/lib/libnccl_m2n.so" \
    "${NCCL_EXTENSIONS_DIR}/python/nccl/m2n/lib/libnccl_m2n.so"
install -Dm644 \
    "${BUILD_DIR}/include/nccl_m2n.h" \
    "${NCCL_EXTENSIONS_DIR}/python/nccl/m2n/include/nccl_m2n.h"

CUDA_HOME="${CUDA_HOME}" uv pip install \
    --python "${PYTHON}" \
    --no-build-isolation \
    --no-cache \
    --no-deps \
    "${NCCL_EXTENSIONS_DIR}/python"

"${PYTHON}" - <<'PY'
import nccl.m2n
from nccl._extensions.bindings._internal.nccl_m2n import _inspect_function_pointers


assert _inspect_function_pointers()
PY

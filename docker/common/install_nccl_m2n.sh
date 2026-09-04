# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

#!/bin/bash

set -euxo pipefail

NCCL_EXTENSIONS_VERSION="0.1.0"
PYTHON="${UV_PROJECT_ENVIRONMENT:-/opt/venv}/bin/python"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --NCCL_EXTENSIONS_VERSION=*)
            NCCL_EXTENSIONS_VERSION="${1#*=}"
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

if [[ ! "${NCCL_EXTENSIONS_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "NCCL_EXTENSIONS_VERSION must be a semantic version such as 0.1.0" >&2
    exit 1
fi
NCCL_EXTENSIONS_TAG="nccl-extensions-v${NCCL_EXTENSIONS_VERSION}"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
test -x "${CUDA_HOME}/bin/nvcc"
test -x "${PYTHON}"

WORK_DIR=$(mktemp -d)
trap 'rm -rf "${WORK_DIR}"' EXIT

NCCL_EXTENSIONS_DIR="${WORK_DIR}/nccl-extensions"
git init "${NCCL_EXTENSIONS_DIR}"
git -C "${NCCL_EXTENSIONS_DIR}" remote add origin https://github.com/NVIDIA/nccl-extensions.git
git -C "${NCCL_EXTENSIONS_DIR}" fetch --depth 1 origin \
    "refs/tags/${NCCL_EXTENSIONS_TAG}:refs/tags/${NCCL_EXTENSIONS_TAG}"
git -C "${NCCL_EXTENSIONS_DIR}" checkout --detach "${NCCL_EXTENSIONS_TAG}^{commit}"

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

# The `nccl.m2n` wheel only declares `lib/cu12/libnccl_m2n.so` and
# `lib/cu13/libnccl_m2n.so` as package data, and the facade resolves the bundled
# library as `nccl/m2n/lib/cu<major>/libnccl_m2n.so`, where <major> comes from
# the installed `cuda.bindings`. Staging outside that directory silently drops
# the library from the wheel and the loader then fails to dlopen it.
CUDA_MAJOR=$("${PYTHON}" -c 'from cuda import bindings; print(bindings.__version__.split(".")[0])')
if [[ "${CUDA_MAJOR}" != "12" && "${CUDA_MAJOR}" != "13" ]]; then
    echo "Unsupported cuda.bindings major for nccl_m2n packaging: ${CUDA_MAJOR}" >&2
    exit 1
fi

install -Dm755 \
    "${BUILD_DIR}/lib/libnccl_m2n.so" \
    "${NCCL_EXTENSIONS_DIR}/python/nccl/m2n/lib/cu${CUDA_MAJOR}/libnccl_m2n.so"
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
from nccl._extensions._runtime import bundled_library
from nccl._extensions.bindings._internal.nccl_m2n import _inspect_function_pointers


# Assert the library actually shipped inside the wheel, so a packaging
# regression fails here instead of silently relying on an ambient
# NCCL_M2N_LIBRARY/NCCL_M2N_HOME/CUDA_HOME fallback at test time.
assert bundled_library("nccl_m2n") is not None, "libnccl_m2n.so was not bundled into nccl.m2n"
assert _inspect_function_pointers()
PY

#!/usr/bin/env bash
# set -euo pipefail

# Run this inside the NGC PyTorch container before training.
# It installs the NCCL/NVSHMEM wheels DeepEP needs, forces the newer NCCL
# runtime to load first, verifies the runtime libraries, and installs DeepEP.

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEEPEP_DIR="${DEEPEP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/DeepEP" && pwd)}"
MIN_NCCL_VERSION="${MIN_NCCL_VERSION:-23004}"

NCCL_PACKAGE="${NCCL_PACKAGE:-nvidia-nccl-cu13>=2.30.4}"
NVSHMEM_PACKAGE="${NVSHMEM_PACKAGE:-nvidia-nvshmem-cu13}"

resolve_nvidia_root() {
    local name="$1"
    local lib_name="$2"

    "${PYTHON_BIN}" - "${name}" "${lib_name}" <<'PY'
import glob
import os
import site
import sys
from importlib.metadata import distributions

name, lib_name = sys.argv[1:3]
seen = set()


def valid_root(root):
    root = os.path.realpath(root)
    if root in seen:
        return False
    seen.add(root)

    include_dir = os.path.join(root, "include")
    lib_dir = os.path.join(root, "lib")
    has_lib = (
        os.path.exists(os.path.join(lib_dir, lib_name)) or
        bool(glob.glob(os.path.join(lib_dir, f"{lib_name}*")))
    )
    return os.path.isdir(include_dir) and has_lib


def print_if_valid(root):
    if valid_root(root):
        print(os.path.realpath(root))
        raise SystemExit(0)


for dist in distributions():
    dist_name = (dist.metadata.get("Name") or "").lower()
    if f"nvidia-{name}" not in dist_name and f"nvidia_{name}" not in dist_name:
        continue

    for file in dist.files or []:
        if lib_name in str(file):
            lib_dir = os.path.dirname(str(file.locate()))
            if os.path.basename(lib_dir) == "lib":
                print_if_valid(os.path.dirname(lib_dir))

    print_if_valid(os.path.join(str(dist._path.parent), "nvidia", name))

for base in [*site.getsitepackages(), site.getusersitepackages(), *sys.path]:
    if base:
        print_if_valid(os.path.join(base, "nvidia", name))

raise SystemExit(f"Cannot resolve nvidia {name} root containing {lib_name}")
PY
}

ensure_unversioned_so() {
    local lib_dir="$1"
    local lib_name="$2"
    local candidate

    if [[ -e "${lib_dir}/${lib_name}" || -L "${lib_dir}/${lib_name}" ]]; then
        return
    fi

    candidate="$(find "${lib_dir}" -maxdepth 1 -name "${lib_name}.*" -print | sort | tail -n 1)"
    if [[ -z "${candidate}" ]]; then
        echo "Missing ${lib_name} or ${lib_name}.* in ${lib_dir}" >&2
        return 1
    fi

    ln -s "$(basename "${candidate}")" "${lib_dir}/${lib_name}"
}

echo "==> Installing DeepEP runtime dependencies"
"${PYTHON_BIN}" -m pip install --no-deps "${NCCL_PACKAGE}" "${NVSHMEM_PACKAGE}"

if [[ -z "${EP_NCCL_ROOT_DIR:-}" ]]; then
    EP_NCCL_ROOT_DIR="$(resolve_nvidia_root nccl libnccl.so)"
fi
if [[ -z "${EP_NVSHMEM_ROOT_DIR:-}" ]]; then
    EP_NVSHMEM_ROOT_DIR="$(resolve_nvidia_root nvshmem libnvshmem_host.so)"
fi

export EP_NCCL_ROOT_DIR
export EP_NVSHMEM_ROOT_DIR
export LD_LIBRARY_PATH="${EP_NCCL_ROOT_DIR}/lib:${EP_NVSHMEM_ROOT_DIR}/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${EP_NCCL_ROOT_DIR}/lib:${EP_NVSHMEM_ROOT_DIR}/lib:${LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="10.0"

echo "==> Using paths"
echo "EP_NCCL_ROOT_DIR=${EP_NCCL_ROOT_DIR}"
echo "EP_NVSHMEM_ROOT_DIR=${EP_NVSHMEM_ROOT_DIR}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "LIBRARY_PATH=${LIBRARY_PATH}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

echo "==> Checking required files"
test -f "${EP_NCCL_ROOT_DIR}/include/nccl.h"
test -e "${EP_NCCL_ROOT_DIR}/lib/libnccl.so" -o -e "${EP_NCCL_ROOT_DIR}/lib/libnccl.so.2"
test -f "${EP_NVSHMEM_ROOT_DIR}/include/nvshmem.h"
test -n "$(find "${EP_NVSHMEM_ROOT_DIR}/lib" -maxdepth 1 -name 'libnvshmem_host.so*' -print -quit)"
test -f "${EP_NVSHMEM_ROOT_DIR}/lib/libnvshmem_device.a"
ensure_unversioned_so "${EP_NCCL_ROOT_DIR}/lib" libnccl.so
ensure_unversioned_so "${EP_NVSHMEM_ROOT_DIR}/lib" libnvshmem_host.so

echo "==> Verifying loaded NCCL runtime"
"${PYTHON_BIN}" - <<PY
import ctypes
import sys

lib = ctypes.CDLL("libnccl.so.2")
version = ctypes.c_int()
lib.ncclGetVersion(ctypes.byref(version))
print(f"runtime NCCL int: {version.value}")
print(f"runtime NCCL: {version.value // 10000}.{(version.value % 10000) // 100}.{version.value % 100}")

if version.value < ${MIN_NCCL_VERSION}:
    raise SystemExit("NCCL runtime is too old; check LD_LIBRARY_PATH ordering")
PY

echo "==> Installing DeepEP editable"
cd "${DEEPEP_DIR}"
"${PYTHON_BIN}" -m pip install --no-build-isolation -ve .

echo "==> Done"
echo "Keep these exports in your training launch environment:"
echo "export EP_NCCL_ROOT_DIR=${EP_NCCL_ROOT_DIR}"
echo "export EP_NVSHMEM_ROOT_DIR=${EP_NVSHMEM_ROOT_DIR}"
echo "export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo 'export LD_LIBRARY_PATH=$EP_NCCL_ROOT_DIR/lib:$EP_NVSHMEM_ROOT_DIR/lib:$LD_LIBRARY_PATH'

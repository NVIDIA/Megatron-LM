# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

#!/bin/bash

set -euxo pipefail

MOK_REVISION="${1:?Usage: install_mok.sh <git commit> <patch> <wheel directory>}"
PATCH_FILE=$(realpath "${2:?Usage: install_mok.sh <git commit> <patch> <wheel directory>}")
WHEEL_DIR=$(realpath -m "${3:?Usage: install_mok.sh <git commit> <patch> <wheel directory>}")
PYTHON="${UV_PROJECT_ENVIRONMENT:-/opt/venv}/bin/python"

if [[ ! "${MOK_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "MOK_COMMIT must be a full Git commit" >&2
    exit 1
fi

WORK_DIR=$(mktemp -d)
trap 'rm -rf "${WORK_DIR}"' EXIT

git init "${WORK_DIR}"
git -C "${WORK_DIR}" remote add origin https://github.com/QiZhangNV/mixture-of-kittens.git
git -C "${WORK_DIR}" fetch --depth 1 origin "${MOK_REVISION}"
git -C "${WORK_DIR}" checkout --detach "${MOK_REVISION}"
git -C "${WORK_DIR}" apply --check "${PATCH_FILE}"
git -C "${WORK_DIR}" apply "${PATCH_FILE}"
bash "${WORK_DIR}/scripts/prepare_thunderkittens.sh"

# The CI Blackwell lane is GB200 (SM100). MoK also supports an explicit SM103 build.
export MOK_ARCH="${MOK_ARCH:-SM100}"
export NVCC="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
export MOK_NVCC="${NVCC}"
# Docker builds have no GPU driver. Stubs are link inputs, never runtime libraries.
export LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"
uv build --python "${PYTHON}" --wheel --no-build-isolation --no-cache --out-dir "${WHEEL_DIR}" "${WORK_DIR}"
uv pip install --python "${PYTHON}" --no-deps --no-cache "${WHEEL_DIR}"/*.whl

# Inspect the installed wheel without loading libcuda on GPU-less image builders.
MOK_EXTENSION=$("${PYTHON}" - <<'PY'
from importlib.metadata import distribution
from sysconfig import get_config_var

extension = distribution("mixture-of-kittens").locate_file("mok/_C" + get_config_var("EXT_SUFFIX"))
assert extension.is_file(), f"MoK native extension missing: {extension}"
print(extension)
PY
)
"${CUDA_HOME:-/usr/local/cuda}/bin/cuobjdump" --list-elf "${MOK_EXTENSION}" > "${WORK_DIR}/cubins.txt"
cat "${WORK_DIR}/cubins.txt"
grep -q "sm_${MOK_ARCH#SM}" "${WORK_DIR}/cubins.txt"

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Install Megatron-LM with a pinned PyPI Transformer Engine release on a CUDA host.
The Transformer Engine PyTorch extension still performs native compilation and
can take 10-15+ minutes depending on CPU resources, memory, cache state, and
build parallelism.

Run from the Megatron-LM repository root:
  bash skills/mcore-transformer-engine-install/scripts/install_te_pypi.sh \
    --torch-backend cu128 --cuda-arch a100

Options:
  --python VERSION        Python version for uv venv (default: 3.12)
  --venv PATH             Virtualenv path (default: .venv)
  --torch-backend BACKEND uv torch backend, e.g. cu128 or cu130 (default: cu128)
  --torch-version VERSION PyTorch version to install (default: 2.10.0)
  --te-version VERSION    Transformer Engine PyPI version (default: 2.11.0)
  --te-spec SPEC          Full TE spec (default: transformer_engine[pytorch]==TE_VERSION)
  --cuda-arch ARCH        b200, gb200, rtx-pro-6000, h100, l4, l40s, a100, sm120, sm100, sm90, sm89, sm80, or auto
  --extras EXTRAS         Editable Megatron extras, excluding te (default: none; use training for training deps)
  --max-jobs N            Native build parallelism (default: 4)
  --repo-root PATH        Megatron-LM repository root (default: current git root)
  --no-smoke              Skip the CUDA Transformer Engine smoke test
  -h, --help              Show this help

Environment overrides:
  NVTE_CUDA_ARCHS and TORCH_CUDA_ARCH_LIST override --cuda-arch detection.
  NVTE_FRAMEWORK defaults to pytorch.
  NVTE_BUILD_THREADS_PER_JOB defaults to 1.
EOF
}

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_PATH="${VENV_PATH:-.venv}"
TORCH_BACKEND="${TORCH_BACKEND:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TE_VERSION="${TE_VERSION:-2.11.0}"
TE_SPEC="${TE_SPEC:-}"
CUDA_ARCH="${CUDA_ARCH:-auto}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-}"
MAX_JOBS="${MAX_JOBS:-4}"
RUN_SMOKE=1
REPO_ROOT="${REPO_ROOT:-}"
PHASE_ACTIVE=""

phase_mark() {
  if [[ -z "${ATE_INSTALL_PHASE_LOG:-}" ]]; then
    return 0
  fi
  local phase="$1"
  local event="$2"
  local status="${3:-}"
  python3 - "${ATE_INSTALL_PHASE_LOG}" "${phase}" "${event}" "${status}" <<'PY'
import datetime
import json
import os
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
epoch = time.time()
payload = {
    "at": datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).isoformat(),
    "epoch": round(epoch, 6),
    "event": sys.argv[3],
    "phase": sys.argv[2],
    "pid": os.getpid(),
}
if sys.argv[4]:
    payload["exit_code"] = int(sys.argv[4])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, sort_keys=True) + "\n")
PY
}

phase_start() {
  PHASE_ACTIVE="$1"
  phase_mark "$1" start
}

phase_end() {
  local phase="$1"
  local status="${2:-0}"
  phase_mark "${phase}" end "${status}"
  if [[ "${PHASE_ACTIVE}" == "${phase}" ]]; then
    PHASE_ACTIVE=""
  fi
}

phase_trap() {
  local status=$?
  if [[ "${status}" -ne 0 && -n "${PHASE_ACTIVE}" ]]; then
    phase_mark "${PHASE_ACTIVE}" end "${status}"
  fi
}

trap phase_trap EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --torch-backend)
      TORCH_BACKEND="$2"
      shift 2
      ;;
    --torch-version)
      TORCH_VERSION="$2"
      shift 2
      ;;
    --te-version)
      TE_VERSION="$2"
      shift 2
      ;;
    --te-spec)
      TE_SPEC="$2"
      shift 2
      ;;
    --cuda-arch)
      CUDA_ARCH="$2"
      shift 2
      ;;
    --extras)
      INSTALL_EXTRAS="$2"
      shift 2
      ;;
    --max-jobs)
      MAX_JOBS="$2"
      shift 2
      ;;
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --no-smoke)
      RUN_SMOKE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${TE_SPEC}" ]]; then
  TE_SPEC="transformer_engine[pytorch]==${TE_VERSION}"
fi

if [[ ",${INSTALL_EXTRAS}," == *",te,"* ]]; then
  echo "Do not include the 'te' extra with this helper." >&2
  echo "It can route uv to the repository-pinned Transformer Engine Git source." >&2
  echo "Use --extras training for training dependencies, or install a TE source/fork path separately." >&2
  exit 2
fi

if [[ -z "${REPO_ROOT}" ]]; then
  REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
fi

cd "${REPO_ROOT}"

if [[ ! -f pyproject.toml ]] || [[ ! -d megatron ]]; then
  echo "Expected Megatron-LM repo root, got: ${REPO_ROOT}" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is required for Transformer Engine native builds but was not found on PATH." >&2
  exit 1
fi

if [[ -z "${CUDA_PATH:-}" ]]; then
  NVCC_PATH="$(command -v nvcc)"
  CUDA_PATH="$(cd "$(dirname "${NVCC_PATH}")/.." && pwd)"
  export CUDA_PATH
fi

PYTHON_BIN="${VENV_PATH}/bin/python"

phase_start "torch-bootstrap"

if [[ -x "${PYTHON_BIN}" ]]; then
  EXISTING_PYTHON_VERSION="$("${PYTHON_BIN}" - <<'PY'
import sys

print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  if [[ "${EXISTING_PYTHON_VERSION}" != "${PYTHON_VERSION}" ]]; then
    echo "Existing virtualenv at ${VENV_PATH} uses Python ${EXISTING_PYTHON_VERSION}, expected ${PYTHON_VERSION}." >&2
    echo "Remove it or rerun with UV_VENV_CLEAR=1 if this helper owns that environment." >&2
    exit 1
  fi
  echo "Reusing existing virtualenv at ${VENV_PATH}."
else
  uv venv --python "${PYTHON_VERSION}" "${VENV_PATH}"
fi

uv pip install --no-config --python "${PYTHON_BIN}" --torch-backend="${TORCH_BACKEND}" \
  "torch==${TORCH_VERSION}" "setuptools>=80,<82" wheel packaging pybind11 Cython \
  hatchling cmake ninja nvidia-mathdx numpy

"${PYTHON_BIN}" - <<'PY'
import torch

print("torch:", torch.__version__, "torch cuda:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA is not visible to PyTorch"
print("gpu:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
PY

phase_end "torch-bootstrap" 0
phase_start "megatron+te-install"

VENV_SITE="$("${PYTHON_BIN}" - <<'PY'
import site

print(site.getsitepackages()[0])
PY
)"

for INCLUDE_DIR in "${VENV_SITE}"/nvidia/*/include; do
  if [[ -d "${INCLUDE_DIR}" ]]; then
    export CPATH="${INCLUDE_DIR}:${CPATH:-}"
  fi
done

for LIB_DIR in "${VENV_SITE}"/nvidia/*/lib; do
  if [[ -d "${LIB_DIR}" ]]; then
    export LIBRARY_PATH="${LIB_DIR}:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${LIB_DIR}:${LD_LIBRARY_PATH:-}"
  fi
done

if [[ -d "${VENV_SITE}/nvidia/cudnn" ]]; then
  export CUDNN_PATH="${CUDNN_PATH:-${VENV_SITE}/nvidia/cudnn}"
  export CUDNN_HOME="${CUDNN_HOME:-${CUDNN_PATH}}"
  export LD_LIBRARY_PATH="${CUDNN_PATH}/lib:${LD_LIBRARY_PATH:-}"
fi

case "${CUDA_ARCH}" in
  rtx-pro-6000|rtxpro6000|rtx-pro-blackwell|sm120)
    DEFAULT_NVTE_CUDA_ARCHS=120
    DEFAULT_TORCH_CUDA_ARCH_LIST=12.0
    ;;
  b200|gb200|sm100)
    DEFAULT_NVTE_CUDA_ARCHS=100
    DEFAULT_TORCH_CUDA_ARCH_LIST=10.0
    ;;
  h100|sm90)
    DEFAULT_NVTE_CUDA_ARCHS=90
    DEFAULT_TORCH_CUDA_ARCH_LIST=9.0
    ;;
  l4|l40s|sm89)
    DEFAULT_NVTE_CUDA_ARCHS=89
    DEFAULT_TORCH_CUDA_ARCH_LIST=8.9
    ;;
  a100|sm80)
    DEFAULT_NVTE_CUDA_ARCHS=80
    DEFAULT_TORCH_CUDA_ARCH_LIST=8.0
    ;;
  auto)
    read -r DEFAULT_NVTE_CUDA_ARCHS DEFAULT_TORCH_CUDA_ARCH_LIST < <("${PYTHON_BIN}" - <<'PY'
import torch

major, minor = torch.cuda.get_device_capability(0)
print(f"{major}{minor} {major}.{minor}")
PY
)
    ;;
  *)
    echo "Unsupported --cuda-arch '${CUDA_ARCH}'. Use b200, gb200, rtx-pro-6000, h100, l4, l40s, a100, sm120, sm100, sm90, sm89, sm80, or auto." >&2
    exit 2
    ;;
esac

export MAX_JOBS
export NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-1}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-${DEFAULT_NVTE_CUDA_ARCHS}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${DEFAULT_TORCH_CUDA_ARCH_LIST}}"

echo "Using CUDA_PATH=${CUDA_PATH}"
echo "Using CUDNN_PATH=${CUDNN_PATH:-<not set>}"
echo "Using NVTE_FRAMEWORK=${NVTE_FRAMEWORK}"
echo "Using NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS}"
echo "Using TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "Using MAX_JOBS=${MAX_JOBS}"
echo "Using NVTE_BUILD_THREADS_PER_JOB=${NVTE_BUILD_THREADS_PER_JOB}"
if [[ -n "${INSTALL_EXTRAS}" ]]; then
  echo "Installing Megatron editable extras: ${INSTALL_EXTRAS}"
else
  echo "Installing Megatron editable extras: <none>"
fi

if [[ -n "${INSTALL_EXTRAS}" ]]; then
  uv pip install --no-config --python "${PYTHON_BIN}" -e ".[${INSTALL_EXTRAS}]"
else
  uv pip install --no-config --python "${PYTHON_BIN}" -e .
fi

CHECK_TRAINING_IMPORT=0
if [[ ",${INSTALL_EXTRAS}," == *",training,"* ]]; then
  CHECK_TRAINING_IMPORT=1
fi
export CHECK_TRAINING_IMPORT

echo "Starting pinned PyPI Transformer Engine install: ${TE_SPEC}"
echo "Transformer Engine native compilation can take 10-15+ minutes depending on CPU resources, memory, cache state, and build parallelism."
echo "Some build steps may be quiet for a while; wait for the final smoke test before deciding it is stuck."

uv pip install --no-config --python "${PYTHON_BIN}" --no-build-isolation "${TE_SPEC}"

if [[ "${RUN_SMOKE}" -eq 1 ]]; then
  "${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata
import os

import torch
import megatron.core
from transformer_engine.pytorch import Linear

if os.environ.get("CHECK_TRAINING_IMPORT") == "1":
    import megatron.training

assert torch.cuda.is_available(), "CUDA is not visible to PyTorch"
layer = Linear(8, 8).cuda()
x = torch.randn(2, 8, device="cuda")
y = layer(x)
torch.cuda.synchronize()
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("transformer-engine:", metadata.version("transformer-engine"))
print("megatron cuda+te smoke: ok", tuple(y.shape))
PY
fi

phase_end "megatron+te-install" 0

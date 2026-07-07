#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Install Megatron-LM with Transformer Engine from source on a CUDA host.
The Transformer Engine source build can take 10-15+ minutes depending on CPU
resources, memory, cache state, and build parallelism.

Run from the Megatron-LM repository root:
  bash skills/mcore-transformer-engine-install/scripts/install_te_source.sh \
    --torch-backend cu128 --cuda-arch h100

Options:
  --python VERSION        Python version for uv venv (default: 3.12)
  --venv PATH             Virtualenv path (default: .venv)
  --torch-backend BACKEND uv torch backend, e.g. cu128 or cu130 (default: cu128)
  --cuda-arch ARCH        b200, gb200, rtx-pro-6000, h100, l4, l40s, a100, sm120, sm100, sm90, sm89, sm80, or auto
  --extras EXTRAS         Editable install extras (default: training,te)
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
CUDA_ARCH="${CUDA_ARCH:-auto}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-training,te}"
MAX_JOBS="${MAX_JOBS:-4}"
RUN_SMOKE=1
REPO_ROOT="${REPO_ROOT:-}"

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
  echo "nvcc is required for Transformer Engine source builds but was not found on PATH." >&2
  exit 1
fi

if [[ -z "${CUDA_PATH:-}" ]]; then
  NVCC_PATH="$(command -v nvcc)"
  CUDA_PATH="$(cd "$(dirname "${NVCC_PATH}")/.." && pwd)"
  export CUDA_PATH
fi

uv venv --python "${PYTHON_VERSION}" "${VENV_PATH}"
PYTHON_BIN="${VENV_PATH}/bin/python"

uv pip install --no-config --python "${PYTHON_BIN}" --torch-backend="${TORCH_BACKEND}" \
  "torch>=2.6.0" "setuptools>=80" wheel packaging pybind11 Cython hatchling cmake ninja nvidia-mathdx

"${PYTHON_BIN}" - <<'PY'
import torch

print("torch:", torch.__version__, "torch cuda:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA is not visible to PyTorch"
print("gpu:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
PY

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
echo "Starting Megatron + Transformer Engine source install."
echo "Transformer Engine native compilation can take 10-15+ minutes depending on CPU resources, memory, cache state, and build parallelism."
echo "Some build steps may be quiet for a while; wait for the final smoke test before deciding it is stuck."

uv pip install --python "${PYTHON_BIN}" --no-build-isolation -e ".[${INSTALL_EXTRAS}]"

if [[ "${RUN_SMOKE}" -eq 1 ]]; then
  "${PYTHON_BIN}" - <<'PY'
import torch
import megatron.core
import megatron.training
from transformer_engine.pytorch import Linear

assert torch.cuda.is_available(), "CUDA is not visible to PyTorch"
layer = Linear(8, 8).cuda()
x = torch.randn(2, 8, device="cuda")
y = layer(x)
torch.cuda.synchronize()
print("megatron cuda+te smoke: ok", tuple(y.shape))
PY
fi

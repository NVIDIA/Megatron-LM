#!/usr/bin/env bash
set -euo pipefail

SOURCE="${MAGI_ATTENTION_SOURCE:?Set MAGI_ATTENTION_SOURCE to a MagiAttention v1.1.1 checkout}"
VENV="${MAGI_ATTENTION_VENV:?Set MAGI_ATTENTION_VENV to the persistent test venv path}"
ARCH="${MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY:-}"
MAX_JOBS_VALUE="${MAX_JOBS:-32}"

if [[ -z "${ARCH}" ]]; then
    ARCH="$(python -c 'import torch; print(torch.cuda.get_device_capability()[0] * 10)')"
fi
if [[ "${ARCH}" != "90" && "${ARCH}" != "100" ]]; then
    echo "Unsupported compute capability sm${ARCH}: expected 90 (Hopper) or 100 (Blackwell)." >&2
    exit 2
fi
MARKER="${VENV}/.magi_attention_v1.1.1_sm${ARCH}_complete"

# sm100 uses the FA4 (flash_attn_cute) kernel backend; sm90 uses the native
# FFA kernels, which are prebuilt at install time instead.
if [[ "${ARCH}" == "100" ]]; then
    IMPORT_CHECK="import flash_attn_cute, magi_attention"
else
    IMPORT_CHECK="import magi_attention, magi_attention.magi_attn_ext"
fi

mkdir -p "${TMPDIR:-/tmp}"

if [[ ! -f "${SOURCE}/pyproject.toml" ]]; then
    echo "MagiAttention source not found at ${SOURCE}" >&2
    exit 2
fi

if [[ -f "${MARKER}" ]]; then
    "${VENV}/bin/python" -c \
        "${IMPORT_CHECK}; assert magi_attention.__version__ == '1.1.1'; print(magi_attention.__version__)"
    exit 0
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
    python -m venv --system-site-packages "${VENV}"
fi

export PATH="${VENV}/bin:${PATH}"
export MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY="${ARCH}"
export MAX_JOBS="${MAX_JOBS_VALUE}"
export NVCC_THREADS="${NVCC_THREADS:-4}"
if [[ "${ARCH}" == "100" ]]; then
    export MAGI_ATTENTION_PREBUILD_FFA=0
    export MAGI_ATTENTION_FA4_BACKEND=1
    export FLASH_ATTN_CUDA_ARCHS="${ARCH}"
else
    export MAGI_ATTENTION_PREBUILD_FFA=1
fi

python -m pip install --upgrade \
    pip setuptools wheel "packaging==25.0" ninja versioningit "pytest==8.3.5"
python -m pip install -r "${SOURCE}/requirements.txt"

if [[ "${ARCH}" == "100" ]]; then
    # Some NGC images carry CUTLASS DSL dist-info and payload files without the
    # .pth file that exposes the ``cutlass`` package. Install the same pinned
    # payload into the venv when that split-package image state is detected.
    if ! python -c "import cutlass" >/dev/null 2>&1; then
        python -m pip install --force-reinstall --no-deps \
            "nvidia-cutlass-dsl-libs-base==4.4.2" "nvidia-cutlass-dsl==4.4.2"
    fi

    if ! python -c "import flash_attn_cute" >/dev/null 2>&1; then
        (
            cd "${SOURCE}"
            bash scripts/install_flash_attn_cute.sh "sm${ARCH}"
        )
    fi
fi

if ! python -c "${IMPORT_CHECK}; assert magi_attention.__version__ == '1.1.1'" \
    >/dev/null 2>&1; then
    python -m pip install --no-build-isolation "${SOURCE}"
fi
python -c \
    "${IMPORT_CHECK}; assert magi_attention.__version__ == '1.1.1'; print(magi_attention.__version__)"
touch "${MARKER}"

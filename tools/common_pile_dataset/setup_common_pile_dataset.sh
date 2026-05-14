#!/bin/bash
# Setup script to create Common Pile CI test dataset on the remote HPC machine.
#
# This script:
#   1. Clones the Megatron-LM repo (if needed)
#   2. Installs Python dependencies (if needed)
#   3. Runs create_common_pile_ci_dataset.py to download, preprocess, and save data
#
# Usage:
#   scp tools/common_pile_dataset/setup_common_pile_dataset.sh \
#       tools/common_pile_dataset/create_common_pile_ci_dataset.py \
#       <user>@<hpc-host>:/tmp/
#   ssh <user>@<hpc-host> 'bash /tmp/setup_common_pile_dataset.sh'

set -euo pipefail

# Use Python 3.10+ (required by latest Megatron-LM for PEP 604 type syntax)
PYTHON="/usr/bin/python3.10"
if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python 3.10+ required but ${PYTHON} not found"
    exit 1
fi
echo "Using Python: ${PYTHON} ($("${PYTHON}" --version))"

OUTPUT_DIR="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci/text/common_pile/v01_filtered_data"
EXISTING_VOCAB="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci/text/the_pile"
NUM_DOCUMENTS=12000000
DATASET_NAME="common-pile/comma_v0.1_training_dataset"
WORK_DIR="/tmp/mcore_dataset_setup_$$"

# Redirect HuggingFace cache to lustre so it doesn't fill up /home
export HF_HOME="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci/.hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

echo "============================================================"
echo "Common Pile CI Dataset Setup"
echo "============================================================"
echo "Output:     ${OUTPUT_DIR}"
echo "Vocab from: ${EXISTING_VOCAB}"
echo "Documents:  ${NUM_DOCUMENTS}"
echo "Dataset:    ${DATASET_NAME}"
echo "Work dir:   ${WORK_DIR}"
echo "HF cache:   ${HF_HOME}"
echo "============================================================"

# Create work directory
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Check if create_common_pile_ci_dataset.py was scp'd alongside this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/create_common_pile_ci_dataset.py" ]; then
    DATASET_SCRIPT="${SCRIPT_DIR}/create_common_pile_ci_dataset.py"
    echo "Found dataset script at: ${DATASET_SCRIPT}"
else
    DATASET_SCRIPT=""
fi

# Clone Megatron-LM — full shallow clone needed because preprocess_data.py
# imports from megatron.core which pulls in tensor_parallel and other submodules
MEGATRON_DIR="${WORK_DIR}/Megatron-LM"
if [ ! -d "${MEGATRON_DIR}" ]; then
    echo ""
    echo "Cloning Megatron-LM repository (full shallow clone)..."
    git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git "${MEGATRON_DIR}" 2>&1
fi

# Patch megatron/training/__init__.py to avoid importing the full training stack.
# preprocess_data.py only needs _add_tokenizer_args from arguments.py, but the
# __init__.py eagerly imports initialize_megatron which pulls in triton, apex, etc.
TRAINING_INIT="${MEGATRON_DIR}/megatron/training/__init__.py"
if [ -f "${TRAINING_INIT}" ] && grep -q "from .initialize" "${TRAINING_INIT}"; then
    echo "Patching megatron/training/__init__.py to skip heavy imports..."
    sed -i 's/^from \.initialize/#from .initialize/' "${TRAINING_INIT}"
    sed -i 's/^from \.training/#from .training/' "${TRAINING_INIT}"
fi

# Use the script from the repo if we didn't have it locally
if [ -z "${DATASET_SCRIPT}" ]; then
    DATASET_SCRIPT="${MEGATRON_DIR}/tools/common_pile_dataset/create_common_pile_ci_dataset.py"
    if [ ! -f "${DATASET_SCRIPT}" ]; then
        echo "ERROR: create_common_pile_ci_dataset.py not found"
        exit 1
    fi
fi

# Create a virtual environment to avoid system package conflicts
VENV_DIR="${WORK_DIR}/venv"
if [ ! -d "${VENV_DIR}" ]; then
    echo ""
    echo "Creating virtual environment..."
    "${PYTHON}" -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
PYTHON="$(which python)"
echo "Using venv Python: ${PYTHON} ($(${PYTHON} --version))"

# Install Python dependencies
echo ""
echo "Checking Python dependencies..."
"${PYTHON}" -c "import datasets" 2>/dev/null || {
    echo "Installing 'datasets' library..."
    pip install datasets --quiet
}
"${PYTHON}" -c "import nltk" 2>/dev/null || {
    echo "Installing 'nltk' library..."
    pip install nltk --quiet
}
"${PYTHON}" -c "import torch" 2>/dev/null || {
    echo "Installing PyTorch (CPU-only, needed for Megatron preprocessing)..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
}
"${PYTHON}" -c "import transformers" 2>/dev/null || {
    echo "Installing 'transformers' library (needed for tokenizer in preprocessing)..."
    pip install transformers --quiet
}

# Download NLTK punkt tokenizer data (needed for BERT sentence splitting)
echo "Ensuring NLTK punkt tokenizer data is available..."
"${PYTHON}" -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true

# Create the output directory
mkdir -p "${OUTPUT_DIR}"

# Run the dataset creation script
echo ""
echo "============================================================"
echo "Running dataset creation..."
echo "============================================================"

"${PYTHON}" "${DATASET_SCRIPT}" \
    --output-dir "${OUTPUT_DIR}" \
    --megatron-dir "${MEGATRON_DIR}" \
    --num-documents "${NUM_DOCUMENTS}" \
    --dataset-name "${DATASET_NAME}" \
    --copy-vocab-from "${EXISTING_VOCAB}" \
    --keep-jsonl \
    --workers 4

echo ""
echo "============================================================"
echo "Done! Dataset saved to: ${OUTPUT_DIR}"
echo "============================================================"
echo ""
echo "Final directory listing:"
find "${OUTPUT_DIR}" -type f -exec ls -lh {} \;

# Cleanup
echo ""
echo "Cleaning up work directory: ${WORK_DIR}"
rm -rf "${WORK_DIR}"

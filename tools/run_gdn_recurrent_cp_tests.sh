#!/usr/bin/env bash
set -euo pipefail
# Run the focused, self-contained real-NCCL suite on 2 or 4 GPUs.
NPROC=${1:-2}
PYTHON_BIN=${PYTHON_BIN:-python}
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT/test-deps:$TASK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-2}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
mkdir -p test-results
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node="$NPROC" \
    -m pytest --noconftest -o addopts="" -x -s \
    tests/unit_tests/ssm/test_gdn_chunkwise_cp.py \
    tests/unit_tests/ssm/test_gdn_recurrent_cp.py \
    tests/unit_tests/determinism/kernels/test_recurrent_gdn_cp_replay.py

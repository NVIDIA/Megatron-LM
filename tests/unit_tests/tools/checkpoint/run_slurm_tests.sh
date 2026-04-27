#!/bin/bash
# Run GPT <-> Mamba checkpoint conversion tests on SLURM.
#
# Covers:
#   Phase 1 - Unit tests (pattern parsing, key mapping, SSM init, round-trip,
#             and the new GPT-compatibility whitelist)
#   Phase 2 - Integration tests (legacy TP=1/PP=1 on-disk round-trip)
#   Phase 3 - Parallelism matrix (TP / PP / FSDP and all combinations,
#             across legacy and torch_dist / fsdp_dtensor formats;
#             hybrid patterns exercised: pure-attention, M*-, M*-M*-,
#             alternating, and pure-SSM)
#
# Single-node mode (default) exercises the full matrix on one GPU.
# Multi-node mode launches the same pytest invocation on N nodes to verify
# the converter is deterministic across nodes and that dist-checkpoint load
# works from a shared filesystem.
#
# Usage:
#   bash run_slurm_tests.sh                 # single-node, default repo path
#   NODES=2 bash run_slurm_tests.sh         # 2 nodes
#   MEGATRON_LM_DIR=/path bash run_slurm_tests.sh
#
# Environment knobs:
#   MEGATRON_LM_DIR   Path to the Megatron-LM checkout (default: this repo root)
#   CONTAINER_IMAGE   Container image                  (default: nemo:26.04)
#   NODES             Number of nodes                  (default: 1)
#   GPUS_PER_NODE     GPUs per node                    (default: 1)
#   PARTITION         SLURM partition                  (default: batch)
#   ACCOUNT           SLURM account                 (default: coreai_dlalgo_genai)
#   TIME              SLURM time limit                 (default: 00:45:00)

set -euo pipefail

# Default to the repo that contains this script.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_DEFAULT_REPO="$(cd "${_SCRIPT_DIR}/../../../.." && pwd)"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/nemo:26.02}"
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-${_DEFAULT_REPO}}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_mcore}"
TIME="${TIME:-00:45:00}"

LOG_DIR="${MEGATRON_LM_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "======================================================"
echo "GPT <-> Mamba Conversion Tests"
echo "  Repo           : ${MEGATRON_LM_DIR}"
echo "  Container      : ${CONTAINER_IMAGE}"
echo "  Nodes          : ${NODES}"
echo "  GPUs per node  : ${GPUS_PER_NODE}"
echo "  Partition      : ${PARTITION}"
echo "  Account        : ${ACCOUNT}"
echo "  Time limit     : ${TIME}"
echo "  Logs           : ${LOG_DIR}"
echo "======================================================"

# The conversion unit tests are CPU-only; the parallelism matrix only needs
# one GPU per test process (it exercises the sharding logic, not kernels).
# On multi-node runs we invoke pytest on every task so each node independently
# validates its view of the checkpoint — i.e. the *same* shared
# checkpoint must round-trip from every node.
srun \
    --job-name=gpt-mamba-conv-test \
    --nodes="${NODES}" \
    --ntasks-per-node=1 \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --cpus-per-gpu=16 \
    --time="${TIME}" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --output="${LOG_DIR}/gpt_mamba_conv_test_%j_%t.out" \
    --error="${LOG_DIR}/gpt_mamba_conv_test_%j_%t.err" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${MEGATRON_LM_DIR}:/opt/megatron-lm" \
    --container-workdir="/opt/megatron-lm" \
    bash -c '
        set -euo pipefail
        export PYTHONPATH=/opt/megatron-lm:${PYTHONPATH:-}

        RANK="${SLURM_PROCID:-0}"
        NODE="${SLURMD_NODENAME:-local}"
        echo "[node=${NODE} rank=${RANK}] Python : $(python --version 2>&1)"
        echo "[node=${NODE} rank=${RANK}] torch  : $(python -c "import torch; print(torch.__version__)")"
        echo "[node=${NODE} rank=${RANK}] cuda   : $(python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())")"
        echo "------------------------------------------------------"

        echo ""
        echo "=== [node=${NODE}] Phase 1: Unit tests ==="
        echo ""
        python -m pytest -vs \
            tests/unit_tests/tools/checkpoint/test_gpt_mamba_conversion.py

        echo ""
        echo "=== [node=${NODE}] Phase 1b: GPT-compatibility whitelist tests ==="
        echo ""
        # Run the whitelist classes in isolation so a regression in the
        # safeguard is easy to spot in CI logs.
        python -m pytest -vs \
            tests/unit_tests/tools/checkpoint/test_gpt_mamba_conversion.py::TestPatternWhitelist \
            tests/unit_tests/tools/checkpoint/test_gpt_mamba_conversion.py::TestSourceArgsWhitelist

        echo ""
        echo "=== [node=${NODE}] Phase 2: Integration (legacy TP=1/PP=1) ==="
        echo ""
        python tests/unit_tests/tools/checkpoint/test_gpt_mamba_conversion_integration.py

        echo ""
        echo "=== [node=${NODE}] Phase 3: Parallelism matrix (TP/PP/FSDP/combos) ==="
        echo ""
        python -m pytest -vs \
            tests/unit_tests/tools/checkpoint/test_gpt_mamba_conversion_parallelism.py
    '

echo "======================================================"
echo "Test complete. Logs: ${LOG_DIR}"
echo "======================================================"

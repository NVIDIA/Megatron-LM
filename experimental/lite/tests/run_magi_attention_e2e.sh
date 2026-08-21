#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORK_DATA="$(dirname "${ROOT}")"
VENV="${MAGI_ATTENTION_VENV:-${WORK_DATA}/magi_attention_test_env_v1.1.1/venv}"
SOURCE="${MAGI_ATTENTION_SOURCE:-${WORK_DATA}/MagiAttention-v1.1.1}"
NPROC="${MLITE_MAGI_NPROC:-4}"
SUITE="${1:-all}"

if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "MagiAttention test environment not found at ${VENV}." >&2
    echo "Run experimental/lite/tests/setup_magi_attention_env.sh first." >&2
    exit 2
fi
ARCH="$("${VENV}/bin/python" -c 'import torch; print(torch.cuda.get_device_capability()[0] * 10)')"
if [[ "${ARCH}" == "100" ]]; then
    KERNEL_BACKEND="fa4"
    UPSTREAM_BACKEND_FILTER="*FA4*"
else
    KERNEL_BACKEND="ffa"
    UPSTREAM_BACKEND_FILTER="*FFA*"
fi
if [[ ! -f "${VENV}/.magi_attention_v1.1.1_sm${ARCH}_complete" ]]; then
    echo "MagiAttention test environment is incomplete for sm${ARCH}: ${VENV}" >&2
    exit 2
fi
if [[ "${NPROC}" != "2" && "${NPROC}" != "4" ]]; then
    echo "MLITE_MAGI_NPROC must be 2 or 4, got ${NPROC}." >&2
    exit 2
fi
if [[ "${SUITE}" != "all" && "${SUITE}" != "upstream" && "${SUITE}" != "lite" ]]; then
    echo "Usage: $0 [all|upstream|lite]" >&2
    exit 2
fi

export PYTHONPATH="${ROOT}:${ROOT}/experimental/lite:${PYTHONPATH:-}"
unset MAGI_ATTENTION_SDPA_BACKEND MAGI_ATTENTION_FA4_BACKEND
export MAGI_ATTENTION_KERNEL_BACKEND="${KERNEL_BACKEND}"
export MAGI_ATTENTION_NATIVE_GRPCOLL=0
export MAGI_ATTENTION_HIERARCHICAL_COMM=0
export MAGI_ATTENTION_QO_COMM=0
export MAGI_ATTENTION_DETERMINISTIC_MODE=0
export MAGI_ATTENTION_FORWARD_HIGH_PRECISION_REDUCE=0
export MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE=0
export MAGI_ATTENTION_BWD_HIDE_TAIL_REDUCE=0

if [[ "${SUITE}" == "all" || "${SUITE}" == "upstream" ]]; then
    if [[ ! -f "${SOURCE}/tests/test_pipeline.py" ]]; then
        echo "MagiAttention source not found at ${SOURCE}." >&2
        exit 2
    fi
    export MAGI_ATTENTION_TEST_WORLD_SIZE="${NPROC}"
    export MAGI_ATTENTION_TEST_ATTN_CONFIG="sdpa_varlen_block_causal_960"
    export MAGI_ATTENTION_TEST_NUM_HEADS="8_2"
    export MAGI_ATTENTION_TEST_HEAD_DIM="64"
    export MAGI_ATTENTION_TEST_DTYPE="*bfloat16*"
    export MAGI_ATTENTION_TEST_BACKEND="${UPSTREAM_BACKEND_FILTER}"
    (
        cd "${SOURCE}"
        "${VENV}/bin/python" -m pytest -q -s \
            "tests/test_pipeline.py::TestPipelineWithWorldSize${NPROC}::test_pipeline"
    )
fi

if [[ "${SUITE}" == "all" || "${SUITE}" == "lite" ]]; then
    cd "${ROOT}"
    # The operator-level attention test runs first and gates the model E2E:
    # it isolates the core attention numerics (dispatch -> calc_attn ->
    # undispatch vs an SDPA reference) in its own torchrun so the two tests
    # never share one process-group lifecycle.
    # MLITE_TEST_HARNESS=1 lifts the conftest GPU skip: this runner is the
    # sanctioned harness for the optional MagiAttention suite (own venv).
    MLITE_TEST_HARNESS=1 "${VENV}/bin/python" -m torch.distributed.run \
        --standalone --nproc_per_node="${NPROC}" -m pytest -q -s \
        experimental/lite/tests/smoke/primitive/test_magi_attention_operator.py
    MLITE_TEST_HARNESS=1 "${VENV}/bin/python" -m torch.distributed.run \
        --standalone --nproc_per_node="${NPROC}" -m pytest -q -s \
        experimental/lite/tests/smoke/model/test_magi_attention_e2e.py
fi

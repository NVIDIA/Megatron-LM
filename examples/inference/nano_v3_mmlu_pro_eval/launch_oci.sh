#!/usr/bin/env bash
# Launch the Nano-v3 evaluation matrix in one OCI-HSG interactive allocation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

[[ -f "${EVAL_IMAGE}" ]] || { echo "Missing container image: ${EVAL_IMAGE}" >&2; exit 1; }
[[ -d "${MEGATRON_ROOT}" ]] || { echo "Missing Megatron worktree: ${MEGATRON_ROOT}" >&2; exit 1; }

SRUN_ARGS=(
    --nodes=1
    --ntasks=1
    --gpus-per-node=4
)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Starting a container step in existing allocation ${SLURM_JOB_ID}."
    SRUN_ARGS+=(--jobid="${SLURM_JOB_ID}" --overlap)
else
    echo "Allocating one OCI-HSG node with four GPUs."
    SRUN_ARGS+=(
        --partition="${SLURM_PARTITION}"
        --account="${SLURM_ACCOUNT}"
        --qos="${SLURM_QOS}"
        --time="${SLURM_TIME}"
        --exclusive
    )
fi

echo "Modes: ${MODES}"
echo "Container: ${EVAL_IMAGE}"

RUN_ID="${RUN_ID:-$(date +'%Y-%m-%d_%H-%M-%S')}"
export RUN_ID
RUN_OUTPUT_DIR="${EVAL_OUTPUT_BASE}/${RUN_ID}"
export EVAL_CONTROL_DIR="${RUN_OUTPUT_DIR}/control"
mkdir -p "${EVAL_CONTROL_DIR}"
MATRIX_LOG="${RUN_OUTPUT_DIR}/matrix.log"

srun \
    "${SRUN_ARGS[@]}" \
    --container-image="${EVAL_IMAGE}" \
    --container-mounts="/home:/home,/lustre:/lustre" \
    --container-workdir="${MEGATRON_ROOT}" \
    bash "${SCRIPT_DIR}/run_matrix.sh" >"${MATRIX_LOG}" 2>&1 &
SRUN_PID=$!

cleanup() {
    if kill -0 "${SRUN_PID}" 2>/dev/null; then
        kill -TERM "${SRUN_PID}" 2>/dev/null || true
        wait "${SRUN_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

wait_for_ready_file() {
    local mode="$1"
    local ready_file="${EVAL_CONTROL_DIR}/${mode}.ready"
    local started="${SECONDS}"
    until [[ -f "${ready_file}" ]]; do
        if ! kill -0 "${SRUN_PID}" 2>/dev/null; then
            echo "Container matrix exited before ${mode} became ready; see ${MATRIX_LOG}." >&2
            return 1
        fi
        if (( SECONDS - started >= STARTUP_TIMEOUT_SECONDS )); then
            echo "Timed out waiting for ${mode}; see ${MATRIX_LOG}." >&2
            return 1
        fi
        sleep 2
    done
}

run_host_client() {
    local mode="$1"
    local ready_file="${EVAL_CONTROL_DIR}/${mode}.ready"
    local client_node
    local base_url
    local client_batch_size="${BATCH_SIZE}"
    local eval_limit_arg="${EVAL_LIMIT:-__FULL__}"
    local handoff_log="${RUN_OUTPUT_DIR}/runtime/${mode}/client-handoff.log"

    if [[ "${mode}" == "dynamo" ]]; then
        client_batch_size="${DYNAMO_BATCH_SIZE}"
    fi
    # lm-eval pads generate requests to the configured batch size. For a
    # numeric smoke-test limit, avoid turning five examples into hundreds of
    # duplicate server requests. Full evaluations retain the configured size.
    if [[ "${eval_limit_arg}" =~ ^[0-9]+$ ]] \
        && (( client_batch_size > eval_limit_arg )); then
        client_batch_size="${eval_limit_arg}"
    fi

    client_node="$(sed -n 's/^client_node=//p' "${ready_file}")"
    base_url="$(sed -n 's/^base_url=//p' "${ready_file}")"
    if [[ -z "${client_node}" || -z "${base_url}" ]]; then
        echo "Malformed readiness file: ${ready_file}" >"${handoff_log}"
        return 1
    fi

    echo "Running ${mode} client on ${client_node}; target ${base_url}; batch size ${client_batch_size}." \
        | tee "${handoff_log}"
    ssh \
        -o BatchMode=yes \
        -o ConnectTimeout=30 \
        "${client_node}" \
        bash "${SCRIPT_DIR}/run_eval_client.sh" \
        "${mode}" "${base_url}" "${RUN_OUTPUT_DIR}" \
        "${eval_limit_arg}" "${client_batch_size}" >>"${handoff_log}" 2>&1
}

IFS=',' read -ra SELECTED_MODES <<<"${MODES}"
for mode in "${SELECTED_MODES[@]}"; do
    mode="${mode//[[:space:]]/}"
    if ! wait_for_ready_file "${mode}"; then
        touch "${EVAL_CONTROL_DIR}/${mode}.client-failed"
        tail -n 120 "${MATRIX_LOG}" >&2 || true
        wait "${SRUN_PID}" 2>/dev/null || true
        exit 1
    fi
    if run_host_client "${mode}"; then
        touch "${EVAL_CONTROL_DIR}/${mode}.client-succeeded"
    else
        touch "${EVAL_CONTROL_DIR}/${mode}.client-failed"
        echo "Client failed for ${mode}; see ${RUN_OUTPUT_DIR}/runtime/${mode}/client-handoff.log." >&2
        tail -n 120 "${RUN_OUTPUT_DIR}/runtime/${mode}/client-handoff.log" >&2 || true
        wait "${SRUN_PID}" 2>/dev/null || true
        exit 1
    fi
done

if ! wait "${SRUN_PID}"; then
    echo "Container matrix failed; see ${MATRIX_LOG}." >&2
    tail -n 120 "${MATRIX_LOG}" >&2 || true
    exit 1
fi
trap - EXIT INT TERM
echo "Completed evaluation run: ${RUN_OUTPUT_DIR}"

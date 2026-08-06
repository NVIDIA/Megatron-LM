#!/usr/bin/env bash
# Run baseline, Dynamo disaggregation, native NCCL, and native NIXL sequentially.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${MEGATRON_ROOT}"

RUN_ID="${RUN_ID:-$(date +'%Y-%m-%d_%H-%M-%S')}"
RUN_OUTPUT_DIR="${EVAL_OUTPUT_BASE}/${RUN_ID}"
RUNTIME_DIR="${RUN_OUTPUT_DIR}/runtime"
CONTROL_DIR="${EVAL_CONTROL_DIR:-${RUN_OUTPUT_DIR}/control}"
mkdir -p "${RUNTIME_DIR}" "${CONTROL_DIR}"

for required_path in \
    "${MODEL_CHECKPOINT}" \
    "${PRETRAINED_CHECKPOINT}" \
    "${TOKENIZER_MODEL}" \
    "${HF_HOME}" \
    "${LM_EVAL_HARNESS_PATH}/env/bin/lm_eval"; do
    [[ -e "${required_path}" ]] || { echo "Required path is missing: ${required_path}" >&2; exit 1; }
done

for command_name in python curl nats-server etcd setsid; do
    command -v "${command_name}" >/dev/null || {
        echo "Required command is missing from the container: ${command_name}" >&2
        exit 1
    }
done

if ! PYTHONPATH="${EVAL_PYTHON_DEPS_DIR}:${PYTHONPATH:-}" \
    python -c 'import hypercorn, quart' >/dev/null 2>&1; then
    echo "Installing the locked Quart HTTP dependency into ${EVAL_PYTHON_DEPS_DIR}."
    mkdir -p "${EVAL_PYTHON_DEPS_DIR}"
    python -m pip install \
        --disable-pip-version-check \
        --upgrade \
        --target "${EVAL_PYTHON_DEPS_DIR}" \
        'quart==0.20.0'
fi
export PYTHONPATH="${EVAL_PYTHON_DEPS_DIR}:${PYTHONPATH:-}"
python -c 'import hypercorn, quart' || {
    echo "Quart/Hypercorn HTTP backend is unavailable after dependency setup." >&2
    exit 1
}

mode_is_selected() {
    local requested_mode="$1"
    local compact_modes="${MODES//[[:space:]]/}"
    [[ ",${compact_modes}," == *",${requested_mode},"* ]]
}

ensure_nixl_cuda_variant() {
    local cuda_version
    local cuda_major
    local nixl_version
    local nixl_variant
    local expected_variant
    local package
    local variant_core_dir
    local variant_library_dir
    local variant_plugin_dir
    local variant_ucx_module_dir

    IFS='|' read -r cuda_version nixl_version nixl_variant < <(
        python "${SCRIPT_DIR}/verify_nixl_cuda.py" --runtime-info
    )
    [[ "${cuda_version}" != "none" ]] || {
        echo "PyTorch does not report a CUDA runtime; NIXL cannot be validated." >&2
        return 1
    }
    cuda_major="${cuda_version%%.*}"
    expected_variant="nixl_cu${cuda_major}"
    package="nixl-cu${cuda_major}==${nixl_version}"

    if [[ "${nixl_variant}" != "${expected_variant}" ]]; then
        echo "NIXL selected ${nixl_variant} for CUDA ${cuda_version}; installing ${package}."
        python -m pip install \
            --disable-pip-version-check \
            --no-deps \
            --upgrade \
            --target "${EVAL_PYTHON_DEPS_DIR}" \
            "${package}"
        IFS='|' read -r cuda_version nixl_version nixl_variant < <(
            python "${SCRIPT_DIR}/verify_nixl_cuda.py" --runtime-info
        )
    fi

    [[ "${nixl_variant}" == "${expected_variant}" ]] || {
        echo "Expected ${expected_variant}, but Python still imports ${nixl_variant}." >&2
        return 1
    }

    variant_core_dir="${EVAL_PYTHON_DEPS_DIR}/.${expected_variant}.mesonpy.libs"
    variant_library_dir="${EVAL_PYTHON_DEPS_DIR}/${expected_variant}.libs"
    variant_plugin_dir="${variant_core_dir}/plugins"
    variant_ucx_module_dir="${variant_library_dir}/ucx"
    [[ -f "${variant_plugin_dir}/libplugin_UCX.so" ]] || {
        echo "The selected NIXL wheel is missing ${variant_plugin_dir}/libplugin_UCX.so." >&2
        return 1
    }
    [[ -f "${variant_ucx_module_dir}/libuct_cuda.so" ]] || {
        echo "The selected NIXL wheel is missing ${variant_ucx_module_dir}/libuct_cuda.so." >&2
        return 1
    }

    # The base image exports paths for its CUDA-12 NIXL/UCX installation.
    # Override both plugin searches so the CUDA-13 wheel's UCX plugin and
    # dynamically loaded CUDA transports are used together.
    export NIXL_PLUGIN_DIR="${variant_plugin_dir}"
    export UCX_MODULE_DIR="${variant_ucx_module_dir}"
    export UCX_TLS="${NIXL_UCX_TLS:-tcp,cuda_ipc,cuda_copy,cma,shm,self}"
    export UCX_MEMTYPE_CACHE="n"
    export LD_LIBRARY_PATH="${variant_core_dir}:${variant_library_dir}:${LD_LIBRARY_PATH:-}"
    export NIXL_CUDA_VARIANT="${nixl_variant}"
    export NIXL_RUNTIME_VERSION="${nixl_version}"

    if ! CUDA_VISIBLE_DEVICES=0 \
        python "${SCRIPT_DIR}/verify_nixl_cuda.py" \
        >"${RUN_OUTPUT_DIR}/nixl-cuda-preflight.log" 2>&1; then
        echo "NIXL CUDA transfer preflight failed:" >&2
        tail -n 100 "${RUN_OUTPUT_DIR}/nixl-cuda-preflight.log" >&2 || true
        return 1
    fi
    tail -n 1 "${RUN_OUTPUT_DIR}/nixl-cuda-preflight.log"
}

if mode_is_selected dynamo; then
    python -c 'import dynamo' || {
        echo "The selected container must provide Dynamo." >&2
        exit 1
    }
fi
if mode_is_selected dynamo || mode_is_selected native_nixl; then
    python -c 'import nixl' || {
        echo "The selected container must provide NIXL." >&2
        exit 1
    }
    ensure_nixl_cuda_variant || exit 1
fi

export NATS_SERVER="${NATS_SERVER:-nats://127.0.0.1:4222}"
export ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:2379}"
export PYTHONUNBUFFERED=1

PIDS=()
CURRENT_MODE=""

log() {
    printf '[matrix %s] %s\n' "$(date +%H:%M:%S)" "$*"
}

start_background() {
    local log_file="$1"
    shift
    setsid "$@" >"${log_file}" 2>&1 &
    PIDS+=("$!")
}

cleanup_current_mode() {
    local pid
    if (( ${#PIDS[@]} == 0 )); then
        return
    fi
    log "Stopping ${CURRENT_MODE:-current mode} processes"
    for pid in "${PIDS[@]}"; do
        kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    done
    for _ in {1..30}; do
        local alive=0
        for pid in "${PIDS[@]}"; do
            kill -0 "${pid}" 2>/dev/null && alive=1
        done
        (( alive == 0 )) && break
        sleep 1
    done
    for pid in "${PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
        fi
        wait "${pid}" 2>/dev/null || true
    done
    PIDS=()
}

trap cleanup_current_mode EXIT
trap 'cleanup_current_mode; exit 130' INT TERM

fail_if_component_exited() {
    local pid
    for pid in "${PIDS[@]}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            log "A ${CURRENT_MODE} component exited during startup"
            return 1
        fi
    done
}

fail_if_cuda_graph_fallback() {
    local mode_dir="${RUNTIME_DIR}/${CURRENT_MODE}"
    local log_file
    for log_file in "${mode_dir}"/*.log; do
        [[ -f "${log_file}" ]] || continue
        if grep -Fq "cuda graph OFF" "${log_file}"; then
            log "CUDA graph fallback detected in ${log_file}"
            grep -F "cuda graph OFF" "${log_file}" | tail -n 5 >&2 || true
            return 1
        fi
    done
}

wait_for_url() {
    local url="$1"
    local description="$2"
    local started="${SECONDS}"
    until curl -fsS "${url}" >/dev/null 2>&1; do
        fail_if_component_exited || return 1
        if (( SECONDS - started >= STARTUP_TIMEOUT_SECONDS )); then
            log "Timed out waiting for ${description}: ${url}"
            return 1
        fi
        sleep 2
    done
    log "Ready: ${description}"
}

wait_for_file() {
    local path="$1"
    local description="$2"
    local started="${SECONDS}"
    until [[ -s "${path}" ]]; do
        fail_if_component_exited || return 1
        if (( SECONDS - started >= STARTUP_TIMEOUT_SECONDS )); then
            log "Timed out waiting for ${description}; expected ${path}"
            return 1
        fi
        sleep 2
    done
    log "Ready: ${description}"
}

dump_mode_logs() {
    local mode_dir="$1"
    local log_file
    for log_file in "${mode_dir}"/*.log; do
        [[ -f "${log_file}" ]] || continue
        echo "===== tail: ${log_file} =====" >&2
        tail -n 100 "${log_file}" >&2 || true
    done
}

COMMON_MODEL_ARGS=(
    --tensor-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --model-provider hybrid
    --inference-max-seq-length "${INFERENCE_MAX_SEQ_LENGTH}"
    --load "${MODEL_CHECKPOINT}"
    --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}"
    --tokenizer-model "${TOKENIZER_MODEL}"
    --micro-batch-size 1
    --moe-router-dtype fp32
    --moe-token-dispatcher-type alltoall
    --use-checkpoint-args
    --bf16
    --attention-backend flash
    --transformer-impl inference_optimized
    --te-rng-tracker
    --inference-rng-tracker
    --cuda-graph-impl local
    --dist-ckpt-strictness log_unexpected
    --inference-dynamic-batching-buffer-size-gb "${INFERENCE_BUFFER_SIZE_GB}"
    --inference-dynamic-batching-max-tokens "${INFERENCE_MAX_TOKENS}"
    --enable-chunked-prefill
    --inference-dynamic-batching-prefix-caching
    --inference-dynamic-batching-prefix-caching-eviction-policy lru
    --inference-dynamic-batching-prefix-caching-mamba-gb "${MAMBA_PREFIX_CACHE_GB}"
    --inference-logging-step-interval "${INFERENCE_LOGGING_STEP_INTERVAL}"
    --inference-dynamic-batching-num-cuda-graphs "${INFERENCE_NUM_CUDA_GRAPHS}"
    --inference-cuda-graph-all-prefills
    --inference-cuda-graph-scope block
    --inference-dynamic-batching-max-requests "${INFERENCE_MAX_REQUESTS}"
    --return-log-probs
)

publish_client_ready() {
    local mode="$1"
    local base_url="http://127.0.0.1:${SERVER_PORT}"
    local ready_file="${CONTROL_DIR}/${mode}.ready"
    local temporary_file="${ready_file}.tmp.$$"
    if [[ "${mode}" == "dynamo" ]]; then
        base_url="${base_url}/v1"
    fi
    {
        echo "client_node=${SLURMD_NODENAME:-$(hostname)}"
        echo "base_url=${base_url}"
    } >"${temporary_file}"
    mv "${temporary_file}" "${ready_file}"
    log "Published client readiness: ${ready_file}"
}

wait_for_client() {
    local mode="$1"
    local success_file="${CONTROL_DIR}/${mode}.client-succeeded"
    local failure_file="${CONTROL_DIR}/${mode}.client-failed"
    local started="${SECONDS}"
    until [[ -f "${success_file}" ]]; do
        if [[ -f "${failure_file}" ]]; then
            log "Host client failed for ${mode}"
            return 1
        fi
        fail_if_cuda_graph_fallback || return 1
        fail_if_component_exited || return 1
        if (( SECONDS - started >= CLIENT_TIMEOUT_SECONDS )); then
            log "Timed out waiting for the host client for ${mode}"
            return 1
        fi
        sleep 2
    done
    fail_if_cuda_graph_fallback || return 1
    log "Host client completed: ${mode}"
}

launch_no_disagg() {
    local mode_dir="$1"
    start_background "${mode_dir}/server.log" \
        env CUDA_VISIBLE_DEVICES=0,1,2,3 \
        python -m torch.distributed.run --nproc-per-node=4 \
        -m tools.run_dynamic_text_generation_server \
        "${COMMON_MODEL_ARGS[@]}" \
        --expert-model-parallel-size 4 \
        --port "${SERVER_PORT}"
    wait_for_url "http://127.0.0.1:${SERVER_PORT}/health" "aggregated Megatron server"
}

resolve_dynamo_model_metadata() {
    if [[ -d "${DYNAMO_MODEL}" ]]; then
        printf '%s\n' "${DYNAMO_MODEL}"
        return
    fi
    python -c \
        'import asyncio, sys
from dynamo.llm import fetch_model
async def main():
    return await fetch_model(sys.argv[1], ignore_weights=True)
print(asyncio.run(main()))' \
        "${DYNAMO_MODEL}" | tail -n 1
}

launch_dynamo() {
    local mode_dir="$1"
    local metadata_dir
    metadata_dir="$(resolve_dynamo_model_metadata)"
    [[ -d "${metadata_dir}" ]] || {
        echo "Dynamo metadata resolution did not return a directory: ${metadata_dir}" >&2
        return 1
    }

    start_background "${mode_dir}/nats.log" \
        nats-server --jetstream --store_dir "${mode_dir}/nats-data" --port 4222 -m 8222
    start_background "${mode_dir}/etcd.log" \
        etcd --data-dir "${mode_dir}/etcd-data" \
        --listen-client-urls http://0.0.0.0:2379 \
        --advertise-client-urls http://0.0.0.0:2379
    wait_for_url "http://127.0.0.1:8222/healthz" "NATS"
    wait_for_url "http://127.0.0.1:2379/health" "etcd"

    start_background "${mode_dir}/worker-prefill.log" \
        env CUDA_VISIBLE_DEVICES=0,1 \
        python -m megatron.inference.integrations.dynamo \
        --role prefill \
        --component prefill \
        --model "${metadata_dir}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --nproc-per-node 2 \
        --coordinator-host 127.0.0.1 \
        --coordinator-port 5555 \
        --worker-id-file "${mode_dir}/prefill-worker.json" \
        --megatron-root "${MEGATRON_ROOT}" \
        -- \
        "${COMMON_MODEL_ARGS[@]}" \
        --expert-model-parallel-size 2 \
        --inference-dynamic-batching-prefix-caching-eviction-policy \
        "${DYNAMO_PREFILL_EVICTION_POLICY:-lru}"

    start_background "${mode_dir}/worker-decode.log" \
        env CUDA_VISIBLE_DEVICES=2,3 \
        python -m megatron.inference.integrations.dynamo \
        --role decode \
        --component backend \
        --model "${metadata_dir}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --nproc-per-node 2 \
        --coordinator-host 127.0.0.1 \
        --coordinator-port 5556 \
        --worker-id-file "${mode_dir}/decode-worker.json" \
        --megatron-root "${MEGATRON_ROOT}" \
        -- \
        "${COMMON_MODEL_ARGS[@]}" \
        --expert-model-parallel-size 2 \
        --inference-dynamic-batching-prefix-caching-eviction-policy \
        "${DYNAMO_DECODE_EVICTION_POLICY:-lru}"

    wait_for_file "${mode_dir}/prefill-worker.json" "Dynamo prefill worker"
    wait_for_file "${mode_dir}/decode-worker.json" "Dynamo decode worker"

    start_background "${mode_dir}/frontend.log" \
        python -m dynamo.frontend \
        --http-port "${SERVER_PORT}" \
        --router-mode kv \
        --request-plane nats \
        --event-plane nats
    wait_for_url "http://127.0.0.1:${SERVER_PORT}/v1/models" "Dynamo frontend"
}

launch_native() {
    local mode_dir="$1"
    local backend="$2"
    local cuda_device_max_connections="${CUDA_DEVICE_MAX_CONNECTIONS}"
    local shard_spec
    shard_spec="tp=1,pp=1,ep=2,expt_tp=1,dp=2,role=prefill+tp=1,pp=1,ep=2,expt_tp=1,dp=2,role=decode"

    if [[ "${backend}" == "nccl" ]]; then
        cuda_device_max_connections="${NATIVE_NCCL_CUDA_DEVICE_MAX_CONNECTIONS}"
    fi

    start_background "${mode_dir}/server.log" \
        env CUDA_VISIBLE_DEVICES=0,1,2,3 \
        CUDA_DEVICE_MAX_CONNECTIONS="${cuda_device_max_connections}" \
        python -m torch.distributed.run --nproc-per-node=4 \
        -m examples.inference.launch_inference_server \
        "${COMMON_MODEL_ARGS[@]}" \
        --expert-model-parallel-size 2 \
        --inference-shards "${shard_spec}" \
        --disagg-kv-transport-backend "${backend}" \
        --port "${SERVER_PORT}"
    wait_for_url "http://127.0.0.1:${SERVER_PORT}/health" "native ${backend} server"
}

run_mode() {
    local mode="$1"
    local mode_dir="${RUNTIME_DIR}/${mode}"
    CURRENT_MODE="${mode}"
    PIDS=()
    mkdir -p "${mode_dir}"
    log "Starting mode: ${mode}"

    case "${mode}" in
        no_disagg) launch_no_disagg "${mode_dir}" || return 1 ;;
        dynamo) launch_dynamo "${mode_dir}" || return 1 ;;
        native_nccl) launch_native "${mode_dir}" nccl || return 1 ;;
        native_nixl) launch_native "${mode_dir}" nixl || return 1 ;;
        *) echo "Unknown mode: ${mode}" >&2; return 2 ;;
    esac

    publish_client_ready "${mode}"
    if ! wait_for_client "${mode}"; then
        dump_mode_logs "${mode_dir}"
        return 1
    fi
    cleanup_current_mode
    log "Completed mode: ${mode}"
}

{
    echo "run_id=${RUN_ID}"
    echo "commit=$(git rev-parse HEAD)"
    echo "branch=$(git branch --show-current)"
    echo "hostname=$(hostname)"
    echo "modes=${MODES}"
    echo "batch_size=${BATCH_SIZE}"
    echo "dynamo_batch_size=${DYNAMO_BATCH_SIZE}"
    echo "eval_limit=${EVAL_LIMIT:-full}"
    echo "inference_num_cuda_graphs=${INFERENCE_NUM_CUDA_GRAPHS}"
    echo "inference_cuda_graph_all_prefills=true"
    echo "inference_logging_step_interval=${INFERENCE_LOGGING_STEP_INTERVAL}"
    echo "dynamo_prefill_eviction_policy=${DYNAMO_PREFILL_EVICTION_POLICY:-lru}"
    echo "dynamo_decode_eviction_policy=${DYNAMO_DECODE_EVICTION_POLICY:-lru}"
    echo "cuda_device_max_connections=${CUDA_DEVICE_MAX_CONNECTIONS}"
    echo "native_nccl_cuda_device_max_connections=${NATIVE_NCCL_CUDA_DEVICE_MAX_CONNECTIONS}"
    echo "python_deps_dir=${EVAL_PYTHON_DEPS_DIR}"
    echo "nixl_version=${NIXL_RUNTIME_VERSION:-not-required}"
    echo "nixl_cuda_variant=${NIXL_CUDA_VARIANT:-not-required}"
    echo "hf_datasets_cache=${HF_DATASETS_CACHE}"
    echo "model_checkpoint=${MODEL_CHECKPOINT}"
    echo "pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
    echo "container=${EVAL_IMAGE}"
} >"${RUN_OUTPUT_DIR}/manifest.txt"
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader \
    >"${RUN_OUTPUT_DIR}/gpus.txt" 2>/dev/null || true

IFS=',' read -ra SELECTED_MODES <<<"${MODES}"
for mode in "${SELECTED_MODES[@]}"; do
    mode="${mode//[[:space:]]/}"
    if ! run_mode "${mode}"; then
        dump_mode_logs "${RUNTIME_DIR}/${mode}"
        exit 1
    fi
done

python "${SCRIPT_DIR}/summarize_results.py" "${RUN_OUTPUT_DIR}" \
    | tee "${RUN_OUTPUT_DIR}/summary.txt"
log "All evaluations completed: ${RUN_OUTPUT_DIR}"

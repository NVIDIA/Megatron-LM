#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:4
#
# Serve Nemotron nano-3.5, Super v3 or Ultra v3 with vLLM (4 GPUs per node), as the vLLM
# counterpart of launch_minf.sh: same experiment directory, same port (5000), and the same
# handshake with client.sh (config.env, then <exp dir>/host when ready; <exp dir>/done stops it).
#
# One vLLM engine spans the whole job: TP within each data-parallel rank, DP over the rest of the
# GPUs, and --enable-expert-parallel, so the experts are sharded over all of them (EP = all GPUs).
# Node 0 runs the API server; the other nodes join it headless. Multi-node jobs need
# `--segment <N>` with `-N <N>` to stay in one NVLink domain (run.sh passes it).
#
# Run from the Megatron-LM repo root (or through run.sh with FRAMEWORK=vllm):
#   source inference-bench/cluster.env
#   sbatch -N 2 --segment 2 --export=ALL,EXP_NAME=vllm,MODEL=superv3 inference-bench/server/launch_vllm.sh
#
# Knobs (environment variables):
#   MODEL           nanov3p5 | superv3 | ultrav3                       (default: nanov3p5)
#   TP              tensor parallel size per DP rank; > 4 spans nodes   (default: 4; ultrav3: 8)
#   PREFIX_CACHING  1 = vLLM prefix caching                            (default: 1)
#   FP8_RECIPE      mxfp8 = quantize the BF16 checkpoint to MXFP8 at load (--quantization mxfp8),
#                   as FP8_RECIPE=mxfp8 does for launch_minf.sh; unset = BF16  (default: unset)
#   EXTRA_ARGS      extra `vllm serve` flags, e.g. "--max-num-seqs 512"; later flags win, so
#                   "--all2all-backend allgather_reducescatter" restores vLLM's default EP dispatch
# Other launch_minf.sh knobs (EP, DISPATCHER, BUFFER_GB, ...) do not apply and are ignored.

SCRIPT=inference-bench/server/launch_vllm.sh

# ── Outer: inside sbatch, launch one task per node in the vLLM container ───────
if [ -z "${INFERENCE_BENCH_INNER:-}" ]; then
    cd "${SLURM_SUBMIT_DIR:-$PWD}"
    source "${INFERENCE_BENCH_CONFIG:-inference-bench/cluster.env}"
    export EXP_DIR="inference-bench/experiments/$(date +%Y%m%d_%H%M%S)_${EXP_NAME:-run}"
    mkdir -p "${EXP_DIR}"
    echo "Experiment directory: ${EXP_DIR}"
    # Nodes run a private copy, as in launch_minf.sh.
    cp "${SCRIPT}" "${EXP_DIR}/launch_vllm.sh"
    export INFERENCE_BENCH_INNER=1
    NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
    srun -l -N "${NUM_NODES}" -n "${NUM_NODES}" --ntasks-per-node 1 --cpus-per-task 144 \
        --container-image "${INFERENCE_BENCH_VLLM_IMAGE:?set in cluster.env}" \
        --container-mounts "${INFERENCE_BENCH_MOUNTS:?set in cluster.env}" \
        --container-workdir="${PWD}" bash "${EXP_DIR}/launch_vllm.sh"
    exit $?
fi

# ── Inner: one node ────────────────────────────────────────────────────────────
NODE_RANK=${SLURM_NODEID:-0}
NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
export VLLM_ENGINE_READY_TIMEOUT_S=3600
# A broken transformers under ~/.local would shadow the container's copy in the API-server workers.
export PYTHONNOUSERSITE=1
export TRITON_CACHE_DIR=/tmp/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
export VLLM_CACHE_ROOT=/tmp/vllm_cache
# FlashInfer autotune results (keyed by FlashInfer version, GPU and model / parallel config) persist
# across jobs when cluster.env sets a shared directory; otherwise every job retunes (~10 min).
[ -n "${INFERENCE_BENCH_FI_AUTOTUNE_DIR:-}" ] && export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=${INFERENCE_BENCH_FI_AUTOTUNE_DIR}

# vLLM's modular TRT-LLM FP8/MXFP8 experts (used with an EP all2all backend) call FlashInfer's
# trtllm_fp8_block_scale_routed_moe without activation_type, so FlashInfer assumes gated SwiGLU and
# rejects non-gated (squared-ReLU) experts such as Nemotron's: "intermediate_size *
# intermediate_size_factor == Mn (5376 vs. 2688)". Pass the layer's activation, as the monolithic
# path already does. Patches this job's container copy of vLLM only.
if [ "${FP8_RECIPE:-}" == "mxfp8" ]; then
    python3 - <<'EOF' || { echo "vLLM MXFP8 activation patch failed"; exit 1; }
import re
import vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe as m

path = m.__file__
src = open(path).read()
call = re.search(r"\n( *)(\w+ = flashinfer\.fused_moe\.trtllm_fp8_block_scale_routed_moe\(\n)", src)
assert call, "routed MoE call not found"
if "activation_type=" in src[call.end():src.index("\n" + call.group(1) + ")", call.end())]:
    print("vLLM MXFP8 activation patch: already passes activation_type")
else:
    indent = call.group(1)
    patched = (
        f"\n{indent}from vllm.model_executor.layers.quantization.utils.flashinfer_utils "
        f"import activation_to_flashinfer_int as _act_to_fi\n"
        f"{indent}{call.group(2)}{indent}    activation_type=_act_to_fi(activation),\n"
    )
    open(path, "w").write(src[:call.start()] + patched + src[call.end():])
    print(f"vLLM MXFP8 activation patch: applied to {path}")
EOF
fi

case "${MODEL:-nanov3p5}" in
    nanov3p5) HF_MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 ;;
    superv3) HF_MODEL=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 ;;
    ultrav3) HF_MODEL=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 ;;
    *) echo "Unknown MODEL ${MODEL}"; exit 1 ;;
esac

# Ultra v3 runs TP8 (a TP group spans two nodes); the others default to TP4, as in launch_minf.sh.
if [ "${MODEL:-nanov3p5}" == "ultrav3" ]; then TP=${TP:-8}; else TP=${TP:-4}; fi
if (( (4 * NUM_NODES) % TP )); then
    echo "TP=${TP} must divide the job's $((4 * NUM_NODES)) GPUs"
    exit 1
fi
DP=$((4 * NUM_NODES / TP))

ARGS=(
    --model "${HF_MODEL}" --dtype bfloat16 --trust-remote-code --port 5000
    --max-model-len 262144
    --tensor-parallel-size "${TP}" --data-parallel-size "${DP}" --enable-expert-parallel
    # The EP dispatch InferenceX's vLLM GB200 recipes use. With one DP rank (EP = TP) vLLM ignores
    # it and skips EP communication (tokens replicated across TP; the TP all-reduce combines).
    --all2all-backend flashinfer_nvlink_one_sided
)
# The API server (node 0) splits reasoning from the answer; headless nodes run no API server.
[ "${NODE_RANK}" == "0" ] && ARGS+=(--reasoning-parser nemotron_v3)
[ "${FP8_RECIPE:-}" == "mxfp8" ] && ARGS+=(--quantization mxfp8)
if [ "${PREFIX_CACHING:-1}" == "1" ]; then
    ARGS+=(--enable-prefix-caching)
else
    ARGS+=(--no-enable-prefix-caching)
fi

# Multi-node: node 0 publishes its address and hosts the API server; the others join headless.
HEAD_IP_FILE="${EXP_DIR}/head_ip"
if [ "${NUM_NODES}" -gt 1 ]; then
    if [ "${NODE_RANK}" == "0" ]; then
        getent hosts "$(hostname)" | awk '{print $1}' | head -1 > "${HEAD_IP_FILE}"
    else
        until [ -s "${HEAD_IP_FILE}" ]; do sleep 2; done
        ARGS+=(--headless)
    fi
    # One multi-node world over all the job's nodes (TP groups may span nodes). The per-node
    # --data-parallel-size-local wiring does not work here: with one DP rank per node, vLLM
    # switches to a server per node and rejects --headless.
    ARGS+=(--nnodes "${NUM_NODES}" --node-rank "${NODE_RANK}" --master-addr "$(cat "${HEAD_IP_FILE}")"
           --master-port 29501)
fi
# shellcheck disable=SC2206  # EXTRA_ARGS is a flag list
ARGS+=(${EXTRA_ARGS:-})

LOG="${EXP_DIR}/server_node${NODE_RANK}.log"
if [ "${NODE_RANK}" == "0" ]; then
    # client.sh finds this directory by the SLURM_JOB_ID line and reads TOKENIZER from it.
    cat > "${EXP_DIR}/config.env" <<EOF
EXP_NAME=${EXP_NAME:-run}
SLURM_JOB_ID=${SLURM_JOB_ID}
TOKENIZER=${HF_MODEL}
FRAMEWORK=vllm MODEL=${MODEL:-nanov3p5} NODES=${NUM_NODES} TP=${TP} DP=${DP} EP=$((TP * DP)) PREFIX_CACHING=${PREFIX_CACHING:-1} FP8_RECIPE=${FP8_RECIPE:-}
CMD=vllm serve ${ARGS[*]}
EOF
fi

vllm serve "${ARGS[@]}" > "${LOG}" 2>&1 &
SERVER_PID=$!

# Serve until <exp dir>/done appears (written by client.sh) or the server dies.
serve_until_done() {
    while kill -0 "${SERVER_PID}" 2>/dev/null && [ ! -f "${EXP_DIR}/done" ]; do sleep 2; done
    kill "${SERVER_PID}" 2>/dev/null
    wait "${SERVER_PID}"
    exit 0
}

[ "${NODE_RANK}" != "0" ] && serve_until_done

# Node 0: wait for the API server, then advertise it to client.sh.
until grep -q "Application startup complete." "${LOG}" 2>/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Server exited during startup; see ${LOG}"
        touch "${EXP_DIR}/done"
        exit 1
    fi
    sleep 5
done
echo "Server ready."
echo "Server reachable at: http://$(hostname):5000/v1"
hostname > "${EXP_DIR}/host"
serve_until_done

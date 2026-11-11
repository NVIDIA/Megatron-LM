#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:4
#
# Serve Nemotron nano-3.5, Super v3 or Ultra v3 with the mcore dynamic inference server
# (4 GPUs per node). nano-3.5 and Super v3 fit on one node; Ultra v3 needs `sbatch -N 2`.
# Multi-node jobs must sit in one NVLink (NVL72) domain for the NVLS / symmetric-memory dispatch:
# pass `--segment <N>` with `-N <N>` (otherwise Slurm may split the nodes across racks and the
# symmetric-memory init fails with "CUDA driver error: invalid resource handle").
#
# Cluster-specific settings (slurm account/partition/QoS, container image, checkpoint and HF
# cache paths) come from inference-bench/cluster.env; copy cluster.env.example and fill it in.
#
# Run from the Megatron-LM repo root:
#   source inference-bench/cluster.env
#   sbatch inference-bench/server/launch_minf.sh
#   sbatch --export=ALL,EXP_NAME=a2a,DISPATCHER=nvl_a2a,COMBINE_PRECISION=bf16 inference-bench/server/launch_minf.sh
#
# Knobs (environment variables):
#   EXP_NAME           run label                                    (default: run)
#   MODEL              nanov3p5 | superv3 | ultrav3                 (default: nanov3p5)
#   TP                 tensor parallel size; data parallel = world / TP   (default: 4; ultrav3: 8)
#   ETP                expert tensor parallel size                      (default: 1)
#   EP                 expert parallel size; experts are replicated over world / (EP x ETP)
#                      expert-data-parallel copies, so EP < world / ETP keeps a copy per group of
#                      EP x ETP GPUs (narrow EP)                         (default: world / ETP)
#   BUFFER_GB          dynamic-batching KV/state buffer per GPU   (default: 50; ultrav3: 25)
#   MAMBA_RATIO        fraction of BUFFER_GB for live Mamba states; with ~100 MB of state per
#                      request on Ultra, this sets the concurrent-request cap   (default: 0.1)
#   MAMBA_PREFIX_GB    Mamba prefix-cache buffer per GPU, with PREFIX_CACHING=1
#                      (default: 50; ultrav3: 25)
#   DISPATCHER         MoE EP dispatcher: nccl | nvls | nvl_a2a     (default: nvls)
#   COMBINE_PRECISION  MoE combine precision: fp32 | bf16           (default: fp32)
#   GEMM_BACKEND       routed-expert grouped GEMM: vllm | trtllm_bf16_routed | flashinfer | torch
#                      (default: trtllm_bf16_routed)
#   ASYNC              1 = async scheduling                         (default: 1)
#   PREFIX_CACHING     1 = prefix caching, LRU eviction             (default: 1)
#   SAMPLING           sampling backend: flashinfer | torch         (default: flashinfer)
#   MAX_REQUESTS       --inference-dynamic-batching-max-requests    (default: unset)
#   MAX_TOKENS         max tokens per engine step (chunked-prefill budget)  (default: 8192)
#   CUDA_GRAPH_MAX_TOKENS  largest token count captured in CUDA graphs, so prefill / mixed steps up
#                      to it are graphed too (default: 2048; the server's own default is 512)
#   LOG_INTERVAL       engine step-log interval; 1 = log every step  (default: 1000)
#   FP8_RECIPE         mxfp8 = MXFP8 weights, quantized from the BF16 checkpoint at load
#                      (--fp8-format e4m3 --fp8-recipe mxfp8 --fp8-param-gather, which
#                      inference_optimized requires); unset = BF16          (default: unset)
#   SHARED_EXPERT_OVERLAP  1 = run the shared experts on a side stream, overlapped with the
#                      routed experts; 0 = in order                         (default: 1)
#   MCORE_ROUTING_COUNTS  1 = EP rank 0 counts each logged step's expert routing: a summary on
#                      its step log and the raw per-layer, per-expert counts in
#                      <exp dir>/routing_counts.jsonl (MCORE_ROUTING_COUNTS_PATH overrides)
#   EVAL_MODE          1 = --eval-mode: chat responses skip echoing prompt token IDs, which
#                      speeds up evals with long contexts             (default: 1)
#   PROMPT_LOGPROBS    1 = materialize logits for every prompt token so /v1/completions can
#                      return prompt log-probs (echo + logprobs); costs memory (default: 0)
#   BENCH              batch sizes for the in-job client, e.g. "1 16 64". When set, the job
#                      runs the sweep and exits; when unset, it serves until the time limit or
#                      until <exp dir>/done appears. The in-job client shares rank 0's CPUs with
#                      the server; for clean timings run the sweep from server/client.sh instead.
#   BENCH_DATASET      gsm8k (~100-token questions) | sharegpt (repo-root
#                      sharegpt_filtered_benchmark.json, <= 1024 tokens)   (default: gsm8k)
#   BENCH_MIN_ISL      sharegpt only: keep prompts with >= this many tokens (default: 0)
#   BENCH_ISL          join consecutive prompts to ~this many tokens (default: 0 = off)
#   BENCH_EXACT_ISLS   prompt lengths in exact tokens, e.g. "1024 4096 8000" (default: unset).
#                      With BENCH=1 each request is a single prefill step (keep <= 8192).
#   BENCH_OSL / BENCH_ITERS / BENCH_WARMUP   output tokens / timed / warmup iterations
#                                             (default: 1024 / 3 / 1)
#   PD                 1 = prefill/decode disaggregation in this one job: the first PREFILL_NODES
#                      nodes form the prefill pool, the rest the decode pool, each its own
#                      torch.distributed world and coordinator. Both pools run with prefix
#                      caching (required by the KV hand-off) and without the Mamba prefix cache
#                      (MAMBA_PREFIX_GB=0). The prefill pool serves no HTTP and publishes its
#                      coordinator address to <exp dir>/prefill_coordinator; the decode pool's
#                      rank 0 hosts the frontend, which routes each request through both pools.
#                      KV and Mamba state move over NIXL. Any PREFILL_<KNOB> overrides <KNOB> on
#                      the prefill pool (PREFILL_TP, PREFILL_EP, ...). The prefill pool logs into
#                      <exp dir>/prefill/.                        (default: 0 = aggregated)
#   PREFILL_NODES      with PD=1: prefill-pool nodes                         (default: 1)
#   DONE_FILE          file that stops the server when it appears  (default: <exp dir>/done)
#   EXP_DIR            experiment directory (default: a new timestamped one)
#   PROFILE            1 = run every rank under nsys (capture range = cudaProfilerApi) and have the
#                      client wrap the timed iterations of the first BENCH batch size in
#                      /start_profile + /stop_profile; the server exits when profiling stops.
#                      Traces: <exp dir>/trace_rank<r>.nsys-rep            (default: 0)
#   PROFILE_ALL        with PROFILE=1: warm every BENCH batch size first, then profile the timed
#                      iterations of ALL batch sizes in one capture              (default: 0)
#   BENCH_PAUSE        seconds idle between batch sizes, to tell them apart in a trace (default: 0)
#   Prefill-heavy runs: BENCH_OSL=1 with BENCH_DATASET=sharegpt BENCH_MIN_ISL=512, or BENCH_ISL=2048.
#
# Output: inference-bench/experiments/<timestamp>_<EXP_NAME>/ with config.env, server_rank*.log and,
# with BENCH, bench.log + bench.json. The server listens on port 5000 (OpenAI API at /v1).

SCRIPT=inference-bench/server/launch_minf.sh

# ── Outer: inside sbatch, launch one task per GPU in the container ─────────────
if [ -z "${INFERENCE_BENCH_INNER:-}" ]; then
    cd "${SLURM_SUBMIT_DIR:-$PWD}"
    # Exported settings reach the ranks through srun's environment.
    source "${INFERENCE_BENCH_CONFIG:-inference-bench/cluster.env}"
    export EXP_DIR="${EXP_DIR:-inference-bench/experiments/$(date +%Y%m%d_%H%M%S)_${EXP_NAME:-run}}"
    export DONE_FILE="${DONE_FILE:-${EXP_DIR}/done}"
    mkdir -p "${EXP_DIR}"
    [ "${PD:-0}" == "1" ] && mkdir -p "${EXP_DIR}/prefill"
    echo "Experiment directory: ${EXP_DIR}"
    # Ranks run a private copy: bash reads scripts incrementally, so editing the original
    # while a job is running would corrupt that job's ranks.
    cp "${SCRIPT}" "${EXP_DIR}/launch_minf.sh"
    # Per-step expert routing counts (EP rank 0) default into the experiment directory.
    if [ "${MCORE_ROUTING_COUNTS:-0}" == "1" ]; then
        export MCORE_ROUTING_COUNTS_PATH=${MCORE_ROUTING_COUNTS_PATH:-${EXP_DIR}/routing_counts.jsonl}
    fi
    export INFERENCE_BENCH_INNER=1
    NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
    if [ "${MODEL:-nanov3p5}" == "ultrav3" ] && [ "${NUM_NODES}" -lt 2 ]; then
        echo "MODEL=ultrav3 needs at least 2 nodes (sbatch -N 2)"
        exit 1
    fi
    # One task per GPU on every node; global rank 0 hosts the HTTP server.
    srun -l -N "${NUM_NODES}" -n $((4 * NUM_NODES)) --ntasks-per-node 4 --cpus-per-task 36 \
        --container-image "${INFERENCE_BENCH_SERVE_IMAGE:?set in cluster.env}" \
        --container-name "inference_bench_${SLURM_JOB_ID}" \
        --container-mounts "${INFERENCE_BENCH_MOUNTS:?set in cluster.env}" \
        --container-workdir="${PWD}" bash "${EXP_DIR}/launch_minf.sh"
    exit $?
fi

# ── Inner: one rank ────────────────────────────────────────────────────────────
export RANK=${SLURM_PROCID} LOCAL_RANK=${SLURM_LOCALID} WORLD_SIZE=${SLURM_NTASKS}
# Multi-node runs use the rendezvous address the cluster exports; fall back to localhost.
export MASTER_ADDR=${MASTER_ADDR:-localhost} MASTER_PORT=${MASTER_PORT:-29500}
DONE_FILE=${DONE_FILE:-${EXP_DIR}/done}
POOL_NODES=${SLURM_JOB_NUM_NODES:-1}
DISAGG_ROLE=

if [ "${PD:-0}" == "1" ]; then
    # Split the job's nodes into the two pools, each its own torch.distributed world.
    PREFILL_NODES=${PREFILL_NODES:-1}
    PD_COORDINATOR_FILE="${EXP_DIR}/prefill_coordinator"
    if (( SLURM_NODEID < PREFILL_NODES )); then
        DISAGG_ROLE=prefill
        POOL_NODES=${PREFILL_NODES}
        FIRST_TASK=0
        for var in $(compgen -v PREFILL_); do
            [ "${var}" == "PREFILL_NODES" ] || export "${var#PREFILL_}=${!var}"
        done
        EXP_DIR="${EXP_DIR}/prefill"
        EXP_NAME="${EXP_NAME:-run}_prefill"
        MASTER_PORT=29600
    else
        DISAGG_ROLE=decode
        POOL_NODES=$(( ${SLURM_JOB_NUM_NODES} - PREFILL_NODES ))
        FIRST_TASK=$(( 4 * PREFILL_NODES ))
        MASTER_PORT=29500
    fi
    RANK=$(( SLURM_PROCID - FIRST_TASK ))
    WORLD_SIZE=$(( 4 * POOL_NODES ))
    # Pool-local rendezvous: the pool's rank 0 publishes its host.
    MASTER_FILE="${EXP_DIR}/.master_addr_${SLURM_JOB_ID}"
    if [ "${RANK}" == "0" ]; then
        hostname > "${MASTER_FILE}.tmp" && mv "${MASTER_FILE}.tmp" "${MASTER_FILE}"
    fi
    until [ -s "${MASTER_FILE}" ]; do sleep 1; done
    MASTER_ADDR=$(cat "${MASTER_FILE}")
    export RANK WORLD_SIZE MASTER_ADDR MASTER_PORT EXP_DIR EXP_NAME
fi
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TRITON_CACHE_DIR=/tmp/triton_cache_${RANK}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_${RANK}

if [ -n "${DISAGG_ROLE}" ]; then
    # The KV hand-off needs prefix caching; math prompts share no prefixes, so the Mamba prefix
    # cache would only take memory (and the decode pool must not have one).
    PREFIX_CACHING=1
    MAMBA_PREFIX_GB=0
    # The cluster sets UCX_TLS=tcp, which keeps NIXL off the GPU transports (cuda_ipc, cuda_copy,
    # RDMA); let UCX choose.
    unset UCX_TLS
    # NIXL from a pip --target directory when the image has none.
    [ -n "${NIXL_PYTHONPATH:-}" ] && export PYTHONPATH="${NIXL_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
    # `import torch` loads HPC-X's UCX (through UCC); NIXL bundles its own, and with two copies in
    # one process GPU-memory registration fails. Preload NIXL's libucs so a single copy serves both.
    NIXL_UCS=$(python -c 'import glob, importlib.util, os
spec = importlib.util.find_spec("nixl_cu13")
root = os.path.dirname(os.path.dirname(spec.origin)) if spec else ""
print(next(iter(sorted(glob.glob(os.path.join(root, "nixl_cu13.libs", "libucs-*.so*")))), ""))')
    if [ -z "${NIXL_UCS}" ]; then
        echo "DISAGG_ROLE needs NIXL (nixl + nixl-cu13); use the +nixl image or set NIXL_PYTHONPATH"
        touch "${DONE_FILE}"
        exit 1
    fi
    export LD_PRELOAD="${NIXL_UCS}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

# Ultra v3 runs TP8 x EP8 on 2 nodes; the others default to TP4.
if [ "${MODEL:-nanov3p5}" == "ultrav3" ]; then TP=${TP:-8}; else TP=${TP:-4}; fi
ETP=${ETP:-1}
EP=${EP:-$((WORLD_SIZE / ETP))}
if (( WORLD_SIZE % (EP * ETP) )); then
    echo "EP x ETP = $((EP * ETP)) must divide the world size ${WORLD_SIZE}"
    exit 1
fi

ARGS=(
    --micro-batch-size 1 --bf16 --te-rng-tracker --inference-rng-tracker
    --inference-dynamic-batching
    --inference-dynamic-batching-unified-memory-level 0
    --inference-dynamic-batching-max-tokens "${MAX_TOKENS:-8192}"
    --inference-dynamic-batching-mamba-memory-ratio "${MAMBA_RATIO:-0.1}"
    --inference-dynamic-batching-num-cuda-graphs -1
    --inference-dynamic-batching-sampling-backend "${SAMPLING:-flashinfer}"
    --inference-logging-step-interval "${LOG_INTERVAL:-1000}"
    --enable-chunked-prefill --use-flashinfer-fused-rope --sequence-parallel
    --cuda-graph-impl local --cuda-graph-scope full_iteration_inference
    --transformer-impl inference_optimized
    --inference-max-seq-length 262144 --seq-length 262144 --max-position-embeddings 262144
    # Parallelism and MoE dispatch.
    --tensor-model-parallel-size "${TP}"
    --expert-model-parallel-size "${EP}"
    --expert-tensor-parallel-size "${ETP}"
    --inference-moe-token-dispatcher-type "${DISPATCHER:-nvls}"
    --inference-moe-combine-precision "${COMBINE_PRECISION:-fp32}"
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --inference-grouped-gemm-backend "${GEMM_BACKEND:-trtllm_bf16_routed}"
    --inference-use-synchronous-zmq-collectives
    --model-provider mamba
    --dist-ckpt-strictness log_unexpected
    --tokenizer-type HuggingFaceTokenizer --no-use-tokenizer-model-from-checkpoint-args
    --parsers nemotron-v3-reasoning qwen3-coder-tool
    --mamba-inference-ssm-states-dtype fp32 --moe-router-dtype fp32 --mtp-use-repeated-layer
)

case "${MODEL:-nanov3p5}" in
nanov3p5)
    TOKENIZER=${NANOV3P5_TOKENIZER:?set in cluster.env}
    ARGS+=(
    --load "${NANOV3P5_CHECKPOINT:?set in cluster.env}"
    --tokenizer-model "${TOKENIZER}"
    --moe-permute-fusion
    --hybrid-layer-pattern 'MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME/*E'
    --vocab-size 131072 --make-vocab-size-divisible-by 128
    --hidden-size 2688 --ffn-hidden-size 1856
    --num-attention-heads 32 --group-query-attention --num-query-groups 2 --kv-channels 128
    --normalization RMSNorm --norm-epsilon 1e-5 --disable-bias-linear --squared-relu
    --position-embedding-type none --untie-embeddings-and-output-weights
    --mamba-state-dim 128 --mamba-head-dim 64 --mamba-num-groups 8 --mamba-num-heads 64
    --num-experts 128 --moe-ffn-hidden-size 1856 --moe-router-topk 6
    --moe-shared-expert-intermediate-size 3712
    --moe-router-score-function sigmoid --moe-router-enable-expert-bias
    --moe-router-topk-scaling-factor 2.5
    )
    ;;
ultrav3)
    # Nemotron 3 Ultra 550B-A55B (latent MoE), converted with convert/convert_ultra_tp8ep8.sh.
    # ~1.1 TB of bf16 weights: at TP8 x EP8 that is ~140 GB per GPU, hence the smaller buffers.
    TOKENIZER=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
    DEFAULT_BUFFER_GB=25
    DEFAULT_MAMBA_PREFIX_GB=25
    ARGS+=(
    --load "${ULTRAV3_CHECKPOINT:-${ULTRA_MCORE_OUT:?set ULTRAV3_CHECKPOINT in cluster.env}}"
    --tokenizer-model "${TOKENIZER}"
    --hybrid-layer-pattern 'MEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME/*E'
    --vocab-size 131072 --make-vocab-size-divisible-by 128
    --hidden-size 8192 --ffn-hidden-size 5120
    --num-attention-heads 64 --group-query-attention --num-query-groups 2 --kv-channels 128
    --normalization RMSNorm --norm-epsilon 1e-5 --disable-bias-linear --squared-relu
    --position-embedding-type none --untie-embeddings-and-output-weights
    --mamba-state-dim 128 --mamba-head-dim 64 --mamba-num-groups 8 --mamba-num-heads 256
    --num-experts 512 --moe-ffn-hidden-size 5120 --moe-router-topk 22 --moe-latent-size 2048
    --moe-shared-expert-intermediate-size 10240
    --moe-router-score-function sigmoid --moe-router-enable-expert-bias
    --moe-router-topk-scaling-factor 5.0
    )
    ;;
superv3)
    # Nemotron 3 Super 120B-A12B (latent MoE); architecture comes from the checkpoint.
    TOKENIZER=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
    ARGS+=(
    --pretrained-checkpoint "${SUPERV3_PRETRAINED_CHECKPOINT:?set in cluster.env}"
    --load "${SUPERV3_CHECKPOINT:?set in cluster.env}"
    --use-checkpoint-args
    --tokenizer-model "${TOKENIZER}"
    )
    ;;
*)
    echo "Unknown MODEL=${MODEL}; expected nanov3p5, superv3 or ultrav3"
    exit 1
    ;;
esac
ARGS+=(--inference-dynamic-batching-buffer-size-gb "${BUFFER_GB:-${DEFAULT_BUFFER_GB:-50}}")
[ "${ASYNC:-1}" == "1" ] && ARGS+=(--inference-dynamic-batching-async-sched-mode async)
[ "${PREFIX_CACHING:-1}" == "1" ] && ARGS+=(
    --inference-dynamic-batching-prefix-caching
    --inference-dynamic-batching-prefix-caching-eviction-policy lru
)
[ "${PREFIX_CACHING:-1}" == "1" ] && [ "${MAMBA_PREFIX_GB:-${DEFAULT_MAMBA_PREFIX_GB:-50}}" != "0" ] && ARGS+=(
    --inference-dynamic-batching-prefix-caching-mamba-gb "${MAMBA_PREFIX_GB:-${DEFAULT_MAMBA_PREFIX_GB:-50}}"
)
[ -n "${DISAGG_ROLE}" ] && ARGS+=(
    --disagg-role "${DISAGG_ROLE}" --kv-transfer-backend nixl
    --disagg-prefill-coordinator-file "${PD_COORDINATOR_FILE}"
)
[ -n "${MAX_REQUESTS:-}" ] && ARGS+=(--inference-dynamic-batching-max-requests "${MAX_REQUESTS}")
[ "${EVAL_MODE:-1}" == "1" ] && ARGS+=(--eval-mode)
# The server returns log-probs by default, which materializes logits for every prompt token
# (gathered across TP); skip prompt log-probs so only each request's last-token logits are kept.
[ "${PROMPT_LOGPROBS:-0}" == "1" ] || ARGS+=(--skip-prompt-log-probs)
ARGS+=(--inference-cuda-graph-max-tokens "${CUDA_GRAPH_MAX_TOKENS:-2048}")
# FP8_RECIPE=mxfp8: MXFP8 expert weights (quantized from the BF16 checkpoint at load).
[ "${FP8_RECIPE:-}" == "mxfp8" ] && ARGS+=(--fp8-format e4m3 --fp8-recipe mxfp8 --fp8-param-gather)
[ "${SHARED_EXPERT_OVERLAP:-1}" == "1" ] && ARGS+=(--moe-shared-expert-overlap)

# Log to node-local disk; frequent writes to lustre stall the rank (and, through EP, every rank).
# Copied to EXP_DIR as server_rank<N>.log when the rank exits, including on failure.
# The log lives in the container's /tmp; to follow it live:
#   srun --jobid <id> --overlap --container-name inference_bench_<id> tail -f /tmp/inference_bench_server_rank0_<id>.log
LOG="/tmp/inference_bench_server_rank${RANK}_${SLURM_JOB_ID:-local}.log"
trap 'cp "${LOG}" "${EXP_DIR}/server_rank${RANK}.log" 2>/dev/null' EXIT
if [ "${RANK}" == "0" ]; then
    cat > "${EXP_DIR}/config.env" <<EOF
EXP_NAME=${EXP_NAME:-run}
SLURM_JOB_ID=${SLURM_JOB_ID}
IMAGE=${INFERENCE_BENCH_SERVE_IMAGE:-}
TOKENIZER=${TOKENIZER}
MODEL=${MODEL:-nanov3p5} NODES=${POOL_NODES} TP=${TP} EP=${EP} ETP=${ETP} BUFFER_GB=${BUFFER_GB:-${DEFAULT_BUFFER_GB:-50}} MAMBA_PREFIX_GB=${MAMBA_PREFIX_GB:-${DEFAULT_MAMBA_PREFIX_GB:-50}} MAMBA_RATIO=${MAMBA_RATIO:-0.1}
DISPATCHER=${DISPATCHER:-nvls} COMBINE_PRECISION=${COMBINE_PRECISION:-fp32} GEMM_BACKEND=${GEMM_BACKEND:-trtllm_bf16_routed}
ASYNC=${ASYNC:-1} PREFIX_CACHING=${PREFIX_CACHING:-1} SAMPLING=${SAMPLING:-flashinfer} EVAL_MODE=${EVAL_MODE:-1} PROMPT_LOGPROBS=${PROMPT_LOGPROBS:-0} PROFILE=${PROFILE:-0} LOG_INTERVAL=${LOG_INTERVAL:-1000} MAX_TOKENS=${MAX_TOKENS:-8192} CUDA_GRAPH_MAX_TOKENS=${CUDA_GRAPH_MAX_TOKENS:-2048} FP8_RECIPE=${FP8_RECIPE:-} SHARED_EXPERT_OVERLAP=${SHARED_EXPERT_OVERLAP:-1}
PD=${PD:-0} DISAGG_ROLE=${DISAGG_ROLE} PREFILL_NODES=${PREFILL_NODES:-} GPUS=$((4 * POOL_NODES))
BENCH=${BENCH:-} BENCH_DATASET=${BENCH_DATASET:-gsm8k} BENCH_MIN_ISL=${BENCH_MIN_ISL:-0} BENCH_ISL=${BENCH_ISL:-0} BENCH_EXACT_ISLS=${BENCH_EXACT_ISLS:-}
BENCH_OSL=${BENCH_OSL:-1024} BENCH_ITERS=${BENCH_ITERS:-3} BENCH_WARMUP=${BENCH_WARMUP:-1}
CMD=python -m tools.run_dynamic_text_generation_server ${ARGS[*]}
EOF
fi

PRE=()
if [ "${PROFILE:-0}" == "1" ]; then
    PRE=(nsys profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none
         --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown
         --force-overwrite true -o "${EXP_DIR}/trace_rank${RANK}")
fi
"${PRE[@]}" python -m tools.run_dynamic_text_generation_server "${ARGS[@]}" > "${LOG}" 2>&1 &
SERVER_PID=$!

# Serve until DONE_FILE (<exp dir>/done) appears (written by the in-job sweep or by client.sh) or the server
# dies. Under PROFILE the server shuts itself down when profiling stops; killing it early would
# cut off nsys while it writes the report.
serve_until_done() {
    if [ "${PROFILE:-0}" == "1" ]; then
        wait "${SERVER_PID}"
        exit 0
    fi
    while kill -0 "${SERVER_PID}" 2>/dev/null && [ ! -f "${DONE_FILE}" ]; do sleep 2; done
    # A rank whose server died ends the deployment (in PD, both pools).
    kill -0 "${SERVER_PID}" 2>/dev/null || touch "${DONE_FILE}"
    kill "${SERVER_PID}" 2>/dev/null
    # The server's SIGTERM exit status (143) would make srun cancel the step as a failure.
    wait "${SERVER_PID}"
    exit 0
}

[ "${RANK}" != "0" ] && serve_until_done
# The prefill pool of a PD deployment serves no HTTP; the decode pool's frontend reaches its
# coordinator through PD_COORDINATOR_FILE.
[ "${DISAGG_ROLE}" == "prefill" ] && serve_until_done

# Rank 0: wait for the server, then either run the client sweep or keep serving.
until grep -q "Running on http://0.0.0.0:5000" "${LOG}" 2>/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Server exited during startup; see ${LOG}"
        touch "${DONE_FILE}"
        exit 1
    fi
    if [ -f "${DONE_FILE}" ]; then
        echo "Stopped during startup (${DONE_FILE} appeared)"
        kill "${SERVER_PID}" 2>/dev/null
        exit 1
    fi
    sleep 5
done
echo "Server ready."
echo "Server reachable at: http://$(hostname):5000/v1"
hostname > "${EXP_DIR}/host"

if [ -n "${BENCH:-}" ]; then
    python3 -u inference-bench/server/client.py \
        --server http://localhost:5000 \
        --batch-sizes ${BENCH} \
        --dataset "${BENCH_DATASET:-gsm8k}" \
        --min-isl "${BENCH_MIN_ISL:-0}" \
        --isl "${BENCH_ISL:-0}" \
        --exact-isls ${BENCH_EXACT_ISLS:-} --tokenizer "${TOKENIZER}" \
        --osl "${BENCH_OSL:-1024}" \
        --iters "${BENCH_ITERS:-3}" \
        --warmup "${BENCH_WARMUP:-1}" \
        --out "${EXP_DIR}/bench.json" \
        --pause "${BENCH_PAUSE:-0}" \
        $([ "${PROFILE:-0}" == "1" ] && { [ "${PROFILE_ALL:-0}" == "1" ] && echo --profile-all || echo --profile; }) \
        2>&1 | tee "${EXP_DIR}/bench.log"
    touch "${DONE_FILE}"
fi
serve_until_done

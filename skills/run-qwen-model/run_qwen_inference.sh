#!/usr/bin/env bash
# Run Qwen inference (server + coherence + optional gsm8k benchmark) on a Slurm
# cluster via cog. See skills/run-qwen-model/SKILL.md.
set -euo pipefail

MODEL="${1:-qwen3-30b-a3b}"
shift || true

FULL_BENCH=0
CHECKPOINT_OVERRIDE=""
NUM_WARMUP_ITERS="${NUM_WARMUP_ITERS:-2}"
NUM_TIMED_ITERS="${NUM_TIMED_ITERS:-5}"
EXPERIMENT_ID="${EXPERIMENT_ID:-unassigned}"
EXPERIMENT_HYPOTHESIS="${EXPERIMENT_HYPOTHESIS:-not-recorded}"
# Install the FA4 (flash-attn-4) beta into the run venv before launching.
# Requires an FA4-aware megatron (PR #5804: num_splits=0 for FA4 inference).
INSTALL_FA4="${INSTALL_FA4:-0}"
REQUIRE_FA4="${REQUIRE_FA4:-0}"
# Fixed Qwen3-30B baseline defaults. Override only for a recorded experiment.
INFERENCE_MOE_DISPATCHER="${INFERENCE_MOE_DISPATCHER:-nvls}"
SYNC_ZMQ_COLLECTIVES="${SYNC_ZMQ_COLLECTIVES:-1}"
QWEN30B_TP="${QWEN30B_TP:-1}"
QWEN30B_EP="${QWEN30B_EP:-4}"
QWEN30B_ETP="${QWEN30B_ETP:-1}"
MOE_ROUTER_FUSION="${MOE_ROUTER_FUSION:-0}"
MOE_PERMUTE_FUSION="${MOE_PERMUTE_FUSION:-0}"
BENCH_SIZES_OVERRIDE="${BENCH_SIZES_OVERRIDE:-256}"
BENCH_OUTPUT_TOKENS="${BENCH_OUTPUT_TOKENS:-1024}"
INFERENCE_GROUPED_GEMM_BACKEND="${INFERENCE_GROUPED_GEMM_BACKEND:-vllm}"
DYNAMIC_BATCHING_BUFFER_GB="${DYNAMIC_BATCHING_BUFFER_GB:-}"
DYNAMIC_BATCHING_MAX_TOKENS="${DYNAMIC_BATCHING_MAX_TOKENS:-4096}"
NUM_CUDA_GRAPHS="${NUM_CUDA_GRAPHS:--1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
CUDA_GRAPH_SIZING_DISTRIBUTION="${CUDA_GRAPH_SIZING_DISTRIBUTION:-exponential}"
DYNAMIC_BATCHING_MAX_REQUESTS="${DYNAMIC_BATCHING_MAX_REQUESTS:-256}"
DYNAMIC_BATCHING_ASYNC_SCHED_MODE="${DYNAMIC_BATCHING_ASYNC_SCHED_MODE:-legacy}"
DYNAMIC_BATCHING_SAMPLING_BACKEND="${DYNAMIC_BATCHING_SAMPLING_BACKEND:-torch}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
trap 'echo "Record ${EXPERIMENT_ID} in skills/run-qwen-model/EXPERIMENTS.md, including failures."' EXIT
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full-benchmark) FULL_BENCH=1; shift ;;
    --checkpoint) CHECKPOINT_OVERRIDE="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Prefer oci-hsg env when present; fall back to default cog env.
_USER_REPO="${COG_MEGATRON_REPO:-}"
_SCRIPT_REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -f "${HOME}/.cog/setup.env.oci-hsg" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.cog/setup.env.oci-hsg"
elif [[ -f "${HOME}/.cog/setup.env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.cog/setup.env"
else
  echo "ERROR: no ~/.cog/setup.env — run cog-setup-and-help skill" >&2
  exit 1
fi
# Precedence: caller export > the checkout this script lives in > setup.env.
# setup.env is machine-wide and can name a different (stale) checkout, which
# would sync and benchmark code you did not edit — silently.
if [[ -n "$_USER_REPO" ]]; then
  export COG_MEGATRON_REPO="$_USER_REPO"
elif [[ -n "$_SCRIPT_REPO" ]]; then
  if [[ -n "${COG_MEGATRON_REPO:-}" && "$COG_MEGATRON_REPO" != "$_SCRIPT_REPO" ]]; then
    echo "WARNING: ~/.cog/setup.env names COG_MEGATRON_REPO=$COG_MEGATRON_REPO" >&2
    echo "         but this script lives in $_SCRIPT_REPO — benchmarking the latter." >&2
    echo "         Export COG_MEGATRON_REPO explicitly to override." >&2
  fi
  export COG_MEGATRON_REPO="$_SCRIPT_REPO"
fi

: "${COG_MEGATRON_REPO:?COG_MEGATRON_REPO not set}"
: "${COG_SSH_HOST:?COG_SSH_HOST not set}"

export COG_ARTIFACTS_ROOT="${COG_ARTIFACTS_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci}"

if ! ssh -o BatchMode=yes "$COG_SSH_HOST" "test -d '$COG_ARTIFACTS_ROOT/model'"; then
  echo "ERROR: artifacts root '$COG_ARTIFACTS_ROOT' not reachable on $COG_SSH_HOST" >&2
  exit 1
fi

# --- Fixed Qwen3-30B-A3B model ---
case "$MODEL" in
  qwen3-30b-a3b|30b|qwen-30b)
    MODEL_TAG="qwen3-30b-a3b"
    NPROC=4
    if [[ -n "$CHECKPOINT_OVERRIDE" ]]; then
      CKPT_ABS="$CHECKPOINT_OVERRIDE"
    elif [[ -n "${QWEN30B_CKPT:-}" ]]; then
      CKPT_ABS="$QWEN30B_CKPT"
    else
      echo "ERROR: Qwen3-30B-A3B checkpoint required. Pass --checkpoint or set QWEN30B_CKPT" >&2
      exit 1
    fi
    TOKENIZER="${QWEN30B_TOKENIZER:-/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/checkpoints/qwen3-30b-a3b-hf}"
    QWEN_MODEL_ARGS="--model-provider gpt --num-layers 48 --hidden-size 2048 --ffn-hidden-size 6144 --num-attention-heads 32 --group-query-attention --num-query-groups 4 --kv-channels 128 --num-experts 128 --moe-router-topk 8 --moe-ffn-hidden-size 768 --moe-grouped-gemm --moe-router-dtype fp32 --moe-router-pre-softmax --moe-token-dispatcher-type alltoall --swiglu --normalization RMSNorm --norm-epsilon 1e-6 --position-embedding-type rope --rotary-base 1000000 --qk-layernorm --disable-bias-linear --untie-embeddings-and-output-weights --no-gradient-accumulation-fusion --make-vocab-size-divisible-by 1187 --tensor-model-parallel-size $QWEN30B_TP --pipeline-model-parallel-size 1 --expert-model-parallel-size $QWEN30B_EP --expert-tensor-parallel-size $QWEN30B_ETP --inference-moe-token-dispatcher-type $INFERENCE_MOE_DISPATCHER --inference-grouped-gemm-backend $INFERENCE_GROUPED_GEMM_BACKEND"
    BENCH_DATASET="gsm8k"
    BUFFER_GB=40
    if [[ "$QWEN30B_TP" -gt 1 ]]; then
      SEQ_PARALLEL_FLAG="--sequence-parallel"
    else
      SEQ_PARALLEL_FLAG=""
    fi
    ;;
  *)
    echo "ERROR: unknown model '$MODEL'. Use qwen3-30b-a3b" >&2
    exit 1
    ;;
esac

ALLOC_GPUS="$NPROC"
if [[ -n "${COG_GPUS_PER_NODE_LIMIT:-}" ]] && [[ "$ALLOC_GPUS" -lt "$COG_GPUS_PER_NODE_LIMIT" ]]; then
  ALLOC_GPUS="$COG_GPUS_PER_NODE_LIMIT"
fi

if [[ -n "$DYNAMIC_BATCHING_BUFFER_GB" ]]; then
  BUFFER_GB="$DYNAMIC_BATCHING_BUFFER_GB"
fi

if [[ -n "$BENCH_SIZES_OVERRIDE" ]]; then
  BENCH_SIZES="$BENCH_SIZES_OVERRIDE"
elif [[ "$FULL_BENCH" == "1" ]]; then
  BENCH_SIZES="16 64 256"
else
  BENCH_SIZES="16"
fi
BENCH_ITERS="--num-iters $NUM_TIMED_ITERS --num-warmup-iters $NUM_WARMUP_ITERS"
BENCH_CLIENT='te''sts/performance_tests/client/static_benchmark.py'

if [[ "$SYNC_ZMQ_COLLECTIVES" == "1" ]]; then
  SYNC_ZMQ_FLAG="--inference-use-synchronous-zmq-collectives"
else
  SYNC_ZMQ_FLAG="--no-inference-use-synchronous-zmq-collectives"
fi
FUSION_FLAGS=""
if [[ "$MOE_ROUTER_FUSION" == "1" ]]; then FUSION_FLAGS+=" --moe-router-fusion"; fi
if [[ "$MOE_PERMUTE_FUSION" == "1" ]]; then FUSION_FLAGS+=" --moe-permute-fusion"; fi
if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
  CHUNKED_PREFILL_FLAG="--enable-chunked-prefill"
else
  CHUNKED_PREFILL_FLAG=""
fi
MAX_REQUESTS_FLAG=""
if [[ -n "$DYNAMIC_BATCHING_MAX_REQUESTS" ]]; then
  MAX_REQUESTS_FLAG="--inference-dynamic-batching-max-requests $DYNAMIC_BATCHING_MAX_REQUESTS"
fi

RUN_SUFFIX=$(printf '%s' "$EXPERIMENT_ID" | tr '[:upper:]_' '[:lower:]-' | tr -cd '[:alnum:]-')
RUN_NAME="qwen-${MODEL_TAG}-${RUN_SUFFIX:-run}-$(date +%Y%m%d-%H%M%S)"
ARTIFACTS="$COG_ARTIFACTS_ROOT"
export COG_EXTRA_MOUNTS="$ARTIFACTS:$ARTIFACTS"
CODE_REVISION=$(git -C "$COG_MEGATRON_REPO" rev-parse HEAD)
if [[ -n "$(git -C "$COG_MEGATRON_REPO" status --porcelain)" ]]; then
  CODE_STATE=dirty
else
  CODE_STATE=clean
fi

echo "Submitting $RUN_NAME: model=$MODEL_TAG nproc=$NPROC alloc_gpus=$ALLOC_GPUS ckpt=$CKPT_ABS"
echo "Experiment $EXPERIMENT_ID: $EXPERIMENT_HYPOTHESIS"
echo "Code: $CODE_REVISION ($CODE_STATE); warmups=$NUM_WARMUP_ITERS timed_iters=$NUM_TIMED_ITERS"

cog --pretty submit \
  --repo "$COG_MEGATRON_REPO" \
  --cluster-name "${COG_CLUSTER_NAME:-oci-hsg}" \
  --run-name "$RUN_NAME" \
  --gpus "$ALLOC_GPUS" --nodes 1 --ntasks-per-node 1 \
  --time 01:00:00 \
  --partition "${COG_BATCH_PARTITION:-batch}" \
  --command "$(cat <<EOF
set -euo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CHECKPOINT_LOAD_PATH=$ARTIFACTS
CKPT='$CKPT_ABS'
TOKENIZER='$TOKENIZER'
EXTRA=\$RUN_DIR/extra_pkgs
mkdir -p "\$EXTRA"
export PYTHONPATH="\$EXTRA:\$PYTHONPATH"
# Fallback deps for server + benchmark (hypercorn is normally in uv lock).
python -m pip install --quiet --no-cache-dir --target="\$EXTRA" hypercorn aiohttp 2>/dev/null || true

SERVER_LOG="\$RUN_DIR/server.log"
BENCH_LOG="\$RUN_DIR/benchmark.log"

cat > "\$BENCH_LOG" <<'META'
===== EXPERIMENT METADATA =====
experiment_id=$EXPERIMENT_ID
hypothesis=$EXPERIMENT_HYPOTHESIS
code_revision=$CODE_REVISION
code_state=$CODE_STATE
model=$MODEL_TAG
checkpoint=$CKPT_ABS
tokenizer=$TOKENIZER
world_size=$NPROC
allocated_gpus=$ALLOC_GPUS
dataset=$BENCH_DATASET
batch_sizes=$BENCH_SIZES
num_output_tokens=$BENCH_OUTPUT_TOKENS
warmup_iters=$NUM_WARMUP_ITERS
timed_iters=$NUM_TIMED_ITERS
install_fa4=$INSTALL_FA4
require_fa4=$REQUIRE_FA4
inference_moe_dispatcher=$INFERENCE_MOE_DISPATCHER
sync_zmq_collectives=$SYNC_ZMQ_COLLECTIVES
tp=$QWEN30B_TP
ep=$QWEN30B_EP
etp=$QWEN30B_ETP
moe_router_fusion=$MOE_ROUTER_FUSION
moe_permute_fusion=$MOE_PERMUTE_FUSION
inference_grouped_gemm_backend=$INFERENCE_GROUPED_GEMM_BACKEND
dynamic_batching_buffer_gb=$BUFFER_GB
dynamic_batching_max_tokens=$DYNAMIC_BATCHING_MAX_TOKENS
num_cuda_graphs=$NUM_CUDA_GRAPHS
enable_chunked_prefill=$ENABLE_CHUNKED_PREFILL
cuda_graph_sizing_distribution=$CUDA_GRAPH_SIZING_DISTRIBUTION
dynamic_batching_max_requests=$DYNAMIC_BATCHING_MAX_REQUESTS
dynamic_batching_async_sched_mode=$DYNAMIC_BATCHING_ASYNC_SCHED_MODE
dynamic_batching_sampling_backend=$DYNAMIC_BATCHING_SAMPLING_BACKEND
extra_server_args=$EXTRA_SERVER_ARGS
META

if [[ "$INSTALL_FA4" == "1" ]]; then
  echo "===== Installing flash-attn-4 beta into run venv =====" | tee -a "\$BENCH_LOG"
  PYBIN="\$(command -v python)"
  echo "python=\$PYBIN" | tee -a "\$BENCH_LOG"
  FA4_PKGS="flash-attn-4[cu13]==4.0.0b20 quack-kernels==0.5.3 nvidia-cutlass-dsl[cu13]==4.6.0.dev0 nvidia-cutlass-dsl-libs-base==4.6.0.dev0 nvidia-cutlass-dsl-libs-cu13==4.6.0.dev0 apache-tvm-ffi==0.1.12 torch-c-dlpack-ext==0.1.5"
  set -f  # keep word-splitting on \$FA4_PKGS but stop [cu13] being glob-expanded
  if command -v uv >/dev/null 2>&1; then
    # Install into the active run venv (same site-packages as flash_attn 2.x).
    uv pip install --python "\$PYBIN" --prerelease=allow --no-deps \$FA4_PKGS 2>&1 | tee -a "\$BENCH_LOG"
  else
    "\$PYBIN" -m pip install --pre --no-deps --no-cache-dir \$FA4_PKGS 2>&1 | tee -a "\$BENCH_LOG"
  fi
  set +f
  echo "----- FA4 verification -----" | tee -a "\$BENCH_LOG"
  if ! python - <<'PYFA4' 2>&1 | tee -a "\$BENCH_LOG"
from importlib.metadata import version
import flash_attn
print("flash_attn", getattr(flash_attn, "__version__", "?"))
print("flash-attn-4", version("flash-attn-4"))
from flash_attn.cute import flash_attn_varlen_func  # noqa: F401
print("FA4_IMPORT_OK")
PYFA4
  then
    echo "FA4 install/verify FAILED — aborting run so we do not silently benchmark without FA4." | tee -a "\$BENCH_LOG"
    exit 1
  fi
fi

if [[ "$REQUIRE_FA4" == "1" && "$INSTALL_FA4" != "1" ]]; then
  echo "----- FA4 verification (no install) -----" | tee -a "\$BENCH_LOG"
  if ! python - <<'PYFA4' 2>&1 | tee -a "\$BENCH_LOG"
from importlib.metadata import version
from packaging.version import Version
import flash_attn
from flash_attn.cute import flash_attn_varlen_func  # noqa: F401
installed = version("flash-attn-4")
print("flash_attn", getattr(flash_attn, "__version__", "?"))
print("flash-attn-4", installed)
assert Version(installed) >= Version("4.0.0b20")
print("FA4_IMPORT_OK")
PYFA4
  then
    echo "Required FA4 beta is unavailable — aborting rather than benchmarking FA2." | tee -a "\$BENCH_LOG"
    exit 1
  fi
fi

python -m torch.distributed.run --nproc-per-node $NPROC --log-dir "\$RUN_DIR/torchrun_logs" \\
  -m examples.inference.launch_inference_server \\
  --load "\$CKPT" \\
  --dist-ckpt-strictness log_unexpected \\
  --tokenizer-type HuggingFaceTokenizer \\
  --tokenizer-model "\$TOKENIZER" \\
  --no-use-tokenizer-model-from-checkpoint-args \\
  --micro-batch-size 1 --bf16 --te-rng-tracker --inference-rng-tracker \\
  --transformer-impl inference_optimized \\
  $SEQ_PARALLEL_FLAG \\
  --inference-dynamic-batching \\
  --inference-dynamic-batching-unified-memory-level 0 \\
  --use-flashinfer-fused-rope \\
  --inference-dynamic-batching-max-tokens $DYNAMIC_BATCHING_MAX_TOKENS \\
  $MAX_REQUESTS_FLAG \\
  --inference-dynamic-batching-cuda-graph-sizing-distribution $CUDA_GRAPH_SIZING_DISTRIBUTION \\
  --inference-dynamic-batching-async-sched-mode $DYNAMIC_BATCHING_ASYNC_SCHED_MODE \\
  --inference-dynamic-batching-sampling-backend $DYNAMIC_BATCHING_SAMPLING_BACKEND \\
  $CHUNKED_PREFILL_FLAG \\
  --seq-length 4096 --max-position-embeddings 4096 --inference-max-seq-length 4096 \\
  --inference-dynamic-batching-buffer-size-gb $BUFFER_GB \\
  --inference-dynamic-batching-num-cuda-graphs $NUM_CUDA_GRAPHS \\
  --cuda-graph-impl local \\
  --cuda-graph-scope full_iteration_inference \\
  $SYNC_ZMQ_FLAG \\
  $FUSION_FLAGS \\
  --inference-logging-step-interval 100 \\
  --port 5000 \\
  $QWEN_MODEL_ARGS \\
  $EXTRA_SERVER_ARGS \\
  > "\$SERVER_LOG" 2>&1 &
SERVER_PID=\$!

READY=0
for i in \$(seq 1 240); do
  if grep -q "Running on http://0.0.0.0:5000" "\$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 \$SERVER_PID 2>/dev/null; then echo "SERVER DIED"; tail -80 "\$SERVER_LOG"; exit 1; fi
  sleep 5
done
if [[ "\$READY" != "1" ]]; then echo "SERVER TIMEOUT"; tail -80 "\$SERVER_LOG"; kill \$SERVER_PID 2>/dev/null; exit 1; fi

echo "===== SERVER READY =====" | tee -a "\$BENCH_LOG"
python - <<'PYEOF' 2>&1 | tee -a "\$BENCH_LOG"
import json, urllib.request
URL="http://localhost:5000/v1/completions"
print("===== COHERENCE (temperature=0) =====")
for p in ["Question: What is 2+2? Answer:","The capital of France is","Q: 3 cows + 2 cows = ? A:","Once upon a time"]:
    b=json.dumps({"model":"qwen","prompt":p,"max_tokens":48,"temperature":0.0}).encode()
    try:
        txt=json.loads(urllib.request.urlopen(urllib.request.Request(URL,data=b,headers={"Content-Type":"application/json"}),timeout=300).read())["choices"][0]["text"]
    except Exception as e:
        txt=f"<ERR {e}>"
    print(f"PROMPT {p!r} -> {txt!r}")
PYEOF

for BS in $BENCH_SIZES; do
  echo "===== static_benchmark dataset=$BENCH_DATASET BS=\$BS OSL=1024 =====" | tee -a "\$BENCH_LOG"
  python -u $BENCH_CLIENT \\
    --server-url "http://localhost:5000/v1" --model qwen \\
    --batch-size \$BS --dataset $BENCH_DATASET --num-output-tokens $BENCH_OUTPUT_TOKENS \\
    $BENCH_ITERS 2>&1 | tee -a "\$BENCH_LOG"
done

kill \$SERVER_PID 2>/dev/null || true
wait \$SERVER_PID 2>/dev/null || true
echo "===== DONE ====="
tail -30 "\$BENCH_LOG"
EOF
)"

#!/usr/bin/env bash
# Profile mcore Qwen3-30B-A3B inference under Nsight Systems (nsys) via cog and
# export a .sqlite for A/B analysis. Uses the fixed baseline:
# inference_optimized, TP=1, EP=4, vLLM grouped-GEMM, full-iteration CUDA
# graphs, BS=256. OSL stays short so the trace remains bounded.
#
# Usage:
#   source ~/.cog/setup.env.oci-hsg
#   # Deploys the checkout this script lives in; export COG_MEGATRON_REPO to override.
#   QWEN30B_CKPT=/lustre/.../qwen3-30b-a3b-mcore \
#   PROFILE_BS=256 PROFILE_OSL=128 \
#   bash skills/run-qwen-model/profile_qwen_mcore.sh
set -euo pipefail

PROFILE_BS="${PROFILE_BS:-256}"
PROFILE_OSL="${PROFILE_OSL:-128}"

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
# would profile code you did not edit — silently.
if [[ -n "$_USER_REPO" ]]; then
  export COG_MEGATRON_REPO="$_USER_REPO"
elif [[ -n "$_SCRIPT_REPO" ]]; then
  if [[ -n "${COG_MEGATRON_REPO:-}" && "$COG_MEGATRON_REPO" != "$_SCRIPT_REPO" ]]; then
    echo "WARNING: ~/.cog/setup.env names COG_MEGATRON_REPO=$COG_MEGATRON_REPO" >&2
    echo "         but this script lives in $_SCRIPT_REPO — profiling the latter." >&2
    echo "         Export COG_MEGATRON_REPO explicitly to override." >&2
  fi
  export COG_MEGATRON_REPO="$_SCRIPT_REPO"
fi

: "${COG_MEGATRON_REPO:?COG_MEGATRON_REPO not set}"
: "${COG_SSH_HOST:?COG_SSH_HOST not set}"
export COG_ARTIFACTS_ROOT="${COG_ARTIFACTS_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci}"

CKPT_ABS="${QWEN30B_CKPT:-${COG_SCRATCH_ROOT}/checkpoints/qwen3-30b-a3b-mcore}"
TOKENIZER="${QWEN30B_TOKENIZER:-${COG_SCRATCH_ROOT}/checkpoints/qwen3-30b-a3b-hf}"

NPROC=4
ALLOC_GPUS="$NPROC"
if [[ -n "${COG_GPUS_PER_NODE_LIMIT:-}" ]] && [[ "$ALLOC_GPUS" -lt "$COG_GPUS_PER_NODE_LIMIT" ]]; then
  ALLOC_GPUS="$COG_GPUS_PER_NODE_LIMIT"
fi

QWEN_MODEL_ARGS="--model-provider gpt --num-layers 48 --hidden-size 2048 --ffn-hidden-size 6144 --num-attention-heads 32 --group-query-attention --num-query-groups 4 --kv-channels 128 --num-experts 128 --moe-router-topk 8 --moe-ffn-hidden-size 768 --moe-grouped-gemm --moe-router-dtype fp32 --moe-router-pre-softmax --moe-token-dispatcher-type alltoall --swiglu --normalization RMSNorm --norm-epsilon 1e-6 --position-embedding-type rope --rotary-base 1000000 --qk-layernorm --disable-bias-linear --untie-embeddings-and-output-weights --no-gradient-accumulation-fusion --make-vocab-size-divisible-by 1187 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 4 --expert-tensor-parallel-size 1 --inference-moe-token-dispatcher-type nvls --inference-grouped-gemm-backend vllm"

RUN_NAME="qwen-30b-nsys-$(date +%Y%m%d-%H%M%S)"
ARTIFACTS="$COG_ARTIFACTS_ROOT"
export COG_EXTRA_MOUNTS="$ARTIFACTS:$ARTIFACTS"

echo "Submitting $RUN_NAME: mcore EP4/TP1 nsys profile BS=$PROFILE_BS OSL=$PROFILE_OSL ckpt=$CKPT_ABS"

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
python -m pip install --quiet --no-cache-dir --target="\$EXTRA" hypercorn aiohttp 2>/dev/null || true

SERVER_LOG="\$RUN_DIR/server.log"
PROF_BASE="\$RUN_DIR/mcore_profile"

if ! command -v nsys >/dev/null 2>&1; then
  echo "ERROR: nsys not found in image" >&2; exit 1
fi
nsys --version

# Launch the server UNDER nsys. --cuda-graph-trace=node is required so kernels
# inside the full_iteration_inference CUDA graph are individually traced.
# Do NOT add osrt or --sample=process-tree: on this workload (MoE decode under
# full-iteration CUDA graphs) either one hangs nsys in finalization and the
# resulting qdstrm is rejected by QdstrmImporter. Bisected over several
# sessions; see optimize-inference-siddharth/references/measuring.md.
nsys profile \\
  --trace=cuda,nvtx \\
  --sample=none --cpuctxsw=none \\
  --cuda-graph-trace=node \\
  --force-overwrite=true \\
  -o "\$PROF_BASE" \\
  python -m torch.distributed.run --nproc-per-node $NPROC --log-dir "\$RUN_DIR/torchrun_logs" \\
  -m examples.inference.launch_inference_server \\
  --load "\$CKPT" \\
  --dist-ckpt-strictness log_unexpected \\
  --tokenizer-type HuggingFaceTokenizer \\
  --tokenizer-model "\$TOKENIZER" \\
  --no-use-tokenizer-model-from-checkpoint-args \\
  --micro-batch-size 1 --bf16 --te-rng-tracker --inference-rng-tracker \\
  --transformer-impl inference_optimized \\
  --inference-dynamic-batching \\
  --inference-dynamic-batching-unified-memory-level 0 \\
  --use-flashinfer-fused-rope \\
  --inference-dynamic-batching-max-tokens 4096 \\
  --enable-chunked-prefill \\
  --seq-length 4096 --max-position-embeddings 4096 --inference-max-seq-length 4096 \\
  --inference-dynamic-batching-buffer-size-gb 40 \\
  --inference-dynamic-batching-max-requests 256 \\
  --inference-dynamic-batching-num-cuda-graphs -1 \\
  --cuda-graph-impl local \\
  --cuda-graph-scope full_iteration_inference \\
  --inference-use-synchronous-zmq-collectives \\
  --inference-logging-step-interval 100 \\
  --port 5000 \\
  $QWEN_MODEL_ARGS \\
  > "\$SERVER_LOG" 2>&1 &
NSYS_PID=\$!

READY=0
for i in \$(seq 1 300); do
  if grep -q "Running on http://0.0.0.0:5000" "\$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 \$NSYS_PID 2>/dev/null; then echo "SERVER/NSYS DIED"; tail -100 "\$SERVER_LOG"; exit 1; fi
  sleep 5
done
if [[ "\$READY" != "1" ]]; then echo "SERVER TIMEOUT"; tail -100 "\$SERVER_LOG"; kill \$NSYS_PID 2>/dev/null; exit 1; fi
echo "===== SERVER READY (profiling) ====="

# One short warmup call (BS=8) so CUDA graphs are captured and the allocator is
# warm; steady-state decode after this is what the analysis anchors on.
python -u tests/performance_tests/client/static_benchmark.py \\
  --server-url "http://localhost:5000/v1" --model qwen \\
  --batch-size 8 --dataset gsm8k --num-output-tokens 32 \\
  --num-iters 1 --num-warmup-iters 0 || true

echo "===== PROFILED BENCHMARK BS=$PROFILE_BS OSL=$PROFILE_OSL ====="
python -u tests/performance_tests/client/static_benchmark.py \\
  --server-url "http://localhost:5000/v1" --model qwen \\
  --batch-size $PROFILE_BS --dataset gsm8k --num-output-tokens $PROFILE_OSL \\
  --num-iters 1 --num-warmup-iters 0 2>&1 | tee "\$RUN_DIR/profile_bench.log"

# Stop nsys gracefully (SIGINT) so it finalizes and writes the .nsys-rep.
echo "===== stopping nsys ====="
kill -INT \$NSYS_PID 2>/dev/null || true
wait \$NSYS_PID 2>/dev/null || true

ls -la "\$RUN_DIR"/mcore_profile.* || true
echo "===== exporting sqlite ====="
nsys export --type sqlite --force-overwrite=true \\
  --output "\$PROF_BASE.sqlite" "\$PROF_BASE.nsys-rep"
ls -la "\$PROF_BASE.sqlite"
echo "===== PROFILE DONE ====="
echo "REP=\$PROF_BASE.nsys-rep"
echo "SQLITE=\$PROF_BASE.sqlite"
EOF
)"

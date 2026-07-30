#!/usr/bin/env bash
# In-session mcore Qwen3-30B nsys profile (bypasses cog repo-sync).
# Wraps the inference server under nsys, runs BS256 at PROFILE_OSL, exports sqlite.
# Decode is OSL-invariant at fixed BS256 (all requests run in lockstep), so a
# bounded OSL still captures the exact OSL1024 steady-state decode kernels; we
# just need a clean decode-only window (no prefill) in analysis.
set -uo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1
PROFILE_BS="${PROFILE_BS:-256}"
PROFILE_OSL="${PROFILE_OSL:-256}"

SCRATCH=/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space
CKPT=$SCRATCH/checkpoints/qwen3-30b-a3b-mcore
TOKENIZER=$SCRATCH/checkpoints/qwen3-30b-a3b-hf
export CHECKPOINT_LOAD_PATH="$CKPT"

VENV=$SCRATCH/envs/megatron_lm/dd356431262b5db4/.venv
PYBIN=$VENV/bin/python

RUN_DIR=$SCRATCH/sessions/qwen-moe-kernel/prof/$(date +%s)
mkdir -p "$RUN_DIR/torchrun_logs"
EXTRA="$RUN_DIR/extra_pkgs"; mkdir -p "$EXTRA"
export PYTHONPATH="$EXTRA:${PYTHONPATH:-}"
$PYBIN -m pip install --quiet --no-cache-dir --target="$EXTRA" hypercorn aiohttp 2>/dev/null || true
echo "RUN_DIR=$RUN_DIR"; echo "PYBIN=$PYBIN"
$PYBIN -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__)" 2>&1 | tail -1

SERVER_LOG="$RUN_DIR/server.log"
PROF_BASE="$RUN_DIR/mcore_profile"

command -v nsys >/dev/null 2>&1 || { echo "ERROR: nsys not found"; exit 3; }
nsys --version

QWEN_MODEL_ARGS="--model-provider gpt --num-layers 48 --hidden-size 2048 --ffn-hidden-size 6144 --num-attention-heads 32 --group-query-attention --num-query-groups 4 --kv-channels 128 --num-experts 128 --moe-router-topk 8 --moe-ffn-hidden-size 768 --moe-grouped-gemm --moe-router-dtype fp32 --moe-router-pre-softmax --moe-token-dispatcher-type alltoall --swiglu --normalization RMSNorm --norm-epsilon 1e-6 --position-embedding-type rope --rotary-base 1000000 --qk-layernorm --disable-bias-linear --untie-embeddings-and-output-weights --no-gradient-accumulation-fusion --make-vocab-size-divisible-by 1187 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 4 --expert-tensor-parallel-size 1 --inference-moe-token-dispatcher-type nvls --inference-grouped-gemm-backend vllm"

# Launch server UNDER nsys. --cuda-graph-trace=node traces kernels inside the
# full_iteration_inference CUDA graph. trace=cuda,nvtx only (no osrt) to bound size.
nsys profile \
  --trace=cuda,nvtx \
  --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  -o "$PROF_BASE" \
  $PYBIN -m torch.distributed.run --nproc-per-node 4 --log-dir "$RUN_DIR/torchrun_logs" \
  -m examples.inference.launch_inference_server \
  --load "$CKPT" \
  --dist-ckpt-strictness log_unexpected \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "$TOKENIZER" \
  --no-use-tokenizer-model-from-checkpoint-args \
  --micro-batch-size 1 --bf16 --te-rng-tracker --inference-rng-tracker \
  --transformer-impl inference_optimized \
  --inference-dynamic-batching \
  --inference-dynamic-batching-unified-memory-level 0 \
  --use-flashinfer-fused-rope \
  --inference-dynamic-batching-max-tokens 4096 \
  --inference-dynamic-batching-max-requests 256 \
  --inference-dynamic-batching-cuda-graph-sizing-distribution exponential \
  --inference-dynamic-batching-async-sched-mode "${ASYNC_SCHED:-legacy}" \
  --inference-dynamic-batching-sampling-backend torch \
  --enable-chunked-prefill \
  --seq-length 4096 --max-position-embeddings 4096 --inference-max-seq-length 4096 \
  --inference-dynamic-batching-buffer-size-gb 40 \
  --inference-dynamic-batching-num-cuda-graphs -1 \
  --cuda-graph-impl local \
  --cuda-graph-scope full_iteration_inference \
  --inference-use-synchronous-zmq-collectives \
  --inference-logging-step-interval 100 \
  --port 5000 \
  $QWEN_MODEL_ARGS \
  > "$SERVER_LOG" 2>&1 &
NSYS_PID=$!

READY=0
for i in $(seq 1 360); do
  if grep -q "Running on http://0.0.0.0:5000" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 $NSYS_PID 2>/dev/null; then echo "SERVER/NSYS DIED"; tail -100 "$SERVER_LOG"; exit 1; fi
  sleep 5
done
if [[ "$READY" != "1" ]]; then echo "SERVER TIMEOUT"; tail -100 "$SERVER_LOG"; kill $NSYS_PID 2>/dev/null; exit 1; fi
echo "===== SERVER READY (profiling) ====="

# Warmup (BS8) so CUDA graphs are captured + allocator warm.
$PYBIN -u tests/performance_tests/client/static_benchmark.py \
  --server-url "http://localhost:5000/v1" --model qwen \
  --batch-size 8 --dataset gsm8k --num-output-tokens 32 \
  --num-iters 1 --num-warmup-iters 0 || true

echo "===== PROFILED BENCHMARK BS=$PROFILE_BS OSL=$PROFILE_OSL ====="
$PYBIN -u tests/performance_tests/client/static_benchmark.py \
  --server-url "http://localhost:5000/v1" --model qwen \
  --batch-size $PROFILE_BS --dataset gsm8k --num-output-tokens $PROFILE_OSL \
  --num-iters 1 --num-warmup-iters 0 2>&1 | tee "$RUN_DIR/profile_bench.log"

echo "===== stopping nsys ====="
kill -INT $NSYS_PID 2>/dev/null || true
wait $NSYS_PID 2>/dev/null || true
ls -la "$RUN_DIR"/mcore_profile.* || true
echo "===== exporting sqlite ====="
nsys export --type sqlite --force-overwrite=true --output "$PROF_BASE.sqlite" "$PROF_BASE.nsys-rep"
ls -la "$PROF_BASE.sqlite"
echo "===== PROFILE DONE ====="
echo "REP=$PROF_BASE.nsys-rep"
echo "SQLITE=$PROF_BASE.sqlite"

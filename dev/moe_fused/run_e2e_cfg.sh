#!/usr/bin/env bash
# Parameterized in-session mcore Qwen3-30B BS256/OSL1024 benchmark.
# Config levers are env-overridable so we can sweep them without editing the file:
#   ASYNC_SCHED       legacy|serial            (default legacy)
#   SAMPLING_BACKEND  torch|flashinfer         (default torch)
#   NUM_CUDA_GRAPHS   int or -1                (default -1)
#   CG_SIZING         exponential|linear       (default exponential)
#   HISTO_COUNT       0|1  -> MCORE_MOE_HISTOGRAM_COUNT (default 0)
#   TAG               label for the run dir + banner (default cfg)
#   OSL               output tokens (default 1024)
#   EXTRA_SERVER_ARGS appended verbatim to the server argv (default empty)
set -uo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1

ASYNC_SCHED="${ASYNC_SCHED:-legacy}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-torch}"
NUM_CUDA_GRAPHS="${NUM_CUDA_GRAPHS:--1}"
CG_SIZING="${CG_SIZING:-exponential}"
HISTO_COUNT="${HISTO_COUNT:-0}"
DISPATCHER="${DISPATCHER:-nvls}"
GEMM_BACKEND="${GEMM_BACKEND:-vllm}"
TAG="${TAG:-cfg}"
OSL="${OSL:-1024}"
BS="${BS:-256}"
NITERS="${NITERS:-5}"
NWARMUP="${NWARMUP:-2}"
SKIP_COHERENCE="${SKIP_COHERENCE:-0}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
export MCORE_MOE_HISTOGRAM_COUNT="$HISTO_COUNT"

CKPT=/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/checkpoints/qwen3-30b-a3b-mcore
TOKENIZER=/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/checkpoints/qwen3-30b-a3b-hf
export CHECKPOINT_LOAD_PATH="$CKPT"

VENV=/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/envs/megatron_lm/dd356431262b5db4/.venv
PYBIN=$VENV/bin/python

RUN_DIR=/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/sessions/qwen-opt/e2e/${TAG}-$(date +%s)
mkdir -p "$RUN_DIR/torchrun_logs"
EXTRA="$RUN_DIR/extra_pkgs"; mkdir -p "$EXTRA"
export PYTHONPATH="$EXTRA:${PYTHONPATH:-}"
$PYBIN -m pip install --quiet --no-cache-dir --target="$EXTRA" hypercorn aiohttp 2>/dev/null || true

SERVER_LOG="$RUN_DIR/server.log"
BENCH_LOG="$RUN_DIR/benchmark.log"
echo "================ CONFIG ($TAG) ================"
echo "ASYNC_SCHED=$ASYNC_SCHED SAMPLING_BACKEND=$SAMPLING_BACKEND NUM_CUDA_GRAPHS=$NUM_CUDA_GRAPHS CG_SIZING=$CG_SIZING HISTO_COUNT=$HISTO_COUNT DISPATCHER=$DISPATCHER GEMM_BACKEND=$GEMM_BACKEND OSL=$OSL"
echo "EXTRA_SERVER_ARGS=$EXTRA_SERVER_ARGS"
echo "RUN_DIR=$RUN_DIR"
echo "==============================================="

QWEN_MODEL_ARGS="--model-provider gpt --num-layers 48 --hidden-size 2048 --ffn-hidden-size 6144 --num-attention-heads 32 --group-query-attention --num-query-groups 4 --kv-channels 128 --num-experts 128 --moe-router-topk 8 --moe-ffn-hidden-size 768 --moe-grouped-gemm --moe-router-dtype fp32 --moe-router-pre-softmax --moe-token-dispatcher-type alltoall --swiglu --normalization RMSNorm --norm-epsilon 1e-6 --position-embedding-type rope --rotary-base 1000000 --qk-layernorm --disable-bias-linear --untie-embeddings-and-output-weights --no-gradient-accumulation-fusion --make-vocab-size-divisible-by 1187 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 4 --expert-tensor-parallel-size 1 --inference-moe-token-dispatcher-type $DISPATCHER --inference-grouped-gemm-backend $GEMM_BACKEND"

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
  --inference-dynamic-batching-cuda-graph-sizing-distribution "$CG_SIZING" \
  --inference-dynamic-batching-async-sched-mode "$ASYNC_SCHED" \
  --inference-dynamic-batching-sampling-backend "$SAMPLING_BACKEND" \
  --enable-chunked-prefill \
  --seq-length 4096 --max-position-embeddings 4096 --inference-max-seq-length 4096 \
  --inference-dynamic-batching-buffer-size-gb 40 \
  --inference-dynamic-batching-num-cuda-graphs "$NUM_CUDA_GRAPHS" \
  --cuda-graph-impl local \
  --cuda-graph-scope full_iteration_inference \
  --inference-use-synchronous-zmq-collectives \
  --inference-logging-step-interval 100 \
  --port 5000 \
  $EXTRA_SERVER_ARGS \
  $QWEN_MODEL_ARGS \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

READY=0
for i in $(seq 1 360); do
  if grep -q "Running on http://0.0.0.0:5000" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "SERVER DIED"; tail -80 "$SERVER_LOG"; exit 1; fi
  sleep 5
done
if [[ "$READY" != "1" ]]; then echo "SERVER TIMEOUT"; tail -80 "$SERVER_LOG"; kill $SERVER_PID 2>/dev/null; exit 1; fi
echo "===== SERVER READY ====="

if [[ "$SKIP_COHERENCE" != "1" ]]; then
$PYBIN - <<'PYEOF'
import json, urllib.request
URL="http://localhost:5000/v1/completions"
print("===== COHERENCE (temperature=0) =====")
for p in ["Question: What is 2+2? Answer:","The capital of France is","Q: 3 cows + 2 cows = ? A:"]:
    b=json.dumps({"model":"qwen","prompt":p,"max_tokens":48,"temperature":0.0}).encode()
    try:
        txt=json.loads(urllib.request.urlopen(urllib.request.Request(URL,data=b,headers={"Content-Type":"application/json"}),timeout=300).read())["choices"][0]["text"]
    except Exception as e:
        txt=f"<ERR {e}>"
    print(f"PROMPT {p!r} -> {txt!r}")
PYEOF
fi

echo "===== static_benchmark BS=$BS OSL=$OSL ($TAG) ====="
$PYBIN -u tests/performance_tests/client/static_benchmark.py \
  --server-url "http://localhost:5000/v1" --model qwen \
  --batch-size "$BS" --dataset gsm8k --num-output-tokens "$OSL" \
  --num-iters "$NITERS" --num-warmup-iters "$NWARMUP" 2>&1 | tee "$BENCH_LOG"

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "===== DONE ($TAG) RUN_DIR=$RUN_DIR ====="

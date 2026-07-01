#!/bin/bash
# Boot the Gemma4 E4B KV-cache eval server in the background on this node.
#
# Reads the MLM checkpoint via TP=1 by default (the Gemma4SelfAttention port
# supports parallelism<=2; set TP=2 NPROC=2 CUDA_VISIBLE_DEVICES=0,1 to shard).
# Serves OpenAI-compatible /v1/chat/completions on PORT.
# Logs to /tmp/g4_eval/server.log.
#
# Usage:
#   bash examples/gemma4/eval/launch_eval_server.sh           # default port 5082
#   PORT=8000 bash examples/gemma4/eval/launch_eval_server.sh  # custom port
#
# Health check:
#   curl http://127.0.0.1:${PORT:-5082}/v1/health
#
# Stop:
#   pkill -f gemma4_eval_server

set -euo pipefail

MLM_ROOT="${MLM_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/huvu/codes/gemma4_elastic/Megatron-LM_gemma4_e4b_tp_sp}"
LOAD_CKPT="${LOAD_CKPT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/code_dev/shared-state/implementations/HYBRID-gemma4-mlm/mlm_ckpt_mg}"
GEMMA4_HF="${GEMMA4_HF:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/gemma4-playground/weights/gemma-4-E4B-it}"
PORT="${PORT:-5082}"
MASTER_PORT="${MASTER_PORT:-12410}"
LOG="${LOG:-/tmp/g4_eval/server.log}"
NPROC="${NPROC:-1}"          # = TP. Don't change unless you also change TP below.
TP="${TP:-1}"                # Gemma4SelfAttention is a parallelism<=2 port (set TP=2 to shard).
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SEQLEN="${SEQLEN:-4096}"

mkdir -p "$(dirname "$LOG")"
: >"$LOG"

cd "$MLM_ROOT"

# Hopper + TP>1 (non-FSDP) requires CUDA_DEVICE_MAX_CONNECTIONS=1.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$MLM_ROOT:${PYTHONPATH:-}"

nohup python -m torch.distributed.run --nproc-per-node="$NPROC" --master-port="$MASTER_PORT" \
  examples/gemma4/eval/gemma4_eval_server.py \
    --use-mcore-models --transformer-impl local \
    --num-layers 42 --hidden-size 2560 --ffn-hidden-size 10240 \
    --num-attention-heads 8 --group-query-attention --num-query-groups 2 \
    --normalization RMSNorm --norm-epsilon 1e-6 --qk-layernorm \
    --disable-bias-linear --position-embedding-type none \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --tensor-model-parallel-size "$TP" --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --seq-length "$SEQLEN" --max-position-embeddings "$SEQLEN" \
    --micro-batch-size 1 --bf16 \
    --load "$LOAD_CKPT" \
    --ckpt-format torch_dist --ckpt-fully-parallel-load \
    --no-load-optim --no-load-rng \
    --tokenizer-type NullTokenizer --vocab-size 262144 \
    --distributed-timeout-minutes 30 --disable-gloo-process-groups \
    --num-workers 1 --logging-level 20 \
    --port "$PORT" --host 0.0.0.0 \
    --gemma4-hf-tokenizer "$GEMMA4_HF" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
  >"$LOG" 2>&1 </dev/null &
disown
echo "launched on port $PORT, logs at $LOG"
echo "poll readiness:  until curl -sf http://127.0.0.1:$PORT/v1/health; do sleep 5; done"

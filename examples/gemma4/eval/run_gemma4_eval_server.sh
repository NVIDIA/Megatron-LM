#!/bin/bash
# ============================================================================
# Gemma 4 E4B evaluation harness.
#
# Single-node SLURM job that:
#   1. Boots gemma4_eval_server.py in the background (TP=2, single-flight,
#      single-batch KV cache). Server listens on http://localhost:5000.
#   2. Polls /v1/health until the model is loaded.
#   3. Runs `nel eval run <NEL_CONFIG>` against the local server.
#   4. Tears the server down.
#
# Sized to mirror gemma4_e4b_load_interactive.sh: real E4B dims, TP=2 (the
# Gemma4SelfAttention port is parallelism<=2; see that script's header), PP=1
# (the cross-layer KV bus assumes a single pipeline stage).
#
# Required env vars (set on the submit command line or in this file):
#   IMAGE          path to the .sqsh container image
#   MLM_ROOT       absolute path to this Megatron-LM checkout
#   EVAL_ROOT      absolute path to the NeMo Evaluator checkout (pip -e'd)
#   LOAD_CKPT      absolute path to the MLM dist-checkpoint (mlm_ckpt_mg/)
#   GEMMA4_HF      path or HF id of the original Gemma4 release (tokenizer + chat template)
#   NEL_CONFIG     absolute path to the NEL YAML config
#   RESULTS_DIR    absolute path for nel results (writable)
# ============================================================================
set -euo pipefail

# ---- SLURM hooks (set values appropriate for your account/partition) -------
#SBATCH --job-name=gemma4-mlm-eval
#SBATCH --account=coreai_dlalgo_genai
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK SLURM_CPU_BIND SLURM_DISTRIBUTION

: "${IMAGE:?set IMAGE to a .sqsh container path}"
: "${MLM_ROOT:?set MLM_ROOT to the Megatron-LM checkout}"
: "${EVAL_ROOT:?set EVAL_ROOT to the Evaluator checkout (pip -e installed)}"
: "${LOAD_CKPT:?set LOAD_CKPT to the MLM dist-checkpoint mlm_ckpt_mg/}"
: "${GEMMA4_HF:?set GEMMA4_HF to the original Gemma4 HF release path/id}"
: "${NEL_CONFIG:?set NEL_CONFIG to the NEL YAML}"
: "${RESULTS_DIR:?set RESULTS_DIR to a writable directory for nel results}"

mkdir -p "$(dirname logs/.)" "$RESULTS_DIR"

# ---- Parallelism ----------------------------------------------------------
# TP=2 max for Gemma4SelfAttention (num_query_groups=2). PP=1 always.
NPROC=${NPROC:-2}
TP=${TP:-2}
PP=${PP:-1}
if [ "$TP" -gt 2 ]; then
  echo "ERROR: TP=$TP unsupported (Gemma4SelfAttention is parallelism<=2)." >&2
  exit 1
fi
if [ "$NPROC" -ne "$TP" ]; then
  echo "WARNING: NPROC=$NPROC != TP=$TP; the eval server is single-flight, DP > 1 is wasted GPUs." >&2
fi

# ---- Real E4B model dims (must match the checkpoint param shapes) ---------
NUM_LAYERS=${NUM_LAYERS:-42}
HIDDEN=${HIDDEN:-2560}
FFN=${FFN:-10240}
HEADS=${HEADS:-8}
KV_GROUPS=${KV_GROUPS:-2}
VOCAB=${VOCAB:-262144}
SEQLEN=${SEQLEN:-4096}
SERVER_PORT=${SERVER_PORT:-5000}

# ---- Distributed env ------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29501}
export NNODES=${SLURM_NNODES}
export GPUS_PER_NODE=${NPROC}
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

server_opts="\
    --use-mcore-models \
    --transformer-impl local \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN} \
    --ffn-hidden-size ${FFN} \
    --num-attention-heads ${HEADS} \
    --group-query-attention \
    --num-query-groups ${KV_GROUPS} \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --qk-layernorm \
    --disable-bias-linear \
    --position-embedding-type none \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size 1 \
    --seq-length ${SEQLEN} \
    --max-position-embeddings ${SEQLEN} \
    --micro-batch-size 1 \
    --bf16 \
    --load ${LOAD_CKPT} \
    --ckpt-format torch_dist \
    --ckpt-fully-parallel-load \
    --no-load-optim \
    --no-load-rng \
    --tokenizer-type NullTokenizer \
    --vocab-size ${VOCAB} \
    --distributed-timeout-minutes 2880 \
    --disable-gloo-process-groups \
    --num-workers 1 \
    --logging-level 20 \
    --port ${SERVER_PORT} \
    --host 0.0.0.0 \
    --gemma4-hf-tokenizer ${GEMMA4_HF} \
"

srun --container-image="$IMAGE" \
     --container-mounts=/lustre:/lustre \
     --no-container-mount-home \
     --ntasks="${NNODES}" --ntasks-per-node=1 \
     bash -c '
set -euo pipefail
# Hopper + TP>1 (non-FSDP) requires CUDA_DEVICE_MAX_CONNECTIONS=1.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH='"${MLM_ROOT}"':${PYTHONPATH:-}

cd '"${MLM_ROOT}"'

# Start the server in the background. All ranks call torchrun together; rank 0
# also hosts the HTTP server.
python -m torch.distributed.run \
  --nnodes='"${NNODES}"' \
  --nproc-per-node='"${GPUS_PER_NODE}"' \
  --node-rank=${SLURM_NODEID} \
  --master-addr='"${MASTER_ADDR}"' \
  --master-port='"${MASTER_PORT}"' \
  examples/gemma4/eval/gemma4_eval_server.py '"${server_opts}"' &
SERVER_PID=$!

# Wait for /v1/health to flip to 200 (model load + first CUDA init takes a few
# minutes on E4B). Cap the wait at ~30 min.
echo "waiting for http://localhost:'"${SERVER_PORT}"'/v1/health"
for i in $(seq 1 360); do
  if curl -sf "http://localhost:'"${SERVER_PORT}"'/v1/health" >/dev/null; then
    echo "server is ready (after ${i}*5s)"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: server process exited before /v1/health became ready" >&2
    exit 1
  fi
  sleep 5
done

# Run the eval against the local server. NEL is expected to live at EVAL_ROOT
# with `pip install -e .` already done.
echo "running nel eval run against the local server"
cd '"${EVAL_ROOT}"'
nel eval run '"${NEL_CONFIG}"' || echo "nel eval run returned non-zero"

# Tear down. /v1/health was up so SERVER_PID is still alive.
echo "stopping server"
kill -INT "$SERVER_PID" || true
wait "$SERVER_PID" || true
'

echo "DONE"

#!/usr/bin/env bash
# Run the vLLM Qwen3-30B-A3B comparison at one batch size on OCI.
set -euo pipefail

if [[ -f "${HOME}/.cog/setup.env.oci-hsg" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.cog/setup.env.oci-hsg"
else
  echo "ERROR: ~/.cog/setup.env.oci-hsg not found" >&2
  exit 1
fi

: "${COG_SSH_HOST:?}"
: "${COG_SCRATCH_ROOT:?}"

EXPERIMENT_ID="${EXPERIMENT_ID:-unassigned}"
EXPERIMENT_HYPOTHESIS="${EXPERIMENT_HYPOTHESIS:-not-recorded}"
BENCH_BS="${BENCH_BS:-256}"
BENCH_OUTPUT_TOKENS="${BENCH_OUTPUT_TOKENS:-1024}"
NUM_WARMUP_ITERS="${NUM_WARMUP_ITERS:-2}"
NUM_TIMED_ITERS="${NUM_TIMED_ITERS:-5}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-512}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"

IMG="${VLLM_IMAGE:-$COG_SCRATCH_ROOT/images/87e4947c6ce36433.sqsh}"
HF_CKPT="${QWEN30B_HF:-$COG_SCRATCH_ROOT/checkpoints/qwen3-30b-a3b-hf}"
RUN_SUFFIX=$(printf '%s' "$EXPERIMENT_ID" | tr '[:upper:]_' '[:lower:]-' | tr -cd '[:alnum:]-')
RUN_NAME="vllm-qwen30b-${RUN_SUFFIX:-run}-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$COG_SCRATCH_ROOT/runs/$RUN_NAME"
REMOTE_SCRIPT="$RUN_DIR/run.sbatch"

ssh -o BatchMode=yes "$COG_SSH_HOST" "mkdir -p '$RUN_DIR'; cat > '$REMOTE_SCRIPT'" <<EOF
#!/bin/bash
#SBATCH --job-name=$EXPERIMENT_ID
#SBATCH --account=${COG_RUNTIME_ACCOUNT}
#SBATCH --partition=${COG_BATCH_PARTITION}
#SBATCH --qos=${SLURM_QOS:-interactive}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=$RUN_DIR/slurm-%j.out
#SBATCH --error=$RUN_DIR/slurm-%j.err
set -euo pipefail

# Locate the benchmark client: an explicit MLM_REPO, else the most recently
# synced mcore workspace. Use "sed -n 1p", not "head -1": head closes the pipe
# after one line, which SIGPIPEs ls, and under pipefail that aborts the job
# with exit 141 before a single line is logged. sed drains its input instead.
MLM_HOST="${MLM_REPO:-}"
if [[ -z "\$MLM_HOST" ]]; then
  MLM_HOST=\$(ls -dt "$COG_SCRATCH_ROOT"/workspaces/megatron_lm/*/repo 2>/dev/null | sed -n 1p)
fi
if [[ -z "\$MLM_HOST" || ! -f "\$MLM_HOST/tests/performance_tests/client/static_benchmark.py" ]]; then
  echo "ERROR: no benchmark client under '\$MLM_HOST'. Set MLM_REPO to a synced Megatron-LM workspace." >&2
  exit 1
fi
echo "benchmark_client_repo=\$MLM_HOST"
srun --container-image="$IMG" --container-mounts=/lustre:/lustre --no-container-mount-home bash -c '
set -euo pipefail
export HF_HOME="$RUN_DIR/hf_home"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OUTLINES_CACHE_DIR=/tmp/outlines_\$\$
VENV=/opt/ray_venvs/nemo_rl.experience.sync_rollout_actor.SyncRolloutActor
PY=\$VENV/bin/python
VLLM=\$VENV/bin/vllm
MLM='"\$MLM_HOST"'
BENCH=\$MLM/tests/performance_tests/client/static_benchmark.py
SERVER_LOG="$RUN_DIR/server.log"
BENCH_LOG="$RUN_DIR/benchmark.log"

cat > "\$BENCH_LOG" <<META
experiment_id=$EXPERIMENT_ID
hypothesis=$EXPERIMENT_HYPOTHESIS
engine=vllm
batch_size=$BENCH_BS
num_output_tokens=$BENCH_OUTPUT_TOKENS
warmup_iters=$NUM_WARMUP_ITERS
timed_iters=$NUM_TIMED_ITERS
max_num_seqs=$VLLM_MAX_NUM_SEQS
gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION
META

\$VLLM serve "$HF_CKPT" --served-model-name qwen \
  --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
  --max-model-len 4096 --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" --trust-remote-code \
  --port 5000 --host 0.0.0.0 > "\$SERVER_LOG" 2>&1 &
SERVER_PID=\$!
trap '"'"'kill "\$SERVER_PID" 2>/dev/null || true'"'"' EXIT

READY=0
for _ in \$(seq 1 360); do
  if grep -qE "Application startup complete|Uvicorn running on http://0.0.0.0:5000" "\$SERVER_LOG"; then
    READY=1
    break
  fi
  if ! kill -0 "\$SERVER_PID" 2>/dev/null; then
    echo "VLLM SERVER DIED"
    tail -100 "\$SERVER_LOG"
    exit 1
  fi
  sleep 5
done
if [[ "\$READY" != 1 ]]; then
  echo "VLLM SERVER TIMEOUT"
  tail -100 "\$SERVER_LOG"
  exit 1
fi

\$PY -u "\$BENCH" --server-url http://localhost:5000/v1 --model qwen \
  --batch-size "$BENCH_BS" --dataset gsm8k --num-output-tokens "$BENCH_OUTPUT_TOKENS" \
  --num-iters "$NUM_TIMED_ITERS" --num-warmup-iters "$NUM_WARMUP_ITERS" \
  2>&1 | tee -a "\$BENCH_LOG"
'
EOF

JOB_ID=$(ssh -o BatchMode=yes "$COG_SSH_HOST" "cd '$RUN_DIR' && sbatch --parsable '$REMOTE_SCRIPT'")
echo "experiment_id=$EXPERIMENT_ID"
echo "job_id=$JOB_ID"
echo "run_dir=$RUN_DIR"

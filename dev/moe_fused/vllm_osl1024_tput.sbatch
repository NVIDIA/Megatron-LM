#!/bin/bash
#SBATCH --job-name=vllm-qwen30b-osl1024
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -x
SCRATCH=/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space
IMG=$SCRATCH/images/87e4947c6ce36433.sqsh
HF_CKPT=$SCRATCH/checkpoints/qwen3-30b-a3b-hf
EXP=$SCRATCH/runs/vllm-qwen30b-osl1024-$SLURM_JOB_ID
MLM=$SCRATCH/workspaces/megatron_lm/a2d89809f051082a/repo
mkdir -p "$EXP"

srun --container-image="$IMG" \
  --container-mounts=/lustre:/lustre \
  --no-container-mount-home \
  bash -c '
set -x
export HF_HOME='"$EXP"'/hf_home
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OUTLINES_CACHE_DIR=/tmp/outlines_$$
export CUDA_DEVICE_MAX_CONNECTIONS=1
EXP='"$EXP"'
HF_CKPT='"$HF_CKPT"'
MLM='"$MLM"'
VENV=/opt/ray_venvs/nemo_rl.experience.sync_rollout_actor.SyncRolloutActor
PY=$VENV/bin/python
VLLM=$VENV/bin/vllm
BENCH=$MLM/tests/performance_tests/client/static_benchmark.py

$VLLM serve "$HF_CKPT" --served-model-name qwen \
  --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
  --max-model-len 4096 --max-num-seqs 512 \
  --gpu-memory-utilization 0.9 --trust-remote-code \
  --port 5000 --host 0.0.0.0 \
  > "$EXP/vqserver.log" 2>&1 &
SERVER_PID=$!

READY=0
for i in $(seq 1 360); do
  if grep -qE "Application startup complete|Uvicorn running on http://0.0.0.0:5000" "$EXP/vqserver.log" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "SERVER DIED"; tail -100 "$EXP/vqserver.log"; exit 1; fi
  sleep 5
done
if [ "$READY" != "1" ]; then echo "VLLM NOT READY"; tail -100 "$EXP/vqserver.log"; kill $SERVER_PID 2>/dev/null; exit 1; fi
echo "===== VLLM READY ====="

echo "===== THROUGHPUT BS=256 OSL=1024 (2 warmup + 5 timed) ====="
$PY -u "$BENCH" --server-url "http://localhost:5000/v1" --model qwen \
  --batch-size 256 --dataset gsm8k --num-output-tokens 1024 \
  --num-iters 5 --num-warmup-iters 2 2>&1 | tee "$EXP/vqtput.log"

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "===== DONE ===== EXP=$EXP"
'
echo "JOB DONE rc=$?"

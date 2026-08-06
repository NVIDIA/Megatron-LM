#!/usr/bin/env bash
# Profile vLLM Qwen3-30B-A3B inference under Nsight Systems (nsys) on oci-hsg and
# export a .sqlite for A/B analysis. Uses TP=1, DP=4, expert parallel and wraps
# `vllm serve`
# with nsys at fixed BS256, profiles a short OSL, and exports sqlite.
#
# Usage:
#   source ~/.cog/setup.env.oci-hsg
#   PROFILE_BS=256 PROFILE_OSL=128 bash skills/run-qwen-model/profile_qwen_vllm.sh
set -euo pipefail

PROFILE_BS="${PROFILE_BS:-256}"
PROFILE_OSL="${PROFILE_OSL:-128}"

if [[ -f "${HOME}/.cog/setup.env.oci-hsg" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.cog/setup.env.oci-hsg"
else
  echo "ERROR: ~/.cog/setup.env.oci-hsg not found" >&2; exit 1
fi
: "${COG_SSH_HOST:?}"; : "${COG_SCRATCH_ROOT:?}"

IMG="${VLLM_IMAGE:-$COG_SCRATCH_ROOT/images/87e4947c6ce36433.sqsh}"
HF_CKPT="${QWEN30B_HF:-$COG_SCRATCH_ROOT/checkpoints/qwen3-30b-a3b-hf}"
ACCOUNT="${COG_RUNTIME_ACCOUNT:-coreai_dlalgo_llm}"
PART="${COG_BATCH_PARTITION:-batch}"
EXP="$COG_SCRATCH_ROOT/runs/vllm-qwen30b-nsys-$(date +%Y%m%d-%H%M%S)"

echo "Creating vLLM nsys sbatch on $COG_SSH_HOST: BS=$PROFILE_BS OSL=$PROFILE_OSL img=$IMG"

# Push the sbatch to the cluster. Outer heredoc quoted so it is sent verbatim;
# $PROFILE_BS / $PROFILE_OSL / paths are substituted by expanding them into the
# stream via a small template below.
ssh -o BatchMode=yes "$COG_SSH_HOST" "mkdir -p '$EXP'; cat > '$EXP/vllm_nsys.sbatch'" <<EOF
#!/bin/bash
#SBATCH --job-name=vllm-qwen30b-nsys
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PART
#SBATCH --qos=${SLURM_QOS:-interactive}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=$EXP/slurm-%j.out
#SBATCH --error=$EXP/slurm-%j.err
set -x
IMG=$IMG
EXP=$EXP
HF_CKPT=$HF_CKPT
PROFILE_BS=$PROFILE_BS
PROFILE_OSL=$PROFILE_OSL
# sed -n 1p rather than head -1: head closes the pipe early and SIGPIPEs ls,
# which is fatal under pipefail. Override MLM_REPO to pin a specific workspace
# instead of taking whichever one was modified most recently.
MLM_HOST=\${MLM_REPO:-\$(ls -dt $COG_SCRATCH_ROOT/workspaces/megatron_lm/*/repo 2>/dev/null | sed -n 1p)}
mkdir -p "\$EXP"
rm -f "\$EXP/vqdone_status" "\$EXP/vqserver.log" "\$EXP/vqprof.log"

srun --container-image="\$IMG" \\
  --container-mounts=/lustre:/lustre \\
  --no-container-mount-home \\
  bash -c '
set -x
export HF_HOME='"\$EXP"'/hf_home
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OUTLINES_CACHE_DIR=/tmp/outlines_\$\$
export CUDA_DEVICE_MAX_CONNECTIONS=1
EXP='"\$EXP"'
HF_CKPT='"\$HF_CKPT"'
MLM='"\$MLM_HOST"'
PROFILE_BS='"\$PROFILE_BS"'
PROFILE_OSL='"\$PROFILE_OSL"'
VENV=/opt/ray_venvs/nemo_rl.experience.sync_rollout_actor.SyncRolloutActor
PY=\$VENV/bin/python
VLLM=\$VENV/bin/vllm
BENCH=\$MLM/tests/performance_tests/client/static_benchmark.py
PROF_BASE=\$EXP/vllm_profile

if ! command -v nsys >/dev/null 2>&1; then echo "ERROR: nsys not in vLLM image"; exit 3; fi
nsys --version

# osrt is omitted deliberately — see the note in profile_qwen_mcore.sh. It is
# not needed for the GPU-kernel comparison and only inflates the trace.
nsys profile \\
  --trace=cuda,nvtx \\
  --sample=none --cpuctxsw=none \\
  --cuda-graph-trace=node \\
  --force-overwrite=true \\
  -o "\$PROF_BASE" \\
  \$VLLM serve "\$HF_CKPT" --served-model-name qwen \\
    --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \\
    --max-model-len 4096 --max-num-seqs 512 \\
    --gpu-memory-utilization 0.9 --trust-remote-code \\
    --port 5000 --host 0.0.0.0 \\
    > "\$EXP/vqserver.log" 2>&1 &
NSYS_PID=\$!

READY=0
for i in \$(seq 1 360); do
  if grep -qE "Application startup complete|Uvicorn running on http://0.0.0.0:5000" "\$EXP/vqserver.log" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 \$NSYS_PID 2>/dev/null; then echo "SERVER/NSYS DIED"; tail -100 "\$EXP/vqserver.log"; exit 1; fi
  sleep 5
done
if [ "\$READY" != "1" ]; then echo "VLLM NOT READY"; tail -100 "\$EXP/vqserver.log"; kill \$NSYS_PID 2>/dev/null; exit 1; fi
echo "===== VLLM READY (profiling) ====="

\$PY -u "\$BENCH" --server-url "http://localhost:5000/v1" --model qwen \\
  --batch-size 8 --dataset gsm8k --num-output-tokens 32 --num-iters 1 --num-warmup-iters 0 || true

echo "===== PROFILED BENCHMARK BS=\$PROFILE_BS OSL=\$PROFILE_OSL ====="
\$PY -u "\$BENCH" --server-url "http://localhost:5000/v1" --model qwen \\
  --batch-size \$PROFILE_BS --dataset gsm8k --num-output-tokens \$PROFILE_OSL \\
  --num-iters 1 --num-warmup-iters 0 2>&1 | tee "\$EXP/vqprof.log"

echo "===== stopping nsys ====="
kill -INT \$NSYS_PID 2>/dev/null || true
wait \$NSYS_PID 2>/dev/null || true
ls -la "\$EXP"/vllm_profile.* || true
echo "===== exporting sqlite ====="
nsys export --type sqlite --force-overwrite=true --output "\$PROF_BASE.sqlite" "\$PROF_BASE.nsys-rep"
ls -la "\$PROF_BASE.sqlite"
echo done > "\$EXP/vqdone_status"
echo "===== PROFILE DONE ====="
echo "SQLITE=\$PROF_BASE.sqlite"
'
echo "JOB DONE rc=\$?"
EOF

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 — wrote $EXP/vllm_nsys.sbatch on $COG_SSH_HOST, not submitting."
  exit 0
fi

echo "Submitting vLLM nsys sbatch..."
JOBID=$(ssh -o BatchMode=yes "$COG_SSH_HOST" "cd '$EXP' && sbatch --parsable vllm_nsys.sbatch")
echo "vLLM nsys job: $JOBID"
echo "Run dir: $EXP"
echo "$JOBID" > /tmp/vllm_nsys_jobid.txt

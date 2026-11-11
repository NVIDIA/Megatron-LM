#!/bin/bash
# HF -> Megatron Core (torch_dist) import of Nemotron 3 Ultra 550B-A55B (BF16)
# on GPUs, via Megatron Bridge 0.6.1 shipped in the NeMo 26.08.01 container.
# TP=8, EP=8, PP=1, ETP=1 over 2 GB200 nodes x 4 GPUs.
#
# Cluster-specific settings (slurm account/partition/QoS, container image, HF cache and output
# paths) come from inference-bench/cluster.env; copy cluster.env.example and fill it in.
#
# Submit from the repo root:
#   source inference-bench/cluster.env
#   sbatch inference-bench/convert/convert_ultra_tp8ep8.sh
#
#SBATCH -J convert-ultra-tp8ep8
#SBATCH -N 2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node 4
#SBATCH -t 4:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o inference-bench/convert/convert_ultra_tp8ep8_%j.log
#SBATCH -e inference-bench/convert/convert_ultra_tp8ep8_%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source "${INFERENCE_BENCH_CONFIG:-inference-bench/cluster.env}"
CONTAINER=${INFERENCE_BENCH_CONVERT_IMAGE:?set in cluster.env}
OUT=${ULTRA_MCORE_OUT:?set in cluster.env}
# The cached snapshot of the HF checkpoint (download it into HF_HOME first).
HF_REPO=${HF_HOME:?set in cluster.env}/hub/models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
HF_SNAPSHOT=${HF_REPO}/snapshots/$(cat "${HF_REPO}/refs/main")

TP=8
EP=8
PP=1
ETP=1

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1  # keep ~/.local packages from shadowing the container venv
# ~142 GB of weights per GPU out of ~186 GB; reduce fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 10000))
export HF_SNAPSHOT OUT TP EP PP ETP

echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "TP=${TP} EP=${EP} PP=${PP} ETP=${ETP} world=${SLURM_NTASKS}"
echo "src=${HF_SNAPSHOT}"
echo "dst=${OUT}"

# One task per GPU. RANK/WORLD_SIZE/LOCAL_RANK are set explicitly from SLURM so
# the Bridge worker uses them directly (it calls torch.cuda.set_device(LOCAL_RANK)
# and init_process_group(backend="nccl") with env:// rendezvous).
srun --container-image="${CONTAINER}" \
     --container-mounts="${INFERENCE_BENCH_MOUNTS:?set in cluster.env}" \
     --kill-on-bad-exit=1 \
     bash -c '
set -euo pipefail
export RANK=${SLURM_PROCID}
export WORLD_SIZE=${SLURM_NTASKS}
export LOCAL_RANK=${SLURM_LOCALID}
cd /opt/Megatron-Bridge
exec python scripts/conversion/run_conversion.py import \
    --device gpu \
    --hf-model "${HF_SNAPSHOT}" \
    --megatron-path "${OUT}" \
    --torch-dtype bfloat16 \
    --tp "${TP}" --pp "${PP}" --ep "${EP}" --etp "${ETP}" \
    --distributed-timeout-minutes 90
'

echo "done: ${OUT}"

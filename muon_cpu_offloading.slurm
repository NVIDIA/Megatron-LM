#!/bin/bash
#SBATCH --job-name=muon-cpu-offload
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --output=./logs/muon_cpu_offload-%j.out
#SBATCH --error=./logs/muon_cpu_offload-%j.err
#SBATCH --exclusive
#SBATCH --time=0:30:00
#SBATCH --partition=h200-hi-preempt
#SBATCH --exclude=ip-10-1-115-124,ip-10-1-45-16
#SBATCH --wait-all-nodes=1

set -euxo pipefail

IMAGE=/fsx/peng/containers/nemo-rl-v0.5.2-te-tilelang.sqsh

MEGATRON_SRC=/fsx/peng/workspace/megatron-muon-cpu-offload/Megatron-LM
MEGATRON_DST=/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM

MOUNTS=${MEGATRON_SRC}:${MEGATRON_DST}

srun \
  --container-image "${IMAGE}" \
  --container-mounts "${MOUNTS}" \
  --container-remap-root \
  bash -lc "
    set -euxo pipefail
    cd ${MEGATRON_DST}

    export NVIDIA_PYTORCH_VERSION=25.06
    export PYTHONPATH=${MEGATRON_DST}:\${PYTHONPATH:-}
    pip install git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.2.0

    torchrun --nproc-per-node=4 tests/test_muon_cpu_offload.py
  "

echo "Test completed."

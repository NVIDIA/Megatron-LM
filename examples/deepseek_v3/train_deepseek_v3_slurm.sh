#!/bin/bash
#SBATCH --job-name=deepseek-train
#SBATCH --output=logs/slurm/deepseek-job.%j.out
#SBATCH --nodes=8                            # Number of nodes, Adjust as necessary
#SBATCH --ntasks-per-node=1                  # One task per GPU -> total 8 tasks per node
#SBATCH --cpus-per-task=226                  # assign all CPUs to the job
#SBATCH --gres=gpu:8                         # Request 8 GPUs per node
#SBATCH --time=01:00:00                      # Adjust as necessary
#SBATCH --reservation=gpu-40_gpu-41_gpu-43_gpu-44_gpu-46_gpu-47_gpu-50_gpu-55_reservation # modify based on your reservation settings

###############################################################################
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################
echo "get first node"
# Get the list of nodes and the first node (master node)
node_list=$(scontrol show hostnames $SLURM_JOB_NODELIST)
node_array=(${node_list})
master_node=${node_array[0]}

# Set environment variables for distributed training
export SLURM_MASTER_ADDR="${SLURM_MASTER_ADDR:-$master_node}"
export SLURM_MASTER_PORT="${SLURM_MASTER_PORT:-29475}"

# Optional: Print out the values for debugging
echo "MASTER_ADDR=$SLURM_MASTER_ADDR"
echo "MASTER_PORT=$SLURM_MASTER_PORT"
# Define the Docker image
export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:v25.4"} # Change the docker image if needed
echo "Trying 'docker ps'..."
if docker ps; then
    echo "Docker is working."
    export container_command=docker
else
    echo "'docker ps' failed. Trying 'podman ps'..."
    if podman ps; then
        echo "Podman is working."
        export container_command=podman
    else
        echo "Both 'docker ps' and 'podman ps' failed."
        exit 1
    fi
fi
# Pull docker image
${container_command} pull $DOCKER_IMAGE
# Define the mount points
export MEGATRON_DIR=${PWD}
export HOST_MOUNT=${HOST_MOUNT:=${HOME}}                # Before run, change it to your own path 
export CONTAINER_MOUNT=${CONTAINER_MOUNT:=${HOME}}      # Before run, change it to your own path 
export NETWORK_INTERFACE=${NETWORK_INTERFACE:-"bond0"}  # Can be get by run `ip a` 
export DATA_DIR=${DATA_DIR:-${MEGATRON_DIR}/../../data} 
export WANDB_API_KEY=${WANDB_API_KEY:-}

# Run the Docker container with the script
srun bash -c '${container_command} run --rm \
 --env SLURM_MASTER_ADDR=$SLURM_MASTER_ADDR \
 --env SLURM_MASTER_PORT=$SLURM_MASTER_PORT \
 --env "SLURM_PROCID=$SLURM_PROCID" \
 --env SLURM_NODEID=$SLURM_NODEID \
 --env SLURM_NNODES=$SLURM_NNODES \
 --env WANDB_API_KEY=$WANDB_API_KEY \
 --ipc=host --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE  --cap-add=CAP_SYS_ADMIN  \
 --security-opt seccomp=unconfined --group-add video --privileged --device=/dev/infiniband \
 -v $HOST_MOUNT:$CONTAINER_MOUNT \
 $DOCKER_IMAGE /bin/bash -c \
 "echo $(date); \
 cd ${MEGATRON_DIR}; \
 NCCL_SOCKET_IFNAME=${NETWORK_INTERFACE} \
 GLOO_SOCKET_IFNAME=${NETWORK_INTERFACE} \
FORCE_BALANCE=true \
RUN_ENV=slurm \
DATA_DIR=${DATA_DIR} \
MODEL_SIZE=671B \
TRAIN_ITERS=10 \
NUM_LAYERS=32 \
SEQ_LEN=4096 \
MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=32 \
PR=bf16 \
AC=full \
TP=2 PP=16 ETP=1 EP=4 \
GEMM_TUNING=0 \
NVTE_CK_USES_BWD_V3=1 \
USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true \
GPT_LAYER_IN_TE=true \
bash examples/deepseek_v3/train_deepseekv3.sh 2>&1 | tee log_deepseek-v3.txt; \
echo $(date)"'



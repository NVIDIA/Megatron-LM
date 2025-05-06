#!/bin/bash
#SBATCH --job-name=llama-train
#SBATCH --output=logs/slurm/multinode-job.%j.out
#SBATCH --nodes=8                            # Number of nodes, Adjust as necessary
#SBATCH --ntasks-per-node=1                  # One task per GPU -> total 8 tasks per node
#SBATCH --cpus-per-task=226                  # assign all CPUs to the job
#SBATCH --gres=gpu:8                         # Request 8 GPUs per node
#SBATCH --time=01:00:00                      # Adjust as necessary
#SBATCH --reservation=gpu-40_gpu-41_gpu-43_gpu-44_gpu-46_gpu-47_gpu-50_gpu-55_reservation # modify based on your reservation settings

# Determine MASTER_ADDR and MASTER_PORT
MASTER_ADDR=$(srun --ntasks=1 hostname | head -n 1)
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29475}"

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


export CONTAINER_NAME="training_env"
export IMAGE=${IMAGE:-"docker.io/rocm/megatron-lm:v25.4"} # change the docker name accordingly 

export HOST_MOUNT=${HOST_MOUNT:=${HOME}}               # change this path to host dir intend to be attached to the docker
export CONTAINER_MOUNT=${CONTAINER_MOUNT:=${HOME}}     # change this path to development workspace path inside the docker
export MEGATRON_DIR=${PWD}                             # change this path to Megatron-LM inside the docker
export CONTAINER_DIR=${HOME}
export DATA_DIR=${DATA_DIR:-"${HOME}/.cache/data"}
export HF_TOKEN="${HF_TOKEN:-hf_xxxx}"  
export MODEL_NAME=${MODEL_NAME:-"llama2"}
export NETWORK_INTERFACE=${NETWORK_INTERFACE:-"bond0"} # Can be get by run `ip a` 
export WANDB_API_KEY=${WANDB_API_KEY:-}

# Build and launch the Docker container, change podmand command to docker command if the system is using docker instead of podman
srun bash -c '
    ${container_command} pull $IMAGE
    ${container_command} rm $CONTAINER_NAME
    ${container_command} images
    ibdev2netdev
    ${container_command} run -d --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband \
      --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
      -v $HOST_MOUNT:$CONTAINER_MOUNT --shm-size 128G --name $CONTAINER_NAME $IMAGE tail -f /dev/null
'

MODEL_SIZE=$1
MBS=$2
BATCH_SIZE_PER_NODE=$3
export BS=$(( SLURM_NNODES * BATCH_SIZE_PER_NODE ))
SEQ_LENGTH=$4
TOTAL_ITERS=$5
FSDP=$6
RECOMPUTE=$7

echo "MODEL_SIZE: $MODEL_SIZE"
echo "MBS: $MBS"
echo "BS: $BS"
echo "SEQ_LENGTH: $SEQ_LENGTH"
echo "TOTAL_ITERS: $TOTAL_ITERS"
echo "FSDP: $FSDP"
echo "RECOMPUTE: $RECOMPUTE"


# Execute the training inside the Docker container
srun bash -c '
  ${container_command} exec \
    -e NNODES=$SLURM_JOB_NUM_NODES \
    -e NODE_RANK=$SLURM_NODEID \
    -e MASTER_ADDR='"$MASTER_ADDR"' \
    -e MASTER_PORT='"$MASTER_PORT"' \
    -e NUM_PROCESSES='"$NUM_PROCESSES"' \
    -e MBS='"$MBS"' \
    -e BS='"$BS"' \
    -e SEQ_LENGTH='"$SEQ_LENGTH"' \
    -e MODEL_SIZE='"$MODEL_SIZE"' \
    -e TOTAL_ITERS='"$TOTAL_ITERS"' \
    -e RECOMPUTE='"$RECOMPUTE"' \
    -e FSDP='"$FSDP"' \
    -e WANDB_API_KEY='"$WANDB_API_KEY"' \
    '"$CONTAINER_NAME"' \
    bash -c "
      echo Inside container: NODE_RANK=\$NODE_RANK
      echo MASTER_ADDR=\$MASTER_ADDR
      echo MASTER_PORT=\$MASTER_PORT
      echo NUM_PROCESSES=\$NUM_PROCESSES
      echo SEQ_LENGTH=\$SEQ_LENGTH
      echo FSDP=\$FSDP
      echo RECOMPUTE=\$RECOMPUTE
      NCCL_SOCKET_IFNAME=${NETWORK_INTERFACE}
      GLOO_SOCKET_IFNAME=${NETWORK_INTERFACE}
      cd ${MEGATRON_DIR}
      pwd
      ibv_devices
      DATA_CACHE_PATH=${CONTAINER_DIR}/cache \
      DATA_DIR=${DATA_DIR} \
      HF_TOKEN=${HF_TOKEN} \
      NCCL_SOCKET_IFNAME=\$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=\$GLOO_SOCKET_IFNAME NODE_RANK=\$NODE_RANK NNODES=\$NNODES MASTER_ADDR=\$MASTER_ADDR MASTER_PORT=\$MASTER_PORT MODEL_SIZE=\$MODEL_SIZE TOTAL_ITERS=\$TOTAL_ITERS \
      TEE_OUTPUT=1 MBS=\$MBS BS=\$BS \
      RECOMPUTE=\$RECOMPUTE TE_FP8=0 FSDP=\$FSDP TP=1 \
      SEQ_LENGTH=\$SEQ_LENGTH bash examples/llama/train_${MODEL_NAME}.sh
    "
' 

#!/bin/bash
#SBATCH --job-name=mixtral-8X7B-train
#SBATCH --output=logs/slurm/mixtral-8X7B-job.%j.out
#SBATCH --nodes=8                            # Number of nodes, Adjust as necessary
#SBATCH --ntasks-per-node=1                  # One task per GPU -> total 8 tasks per node
#SBATCH --cpus-per-task=226                  # assign all CPUs to the job
#SBATCH --gres=gpu:8                         # Request 8 GPUs per node
#SBATCH --time=01:00:00                      # Adjust as necessary
#SBATCH --reservation=gpu-40_gpu-41_gpu-43_gpu-44_gpu-46_gpu-47_gpu-50_gpu-55_reservation # modify based on your reservation settings

export batch_size_per_node=32 # Set the batch size per node, change it to your own value
export GBS=$(( SLURM_NNODES * batch_size_per_node ))
echo "GBS:" $GBS
echo "get first node"
# Get the list of nodes and the first node (master node)
node_list=$(scontrol show hostnames $SLURM_JOB_NODELIST)
node_array=(${node_list})
master_node=${node_array[0]}

# Set environment variables for distributed training
export SLURM_MASTER_ADDR=$master_node
export SLURM_MASTER_PORT="${SLURM_MASTER_PORT:-29475}"


# Optional: Print out the values for debugging
echo "MASTER_ADDR=$SLURM_MASTER_ADDR"
echo "MASTER_PORT=$SLURM_MASTER_PORT"
# Define the Docker image
export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:v25.4"}

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

export NETWORK_INTERFACE=${NETWORK_INTERFACE:-"bond0"} # Can be get by run `ip a` 

MEGATRON_DIR=${PWD}
# Define the dataset path. Before each run, change the following paths accordingly
export HOST_MOUNT=${HOME}                  # change the path to host dir intend to be attached to the docker
export CONTAINER_MOUNT=${HOME}           # change the path to workspace developing path inside the docker
export MEGATRON_DIR=${MEGATRON_DIR:-"${MEGATRON_DIR}"} # change the path to Megatron-LM inside the docker
export TOKENIZER_MODEL=${TOKENIZER_MODEL:-"${CONTAINER_MOUNT}/path/to/tokenizer.model"}   # change the tokenizer path accordingly
export DATA_DIR=${DATA_DIR:-"${CONTAINER_MOUNT}/path/to/dataset"}                # change the path to dataset location
export WANDB_API_KEY=${WANDB_API_KEY:-}

# Run the Docker container with the script
srun bash -c '${container_command} run --rm \
 --env SLURM_MASTER_ADDR=$SLURM_MASTER_ADDR \
 --env SLURM_MASTER_PORT=$SLURM_MASTER_PORT \
 --env "SLURM_PROCID=$SLURM_PROCID" \
 --env SLURM_NODEID=$SLURM_NODEID \
 --env SLURM_NNODES=$SLURM_NNODES \
 --env WANDB_API_KEY=${WANDB_API_KEY} \
 --ipc=host --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE  --cap-add=CAP_SYS_ADMIN  \
 --security-opt seccomp=unconfined --group-add video --privileged --device=/dev/infiniband \
 -v $HOST_MOUNT:$CONTAINER_MOUNT \
 $DOCKER_IMAGE /bin/bash -c \
 "echo $(date); \
 cd $MEGATRON_DIR; \
 TOKENIZER_MODEL=${TOKENIZER_MODEL} \
 DATA_DIR=${DATA_DIR} \
 NCCL_SOCKET_IFNAME=${NETWORK_INTERFACE} GLOO_SOCKET_IFNAME=${NETWORK_INTERFACE} \
 RECOMPUTE_NUM_LAYERS=0 \
 TEE_OUTPUT=1 MBS=2 GBS=${GBS} TP_SIZE=1 PP_SIZE=1 AC=none \
 PR=bf16 EP_SIZE=8 ETP_SIZE=1 SEQLEN=4096 FORCE_BALANCE=true \
 RUN_ENV=slurm MODEL_SIZE=8x7B bash examples/mixtral/train_mixtral_moe.sh 2>&1 | tee result_8X7B.log; \
 echo $(date)"'



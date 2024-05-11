#!/bin/bash
#SBATCH -J SpeedTest-EXP2
#SBATCH -p gpu
#SBATCH -A r00114
#SBATCH -o speed_test_starter-exp2_%j.txt
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=240g
#SBATCH --time=00:60:00

set -x

# Function to activate a Conda environment, create a directory based on the script name, and execute the script with output redirection
execute_script() {
    local node=$1
    local script_path=$2
    local conda_env=$3
    local output_base_path=$4
    local node_rank=$5
    local run_in_background=$6

    echo "Preparing to execute script on node $node with rank $node_rank..."
    
    # Pre-calculate output directory locally
    local script_name=$(basename "$script_path")
    local output_dir="${script_name%.*}/${NNODES}NODEs"
    local full_output_path="$output_base_path/$output_dir"

    # Build the remote command
    local command="ssh $node \"bash -c \\\"\
        set -x; \
        echo 'Activating Conda environment: $conda_env'; \
        mamba init; \
        source ~/.bashrc; \
        mamba activate $conda_env; \

        mkdir -p $full_output_path; \
        cd $full_output_path; \
        export NODE_RANK=$node_rank; \
        export NNODES=$NNODES; \
        export GPUS_PER_NODE=$GPUS_PER_NODE; \
        export MASTER_ADDR=$MASTER_ADDR; \
        export MASTER_PORT=$MASTER_PORT; \
        export WORLD_SIZE=$WORLD_SIZE; \
        export TENSORBOARD_DIR=$full_output_path/tensorboard; \
        export WANDB_DIR=$full_output_path/wandb; \
        export MEGATRON_PATH=$MEGATRON_PATH; \
        export VOCAB_FILE=$VOCAB_FILE; \
        export MERGE_FILE=$MERGE_FILE; \
        export DATA_PATH=$DATA_PATH; \
        export OMP_NUM_THREADS=$OMP_NUM_THREADS; \
        echo 'Running script: $script_path with NODE_RANK $node_rank and MASTER_ADDR $MASTER_ADDR'; \

        bash $script_path > script_output_node$node_rank.log 2>&1; \
    \\\"\""

    if [[ "$run_in_background" == "yes" ]]; then
        command="$command &"
    fi

    eval $command
}

MEGATRON_PATH="/N/slate/jindjia/LLM/Megatron-SpeedTest"
OUTPUT_LOG_PATH="/N/slate/jindjia/bash_scripts/bytedance/speedtest/output_result"
OUTPUT_LOG_PATH="${OUTPUT_LOG_PATH}/Exp2/${SLURM_JOB_ID}"
SCRIPT_DIRECTORY="/N/slate/jindjia/bash_scripts/bytedance/speedtest/test_scripts/Exp2"
SCRIPT_LIST=("2_7B_Baseline.sh" "2_7B_QGrad.sh" "2_7B_QWeightdiff.sh" "2_7B_QWG.sh" )  # List of scripts
NUM_NODES_LIST=(4 8 )  # Number of Nodes you will use to test on it. (Need to equal or smaller than your allocation)
CONDA_ENV_NAME="megatron-TE"


# Set master node and node list (Slurm only)
if [[ -z "${SLURM_NODELIST}" ]]; then
    NODERANKS=$HOSTNAME
else
    MASTER_RANK=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
    NODERANKS=$(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')
fi

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

VOCAB_FILE=/N/scratch/jindjia/thepile/vocab.json
MERGE_FILE=/N/scratch/jindjia/thepile/merges.txt
DATA_PATH=/N/scratch/jindjia/thepile/pile_text_document

for num_nodes in "${NUM_NODES_LIST[@]}"; do
    NNODES=$num_nodes
    MASTER_PORT=6002
    WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

    selected_nodes=$(echo $NODERANKS | tr ' ' '\n' | head -n $NNODES | tr '\n' ' ')

    for script_name in "${SCRIPT_LIST[@]}"; do
        full_script_path="$SCRIPT_DIRECTORY/$script_name"
        rank=0
        total_nodes="$NNODES"
        for node in $selected_nodes; do
            run_in_background="yes"
            if (( rank == total_nodes - 1 )); then
                run_in_background="" # other nodes except last nodes should run in background, only last node should wait untill finish
            fi
            echo "Executing $full_script_path on $node with rank $rank and $NNODES nodes..."
            execute_script $node $full_script_path $CONDA_ENV_NAME $OUTPUT_LOG_PATH $rank $run_in_background
            ((rank++))
        done
    done
done

echo "Script execution completed on all nodes."

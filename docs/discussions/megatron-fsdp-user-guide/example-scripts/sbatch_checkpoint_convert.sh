#!/bin/bash

# Configuration: Set these paths before running the script
MEGATRON_PATH=${MEGATRON_PATH:-"your_own_megatron_path"} # Path to Megatron-LM repository
CONTAINER_IMAGE=${CONTAINER_IMAGE:-"your_own_container_image"} # Path to .sqsh or docker image url
OUTPUT_PATH=${OUTPUT_PATH:-"your_own_output_path"} # Path for SLURM logs

# Checkpoint conversion command
# Note: Update the checkpoint paths in the command below
RUN_CMD="
cd ${MEGATRON_PATH};
git rev-parse HEAD;
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH};
python3 tools/checkpoint/checkpoint_inspector.py \
    convert-torch-dist-to-fsdp-dtensor --swiglu \
    your_own_path_to_input_torch_dist_checkpoint \
    your_own_path_to_output_fsdp_dtensor_checkpoint \
    --param-to-param-group-map-json your_own_path_to_param_to_param_group_map.json"

# SLURM settings
SLURM_LOGS="${OUTPUT_PATH}/slurm_logs"
mkdir -p ${SLURM_LOGS} || {
    echo "Error: Failed to create SLURM logs directory ${SLURM_LOGS}"
    exit 1
}

# Submit SLURM job
# Note: Update SBATCH parameters below according to your cluster configuration
set +e
sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=your_own_job_name
#SBATCH --partition=your_own_partition
#SBATCH --nodes=your_own_num_nodes
#SBATCH --ntasks-per-node=your_own_tasks_per_node
#SBATCH --gres=gpu:your_own_gpu_per_node
#SBATCH --time=your_own_time
#SBATCH --account=your_own_account
#SBATCH --exclusive
#SBATCH --dependency=singleton

srun --mpi=pmix -l \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=your_own_container_mounts \
    --container-workdir=${MEGATRON_PATH} \
    bash -x -c "${RUN_CMD}" 2>&1 | tee ${SLURM_LOGS}/\${SLURM_JOB_ID}.log

EOF
set -e

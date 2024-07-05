#!/bin/bash

set -ux

#######################################################################################
#
# Script for capturing a reference model.
#
# It will train a model until a target iteration was hit.
#
#
########################################################################################

########################################################################################
# Please adjust to your needs:
########################################################################################

OVERRIDE_GOLDEN_VALUES=true
MODEL="<model>"
MCORE_RELEASE_NUM="<X.Y>"
DATA_PATH="<path-to-datastorage>" 
TRAINING_SCRIPT_PATH="<pretrain-script>.py"
TRAINING_PARAMS_PATH="./tests/functional_tests/model_configs/$MODEL/<training_config>.yaml"
TEST_PARAMS_PATH="./tests/functional_tests/test_configs/$MODEL/"
OUTPUT_PATH="<path-to-modelstorage>/mcore-v$MCORE_RELEASE_NUM/$MODEL" 
IMAGE_TAG="<...>" 
NODES="<...>"
PPP="<...>"
PARTITION="<...>"
ITERATIONS="<...>"
GITLAB_TOKEN="my-super-duper-token"  # Do not track in VCS
WAND_API_KEY="my-super-duper-key" # Do not track in VCS

########################################################################################
# Dont change below
########################################################################################

# Container settings
IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/mcore_ci:$IMAGE_TAG"
MOUNTS="${DATA_PATH}:${DATA_PATH},${OUTPUT_PATH}:${OUTPUT_PATH}"
ARGUMENTS=(
    "TRAINING_SCRIPT_PATH=${TRAINING_SCRIPT_PATH}"
    "TRAINING_PARAMS_PATH=${TRAINING_PARAMS_PATH}"
    "DATA_PATH=${DATA_PATH}"
    "OUTPUT_PATH=${OUTPUT_PATH}"
    "WAND_API_KEY=${WAND_API_KEY}"
)
SLURM_LOGS=$OUTPUT_PATH/slurm_logs/
mkdir -p $SLURM_LOGS

while : 
do
ACTUAL_ITERATIONS=$(cat "$OUTPUT_PATH/checkpoints/latest_checkpointed_iteration.txt" || 0)
if [[ $ACTUAL_ITERATIONS -gt $ITERATIONS ]]; then
    break
fi

# Fire of sbatch
sbatch -W <<EOF
#!/bin/bash

#SBATCH --nodes=$NODES
#SBATCH --account $PPP
#SBATCH --partition $PARTITION
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time "04:00:00"
#SBATCH --job-name=$PPP:mcore:release:$(uuidgen)
#SBATCH --dependency=singleton
#SBATCH --output=/dev/null 
#SBATCH --error=/dev/null
#SBATCH --exclusive

# Prepare SLURM job
echo "SLURM_JOB_ID=\$SLURM_JOB_ID" > "$SLURM_LOGS/\${SLURM_JOB_ID}.log"

srun \
    --ntasks-per-node=1 \
    --container-image=${IMAGE} \
    --container-mounts=${MOUNTS} \
    --container-workdir=/workspace/megatron-lm \
    bash ./tests/functional_tests/shell_test_utils/_run_local_training.sh ${ARGUMENTS[@]} >>"$SLURM_LOGS/\${SLURM_JOB_ID}.log" 2>&1
EOF

done

# Generate golden values
# This code will be added later
# export PYTHONPATH=$(pwd)
# export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
# LOG_INTERVAL=$(cat $TRAINING_PARAMS_PATH | yq '."--log-interval" // 1')
# GOLDEN_VALUES=$(python ./tests/functional_tests/python_test_utils/get_test_results_from_tensorboard_logs.py \
#     --logs-dir $OUTPUT_PATH/tensorboard \
#     --run-name "$MODEL")
# echo "$GOLDEN_VALUES" > "$OUTPUT/$MODEL.json"

# # Write golden values into repo if this run should become a reference
# if [[ $OVERRIDE_GOLDEN_VALUES == true ]]; then
#     echo "$GOLDEN_VALUES" > tests/functional_tests/test_results/release-$MCORE_RELEASE_NUM-$$MODEL.json
# fi

# Finally upload everything to JET
jet artifacts registry add \
    --token $GITLAB_TOKEN \
    --source-path $OUTPUT_PATH \
    "unverified/model/mcore-$MCORE_RELEASE_NUM/$MODEL" 
